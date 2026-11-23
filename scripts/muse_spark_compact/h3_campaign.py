"""Immutable experiment identity and receipt validation for the compact H3 pair."""

from hashlib import sha256
import os
from pathlib import Path
import socket
import time


CONTROL = "repacked29184route49full"
CANDIDATE = "repacked29184routeh333full"
SELECTED_LAYERS = (5, 6, 7, 10, 11, 14, 15, 17, 18, 20, 22, 24, 27, 28, 29, 31,
                   33, 34, 36, 38, 39, 41, 44, 45, 47, 49, 51, 52, 54, 55, 58, 59, 60)
TOKENS = 29_184
WIDTH = 4_096
TOPK = 8
LAYERS = 62
RANKS = 64
STEPS = 100
SNAPSHOT_BYTES = TOKENS * WIDTH * TOPK * 2
HOST_PHASES = ("after_warmup_update_1", "after_first_captured_update_3", "before_trainer_close")
POOL_STATS = ("allocated_bytes.current", "allocated_bytes.peak", "active_bytes.current", "active_bytes.peak")
TORCH_GIT = "daf1a40a5de981f7cab47fed4dd9883b1581cd63"
DIST_MOE_API_SHA = "e8def4834e7cc57c2ff54f052dcc58b14e04dc8ae5b9b74281860a1f04ec3497"
TORCHTITAN_OFFLOAD_SHA = "296fecd743925dbf33f5ab97516e5cd21a5efc47120bf61299e8da9d7d0b880f"


def host_h3_fields(variant):
    if variant not in (CONTROL, CANDIDATE):
        raise ValueError(f"Unexpected compact H3 variant {variant!r}")
    enabled = variant == CANDIDATE
    return {
        "enabled": enabled,
        "exact_host_pool": enabled,
        "selected_zero_based_layers": list(SELECTED_LAYERS) if enabled else [],
        "warmup_replay_parity_layer": SELECTED_LAYERS[0] if enabled else None,
    }


def check_environment(env=os.environ):
    fields = host_h3_fields(env["VARIANT"])
    expected = {
        "WORLD_SIZE": str(RANKS), "LOCAL_WORLD_SIZE": "4", "REPRO_STOP_AFTER": str(STEPS),
        "EXP_NORM_FUSION": "cast", "EXP_SWIGLU_RECOMPUTE": "1", "EXP_LEARNED_ROUTE": "1",
        "EXP_SELECTIVE_HOST_H3": str(int(fields["enabled"])), "EXP_MOE_H3_OFFLOAD": "0",
        "MUSE_CP": "1", "MUSE_TP": "1", "MUSE_EP": "64", "MUSE_SLOT_GIB": "49",
        "MUSE_LOSS_CHUNKS": "16", "MUSE_SAVE_PROJ": "1", "MUSE_PROFILE": "0",
        "MUSE_SEPARATE_RS": "0", "MUSE_UNSHARD_CURRENT": "0", "MUSE_EXPERT_PRESET": "",
        "NCCL_NVLS_ENABLE": "1", "PYTORCH_ALLOC_CONF": "pinned_max_round_threshold_mb:1",
    }
    differing = {name: env.get(name) for name, value in expected.items() if env.get(name) != value}
    if differing:
        raise RuntimeError(f"The matched compact H3 environment differs at {differing}")
    return fields


def check_source(module, expected):
    actual = sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{module.__name__} source pin changed: {actual}")


def install(variant, torch):
    from dist_moe import api
    from dist_moe._buffers import is_fake_symmetric_memory
    from torchtitan.distributed import activation_offload

    if torch.__version__ != "2.16.0.dev20261119+cu134" or torch.version.git_version != TORCH_GIT:
        raise RuntimeError(f"Pinned CUDA PyTorch changed: {torch.__version__} {torch.version.git_version}")
    check_source(api, DIST_MOE_API_SHA)
    check_source(activation_offload, TORCHTITAN_OFFLOAD_SHA)
    if not host_h3_fields(variant)["enabled"]:
        return None

    from torch_h3_runtime import install_with_exact_host_pool

    return install_with_exact_host_pool(
        api, torch, activation_offload, SELECTED_LAYERS, is_fake_symmetric_memory, validate_replay=True,
    )


def collect_host_phase(torch, binder, phase, step, rank, local_rank):
    from host_allocation_receipt import snapshot

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Host allocator snapshots must run outside CUDA graph capture")
    hosts = ()
    if binder is not None:
        (_, _, schedule), = binder.states.values()
        hosts = tuple(schedule.host_buffers.values())
    return {
        "phase": phase, "step": step, "rank": rank, "local_rank": local_rank,
        "host": socket.gethostname(), "utc_timestamp_ns": time.time_ns(), **snapshot(torch, hosts),
    }


def check_host_snapshots(samples, rank, enabled):
    if [sample["phase"] for sample in samples] != list(HOST_PHASES):
        raise ValueError(f"Rank {rank} is missing an untimed host allocator phase")
    hosts = {sample["host"] for sample in samples}
    if len(hosts) != 1 or not next(iter(hosts)):
        raise ValueError(f"Rank {rank} host identity changed")
    count = len(SELECTED_LAYERS) if enabled else 0
    for sample, step in zip(samples, (1, 3, STEPS), strict=True):
        if (sample["rank"], sample["local_rank"], sample["step"]) != (rank, rank % 4, step):
            raise ValueError(f"Rank {rank} host snapshot identity changed")
        if (sample["snapshot_count"], sample["snapshot_payload_bytes"], sample["snapshots_report_pinned"]) != (
            count, count * SNAPSHOT_BYTES, True if count else None,
        ):
            raise ValueError(f"Rank {rank} does not own the exact expected pinned H3 payload")
        pool = sample["pinned_allocator_bytes"]
        if any(not isinstance(pool.get(key), int) or pool[key] <= 0 for key in POOL_STATS):
            raise ValueError(f"Rank {rank} is missing pinned allocator byte counters")
        if pool["allocated_bytes.current"] < count * SNAPSHOT_BYTES:
            raise ValueError(f"Rank {rank} reported fewer pinned bytes than its H3 payload")


def check_completion(receipt, rank, enabled):
    if receipt.get("torch_git_version") != TORCH_GIT:
        raise ValueError(f"Rank {rank} has unexpected PyTorch pin")
    route = receipt["learned_route"]
    if (route["forward_norms"], route["backward_norms"]) != (LAYERS, LAYERS):
        raise ValueError(f"Rank {rank} did not cover all learned-route layers")
    if min(route["forward_calls"], route["backward_calls"]) < LAYERS:
        raise ValueError(f"Rank {rank} did not construct any complete learned-route schedule")
    if receipt.get("training_accelerator_graph") is not True:
        raise ValueError(f"Rank {rank} did not configure the training CUDA graph")
    h3 = receipt.get("host_h3")
    if not enabled:
        if h3 is not None:
            raise ValueError(f"Control rank {rank} unexpectedly installed H3")
        return
    required = {
        "layers": LAYERS, "selected_zero_based_layers": list(SELECTED_LAYERS),
        "host_snapshots_per_schedule": len(SELECTED_LAYERS),
        "prefetched_restores_per_training_schedule": len(SELECTED_LAYERS),
        "snapshot_bytes": SNAPSHOT_BYTES, "reserved_host_bytes": len(SELECTED_LAYERS) * SNAPSHOT_BYTES,
    }
    if h3 is None or any(h3.get(key) != value for key, value in required.items()):
        raise ValueError(f"Rank {rank} did not construct the complete selective H3 schedule")
    if not 1 < h3.get("constructed_schedules", 0) < STEPS:
        raise ValueError(f"Rank {rank} has no evidence of CUDA graph reuse")
    parity = h3["warmup_replay_parity"]
    routes = parity["valid_routes"]
    if (parity["layer"] != SELECTED_LAYERS[0] or not 0 < routes <= TOKENS * TOPK
            or routes % TOPK or parity["matched_words"] != routes * WIDTH):
        raise ValueError(f"Rank {rank} is missing its bitwise valid-H3 warmup parity receipt")
