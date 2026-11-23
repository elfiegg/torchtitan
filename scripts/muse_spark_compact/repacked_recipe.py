"""Opt-in TorchTitan recipe for the phase-preserving Muse Spark C4 replay."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

from h3_campaign import host_h3_fields
from gb300_config import fit_host_memory
from indexed_packed_replay import (
    IndexedPackedReplay,
    LANES,
    ORIGINAL_CAPACITY,
    PLAN_FORMAT,
    SOURCE_SEGMENTS_PER_ROW,
)
from torchtitan.components.data.collators import TextCollator
from torchtitan_recipes.models import muse_spark


CAPACITY = 29_184
STEPS = 100
LOSS_CHUNKS = 16
INPUT_PLAN = "48edfcbac44fbf3ffd4a68285f0a99029ce7275c5cbece738c37abe50b5aa945"
PINNED_SOURCE_COMMIT = "814df1c9e892f0e45bb2af5523c1968c52e7262b"


def build_repacked_config(source_dir, plan_dir, *, slot_gib=49, save_projections=True):
    """Use the normal benchmark controls and rebuild the smaller native CP1 model."""
    source_bytes = (Path(source_dir) / "manifest.json").read_bytes()
    source = json.loads(source_bytes)
    plan = json.loads((Path(plan_dir) / "manifest.json").read_text())
    if (
        source.get("source_commit") != PINNED_SOURCE_COMMIT
        or plan.get("source_manifest_sha256") != sha256(source_bytes).hexdigest()
        or plan.get("format") != PLAN_FORMAT
        or plan.get("phase_preserved") is not True
        or plan.get("capacity") != CAPACITY
        or plan.get("updates") != STEPS
        or plan.get("input_plan_sha256") != INPUT_PLAN
    ):
        raise ValueError("Expected the 100-update, phase-preserving 29,184-token replay plan")
    if slot_gib <= 0:
        raise ValueError("The Dist-MoE activation slot must be positive")

    config = muse_spark.muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark()
    native = muse_spark._muse_spark_816b_dist_moe_spmd(
        seq_len=CAPACITY,
        global_batch_size=LANES,
        data_parallel_shard_degree=LANES,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=LANES,
        fake_backend=False,
        attn_backend="varlen",
        max_num_documents=SOURCE_SEGMENTS_PER_ROW,
        expert_precision="bf16",
    )
    model = native.model
    muse_spark._use_forced_balanced_routing(model)
    if not all(
        isinstance(layer.feed_forward.router, muse_spark.RoundRobinTokenChoiceTopKRouter.Config)
        for layer in model.layers
    ):
        raise RuntimeError("Every decoder layer must use the pinned round-robin router")
    for _, attention, _, _ in model.traverse(muse_spark.VarlenInnerAttention.Config):
        attention.sdpa_backends = ["FLASH_ATTENTION"]
    config.model = model

    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=LANES,
        tensor_parallel_degree=1,
        enable_sequence_parallel=False,
        context_parallel_degree=1,
        context_parallel_load_balancer=None,
        expert_parallel_degree=LANES,
        pipeline_parallel_degree=1,
        num_pp_microbatches=1,
        fsdp_reshard_after_forward="default",
    )
    config.training = replace(
        config.training,
        steps=STEPS,
        num_tokens_per_microbatch_per_dp_rank=CAPACITY,
        num_tokens_per_train_step=LANES * CAPACITY,
    )
    config.dataloader = replace(
        config.dataloader,
        dataset=IndexedPackedReplay(source_dir=str(source_dir), plan_dir=str(plan_dir)),
        max_num_documents=SOURCE_SEGMENTS_PER_ROW,
        per_sample_pad_multiple=1,
        shuffle=False,
        repeat=True,
    )
    config.precision = replace(
        config.precision, dtype="float32", param_dtype="bfloat16", reduce_dtype="bfloat16"
    )
    config.optim.optimizer.optimizers = [
        replace(optimizer, moment_dtype="bfloat16", pinned_host_moment_fraction=0.0, fused=True)
        for optimizer in config.optim.optimizer.optimizers
    ]
    config.loss = replace(config.loss, num_chunks=LOSS_CHUNKS)
    config.dist_moe_runtime = replace(
        config.dist_moe_runtime,
        scratch_capacity_factor=1.0,
        activation_slot_capacity_factor=None,
        activation_slot_bytes=slot_gib * 2**30,
    )
    if save_projections:
        policy = config.activation_remat
        projections = muse_spark._MUSE_SPARK_MOE_PROJECTION_REGIONS
        config.activation_remat = replace(
            policy,
            save_regions=[*policy.save_regions, *projections],
            offload_regions=[*policy.offload_regions, *projections],
        )

    config = replace(config)
    if not (
        config.model.max_context_length == CAPACITY
        and CAPACITY % config.loss.num_chunks == 0
        and CAPACITY % 32 == 0
        and type(config.dataloader.collator) is TextCollator.Config
        and config.training.accelerator_graph_per_microbatch_group
        and not config.training.disable_accelerator_graphs
        and not config.parallelism.fsdp_defer_gradient_reduction
        and not config.debug.spmd_typechecking
        and config.optim.enable_accelerator_graph
        and config.checkpointer is None
    ):
        raise RuntimeError("The pinned benchmark no longer meets the replay shape or graph contract")
    return config


def bf16_repacked():
    """Recipe entry point for the existing campaign launcher."""
    if os.environ.get("WORLD_SIZE", str(LANES)) != str(LANES):
        raise ValueError("The phase-preserving replay requires exactly 64 ranks")
    source_dir = os.environ["MUSE_REPACK_SOURCE"]
    plan_dir = os.environ["MUSE_REPACK_PLAN"]
    config = build_repacked_config(
        source_dir,
        plan_dir,
        slot_gib=int(os.environ.get("MUSE_SLOT_GIB", "49")),
        save_projections=os.environ.get("MUSE_SAVE_PROJ", "1") == "1",
    )
    if os.environ.get("MUSE_PLATFORM") == "gb300":
        config = fit_host_memory(config)
    if os.environ.get("RANK") == "0" and (output_dir := os.environ.get("REPRO_OUTPUT")):
        fields = {
            "variant": os.environ.get("VARIANT", "phase-preserving-repack"),
            "bf16_compute": True,
            "source_dir": source_dir,
            "plan_dir": plan_dir,
            "phase_preserved": True,
            "input_plan_sha256": json.loads((Path(plan_dir) / "manifest.json").read_text())["input_plan_sha256"],
            "norm_fusion": os.environ.get("EXP_NORM_FUSION", "0"),
            "swiglu_recompute": os.environ.get("EXP_SWIGLU_RECOMPUTE", "0"),
            "learned_route": os.environ.get("EXP_LEARNED_ROUTE", "0"),
            "host_h3": host_h3_fields(os.environ["VARIANT"]),
            "physical_tokens_per_gpu": CAPACITY,
            "physical_global_tokens": config.training.num_tokens_per_train_step,
            "workload_equivalent_tokens_per_gpu": ORIGINAL_CAPACITY,
            "parallelism": repr(config.parallelism),
            "dist_moe": repr(config.dist_moe_runtime),
            "precision": repr(config.precision),
            "optimizer": repr(config.optim),
            "loss": repr(config.loss),
            "remat": repr(config.activation_remat),
        }
        (Path(output_dir) / "effective-config.json").write_text(json.dumps(fields, indent=2))
    return config
