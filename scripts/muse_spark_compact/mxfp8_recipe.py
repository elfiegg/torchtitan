"""Native MXFP8 with the 100-update phase-preserving Muse Spark C4 replay."""

from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path

from gb300_config import fit_host_memory
from torchtitan_recipes.models._muse_spark_indexed_replay import (
    IndexedPackedReplay,
    LANES,
    ORIGINAL_CAPACITY,
    PLAN_FORMAT,
    SOURCE_FORMAT,
    SOURCE_SEGMENTS_PER_ROW,
)
from torchtitan.components.data.collators import TextCollator
from torchtitan.models.common.dist_moe import _dist_moe as dist_moe, MXFP8DistMoeRoutedExperts
from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan_recipes.models import muse_spark


CAPACITY = 29_184
STEPS = 100
LAYERS = 62
LOSS_CHUNKS = 16
DEFAULT_SCRATCH_FACTOR = 1.0
EXPERT_WIDTH = 4_096
MINIMUM_REPLAY_SLOT_BYTES = CAPACITY * EXPERT_WIDTH * 2 * LAYERS
PINNED_SOURCE_COMMIT = "814df1c9e892f0e45bb2af5523c1968c52e7262b"
INPUT_PLAN = "48edfcbac44fbf3ffd4a68285f0a99029ce7275c5cbece738c37abe50b5aa945"
NATIVE_BENCHMARK = "muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark"
MXFP8_PROJECTIONS_PER_LAYER = (
    "attention.qkv_linear.wqkv", "attention.wo", "feed_forward.pooler", "feed_forward.unpooler",
)


def check_plan(source_dir, plan_dir):
    source_bytes = (Path(source_dir) / "manifest.json").read_bytes()
    source = json.loads(source_bytes)
    plan = json.loads((Path(plan_dir) / "manifest.json").read_text())
    if (
        source.get("format") != SOURCE_FORMAT
        or source.get("source_commit") != PINNED_SOURCE_COMMIT
        or plan.get("source_manifest_sha256") != sha256(source_bytes).hexdigest()
        or plan.get("format") != PLAN_FORMAT
        or plan.get("phase_preserved") is not True
        or plan.get("capacity") != CAPACITY
        or plan.get("updates") != STEPS
        or plan.get("input_plan_sha256") != INPUT_PLAN
    ):
        raise ValueError("Expected the pinned 100-update, phase-preserving 29,184-token replay plan")


def check_model(model):
    if len(model.layers) != LAYERS or isinstance(model.lm_head, MXFP8Linear.Config):
        raise RuntimeError("Expected 62 decoder layers and the native BF16 language-model head")
    for layer in model.layers:
        moe = layer.feed_forward
        attention = layer.attention
        experts = moe.routed_experts
        if not isinstance(moe.router, muse_spark.RoundRobinTokenChoiceTopKRouter.Config):
            raise RuntimeError("Expected the benchmark round-robin router on every decoder layer")
        if isinstance(moe.router.gate, MXFP8Linear.Config):
            raise RuntimeError("The MXFP8 projection converter changed the router gate")
        if not isinstance(experts, MXFP8DistMoeRoutedExperts.Config):
            raise RuntimeError("Expected native MXFP8 Dist-MoE experts on every decoder layer")
        policy = experts.block_scaled_config
        if (
            policy.format is not dist_moe.BlockScaledFormat.MXFP8_E4M3
            or not policy.fast_math or policy.pipeline != "staged"
            or not experts.inplace_wgrad_accum or experts.top_k != 8
            or experts.w13.in_features != EXPERT_WIDTH or experts.w2.in_features != EXPERT_WIDTH
        ):
            raise RuntimeError("The pinned MXFP8 expert precision or WGRAD policy changed")
        projections = (attention.qkv_linear.wqkv, attention.wo, moe.pooler, moe.unpooler)
        for path, projection in zip(MXFP8_PROJECTIONS_PER_LAYER, projections, strict=True):
            if not isinstance(projection, MXFP8Linear.Config) or (
                projection.in_features % 32 or projection.out_features % 32
                or projection.input_activation_format_for_backward != "bf16"
            ):
                raise RuntimeError(f"Expected the pinned native MXFP8 projection at {path}")


def build_repacked_config(source_dir, plan_dir, *, slot_gib=0, scratch_factor=DEFAULT_SCRATCH_FACTOR):
    check_plan(source_dir, plan_dir)
    if type(slot_gib) is not int or slot_gib < 0:
        raise ValueError("The requested activation slot GiB must be a nonnegative integer")
    if slot_gib and slot_gib * 2**30 < MINIMUM_REPLAY_SLOT_BYTES:
        raise ValueError("An explicit activation slot must cover all mandatory replay inputs")
    if not isfinite(scratch_factor) or scratch_factor < 1:
        raise ValueError("The MXFP8 scratch factor must cover balanced routing")

    config = muse_spark.muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark()
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
        expert_precision="mxfp8",
    )
    model = native.model
    muse_spark._use_forced_balanced_routing(model)
    check_model(model)
    attention_configs = [attention for _, attention, _, _ in model.traverse(muse_spark.VarlenInnerAttention.Config)]
    if len(attention_configs) != LAYERS:
        raise RuntimeError("Expected 62 CP1 varlen-attention configs")
    for attention in attention_configs:
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
        config.training, steps=STEPS,
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
        scratch_capacity_factor=scratch_factor,
        activation_slot_capacity_factor=None if slot_gib else 0.0,
        activation_slot_bytes=slot_gib * 2**30 if slot_gib else None,
    )

    config = replace(config)
    remat = config.activation_remat
    projections = set(muse_spark._MUSE_SPARK_MOE_PROJECTION_REGIONS)
    if not (
        config.model.max_context_length == CAPACITY
        and CAPACITY % config.loss.num_chunks == 0
        and CAPACITY // config.loss.num_chunks % 32 == 0
        and type(config.dataloader.collator) is TextCollator.Config
        and config.training.accelerator_graph_per_microbatch_group
        and not config.training.disable_accelerator_graphs
        and not config.parallelism.fsdp_defer_gradient_reduction
        and config.parallelism.fsdp_symm_mem_scope == "all"
        and not config.debug.spmd_typechecking
        and config.debug.seed == 42
        and config.optim.enable_accelerator_graph
        and config.checkpointer is None
        and remat.offload_block_input
        and projections.issubset(remat.save_regions)
        and projections.issubset(remat.offload_regions)
    ):
        raise RuntimeError("The pinned MXFP8 benchmark no longer meets the compact replay contract")
    return config


def mxfp8_repacked():
    if os.environ.get("WORLD_SIZE", str(LANES)) != str(LANES):
        raise ValueError("The phase-preserving MXFP8 replay requires exactly 64 ranks")
    norm_fusion = os.environ.get("EXP_NORM_FUSION", "0")
    if norm_fusion not in {"0", "cast"} or any(
        os.environ.get(name, "0") != "0"
        for name in ("EXP_SWIGLU_RECOMPUTE", "EXP_LEARNED_ROUTE", "EXP_MOE_H3_OFFLOAD", "EXP_SELECTIVE_HOST_H3")
    ):
        raise ValueError("The compact MXFP8 recipe supports cast-only norm fusion and native block-scaled experts")
    source_dir = os.environ["MUSE_REPACK_SOURCE"]
    plan_dir = os.environ["MUSE_REPACK_PLAN"]
    slot_gib = int(os.environ.get("MUSE_SLOT_GIB", "0"))
    scratch_factor = float(os.environ.get("MUSE_SCRATCH_FACTOR", str(DEFAULT_SCRATCH_FACTOR)))
    config = build_repacked_config(
        source_dir, plan_dir, slot_gib=slot_gib, scratch_factor=scratch_factor,
    )
    if os.environ.get("MUSE_PLATFORM") == "gb300":
        config = fit_host_memory(config)
    if os.environ.get("RANK") == "0" and (output_dir := os.environ.get("REPRO_OUTPUT")):
        fields = {
            "variant": os.environ.get("VARIANT", "compact-mxfp8"),
            "benchmark_recipe": NATIVE_BENCHMARK,
            "bf16_compute": False,
            "expert_compute": "mxfp8",
            "expert_policy": "staged_mxfp8_e4m3_fast_math",
            "mxfp8_projections_per_layer": MXFP8_PROJECTIONS_PER_LAYER,
            "inner_attention": "bf16_varlen_fa4",
            "language_model_head": "native_bf16",
            "source_commit": PINNED_SOURCE_COMMIT,
            "source_dir": source_dir,
            "plan_dir": plan_dir,
            "phase_preserved": True,
            "input_plan_sha256": INPUT_PLAN,
            "norm_fusion": norm_fusion,
            "physical_tokens_per_gpu": CAPACITY,
            "physical_global_tokens": config.training.num_tokens_per_train_step,
            "workload_equivalent_tokens_per_gpu": ORIGINAL_CAPACITY,
            "activation_slot_gib": slot_gib,
            "minimum_replay_slot_bytes": MINIMUM_REPLAY_SLOT_BYTES,
            "scratch_capacity_factor": scratch_factor,
            "parallelism": repr(config.parallelism),
            "dist_moe": repr(config.dist_moe_runtime),
            "precision": repr(config.precision),
            "optimizer": repr(config.optim),
            "loss": repr(config.loss),
            "remat": repr(config.activation_remat),
        }
        (Path(output_dir) / "effective-config.json").write_text(json.dumps(fields, indent=2) + "\n")
    return config
