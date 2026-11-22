# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark 816B-A42B BF16 64-GPU benchmark with the measured optimizations.

``bf16_opt`` starts from ``muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark``
and applies the options selected by ``EXP_*`` environment variables, so one
Slurm allocation can run several variants back to back without editing code
(see ``run_bench.sbatch``). With no ``EXP_*`` variable set it is the measured
baseline.

Torchtitan options (read here):

  EXP_NORM_FUSION=cast    fuse the pre-norm bf16 cast into the compiled norm
  EXP_MOE_POSTNORM=1      fused Triton Dist-MoE post-expert norm
  EXP_PAD_SEGMENT=<S>     split the row-tail padding into <= 16 varlen segments
                          of a multiple of S tokens (S = 128 was measured)
  EXP_RESHARD=always      FSDP reshard after forward (default "never")
  DISTMOE_SLOT_GIB=<n>    Dist-MoE saved-activation slot in GiB (default 16)
  EXP_SAVE_MOE_PROJ=1     save and offload the MoE pooler/unpooler/router
  EXP_CP_LB=headtail      head-tail context-parallel load balancer
  EXP_PROFILE_VARIANTS    comma-separated VARIANT names to profile (ranks 0/1/4/63)
  EXP_FAKE_BACKEND=1      build with FakePG to validate the config without GPUs

Dist-MoE runtime patches (read by ``torchtitan.experiments.dist_moe_patches``):

  EXP_SWIGLU_RECOMPUTE=1  EXP_MOE_H3_OFFLOAD=1  EXP_MOE_TILES=<spec.json>
"""

import json
import os
from dataclasses import replace
from pathlib import Path

from torchtitan_recipes.models import muse_spark

from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.models.common.attention.attention import PaddingSegmentsConfig


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


def bf16_opt():
    rank = int(os.environ.get("RANK", "0"))
    variant = os.environ.get("VARIANT", "")
    slot_gib = int(os.environ.get("DISTMOE_SLOT_GIB", "16"))
    reshard = os.environ.get("EXP_RESHARD", "never")
    cp_lb = os.environ.get("EXP_CP_LB", "none")
    norm_fusion = os.environ.get("EXP_NORM_FUSION", "0")
    pad_segment = int(os.environ.get("EXP_PAD_SEGMENT", "0"))
    profile = variant in os.environ.get("EXP_PROFILE_VARIANTS", "").split(",")

    # muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark(), plus the padding split.
    config = muse_spark._muse_spark_816b_dist_moe_spmd_64_benchmark(
        fake_backend=_flag("EXP_FAKE_BACKEND"),
        expert_precision="bf16",
        padding_segments=(
            PaddingSegmentsConfig(segment_length=pad_segment) if pad_segment else None
        ),
    )

    # Measured baseline: FP32 master weights, BF16 compute, BF16 gradient reduction,
    # BF16 Adam moments.
    config.precision = replace(
        config.precision,
        dtype="float32",
        param_dtype="bfloat16",
        reduce_dtype="bfloat16",
    )
    config.optim.optimizer.optimizers = [
        replace(
            opt, moment_dtype="bfloat16", pinned_host_moment_fraction=0.0, fused=True
        )
        for opt in config.optim.optimizer.optimizers
    ]
    config.dist_moe_runtime = replace(
        config.dist_moe_runtime,
        scratch_capacity_factor=1.0,
        activation_slot_capacity_factor=None,
        activation_slot_bytes=slot_gib * 2**30,
    )
    config.parallelism.fsdp_reshard_after_forward = reshard

    if _flag("EXP_SAVE_MOE_PROJ"):
        regions = list(muse_spark._MUSE_SPARK_MOE_PROJECTION_REGIONS)
        remat = config.activation_remat
        config.activation_remat = replace(
            remat,
            save_regions=[*remat.save_regions, *regions],
            offload_regions=[*remat.offload_regions, *regions],
        )
    if cp_lb == "headtail":
        config.parallelism.context_parallel_load_balancer = (
            HeadTailCPLoadBalancer.Config()
        )
    elif cp_lb != "none":
        raise ValueError(f"EXP_CP_LB={cp_lb!r}")

    if norm_fusion == "cast":
        muse_spark.use_fused_pre_norm_cast(config.model)
    elif norm_fusion != "0":
        raise ValueError(f"EXP_NORM_FUSION={norm_fusion!r}")
    if _flag("EXP_MOE_POSTNORM"):
        muse_spark.use_fused_dist_moe_post_expert_norm(config.model)

    config.profiler = replace(
        config.profiler,
        enable_profiling=profile and rank in {0, 1, 4, 63},
        profile_freq=14,
        profiler_warmup=2,
        profiler_active=2,
        profiler_repeat=1,
        profiler_skip_first=0,
        enable_memory_snapshot=False,
        cuspy=None,
    )
    config = replace(config)
    if not _flag("EXP_FAKE_BACKEND"):
        # Keep the measured recipe's LR/optimizer horizon and batch.
        assert config.training.steps == 100
        assert config.training.num_tokens_per_train_step == 2097152
        assert config.optim.enable_accelerator_graph

    if rank == 0 and "REPRO_OUTPUT" in os.environ:
        facts = {
            "variant": variant,
            "exp_env": {k: v for k, v in os.environ.items() if k.startswith("EXP_")},
            "activation_slot_gib": slot_gib,
            "reshard_after_forward": reshard,
            "cp_load_balancer": cp_lb,
            "profile": profile,
            "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
            "precision": repr(config.precision),
            "optimizer": repr(config.optim),
            "parallelism": repr(config.parallelism),
            "remat": repr(config.activation_remat),
            "dist_moe_runtime": repr(config.dist_moe_runtime),
        }
        (Path(os.environ["REPRO_OUTPUT"]) / "effective-config.json").write_text(
            json.dumps(facts, indent=2)
        )
    return config
