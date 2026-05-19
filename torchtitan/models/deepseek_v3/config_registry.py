# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 named-config registry.

Exposes:

  - ``deepseek_v3_debugmodel*``        : tiny configs for unit/smoke tests
    (untouched from upstream).
  - ``deepseek_v3_16b``                : 16B reference config (untouched).
  - ``deepseek_v3_671b``               : upstream 671B (FP8 + flex attention)
                                          baseline, unchanged.
  - ``deepseek_v3_671b_<backend>_<attn>[_bf16]`` : LLMB benchmarking
    variants, where:
       <backend> in {deepep, hybridep}
       <attn>    in {sdpa, flex, cudnn}
       suffix    "" = FP8 (MXFP8), "_bf16" = BF16

These factory variants are what the LLMB ``launch.sh`` selects via the
unified ``--module deepseek_v3 --config <name>`` CLI for H100, B200, B300,
GB200, and GB300 runs.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry


def deepseek_v3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def deepseek_v3_debugmodel_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel")
    return config


def deepseek_v3_debugmodel_flex_attn() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    return config


def deepseek_v3_debugmodel_flex_attn_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    return config


def deepseek_v3_16b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_registry("16B", attn_backend="flex"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b() -> Trainer.Config:
    """Upstream 671B baseline (FP8 Float8 + flex attention)."""
    compile_config = CompileConfig(enable=True, components=["loss"])
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_registry(
            "671B",
            attn_backend="flex",
            converters=[
                Float8LinearConverter.Config(
                    filter_fqns=["output", "router.gate"],
                    model_compile_enabled=model_compile_enabled,
                ),
                Float8GroupedExpertsConverter.Config(
                    model_compile_enabled=model_compile_enabled
                ),
            ],
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=compile_config,
    )


# ---------------------------------------------------------------------------
# LLMB benchmarking variants
# ---------------------------------------------------------------------------
#
# `_deepseek_v3_671b_with_backends` is the factory used by all of the
# `deepseek_v3_671b_<backend>_<attn>[_bf16]` registry entries.
#
# Differences vs. upstream `deepseek_v3_671b`:
#
#   - `attn_backend`        : parameterised over {sdpa, flex, cudnn}.
#                             cuDNN attention (single-backend SDPA pinned to
#                             CUDNN) is the recommended path on Blackwell.
#   - `moe_comm_backend`    : parameterised over {deepep, hybridep}.
#                             - deepep   : DeepEP-main (NCCL Gin) - the
#                                          portable path for H100/B200/B300.
#                             - hybridep : DeepEP hybrid-ep branch (MNNVL
#                                          fabric) - only works on full
#                                          NVL72 domains, used for GB200/
#                                          GB300.
#   - `precision`           : "fp8" -> MXFP8 (Linear + GroupedExperts) using
#                             the mxfp8_rceil recipe (Blackwell native).
#                             "bf16" -> no converters.
#   - `local_batch_size=8`  : larger micro-batch to soak up Blackwell HBM.
#                             The LLMB launcher overrides this via
#                             `--training.local_batch_size`, so this is just
#                             the default.
#   - `compile = ["loss", "model"]` : model compilation is essential to
#                             realise the MXFP8 speedup; upstream now
#                             handles fullgraph + activation checkpointing
#                             cleanly via
#                             `torch._dynamo.config.skip_fwd_side_effects_
#                             in_bwd_under_checkpoint = True` in
#                             distributed/compile.py.
#   - `activation_checkpoint.mode="full"` : selective AC OOMs at lbs=8 on
#                             B300/GB300 with 256 GPUs; full AC is the
#                             memory-safe default for the benchmark.
#
# The DeepEP-main + FP8 combination additionally requires
# `TT_DEEPEP_ALLOW_UNPADDED_FP8=1` (set in the LLMB launcher), since
# DeepEP-main lacks a post-dispatch padding hook.

_PRECISION_TO_CONVERTERS = {
    # MXFP8 on Blackwell:
    #   - MXFP8LinearConverter (no filter_fqns: convert all eligible Linears)
    #   - MXFP8GroupedExpertsConverter (group GEMM in MXFP8 too)
    # Both use the mxfp8_rceil recipe, which is the Blackwell-native
    # block-scaled FP8 with rounded-ceil scaling factors.
    "fp8": "mxfp8",
    "bf16": None,
}


def _deepseek_v3_671b_with_backends(
    *,
    precision: str = "fp8",
    attn_backend: str = "cudnn",
    moe_comm_backend: str = "deepep",
) -> Trainer.Config:
    if precision not in _PRECISION_TO_CONVERTERS:
        raise ValueError(
            f"Unsupported precision={precision!r}; expected one of "
            f"{sorted(_PRECISION_TO_CONVERTERS)}."
        )

    compile_config = CompileConfig(enable=True, components=["loss", "model"])
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if precision == "fp8":
        quant_configs = [
            MXFP8LinearConverter.Config(
                recipe_name="mxfp8_rceil",
                model_compile_enabled=model_compile_enabled,
            ),
            MXFP8GroupedExpertsConverter.Config(
                recipe_name="mxfp8_rceil",
                model_compile_enabled=model_compile_enabled,
            ),
        ]
    else:  # bf16
        quant_configs = None

    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_registry(
            "671B",
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            converters=quant_configs,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        compile=compile_config,
    )


# DeepEP-main variants (H100, B200, B300, and GB200/GB300 fallback)
def deepseek_v3_671b_deepep_sdpa() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="fp8", attn_backend="sdpa", moe_comm_backend="deepep"
    )


def deepseek_v3_671b_deepep_sdpa_bf16() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="bf16", attn_backend="sdpa", moe_comm_backend="deepep"
    )


def deepseek_v3_671b_deepep_flex() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="fp8", attn_backend="flex", moe_comm_backend="deepep"
    )


def deepseek_v3_671b_deepep_flex_bf16() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="bf16", attn_backend="flex", moe_comm_backend="deepep"
    )


def deepseek_v3_671b_deepep_cudnn() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="fp8", attn_backend="cudnn", moe_comm_backend="deepep"
    )


def deepseek_v3_671b_deepep_cudnn_bf16() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="bf16", attn_backend="cudnn", moe_comm_backend="deepep"
    )


# HybridEP variants (GB200/GB300 full NVL72)
def deepseek_v3_671b_hybridep_sdpa() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="fp8", attn_backend="sdpa", moe_comm_backend="hybridep"
    )


def deepseek_v3_671b_hybridep_sdpa_bf16() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="bf16", attn_backend="sdpa", moe_comm_backend="hybridep"
    )


def deepseek_v3_671b_hybridep_flex() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="fp8", attn_backend="flex", moe_comm_backend="hybridep"
    )


def deepseek_v3_671b_hybridep_flex_bf16() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="bf16", attn_backend="flex", moe_comm_backend="hybridep"
    )


def deepseek_v3_671b_hybridep_cudnn() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="fp8", attn_backend="cudnn", moe_comm_backend="hybridep"
    )


def deepseek_v3_671b_hybridep_cudnn_bf16() -> Trainer.Config:
    return _deepseek_v3_671b_with_backends(
        precision="bf16", attn_backend="cudnn", moe_comm_backend="hybridep"
    )
