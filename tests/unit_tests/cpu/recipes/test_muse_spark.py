# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from hashlib import sha256
from types import ModuleType
from unittest.mock import patch

import pytest
from torchtitan.config import ConfigLoader
from torchtitan.models.common.attention.cp_attention import (
    UlyssesSlidingWindowVarlenInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.models.muse_spark import MUSE_SPARK_SPECIAL_TOKENS
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer

from torchtitan_recipes.models import (
    _muse_spark_indexed_replay as replay,
    muse_spark as recipes,
)
from torchtitan_recipes.tests.models.muse_spark import (
    muse_spark_debugmodel,
    muse_spark_debugmodel_mm,
)


@pytest.mark.parametrize(
    "recipe",
    [
        recipes.muse_spark_816b_a42b_dist_moe_bf16_spmd_64_fake,
        recipes.muse_spark_816b_a42b_dist_moe_bf16_spmd_256_fake,
        recipes.muse_spark_816b_a42b_dist_moe_bf16_spmd_512_fake,
    ],
)
def test_fake_scale_recipe_tokenizer(recipe):
    # Keep Dist-MoE kernels and CUDA-graph checks outside this tokenizer test.
    def configure_dist_moe(config, **kwargs):
        config.training.disable_accelerator_graphs = False
        return config

    with (
        patch.object(recipes, "_with_dist_moe", side_effect=configure_dist_moe),
        patch(
            "torchtitan.distributed.accelerator_graph.accelerator_graphs_supported",
            return_value=False,
        ),
    ):
        config = recipe()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")


def test_production_recipe_keeps_multimodal_tokenizer():
    config = recipes.muse_spark_816b_a42b()
    assert isinstance(config.tokenizer, MuseSparkTokenizer.Config)
    for field, token in MUSE_SPARK_SPECIAL_TOKENS.items():
        assert getattr(config.tokenizer, field) == token


def test_cp16_lora_recipe_uses_ulysses_varlen():
    # Building raises if the Ulysses CP map misses a varlen layer type.
    config = recipes.muse_spark_816b_a42b_ultradata_lora_sft_cp16_128k()
    for layer in config.model.layers:
        assert isinstance(
            layer.attention.inner_attention,
            (
                UlyssesVarlenInnerAttention.Config,
                UlyssesSlidingWindowVarlenInnerAttention.Config,
            ),
        )
        # Building raises if sdpa_backends holds anything but SDPBackend names.
        layer.attention.inner_attention.build()


def test_dist_moe_64_mxfp8_recipe_quantizes_experts_and_projections():
    pytest.importorskip("dist_moe")
    from torchtitan.config.transform.quantization import MXFP8Linear
    from torchtitan.models.common.dist_moe.mxfp8 import MXFP8DistMoeRoutedExperts

    # The MXFP8 converter checks the local GPU while the config is built.
    with patch(
        "torchtitan.config.transform.quantization.has_cuda_capability",
        return_value=True,
    ):
        config = recipes.muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_fake()
    for layer in config.model.layers:
        feed_forward = layer.feed_forward
        assert isinstance(feed_forward.routed_experts, MXFP8DistMoeRoutedExperts.Config)
        assert feed_forward.routed_experts.block_scaled_config.fast_math
        assert isinstance(feed_forward.pooler, MXFP8Linear.Config)
        assert isinstance(feed_forward.unpooler, MXFP8Linear.Config)
        assert isinstance(layer.attention.wo, MXFP8Linear.Config)


def test_compact_bf16_benchmark_config_loader(tmp_path, monkeypatch):
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
    from torchtitan.models.common.moe import RoundRobinTokenChoiceTopKRouter

    source_dir, plan_dir = tmp_path / "source", tmp_path / "plan"
    source_dir.mkdir()
    plan_dir.mkdir()
    source = json.dumps(
        {"format": replay.SOURCE_FORMAT, "source_commit": replay.SOURCE_COMMIT}
    )
    (source_dir / "manifest.json").write_text(source)
    (plan_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": replay.PLAN_FORMAT,
                "source_manifest_sha256": sha256(source.encode()).hexdigest(),
                "input_plan_sha256": replay.INPUT_PLAN_SHA256,
                "phase_preserved": True,
                "capacity": 29_184,
                "updates": 100,
            }
        )
    )
    monkeypatch.setenv("MUSE_REPACK_SOURCE", str(source_dir))
    monkeypatch.setenv("MUSE_REPACK_PLAN", str(plan_dir))
    monkeypatch.setenv("WORLD_SIZE", "64")

    args = [
        "--module",
        "torchtitan_recipes.models.muse_spark",
        "--config",
        "muse_spark_816b_a42b_dist_moe_bf16_spmd_64_compact_benchmark",
    ]
    # The BF16 config transform needs the optional package to be importable.
    # Instantiating its kernels happens later, when the trainer builds the model.
    with (
        monkeypatch.context() as runtime,
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(10, 3)),
    ):
        runtime.setitem(sys.modules, "dist_moe", ModuleType("dist_moe"))
        config = ConfigLoader().load(args)

    assert config.model.max_context_length == 29_184
    assert config.training.steps == 100
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 29_184
    assert config.training.num_tokens_per_train_step == 64 * 29_184
    assert config.parallelism.data_parallel_shard_degree == 64
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 64
    assert config.parallelism.fsdp_symm_mem_scope == "all"
    assert config.loss.num_chunks == 16
    assert config.precision.reduce_dtype == "bfloat16"
    assert all(
        opt.moment_dtype == "bfloat16" for opt in config.optim.optimizer.optimizers
    )
    assert config.dist_moe_runtime.activation_slot_bytes == 49 * 2**30
    assert config.dist_moe_runtime.scratch_capacity_factor == 1.0
    assert "feed_forward.unpooler.*" in config.activation_remat.offload_regions
    assert isinstance(config.dataloader.dataset, replay.IndexedPackedReplay)
    assert all(
        isinstance(layer.feed_forward.router, RoundRobinTokenChoiceTopKRouter.Config)
        and isinstance(layer.feed_forward.routed_experts, DistMoeRoutedExperts.Config)
        for layer in config.model.layers
    )
    assert not config.override.imports


@pytest.mark.parametrize("recipe", [muse_spark_debugmodel, muse_spark_debugmodel_mm])
def test_debug_recipe_tokenizer(recipe):
    config = recipe()
    config.__post_init__()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")
