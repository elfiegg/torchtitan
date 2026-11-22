# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torchtitan.models.common.attention.cp_attention import (
    UlyssesSlidingWindowVarlenInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.models.muse_spark import build_model_config, MUSE_SPARK_SPECIAL_TOKENS
from torchtitan.models.muse_spark.model import MuseSparkRMSNorm
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer

from torchtitan_recipes.models import muse_spark as recipes
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


def test_use_fused_pre_norm_cast():
    default = build_model_config("debugmodel")
    fused = build_model_config("debugmodel")
    recipes.use_fused_pre_norm_cast(fused)

    # Only the two pre-norms of each layer round to branch_dtype; every other norm,
    # and every norm of the default config, keeps its input dtype.
    fused_pre_norms = {
        f"layers.{layer_id}.{name}"
        for layer_id in range(len(fused.layers))
        for name in ("attention_norm", "ffn_norm")
    }
    for model_config, pre_norms in ((default, set()), (fused, fused_pre_norms)):
        for fqn, norm, _, _ in model_config.traverse(
            MuseSparkRMSNorm.Config, recurse=True
        ):
            expected = torch.bfloat16 if fqn in pre_norms else None
            assert norm.output_dtype == expected, fqn

    # Same math as the block's eager cast, rounded once.
    layer, fused_layer = default.layers[0], fused.layers[0]
    norm = layer.attention_norm.build().to(torch.bfloat16)
    torch.nn.init.normal_(norm.weight, std=0.05)
    fused_norm = fused_layer.attention_norm.build().to(torch.bfloat16)
    fused_norm.load_state_dict(norm.state_dict())
    stream_TD = torch.randn(8, layer.attention_norm.normalized_shape)
    grad_output_TD = torch.randn(stream_TD.shape, dtype=torch.bfloat16)
    results = []
    for module, cast in ((norm, True), (fused_norm, False)):
        x = stream_TD.clone().requires_grad_()
        output_TD = module(x)
        if cast:
            output_TD = output_TD.to(torch.bfloat16)
        output_TD.backward(grad_output_TD)
        results.append((output_TD, x.grad, module.weight.grad))
    for expected, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    with pytest.raises(ValueError, match="branch_dtype"):
        replace(
            layer,
            ffn_norm=replace(layer.ffn_norm, output_dtype=torch.float16),
        )


@pytest.mark.parametrize("recipe", [muse_spark_debugmodel, muse_spark_debugmodel_mm])
def test_debug_recipe_tokenizer(recipe):
    config = recipe()
    config.__post_init__()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")
