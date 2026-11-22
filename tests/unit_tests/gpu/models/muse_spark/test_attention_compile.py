# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.muse_spark import build_model_config
from torchtitan.models.muse_spark.model import MuseSparkTransformerBlock


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _run(fn, inputs_):
    leaves = [t.detach().clone().requires_grad_() for t in inputs_]
    outputs = fn(*leaves)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    torch.autograd.backward(outputs, [torch.ones_like(t) for t in outputs])
    return [*outputs, *(t.grad for t in leaves)]


# Debug model layer 0: sliding window with RoPE. Layer 3: global, position temperature, no RoPE.
@pytest.mark.parametrize("layer_id", [0, 3])
@pytest.mark.parametrize(
    "region", ["muse_spark_attention_qk", "muse_spark_attention_output"]
)
def test_attention_region_matches_eager(region, layer_id):
    torch.manual_seed(0)
    layer = build_model_config("debugmodel").layers[layer_id]
    attention = layer.attention.build().cuda()
    num_tokens, num_heads, num_kv_heads = 64, attention.n_heads, attention.n_kv_heads
    head_dim = attention.head_dim
    if region == "muse_spark_attention_qk":
        positions_T = torch.arange(num_tokens, device="cuda")

        def fn(qkv):
            return attention._prepare_query_key_value(qkv, positions_T)

        qkv = torch.randn(
            num_tokens,
            (num_heads + 2 * num_kv_heads) * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        inputs_ = (qkv,)
    else:
        fn = attention._flatten_attention_output
        query_TNH = torch.randn(
            num_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        gate_TO = torch.randn(
            num_tokens, num_heads * head_dim, device="cuda", dtype=torch.bfloat16
        )
        inputs_ = (query_TNH, gate_TO)

    try:
        LocalCompileConfig().apply()
        reference = _run(fn, [t.float() for t in inputs_])
        expected = _run(fn, inputs_)
        LocalCompileConfig(regions=[region]).apply()
        actual = _run(fn, inputs_)
    finally:
        LocalCompileConfig().apply()

    # The compiled region rounds to bf16 once at the end, eager rounds between ops,
    # so the two differ by a few bf16 ulps. Check the compiled path is at least as
    # close as eager to an fp32 run of the same math, for outputs and grads.
    for actual_tensor, expected_tensor, reference_tensor in zip(
        actual, expected, reference, strict=True
    ):
        actual_error = (actual_tensor.float() - reference_tensor).norm()
        eager_error = (expected_tensor.float() - reference_tensor).norm()
        assert actual_error <= 1.05 * eager_error, (actual_error, eager_error)


def test_rmsnorm_region_matches_eager():
    # The residual stream is fp32 and the norm weight bf16, as under FSDP mixed precision.
    norm = build_model_config("debugmodel").layers[0].attention_norm.build().cuda()
    norm = norm.to(torch.bfloat16)
    stream_TD = torch.randn(64, norm.normalized_shape[0], device="cuda")

    def norm_and_weight_grad(regions):
        LocalCompileConfig(regions=regions).apply()
        norm.weight.grad = None
        return [*_run(norm, (stream_TD,)), norm.weight.grad]

    try:
        expected = norm_and_weight_grad([])
        actual = norm_and_weight_grad(["muse_spark_rmsnorm"])
    finally:
        LocalCompileConfig().apply()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2)


class _StubAttention(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(dim) * 0.5)

    def forward(self, x_TD, attention_metadata, positions=None):
        return torch.tanh(x_TD) * self.scale


class _StubFeedForward(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(dim) * 0.5)

    def forward(self, x_TD):
        return torch.sin(x_TD) * self.scale + x_TD


class _NormChainBlock(torch.nn.Module):
    """``MuseSparkTransformerBlock.forward`` with real norms and residual gates
    and cheap differentiable stand-ins for attention and the FFN."""

    forward = MuseSparkTransformerBlock.forward

    def __init__(self, layer: MuseSparkTransformerBlock.Config):
        super().__init__()
        dim = layer.attention_norm.normalized_shape
        self.attention = _StubAttention(dim)
        self.attention_norm = layer.attention_norm.build()
        self.post_attention_norm = layer.post_attention_norm.build()
        self.attention_residual = layer.attention_residual.build()
        self.ffn_norm = layer.ffn_norm.build()
        self.post_ffn_norm = layer.post_ffn_norm.build()
        self.ffn_residual = layer.ffn_residual.build()
        self.feed_forward = _StubFeedForward(dim)
        self.branch_dtype = layer.branch_dtype


def test_fused_pre_norm_cast_is_bitwise():
    """``output_dtype=branch_dtype`` on the pre-norms only moves the cast into
    the compiled region: the block output and every gradient are unchanged."""
    torch.manual_seed(0)
    # A production 816B layer, at the per-rank shape of the 64-GPU recipe:
    # 16384 tokens of the fp32 residual stream, bf16 parameters (FSDP param_dtype).
    layer = build_model_config("816B-A42B").layers[10]
    fused_layer = replace(
        layer,
        attention_norm=replace(
            layer.attention_norm, output_dtype=layer.branch_dtype
        ),
        ffn_norm=replace(layer.ffn_norm, output_dtype=layer.branch_dtype),
    )
    block = _NormChainBlock(layer)
    for name, param in block.named_parameters():
        with torch.no_grad():
            if name.endswith("residual.gate"):
                # Near the flavor's layer-10 init, 0.3 * logit(1 / 22).
                param.copy_(-0.9 + 0.2 * torch.randn_like(param))
            elif name.endswith("norm.weight"):
                torch.nn.init.normal_(param, std=0.05)
    block = block.to(device="cuda", dtype=torch.bfloat16)
    fused_block = _NormChainBlock(fused_layer)
    fused_block = fused_block.to(device="cuda", dtype=torch.bfloat16)
    fused_block.load_state_dict(block.state_dict())
    assert fused_block.attention_norm.output_dtype == torch.bfloat16
    assert block.attention_norm.output_dtype is None

    dim = layer.attention_norm.normalized_shape
    stream_TD = 2.0 * torch.randn(16384, dim, device="cuda")
    grad_output_TD = torch.randn_like(stream_TD)

    def output_and_grads(module):
        module.zero_grad(set_to_none=True)
        x = stream_TD.clone().requires_grad_()
        # No remat region here; the block's annotations are only for torch_remat.
        with patch.object(remat, "recompute_needs_tensor"):
            output_TD = module(x, None)
        output_TD.backward(grad_output_TD)
        return {
            "output": output_TD.detach(),
            "grad x": x.grad,
            **{name: param.grad for name, param in module.named_parameters()},
        }

    try:
        # A fresh cache, so earlier tests cannot exhaust the regions' recompile limits.
        torch._dynamo.reset()
        LocalCompileConfig(
            regions=[
                "muse_spark_rmsnorm",
                "muse_spark_input_scaled_rmsnorm",
                "muse_spark_residual",
            ]
        ).apply()
        expected = output_and_grads(block)
        actual = output_and_grads(fused_block)
        # The fused pre-norms do round inside the region, so the match is not vacuous.
        with torch.no_grad():
            assert fused_block.ffn_norm(stream_TD).dtype == torch.bfloat16
    finally:
        LocalCompileConfig().apply()

    assert expected.keys() == actual.keys()
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(
            actual[name],
            expected_tensor,
            atol=0,
            rtol=0,
            msg=lambda message, name=name: f"{name}: {message}",
        )


# The input-scaled norm also runs on routed tokens, a different count per layer.
@pytest.mark.parametrize(
    "region, norm_name, layer1_tokens",
    [
        ("muse_spark_rmsnorm", "post_attention_norm", 64),
        ("muse_spark_input_scaled_rmsnorm", "post_ffn_norm", 48),
    ],
)
def test_norm_region_under_activation_checkpoint(region, norm_name, layer1_tokens):
    # Two layers' post norms share one region and differ in gain_center.
    torch.manual_seed(0)
    layers = build_model_config("debugmodel").layers[:2]
    norms = [getattr(layer, norm_name).build().cuda() for layer in layers]
    assert norms[0].gain_center != norms[1].gain_center
    for norm in norms:
        torch.nn.init.normal_(norm.weight, std=0.1)
    input_TD = torch.randn(64, norms[0].normalized_shape[0], device="cuda")

    def two_norms(x):
        return norms[1](norms[0](x)[:layer1_tokens])

    def two_norms_checkpointed(x):
        return torch.utils.checkpoint.checkpoint(two_norms, x, use_reentrant=False)

    def output_and_grads(fn):
        # Empty cache, so layer 0 compiles the region first, as in training.
        torch._dynamo.reset()
        for norm in norms:
            norm.weight.grad = None
        return [*_run(fn, (input_TD,)), *(norm.weight.grad for norm in norms)]

    try:
        LocalCompileConfig(regions=[region]).apply()
        expected = output_and_grads(two_norms)
        actual = output_and_grads(two_norms_checkpointed)
    finally:
        LocalCompileConfig().apply()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
