# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Dist-MoE post-expert norm against the compiled module and FP64.

Each case runs the Dist-MoE callback protocol on the fused callback and on the
module (the default callback, with its region compiled): a forward without
grad, then a replay under ``enable_grad`` and ``.backward``.
"""

import pytest
import torch

from torchtitan.config.transform.base import convert_config_type
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.muse_spark import build_model_config
from torchtitan.models.muse_spark.input_scaled_rmsnorm import (
    FusedDistMoeInputScaledRMSNorm,
)
from torchtitan.models.muse_spark.model import MuseSparkRMSNormWithInputScale


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_BF16_EPS = 2.0**-8


def _build(
    config: MuseSparkRMSNormWithInputScale.Config,
) -> FusedDistMoeInputScaledRMSNorm:
    module = convert_config_type(config, FusedDistMoeInputScaledRMSNorm).build()
    torch.nn.init.normal_(module.weight, std=0.05)
    # FSDP param_dtype: the callback sees bf16 parameters.
    return module.to(device="cuda", dtype=torch.bfloat16)


def _inputs(num_rows: int, num_columns: int, dtype: torch.dtype):
    x_RC = (1.5 * torch.randn(num_rows, num_columns, device="cuda")).to(dtype)
    dy_RC = (0.1 * torch.randn(num_rows, num_columns, device="cuda")).to(dtype)
    # Padded routes: Dist-MoE zeroes their input rows, and their gradient is zero.
    x_RC[::97] = 0
    dy_RC[::97] = 0
    return x_RC, dy_RC


def _dist_moe_protocol(callback, module, x_RC, dy_RC):
    """What Dist-MoE does with a postprocess callback, minus the combine."""
    module.weight.grad = None
    with torch.no_grad():
        y_RC = callback(x_RC)
    x_RC = x_RC.detach().requires_grad_()
    with torch.enable_grad():
        replay_y_RC = callback(x_RC)
    replay_y_RC.backward(dy_RC)
    return {
        "y": y_RC,
        "replay y": replay_y_RC.detach(),
        "grad x": x_RC.grad,
        "grad weight": module.weight.grad,
    }


def _fp64_reference(module, x_RC, dy_RC):
    x_RC = x_RC.double().requires_grad_()
    weight_C = module.weight.detach().double().requires_grad_()
    z_RC = (weight_C + module.gain_center) * x_RC
    y_RC = z_RC * torch.rsqrt(z_RC.pow(2).mean(-1, keepdim=True) + module.eps)
    y_RC.backward(dy_RC.double())
    return {
        "y": y_RC.detach(),
        "replay y": y_RC.detach(),
        "grad x": x_RC.grad,
        "grad weight": weight_C.grad,
    }


def _fused_and_compiled(module, x_RC, dy_RC):
    fused_callback = module.to_dist_moe_postprocess()
    # One callback per module, like the default that returns the module itself.
    assert module.to_dist_moe_postprocess() is fused_callback
    try:
        # A fresh cache, so each case compiles the region within its recompile limit.
        torch._dynamo.reset()
        LocalCompileConfig(regions=["muse_spark_input_scaled_rmsnorm"]).apply()
        fused = _dist_moe_protocol(fused_callback, module, x_RC, dy_RC)
        compiled = _dist_moe_protocol(module, module, x_RC, dy_RC)
    finally:
        LocalCompileConfig().apply()
    return fused, compiled


def _relative_error(actual, reference):
    return float((actual.double() - reference).norm() / reference.norm())


def _assert_no_less_accurate(fused, compiled, reference, case):
    for name, reference_tensor in reference.items():
        fused_error = _relative_error(fused[name], reference_tensor)
        compiled_error = _relative_error(compiled[name], reference_tensor)
        assert fused_error <= 1.25 * compiled_error + 1e-6, (
            f"{case} {name}: relative error vs fp64 {fused_error:.3e}, "
            f"compiled region {compiled_error:.3e}"
        )


def test_production_shape_matches_compiled_region():
    torch.manual_seed(0)
    # The 816B post-expert norm at the per-rank shape of the 64-GPU recipe:
    # 16384 tokens x top-8 routes, 4096 compressed columns.
    config = (
        build_model_config("816B-A42B")
        .layers[0]
        .feed_forward.routed_experts.output_postprocess
    )
    module = _build(config)
    x_RC, dy_RC = _inputs(16384 * 8, config.normalized_shape, torch.bfloat16)
    fused, compiled = _fused_and_compiled(module, x_RC, dy_RC)

    # The row reductions run in another order than Inductor's: allow a few bf16 ulps.
    # The weight gradient also sums 131072 rows in another order, so it is checked
    # only against fp64 below.
    for name in ("y", "replay y", "grad x"):
        assert fused[name].dtype == compiled[name].dtype, name
        actual, expected = fused[name].float(), compiled[name].float()
        assert torch.isfinite(actual).all(), name
        difference = (actual - expected).abs()
        relative_l2 = float(difference.norm() / expected.norm())
        # Elementwise, in bf16 ulps of max(|expected|, 1% of max |expected|).
        ulp = torch.maximum(expected.abs(), 0.01 * expected.abs().max()) * _BF16_EPS
        max_ulps = float((difference / ulp).max())
        assert relative_l2 <= _BF16_EPS and max_ulps <= 2.0, (
            f"{name}: relative L2 {relative_l2:.3e}, max {max_ulps:.2f} bf16 ulps"
        )
    _assert_no_less_accurate(
        fused, compiled, _fp64_reference(module, x_RC, dy_RC), "production"
    )


@pytest.mark.parametrize(
    "num_rows, num_columns, dtype",
    [
        (1000, 1000, torch.bfloat16),  # not a power of two
        (7, 4096, torch.bfloat16),  # fewer rows than backward programs
        (513, 4096, torch.float32),
        (64, 8192, torch.bfloat16),
    ],
)
def test_edge_shapes_match_fp64(num_rows, num_columns, dtype):
    torch.manual_seed(0)
    module = _build(
        MuseSparkRMSNormWithInputScale.Config(
            normalized_shape=num_columns, eps=1e-8, gain_center=1.0
        )
    )
    x_RC, dy_RC = _inputs(num_rows, num_columns, dtype)
    fused, compiled = _fused_and_compiled(module, x_RC, dy_RC)
    _assert_no_less_accurate(
        fused,
        compiled,
        _fp64_reference(module, x_RC, dy_RC),
        f"{num_rows}x{num_columns} {dtype}",
    )


@pytest.mark.parametrize("case", ["non-contiguous", "too wide"])
def test_unsupported_inputs_run_the_module(case):
    torch.manual_seed(0)
    num_columns = 16385 if case == "too wide" else 256
    module = _build(
        MuseSparkRMSNormWithInputScale.Config(
            normalized_shape=num_columns, eps=1e-8, gain_center=1.0
        )
    )
    x_RC, dy_RC = _inputs(16, num_columns, torch.bfloat16)
    if case == "non-contiguous":
        x_RC = x_RC.t().contiguous().t()
    expected = _dist_moe_protocol(module, module, x_RC, dy_RC)
    actual = _dist_moe_protocol(module.to_dist_moe_postprocess(), module, x_RC, dy_RC)
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(actual[name], expected_tensor, atol=0, rtol=0)
