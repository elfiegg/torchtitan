# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType
from unittest.mock import Mock

import torch

from torchtitan.models.muse_spark.model import (
    MuseSparkRMSNorm,
    MuseSparkRMSNormWithInputScale,
)


def test_compiled_rmsnorm_output_cast_preserves_numerics():
    reference = MuseSparkRMSNorm.Config(normalized_shape=32).build()
    fused = MuseSparkRMSNorm.Config(
        normalized_shape=32, output_dtype=torch.bfloat16
    ).build()
    generator = torch.Generator().manual_seed(42)
    gain = torch.randn(32, generator=generator, dtype=torch.bfloat16)
    reference.weight = torch.nn.Parameter(gain.clone())
    fused.weight = torch.nn.Parameter(gain.clone())
    input_TD = torch.randn(16, 32, generator=generator, requires_grad=True)
    grad_TD = torch.randn(16, 32, generator=generator, dtype=torch.bfloat16)

    expected = torch.compile(reference, backend="aot_eager", fullgraph=True)(
        input_TD
    ).to(torch.bfloat16)
    actual = torch.compile(fused, backend="aot_eager", fullgraph=True)(input_TD)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_grads = torch.autograd.grad(
        expected, (input_TD, reference.weight), grad_TD
    )
    actual_grads = torch.autograd.grad(actual, (input_TD, fused.weight), grad_TD)
    torch.testing.assert_close(actual_grads, expected_grads, atol=0, rtol=0)


def test_dist_moe_norm_postprocess_resolves_the_live_weight(monkeypatch):
    regular = MuseSparkRMSNormWithInputScale.Config(normalized_shape=32).build()
    assert regular.to_dist_moe_postprocess() is regular

    dist_moe = ModuleType("dist_moe")
    dist_moe.InputScaledRMSNormPostprocess = Mock()
    monkeypatch.setitem(sys.modules, "dist_moe", dist_moe)
    fused = MuseSparkRMSNormWithInputScale.Config(
        normalized_shape=32, fuse_dist_moe_postprocess=True
    ).build()
    assert (
        fused.to_dist_moe_postprocess()
        is dist_moe.InputScaledRMSNormPostprocess.return_value
    )
    kwargs = dist_moe.InputScaledRMSNormPostprocess.call_args.kwargs
    assert kwargs["fallback"] is fused
    assert kwargs["eps"] == fused.eps
    assert kwargs["gain_center"] == fused.gain_center
    assert kwargs["weight_getter"]() is fused.weight
    fused.weight = torch.nn.Parameter(torch.ones(32))
    assert kwargs["weight_getter"]() is fused.weight
