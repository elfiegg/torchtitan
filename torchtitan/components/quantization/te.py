# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TE grouped GEMM converter for MoE layers.

Registers as "quantize.grouped_mm.te" model converter. When applied, wraps
MoE expert weight parameters in TEGroupedMMTensor, which intercepts
torch._grouped_mm and routes to TE's general_grouped_gemm kernel.

Supports two modes (configurable via --quantize.grouped_mm.te.mode):
  - "mxfp8": MXFP8 quantization + grouped GEMM (default)
  - "bf16": BF16 grouped GEMM (no quantization, uses TE's kernel directly)

Usage in config:
    --model.converters="quantize.grouped_mm.te"
    --quantize.grouped_mm.te.fqns="experts"
    --quantize.grouped_mm.te.mode="mxfp8"   # or "bf16"
"""

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger


class TEGroupedMMConverter:
    """Model converter that wraps MoE expert weights in TEGroupedMMTensor.

    This enables TE's grouped GEMM kernel for the expert computation,
    transparent to the rest of the model. Supports MXFP8 and BF16 modes.
    """

    enabled: bool = False

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        te_config = job_config.quantize.grouped_mm.te
        fqns = te_config.fqns
        if not fqns:
            logger.warning(
                "quantize.grouped_mm.te.fqns not set, TE grouped MM converter disabled"
            )
            self.enabled = False
            return

        # fqns can be a string or list
        if isinstance(fqns, str):
            self.moe_fqns = [s.strip() for s in fqns.split(",") if s.strip()]
        else:
            self.moe_fqns = list(fqns)

        if not self.moe_fqns:
            self.enabled = False
            return

        self.mode = te_config.mode
        self.use_fp8 = self.mode == "mxfp8"
        self.enabled = True
        logger.info(
            f"TE grouped GEMM converter enabled: mode={self.mode}, FQNs={self.moe_fqns}"
        )

    def convert(self, model: nn.Module):
        """Wraps expert weight parameters in TEGroupedMMTensor."""
        if not self.enabled:
            return

        from torchtitan.models.moe.te_backend import swap_params_to_te_grouped_mm

        swap_params_to_te_grouped_mm(
            model, fqn_filter=self.moe_fqns, use_fp8=self.use_fp8
        )
        logger.info(
            f"Converted MoE layers matching FQNs {self.moe_fqns} "
            f"to use TE grouped GEMM ({self.mode})"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """TE grouped GEMM doesn't require post-optimizer hooks."""
        return


register_model_converter(TEGroupedMMConverter, "quantize.grouped_mm.te")
