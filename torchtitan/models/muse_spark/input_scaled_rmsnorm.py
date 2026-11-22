# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Triton forward and backward for the Dist-MoE post-expert norm.

Muse Spark normalizes every routed expert output row with
``MuseSparkRMSNormWithInputScale``, ``y = rmsnorm((weight + gain_center) * x)``
without an output gain. Its weight is trained, which Dist-MoE's typed
``RMSNormPostprocess`` supports only for inference, so Dist-MoE runs the module
as an eager callback, and the callback runs the compiled
``muse_spark_input_scaled_rmsnorm`` region. Inductor emits looped reductions
for the forward and backward plus a separate column reduction for the weight
gradient, which reads ``x`` and ``dy`` again.

``FusedDistMoeInputScaledRMSNorm`` returns a callback backed by two Triton
kernels instead:

- Forward: one program per row with the whole row in registers. It reads ``x``
  once, writes ``y`` once, and saves the fp32 ``rstd`` only when grad is
  enabled.
- Backward: persistent programs over contiguous row blocks. Each reads ``x``
  and ``dy`` once, writes ``dx`` once, and accumulates its weight-gradient
  partial sum in fp32 registers; one ``sum(0)`` over the partials follows.

The math is fp32 throughout, as in the compiled region (``weight + gain`` and
``w_eff * x`` in fp32, fp32 sum of squares, ``rsqrt``), with one rounding to the
input dtype. The row reductions and the weight-gradient reduction run in a
different order than Inductor's, so ``y`` and ``dx`` can differ from the
compiled region by a few bf16 ulps, not bitwise.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

from torchtitan.models.muse_spark.model import MuseSparkRMSNormWithInputScale

# Shape suffix legend:
# R = rows (routed token-expert pairs), C = normalized (compressed expert) dimension,
# P = backward programs.

# Rows wider than this would not fit in registers; such inputs use the module.
_MAX_NUM_COLUMNS = 16384
# Backward programs per SM: enough to hide latency while keeping the
# weight-gradient partials (P x C fp32) small.
_BACKWARD_PROGRAMS_PER_SM = 8


@triton.jit
def _input_scaled_rms_norm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    x_row_stride,
    y_row_stride,
    C,
    gain,
    eps,
    STORE_RSTD: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32) + gain
    z = x * w
    rstd = tl.rsqrt(tl.sum(z * z, axis=0) / C + eps)
    tl.store(
        y_ptr + row * y_row_stride + cols,
        (z * rstd).to(y_ptr.dtype.element_ty),
        mask=mask,
    )
    if STORE_RSTD:
        tl.store(rstd_ptr + row, rstd)


@triton.jit
def _input_scaled_rms_norm_backward_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    rstd_ptr,
    dw_partial_ptr,
    x_row_stride,
    dy_row_stride,
    dx_row_stride,
    R,
    C,
    rows_per_program,
    gain,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < C
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32) + gain
    dw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    row0 = pid.to(tl.int64) * rows_per_program
    for i in range(0, rows_per_program):
        row = row0 + i
        valid = row < R
        mask = col_mask & valid
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        rstd = tl.load(rstd_ptr + row, mask=valid, other=0.0)
        z = x * w
        y = z * rstd
        # y = z * r, r = (mean(z^2) + eps)^-1/2
        # dz = r * (dy - y * mean(dy * y)); dx = dz * w; dw += dz * x
        dz = rstd * (dy - y * (tl.sum(dy * y, axis=0) / C))
        tl.store(
            dx_ptr + row * dx_row_stride + cols,
            (dz * w).to(dx_ptr.dtype.element_ty),
            mask=mask,
        )
        dw_acc += dz * x
    tl.store(dw_partial_ptr + pid.to(tl.int64) * BLOCK_C + cols, dw_acc, mask=col_mask)


def _num_warps(block_c: int) -> int:
    # The whole row lives in registers: 16 fp32 elements per thread at C = 4096.
    if block_c <= 1024:
        return 4
    return 8 if block_c <= 4096 else 16


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


class _InputScaledRMSNormFunction(torch.autograd.Function):
    """``rmsnorm((weight + gain) * x)`` over the last dimension of contiguous ``x``."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        x: torch.Tensor,
        weight_C: torch.Tensor,
        gain: float,
        eps: float,
        save_for_backward: bool,
    ) -> torch.Tensor:
        num_columns = x.shape[-1]
        x_RC = x.view(-1, num_columns)
        num_rows = x_RC.shape[0]
        y = torch.empty_like(x)
        y_RC = y.view(-1, num_columns)
        block_c = triton.next_power_of_2(num_columns)
        rstd_R = torch.empty(
            num_rows if save_for_backward else 1,
            dtype=torch.float32,
            device=x.device,
        )
        if num_rows > 0:
            _input_scaled_rms_norm_forward_kernel[(num_rows,)](
                x_RC,
                weight_C,
                y_RC,
                rstd_R,
                x_RC.stride(0),
                y_RC.stride(0),
                num_columns,
                gain,
                eps,
                STORE_RSTD=save_for_backward,
                BLOCK_C=block_c,
                num_warps=_num_warps(block_c),
            )
        if save_for_backward:
            ctx.save_for_backward(x, weight_C, rstd_R)
            ctx.gain = gain
        ctx.saved_state = save_for_backward
        return y

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, dy: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        if not ctx.saved_state:
            raise RuntimeError(
                "fused input-scaled RMSNorm ran without grad; nothing saved for backward"
            )
        x, weight_C, rstd_R = ctx.saved_tensors
        num_columns = x.shape[-1]
        x_RC = x.view(-1, num_columns)
        dy_RC = dy.reshape(-1, num_columns)
        if dy_RC.stride(-1) != 1:
            dy_RC = dy_RC.contiguous()
        num_rows = x_RC.shape[0]
        dx = torch.empty_like(x)
        dx_RC = dx.view(-1, num_columns)
        block_c = triton.next_power_of_2(num_columns)
        num_programs = max(
            1, min(num_rows, _num_sms(x.device.index) * _BACKWARD_PROGRAMS_PER_SM)
        )
        rows_per_program = triton.cdiv(num_rows, num_programs) if num_rows > 0 else 1
        num_programs = triton.cdiv(num_rows, rows_per_program) if num_rows > 0 else 1
        # Every launched program writes its whole partial row; with no rows, nothing
        # launches and the partials must read as zero.
        alloc = torch.empty if num_rows > 0 else torch.zeros
        dw_partial_PC = alloc(
            (num_programs, block_c), dtype=torch.float32, device=x.device
        )
        if num_rows > 0:
            _input_scaled_rms_norm_backward_kernel[(num_programs,)](
                x_RC,
                weight_C,
                dy_RC,
                dx_RC,
                rstd_R,
                dw_partial_PC,
                x_RC.stride(0),
                dy_RC.stride(0),
                dx_RC.stride(0),
                num_rows,
                num_columns,
                rows_per_program,
                ctx.gain,
                BLOCK_C=block_c,
                num_warps=_num_warps(block_c),
            )
        dw_C = None
        if ctx.needs_input_grad[1]:
            dw_C = dw_partial_PC[:, :num_columns].sum(dim=0).to(weight_C.dtype)
        dx_out = dx if ctx.needs_input_grad[0] else None
        return dx_out, dw_C, None, None, None


class _FusedDistMoePostprocess:
    """Dist-MoE callback with the same contract as the module itself.

    Dist-MoE calls it without grad in forward, and again under
    ``torch.enable_grad`` in its backward before calling ``.backward`` on the
    result. The weight is read at call time: FSDP may swap the parameter between
    those calls, exactly as for the module forward.
    """

    __slots__ = ("module",)

    def __init__(self, module: "FusedDistMoeInputScaledRMSNorm"):
        self.module = module

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        module = self.module
        weight_C = module.weight
        if not _is_eligible(x, weight_C) or tuple(module.normalized_shape) != (
            x.shape[-1],
        ):
            return module(x)
        save_for_backward = torch.is_grad_enabled() and (
            x.requires_grad or weight_C.requires_grad
        )
        return _InputScaledRMSNormFunction.apply(
            x,
            weight_C,
            float(module.gain_center),
            float(module.eps),
            save_for_backward,
        )


def _is_eligible(x: torch.Tensor, weight_C: torch.Tensor | None) -> bool:
    """Whether the kernels support these operands; otherwise the module runs."""
    if weight_C is None or isinstance(x, DTensor) or isinstance(weight_C, DTensor):
        return False
    return (
        x.is_cuda
        and weight_C.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.dim() >= 1
        and x.is_contiguous()
        and weight_C.dim() == 1
        and weight_C.is_contiguous()
        and weight_C.shape[0] == x.shape[-1]
        and 0 < x.shape[-1] <= _MAX_NUM_COLUMNS
    )


class FusedDistMoeInputScaledRMSNorm(MuseSparkRMSNormWithInputScale):
    """Input-scaled RMSNorm whose Dist-MoE callback runs fused Triton kernels.

    Only ``to_dist_moe_postprocess`` changes; the module forward and the parameter
    and checkpoint layout are the parent's. The callback falls back to the module
    for operands the kernels do not support: no weight, a DTensor operand,
    a non-CUDA or non-contiguous input, or more than 16384 columns.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MuseSparkRMSNormWithInputScale.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # One callback per module, as when the parent returns ``self``.
        self._dist_moe_postprocess = _FusedDistMoePostprocess(self)

    def to_dist_moe_postprocess(
        self,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the fused Triton callback for this post-expert norm."""
        return self._dist_moe_postprocess
