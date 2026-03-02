# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TE backend for MoE grouped GEMM via a tensor subclass.

TEGroupedMMTensor wraps MoE expert weight parameters. When torch._grouped_mm
is called with a TEGroupedMMTensor weight, the __torch_function__ intercept
routes the GEMM to TE's general_grouped_gemm CUDA kernel.

Two modes are supported:
  - MXFP8 (use_fp8=True): Quantizes inputs/weights with TE's MXFP8Quantizer
    (block_size=32, E8M0 scales) before calling general_grouped_gemm.
  - BF16 (use_fp8=False): Splits inputs/weights into per-expert lists and
    calls general_grouped_gemm directly in BF16.

This is transparent to the rest of the codebase — the original GroupedExperts
module works unchanged:

    # In _run_experts_grouped_mm:
    torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    # → intercepted when w1 is TEGroupedMMTensor, routed to TE GEMM

Architecture:
    - TEGroupedMMTensor: tensor subclass wrapping 3D weight parameters
    - _TEGroupedGEMM: autograd.Function with setup_context for forward + backward
    - te_moe::gemm_fwd/dgrad/wgrad: custom ops (torch.library) with fake
      implementations, enabling torch.compile to trace through the GEMM without
      graph breaks
    - TE's MXFP8Quantizer handles FP8 quantization (when use_fp8=True)
    - TE's general_grouped_gemm handles the CUDA GEMM kernel

torch.compile: fully compatible — the custom ops provide fake (meta)
implementations so dynamo can infer output shapes at tracing time without
executing the TE C++ kernels.

Parallelism: fully compatible with FSDP, EP, TP, DeepEP/HybridEP because
the tensor subclass preserves shape, dtype, and supports FSDP hooks.

Note: _offs_to_m_splits() calls offs.tolist() which triggers a GPU→CPU sync.
This is unavoidable because TE's general_grouped_gemm requires m_splits as
a Python List[int]. This sync is per-GEMM-call (3 per MoE layer per step).
"""

import logging
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch._prims_common import suggest_memory_format
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

logger: logging.Logger = logging.getLogger(__name__)

try:
    # Import from transformer_engine.pytorch FIRST — this triggers
    # load_framework_extension("torch") which dynamically loads the
    # transformer_engine_torch C++ extension into sys.modules.
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    _TE_AVAILABLE = True
except ImportError as e:
    _TE_AVAILABLE = False
    _TE_IMPORT_ERROR = str(e)


def _offs_to_m_splits(offs: torch.Tensor) -> List[int]:
    """Convert cumulative offsets tensor to per-group sizes (m_splits).

    offs is [n1, n1+n2, n1+n2+n3, ...] (cumulative, no leading zero).
    Returns [n1, n2, n3, ...].

    NOTE: calls offs.tolist() which triggers a GPU→CPU sync. This is required
    because TE's general_grouped_gemm accepts m_splits as List[int].
    """
    if offs.numel() == 0:
        return []
    offs_list = offs.tolist()
    splits = [offs_list[0]]
    for i in range(1, len(offs_list)):
        splits.append(offs_list[i] - offs_list[i - 1])
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# MXFP8 helpers
# ──────────────────────────────────────────────────────────────────────────────

_MXFP8_BLOCK = 32  # MXFP8 requires both tensor dims divisible by this


def _ceil_to_block(n: int) -> int:
    """Round *n* up to the nearest multiple of ``_MXFP8_BLOCK`` (32)."""
    return (n + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK * _MXFP8_BLOCK


def _pad_for_mxfp8(
    tensor: torch.Tensor, m_splits: List[int]
) -> Tuple[torch.Tensor, List[int]]:
    """Pad each expert chunk so its row count is divisible by 32.

    MXFP8 quantization requires ``(numel / last_dim) % 32 == 0``.
    With dynamic MoE routing the per-expert token count is arbitrary,
    so we zero-pad each chunk to satisfy the constraint.

    Uses a single zero-initialized buffer and per-expert slice copies
    instead of per-expert ``torch.zeros`` + ``torch.cat`` to minimise
    kernel launch overhead (eliminates ~120K tiny cat/fill kernels per
    training step).

    Returns ``(padded_tensor, padded_splits)`` where *padded_tensor* is a
    single contiguous buffer with zero-padded expert regions and
    *padded_splits* contains the (padded) row count for each expert.
    """
    padded_splits = [_ceil_to_block(m) for m in m_splits]
    # Fast path -- no padding needed
    if padded_splits == m_splits:
        return tensor.contiguous(), m_splits

    padded_total = sum(padded_splits)
    K = tensor.shape[-1]
    # Single allocation -- one memset kernel instead of N torch.zeros
    padded = torch.zeros(padded_total, K, dtype=tensor.dtype, device=tensor.device)
    # Copy each expert's rows via slice assignment (DtoD memcpy, no cat)
    src_off = 0
    dst_off = 0
    for orig, ps in zip(m_splits, padded_splits):
        if orig > 0:
            padded[dst_off : dst_off + orig] = tensor[src_off : src_off + orig]
        src_off += orig
        dst_off += ps
    return padded, padded_splits


def _unpad_mxfp8_output(
    padded_out: torch.Tensor,
    m_splits: List[int],
    padded_splits: List[int],
    out: torch.Tensor,
) -> None:
    """Copy unpadded rows from *padded_out* into *out*.

    After running ``general_grouped_gemm`` with *padded_splits*, the output
    contains interleaved real + padding rows.  This extracts only the real
    rows back into the caller's output buffer.
    """
    if padded_splits == m_splits:
        # No padding was applied — single_output already wrote into *out*
        return
    off_p = 0  # offset into padded_out
    off_o = 0  # offset into out
    for orig, padded in zip(m_splits, padded_splits):
        out[off_o : off_o + orig] = padded_out[off_p : off_p + orig]
        off_p += padded
        off_o += orig


def _mxfp8_quantize_inputs(
    A: torch.Tensor, m_splits: List[int], num_experts: int
) -> Tuple[Any, List[int]]:
    """Quantize input activations per expert group using MXFP8.

    Pads each expert chunk to a multiple of 32 rows (MXFP8 requirement).
    Returns ``(quantized_list, padded_splits)``.
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    padded_A, padded_splits = _pad_for_mxfp8(A, m_splits)
    quantizers = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
        q.internal = True
        q.optimize_for_gemm = True
        quantizers.append(q)
    return tex.split_quantize(padded_A, padded_splits, quantizers), padded_splits


def _mxfp8_quantize_weights(B_t: torch.Tensor, num_experts: int, transpose: bool = True):
    """Quantize weight matrices per expert using MXFP8.

    Args:
        B_t: [E, K, N] weight tensor (already transposed from GroupedExperts).
        transpose: If True, transposes each [K, N] -> [N, K] for TN layout.
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    quantizers = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
        q.internal = True
        quantizers.append(q)
    if transpose:
        # Single contiguous copy for the full [E,K,N]->[E,N,K] transpose
        # instead of 16 individual .t().contiguous() calls (saves 15 DtoD
        # copy kernels per invocation).
        B_t_T = B_t.transpose(-2, -1).contiguous()  # [E, N, K]
        return [q.quantize_impl(B_t_T[i]) for i, q in enumerate(quantizers)]
    else:
        # B_t is [E, K, N]; B_t[i] is already contiguous when B_t is.
        return [q.quantize_impl(B_t[i].contiguous()) for i, q in enumerate(quantizers)]


def _mxfp8_quantize_wgrad(
    A: torch.Tensor, grad_output: torch.Tensor, m_splits: List[int], num_experts: int
) -> Tuple[Any, Any, List[int]]:
    """Quantize inputs (columnwise) and grads (rowwise) for WGRAD.

    Returns ``(inputs_fp8, grads_fp8, padded_splits)``.
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    padded_A, padded_splits = _pad_for_mxfp8(A, m_splits)
    padded_grad, _ = _pad_for_mxfp8(grad_output, m_splits)
    # NT-layout GEMM (wgrad = dOut^T @ X) requires both operands to have
    # columnwise data available.  Quantize with BOTH rowwise+columnwise
    # to produce MXFP8 2D block scaling, then switch inputs to present
    # only the columnwise view (matching TE's expected update_usage pattern).
    input_qs = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
        q.internal = True
        q.optimize_for_gemm = True
        input_qs.append(q)
    grad_qs = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
        q.internal = True
        q.optimize_for_gemm = True
        grad_qs.append(q)
    inputs_fp8 = tex.split_quantize(padded_A, padded_splits, input_qs)
    grads_fp8 = tex.split_quantize(padded_grad, padded_splits, grad_qs)
    for im in inputs_fp8:
        if hasattr(im, "update_usage"):
            im.update_usage(rowwise_usage=False, columnwise_usage=True)
    return inputs_fp8, grads_fp8, padded_splits


# ──────────────────────────────────────────────────────────────────────────────
# Custom ops — torch.library registration for torch.compile compatibility
# ──────────────────────────────────────────────────────────────────────────────
#
# Each TE GEMM operation (forward, dgrad, wgrad) is registered as a custom op
# with a fake (meta) implementation.  This allows torch.compile / dynamo to
# trace through the grouped GEMM without actually executing the TE C++ kernels
# (which cannot handle FakeTensors / symbolic shapes).
#
# At execution time the *real* implementations call TE's general_grouped_gemm;
# at tracing time the *fake* implementations return tensors with the correct
# shape/dtype so dynamo can continue compiling the surrounding graph.
# ──────────────────────────────────────────────────────────────────────────────


@torch.library.custom_op("te_moe::gemm_fwd", mutates_args=())
def _te_gemm_fwd(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    use_fp8: bool,
) -> torch.Tensor:
    """Forward grouped GEMM: out[i] = input[i] @ weight[i]^T  (TN layout).

    Args:
        A: [total_tokens, K] input activations.
        B_t: [E, K, N] weights (already transposed from GroupedExperts).
        offs: [E] int32 cumulative offsets.
        out_dtype: output dtype (e.g. torch.bfloat16).
        use_fp8: If True, MXFP8 quantization; if False, BF16 direct.
    """
    m_splits = _offs_to_m_splits(offs)
    num_experts = len(m_splits)
    total_tokens = sum(m_splits)
    N = B_t.shape[-1]

    # Use A.shape[0] for output rows (not total_tokens) to match the
    # invariant of torch._grouped_mm: output rows == input rows.
    # With HybridEP, A may be a pre-allocated buffer larger than
    # total_tokens; the extra rows are left untouched by TE's kernel.
    out = torch.empty(A.shape[0], N, dtype=out_dtype, device=A.device)

    # Slice A to the actual token count to avoid passing excess buffer
    # rows to TE's split_quantize / general_grouped_gemm.
    A_used = A[:total_tokens] if A.shape[0] > total_tokens else A

    if use_fp8:
        _nan_debug = os.environ.get("TE_MXFP8_NAN_DEBUG", "0") == "1"
        if _nan_debug:
            _has_nan_A = A_used.isnan().any().item()
            _has_nan_B = B_t.isnan().any().item()
            logger.info(
                f"[NaN-debug FWD] A_used={list(A_used.shape)} nan={_has_nan_A} "
                f"B_t={list(B_t.shape)} nan={_has_nan_B} "
                f"m_splits={m_splits} total_tokens={total_tokens}"
            )
        inputmats_fp8, padded_splits = _mxfp8_quantize_inputs(
            A_used, m_splits, num_experts
        )
        weights_fp8 = _mxfp8_quantize_weights(B_t, num_experts, transpose=True)
        if _nan_debug:
            logger.info(
                f"[NaN-debug FWD] quantization done: "
                f"num_inputs={len(inputmats_fp8)} num_weights={len(weights_fp8)} "
                f"padded_splits={padded_splits}"
            )
        padded_total = sum(padded_splits)
        padded_out = torch.empty(padded_total, N, dtype=out_dtype, device=A.device)
        general_grouped_gemm(
            weights_fp8, inputmats_fp8, [padded_out],
            [None] * num_experts, out_dtype,
            single_output=True, m_splits=padded_splits,
        )
        if _nan_debug:
            _has_nan_gemm = padded_out.isnan().any().item()
            logger.info(
                f"[NaN-debug FWD] padded_out={list(padded_out.shape)} nan={_has_nan_gemm} "
                f"padded_splits={padded_splits}"
            )
        _unpad_mxfp8_output(padded_out, m_splits, padded_splits, out)
        if _nan_debug:
            _has_nan_out = out[:total_tokens].isnan().any().item()
            logger.info(
                f"[NaN-debug FWD] final out={list(out.shape)} nan={_has_nan_out}"
            )
    else:
        # Single transpose+contiguous for all experts instead of 16 individual
        B_t_T = B_t.transpose(-2, -1).contiguous()  # [E, N, K]
        weights = [B_t_T[i] for i in range(num_experts)]
        inputs = list(A_used.contiguous().split(m_splits))
        general_grouped_gemm(
            weights, inputs, [out],
            [None] * num_experts, out_dtype,
            single_output=True, m_splits=m_splits,
        )

    return out


@_te_gemm_fwd.register_fake
def _(A, B_t, offs, out_dtype, use_fp8):
    return torch.empty(A.shape[0], B_t.shape[-1], dtype=out_dtype, device=A.device)


@torch.library.custom_op("te_moe::gemm_dgrad", mutates_args=())
def _te_gemm_dgrad(
    grad_output: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    use_fp8: bool,
) -> torch.Tensor:
    """DGRAD: grad_A[i] = grad_out[i] @ B_t[i]  (TN layout)."""
    m_splits = _offs_to_m_splits(offs)
    num_experts = len(m_splits)
    total_tokens = sum(m_splits)
    K = B_t.shape[-2]

    grad_A = torch.empty(grad_output.shape[0], K, dtype=out_dtype, device=grad_output.device)

    # Slice to actual token count (see _te_gemm_fwd for rationale)
    grad_used = grad_output[:total_tokens] if grad_output.shape[0] > total_tokens else grad_output

    if use_fp8:
        grad_fp8, padded_splits = _mxfp8_quantize_inputs(
            grad_used, m_splits, num_experts
        )
        bt_fp8 = _mxfp8_quantize_weights(B_t, num_experts, transpose=False)
        padded_total = sum(padded_splits)
        padded_grad_A = torch.empty(
            padded_total, K, dtype=out_dtype, device=grad_output.device
        )
        general_grouped_gemm(
            bt_fp8, grad_fp8, [padded_grad_A],
            [None] * num_experts, out_dtype,
            single_output=True, m_splits=padded_splits,
            grad=True,
        )
        _unpad_mxfp8_output(padded_grad_A, m_splits, padded_splits, grad_A)
    else:
        bt_weights = [B_t[i].contiguous() for i in range(num_experts)]
        grad_splits = list(grad_used.contiguous().split(m_splits))
        general_grouped_gemm(
            bt_weights, grad_splits, [grad_A],
            [None] * num_experts, out_dtype,
            single_output=True, m_splits=m_splits,
            grad=True,
        )

    return grad_A


@_te_gemm_dgrad.register_fake
def _(grad_output, B_t, offs, out_dtype, use_fp8):
    K = B_t.shape[-2]
    return torch.empty(grad_output.shape[0], K, dtype=out_dtype, device=grad_output.device)


@torch.library.custom_op("te_moe::gemm_wgrad", mutates_args=())
def _te_gemm_wgrad(
    A: torch.Tensor,
    grad_output: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    use_fp8: bool,
) -> torch.Tensor:
    """WGRAD: wgrad[i] = grad_out[i]^T @ A[i]  (NT layout), returns [E, K, N]."""
    m_splits = _offs_to_m_splits(offs)
    num_experts = len(m_splits)
    total_tokens = sum(m_splits)
    N = grad_output.shape[-1]
    K = A.shape[-1]

    wgrad_list = [
        torch.empty(N, K, dtype=out_dtype, device=A.device)
        for _ in range(num_experts)
    ]

    # Slice to actual token count (see _te_gemm_fwd for rationale)
    A_used = A[:total_tokens] if A.shape[0] > total_tokens else A
    grad_used = grad_output[:total_tokens] if grad_output.shape[0] > total_tokens else grad_output

    if use_fp8:
        inputs_fp8, grads_fp8, padded_splits = _mxfp8_quantize_wgrad(
            A_used, grad_used, m_splits, num_experts
        )
        general_grouped_gemm(
            inputs_fp8, grads_fp8, wgrad_list,
            [None] * num_experts, out_dtype,
            layout="NT", m_splits=padded_splits,
            grad=True,
        )
    else:
        input_splits = list(A_used.contiguous().split(m_splits))
        grad_splits = list(grad_used.contiguous().split(m_splits))
        general_grouped_gemm(
            input_splits, grad_splits, wgrad_list,
            [None] * num_experts, out_dtype,
            layout="NT", m_splits=m_splits,
            grad=True,
        )

    # wgrad[i] is [N, K]; B_t is [E, K, N] → gradient is also [E, K, N]
    return torch.stack([w.t() for w in wgrad_list], dim=0)


@_te_gemm_wgrad.register_fake
def _(A, grad_output, offs, out_dtype, use_fp8):
    num_experts = offs.shape[0]
    K = A.shape[-1]
    N = grad_output.shape[-1]
    return torch.empty(num_experts, K, N, dtype=out_dtype, device=A.device)


# ──────────────────────────────────────────────────────────────────────────────
# Autograd function
# ──────────────────────────────────────────────────────────────────────────────


class _TEGroupedGEMM(torch.autograd.Function):
    """Differentiable grouped GEMM using TE's general_grouped_gemm.

    Uses ``setup_context`` and custom ops (``te_moe::gemm_fwd``,
    ``te_moe::gemm_dgrad``, ``te_moe::gemm_wgrad``) so that
    ``torch.compile`` / dynamo can trace through the full forward and
    backward without graph breaks.

    GEMM layout conventions (row-major → cuBLAS column-major mapping):
        Forward (TN):  out[i] = input[i] @ weight[i]^T    [m_i, K] @ [K, N] = [m_i, N]
        DGRAD   (TN):  grad_A[i] = grad_out[i] @ B[i]     [m_i, N] @ [N, K] = [m_i, K]
        WGRAD   (NT):  wgrad[i] = grad_out[i]^T @ A[i]    [N, m_i] @ [m_i, K] = [N, K]
    """

    @staticmethod
    def forward(
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        use_fp8: bool = True,
    ) -> torch.Tensor:
        return _te_gemm_fwd(A, B_t, offs, out_dtype, use_fp8)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, B_t, offs, out_dtype, use_fp8 = inputs
        ctx.save_for_backward(A, B_t, offs)
        ctx.out_dtype = out_dtype
        ctx.use_fp8 = use_fp8

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, B_t, offs = ctx.saved_tensors
        grad_A = _te_gemm_dgrad(grad_output, B_t, offs, ctx.out_dtype, ctx.use_fp8)
        grad_B_t = _te_gemm_wgrad(A, grad_output, offs, ctx.out_dtype, ctx.use_fp8)
        return grad_A, grad_B_t, None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Tensor subclass
# ──────────────────────────────────────────────────────────────────────────────

# Ops that should preserve the TEGroupedMMTensor subclass through dispatch
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
}


class TEGroupedMMTensor(torch.Tensor):
    """Tensor subclass that routes torch._grouped_mm to TE's grouped GEMM.

    Wraps a regular tensor (MoE expert weight). When torch._grouped_mm is
    called with this tensor as the B operand, __torch_function__ intercepts
    and dispatches to _TEGroupedGEMM which uses TE's general_grouped_gemm.

    Args:
        tensor: The underlying weight tensor to wrap.
        use_fp8: If True, use MXFP8 quantization. If False, use BF16 directly.
    """

    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, use_fp8: bool = True):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )
        return self

    def __init__(self, tensor: torch.Tensor, use_fp8: bool = True):
        self._data = tensor
        self._use_fp8 = use_fp8

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # Intercept torch._grouped_mm when B is a TEGroupedMMTensor
        if func.__name__ == cls.grouped_mm_func_name:
            A, B = args[0], args[1]
            assert not isinstance(A, TEGroupedMMTensor), (
                "A (input) should not be a TEGroupedMMTensor"
            )
            assert isinstance(B, TEGroupedMMTensor), (
                "B (weight) should be a TEGroupedMMTensor"
            )
            A_is_2d = A.ndim == 2
            B_is_2d_or_3d = B.ndim == 2 or B.ndim == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None
            if A_is_2d and B_is_2d_or_3d and has_offs:
                offs = kwargs[cls.offs_arg_name]
                out_dtype = kwargs.get("out_dtype", torch.bfloat16)
                use_fp8 = B._use_fp8
                # Unwrap B to plain tensor for the autograd function
                with torch._C.DisableTorchFunctionSubclass():
                    B_plain = B._data
                return _TEGroupedGEMM.apply(A, B_plain, offs, out_dtype, use_fp8)

        # For all other ops, disable torch_function and dispatch normally
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # Unwrap TEGroupedMMTensor args to plain tensors, capture use_fp8
        use_fp8_vals = []

        def unwrap(t):
            use_fp8_vals.append(t._use_fp8)
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            TEGroupedMMTensor, unwrap, (args, kwargs or {})
        )

        # Use the first encountered use_fp8 value (all should be the same)
        use_fp8 = use_fp8_vals[0] if use_fp8_vals else True

        # Detach is a special case
        if func == torch.ops.aten.detach.default:
            return TEGroupedMMTensor(args_unwrapped[0], use_fp8=use_fp8)

        # Perform the op on unwrapped tensors
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # Only re-wrap for ops that should preserve the subclass
        if func not in _ops_to_preserve_subclass:
            return out

        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: TEGroupedMMTensor(x, use_fp8=use_fp8),
            out,
        )

    def __repr__(self):
        mode = "mxfp8" if self._use_fp8 else "bf16"
        return f"TEGroupedMMTensor(shape={list(self.shape)}, dtype={self.dtype}, mode={mode})"

    def __tensor_flatten__(self):
        return ["_data"], {"use_fp8": self._use_fp8}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return TEGroupedMMTensor(
            inner_tensors["_data"], use_fp8=flatten_spec["use_fp8"]
        )

    # ----- FSDP hooks (same pattern as ScaledGroupedMMTensor) -----

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        all_gather_inputs = (self._data.to(mp_policy.param_dtype),)
        all_gather_metadata = ()
        return all_gather_inputs, all_gather_metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs

        if out is not None:
            if isinstance(out, TEGroupedMMTensor):
                out_data = out._data
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, TEGroupedMMTensor
            ):
                out_data = out._local_tensor._data
            else:
                raise RuntimeError(
                    f"Expected out to be TEGroupedMMTensor or DTensor wrapping one, "
                    f"but got {type(out)}"
                )

            if data.dtype == param_dtype:
                assert (
                    data.untyped_storage().data_ptr()
                    == out_data.untyped_storage().data_ptr()
                )
            else:
                assert out_data.dtype == param_dtype
                out_data.copy_(data)
            return

        output = TEGroupedMMTensor(data, use_fp8=self._use_fp8)
        inner_tensors = (data,)
        return output, inner_tensors


# ──────────────────────────────────────────────────────────────────────────────
# Parameter swapping utility
# ──────────────────────────────────────────────────────────────────────────────


def swap_params_to_te_grouped_mm(
    model: nn.Module,
    fqn_filter: Optional[List[str]] = None,
    use_fp8: bool = True,
) -> None:
    """Swap nn.Parameter tensors in matching modules to TEGroupedMMTensor.

    Args:
        model: The model to convert.
        fqn_filter: List of FQN substrings to match (e.g., ["experts"]).
                     If None, all modules are converted.
        use_fp8: If True, use MXFP8 quantization. If False, use BF16 directly.
    """
    if not _TE_AVAILABLE:
        raise ImportError(
            f"TransformerEngine is required for TEGroupedMMTensor but failed to import: {_TE_IMPORT_ERROR}\n"
            "Install from: https://github.com/NVIDIA/TransformerEngine"
        )

    mode_str = "MXFP8" if use_fp8 else "BF16"
    for fqn, module in model.named_modules():
        if fqn_filter is not None:
            if not any(target in fqn for target in fqn_filter):
                continue

        for param_name, param in list(module.named_parameters(recurse=False)):
            if not isinstance(param.data, TEGroupedMMTensor):
                new_param = nn.Parameter(
                    TEGroupedMMTensor(param.data, use_fp8=use_fp8),
                    requires_grad=param.requires_grad,
                )
                setattr(module, param_name, new_param)
                logger.info(
                    f"Swapped {fqn}.{param_name} to TEGroupedMMTensor ({mode_str})"
                )
