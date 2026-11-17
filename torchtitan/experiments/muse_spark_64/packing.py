# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Unqualified, opt-in packing experiment for one pinned FSDP runtime.

No registration into FSDP happens on import. Call install() only in an isolated
qualification runner after source/version checks and representative validation.
"""

import math
from collections.abc import Callable

import torch
import triton
import triton.language as tl

TORCH_GIT = "bf270c3251d8e84723cedbf5b60eaa60cacaa3f2"


@triton.jit
def _pack_bits(
    PTRS,
    OUT,
    ROWS: tl.constexpr,
    NUMELS: tl.constexpr,
    WIDTHS: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_STARTS: tl.constexpr,
    OUT_STARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one tensor chunk per program; BF16 widening is an exact bit shift."""
    program = tl.program_id(0)
    rank = tl.program_id(1).to(tl.int64)
    for i in tl.static_range(len(PTRS)):
        if (program >= BLOCK_STARTS[i]) & (program < BLOCK_STARTS[i + 1]):
            x = (program - BLOCK_STARTS[i]).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            source_x = rank * WIDTHS[i] + x
            if IS_BF16[i]:
                src = PTRS[i].to(tl.pointer_type(tl.uint16))
                bits = tl.load(
                    src + source_x, (x < WIDTHS[i]) & (source_x < NUMELS[i]), other=0
                )
                value = bits.to(tl.uint32) << 16
            else:
                src = PTRS[i].to(tl.pointer_type(tl.uint32))
                value = tl.load(
                    src + source_x, (x < WIDTHS[i]) & (source_x < NUMELS[i]), other=0
                )
            tl.store(
                OUT.to(tl.pointer_type(tl.uint32)) + rank * ROWS + OUT_STARTS[i] + x,
                value,
                x < WIDTHS[i],
            )


def _launch_spec(grads, widths, world_size, out):
    """Build immutable prefix metadata outside the Triton static-range body."""
    block = 1024
    block_starts, out_starts = [0], [0]
    for width in widths:
        block_starts.append(block_starts[-1] + triton.cdiv(width, block))
        out_starts.append(out_starts[-1] + width)
    arguments = (
        tuple(grads),
        out,
        sum(widths),
        tuple(g.numel() for g in grads),
        tuple(widths),
        tuple(g.dtype == torch.bfloat16 for g in grads),
        tuple(block_starts),
        tuple(out_starts),
        block,
    )
    return (block_starts[-1], world_size), arguments


@torch.library.custom_op("kernel_pack::pack", mutates_args={"out"})
def pack(
    grads: list[torch.Tensor], widths: list[int], world_size: int, out: torch.Tensor
) -> None:
    """Mutate a validated flat FP32 buffer; the custom op bumps its version."""
    grid, arguments = _launch_spec(grads, widths, world_size, out)
    _pack_bits[grid](*arguments, num_warps=4)


@pack.register_fake
def _pack_fake(grads, widths, world_size, out):
    """Mutation-only operation with no output metadata."""
    return None


def _ranges_disjoint(tensors: list[torch.Tensor]) -> bool:
    """Match native contiguous storage-range overlap exclusion."""
    ranges = sorted(
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors
    )
    return all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))


def _eligible(grads, shard_dims, world_size) -> bool:
    """Reject all contracts not implemented by this deliberately narrow kernel."""
    return (
        isinstance(world_size, int)
        and 0 < world_size <= 65535
        and 0 < len(grads) <= 16
        and len(grads) == len(shard_dims)
        and all(d == 0 for d in shard_dims)
        and all(
            g.device.type == "cuda"
            and g.layout == torch.strided
            and g.dim() > 0
            and g.numel() > 0
            and g.is_contiguous()
            and not g.is_conj()
            and not g.is_neg()
            and g.dtype in (torch.float32, torch.bfloat16)
            and g.device == grads[0].device
            for g in grads
        )
        and _ranges_disjoint(grads)
    )


def prepare(
    grads: list[torch.Tensor], *, shard_dims: list[int], world_size: int
) -> Callable[[torch.Tensor], None]:
    """Return a one-use copy-in closure; unsupported cases retain native behavior."""
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        _default_reduce_scatter_input_fn,
    )

    if not _eligible(grads, shard_dims, world_size):
        return _default_reduce_scatter_input_fn(
            grads, shard_dims=shard_dims, world_size=world_size
        )
    widths = [
        ((g.shape[0] + world_size - 1) // world_size) * math.prod(g.shape[1:])
        for g in grads
    ]
    if sum((width + 1023) // 1024 for width in widths) > 2147483647:
        return _default_reduce_scatter_input_fn(
            grads, shard_dims=shard_dims, world_size=world_size
        )

    def copy_in(out):
        """Use native output allocation and stream; preserve native view semantics."""
        if (
            out.dtype != torch.float32
            or out.device != grads[0].device
            or not out.is_contiguous()
            or out.is_conj()
            or out.is_neg()
            or out.numel() != world_size * sum(widths)
            or not _ranges_disjoint([*grads, out])
        ):
            torch.ops.fsdp.chunk_cat_mixed_dtype(
                grads, dim=0, num_chunks=world_size, out=out.view(world_size, -1)
            )
            return
        pack(grads, widths, world_size, out)

    return copy_in


def install(model):
    """Opt into the experimental public seam only on the audited Torch commit."""
    if torch.version.git_version != TORCH_GIT:
        raise RuntimeError("Packing candidate requires the audited Torch commit")
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        _default_reduce_scatter_input_fn,
    )

    modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    if not modules:
        raise RuntimeError("Install packing after fully_shard wrapping")
    if any(
        group._prepare_reduce_scatter_inputs is not _default_reduce_scatter_input_fn
        for module in modules
        for group in module._get_fsdp_state()._fsdp_param_groups
    ):
        raise RuntimeError("Refusing to replace an existing custom packing callback")
    model.set_reduce_scatter_input_fn(prepare)
