"""Opt-in integration for the pinned compact BF16 Dist-MoE training path."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import linecache
import textwrap

import torch
from torch.distributed.tensor import DTensor

from triton_candidate import finish_gain, reduce_gain_tiles, route_backward, route_forward


TOKENS, WIDTH, TOPK = 29184, 4096, 8
EXPERTS, EP_SIZE = 256, 64
ROWS_PER_CTA = 4
PARTIALS_PER_TILE, TILE_WIDTH = 256, 32
ROUTES = TOKENS * TOPK
PARTIALS = ROUTES // ROWS_PER_CTA
REDUCTION_TILES = (PARTIALS + PARTIALS_PER_TILE - 1) // PARTIALS_PER_TILE
MIN_RECV_ROWS = ((PARTIALS + REDUCTION_TILES) * 4 + 1) // 2


@dataclass
class Calls:
    forward: int = 0
    backward: int = 0
    forward_norms: set[int] = field(default_factory=set)
    backward_norms: set[int] = field(default_factory=set)

    def receipt(self):
        return {
            "forward_calls": self.forward,
            "backward_calls": self.backward,
            "forward_norms": len(self.forward_norms),
            "backward_norms": len(self.backward_norms),
        }


calls = Calls()


class LearnedRoute:
    """Bind the fixed-shape kernels to the existing BF16 route callback owner."""

    def __init__(self, norm, expert_ids, copy_activations):
        self.norm = norm
        self.expert_ids = expert_ids
        self.copy_activations = copy_activations

    def _gain(self):
        # Resolve the current unsharded parameter in each pass. FSDP may
        # replace the forward view before it enters this backward.
        weight = self.norm.weight
        gain = weight.to_local() if isinstance(weight, DTensor) else weight
        if gain.shape != (WIDTH,) or not gain.is_contiguous() or gain.dtype != torch.bfloat16:
            raise RuntimeError("Learned route expected the unsharded BF16 gain")
        return gain

    def forward(self, raw, scores, *, output_dtype, save, save_buffer, save_offset, save_condition):
        if not save or output_dtype != torch.bfloat16 or save_buffer is None:
            raise RuntimeError("Learned route expected a BF16 training activation slot")
        gain = self._gain()
        self.copy_activations(
            condition=save_condition, lhs=None, lhs_offset=None, rhs=raw,
            rhs_offset=save_offset, activation_buffer=save_buffer, copy_to_buffer=True,
        )
        result = torch.empty((TOKENS, WIDTH), device=raw.device, dtype=output_dtype)
        route_forward[(TOKENS,)](
            raw, gain, scores, self.expert_ids, result, WIDTH, TOPK,
            self.norm.eps, self.norm.gain_center, num_warps=32, enable_fp_fusion=False,
        )
        calls.forward += 1
        calls.forward_norms.add(id(self.norm))
        return result, None, None, output_dtype

    def backward(self, grad_output, raw, scores, published, *, saved, saved_offset,
                 need_recompute, grad_h2_offset, recv_rows):
        grad_output = grad_output.to(torch.bfloat16).contiguous()
        if published.data_ptr() != raw.data_ptr():
            raise RuntimeError("Learned route expected W2 output in the local combine buffer")
        # The data is device-resident and varies per graph replay. Check before
        # writing partials so the existing grad-H2 bank cannot be overrun.
        torch._assert_async(recv_rows.ge(MIN_RECV_ROWS), "Learned route gain workspace does not fit")
        with torch.enable_grad():
            gain = self._gain()
        grad_scores = torch.empty_like(scores)
        grad_gain = torch.empty_like(gain)
        route_backward[(PARTIALS,)](
            raw, saved, saved_offset, need_recompute, grad_output, gain, scores, self.expert_ids,
            published, grad_scores, saved, grad_h2_offset, ROUTES, WIDTH, TOPK,
            ROWS_PER_CTA, self.norm.eps, self.norm.gain_center,
            num_warps=16, enable_fp_fusion=False,
        )
        reduce_gain_tiles[(REDUCTION_TILES, WIDTH // TILE_WIDTH)](
            saved, grad_h2_offset, PARTIALS, WIDTH, PARTIALS_PER_TILE, TILE_WIDTH, num_warps=8,
        )
        finish_gain[(WIDTH // TILE_WIDTH,)](
            saved, grad_h2_offset, grad_gain, PARTIALS, REDUCTION_TILES, WIDTH,
            PARTIALS_PER_TILE, TILE_WIDTH, num_warps=8,
        )
        # The stock callback reaches the same parameter via its nested
        # backward. Supply the gradient to the live edge so FSDP still owns
        # accumulation and reduction.
        torch.autograd.backward(gain, grad_gain)
        calls.backward += 1
        calls.backward_norms.add(id(self.norm))
        return published, grad_scores, True


_FORWARD_ANCHOR = """        num_local_input_tokens = x_TD.shape[0]
        local_rank = dist.get_rank(context.group)
"""
_FORWARD_REPLACEMENT = """        if save_for_backward and not config.inference:
            postprocess = _select_learned_route(
                postprocess, options.experts_output_postprocess,
                x_TD, topk_expert_ids_TK, topk_scores_TK,
                config, dist.get_world_size(context.group),
            )
""" + _FORWARD_ANCHOR
_COPY_ANCHOR = """            if not ctx.postprocess.fuses_saved_copy_into_reduction:
"""
_COPY_REPLACEMENT = """            if not isinstance(ctx.postprocess, _LearnedRoute) and not (
                ctx.postprocess.fuses_saved_copy_into_reduction
            ):
"""
_BACKWARD_BEGIN = """        grad_h3_MD, grad_topk_scores_TK, grad_h3_is_published = (
            ctx.postprocess.backward(
"""
_BACKWARD_END = """        if not grad_h3_is_published:
"""
_DIRECT_BACKWARD = """        if isinstance(ctx.postprocess, _LearnedRoute):
            published = buffers.combine.hdl.get_buffer(
                buffers.combine.hdl.rank, h3_saved_MD.shape, h3_saved_MD.dtype,
            )
            grad_h3_MD, grad_topk_scores_TK, grad_h3_is_published = ctx.postprocess.backward(
                grad_output_TD, h3_saved_MD, topk_scores_TK, published,
                saved=activation_buffer.buffer,
                saved_offset=h3_offset,
                need_recompute=need_recompute,
                grad_h2_offset=backward_plan.grad_h2_offset,
                recv_rows=num_recv_rows_1,
            )
        else:
"""


def _replace_once(source, anchor, replacement, stage):
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"Learned route {stage} anchor matched {count} times")
    return source.replace(anchor, replacement)


def patch_forward(source):
    return _replace_once(source, _FORWARD_ANCHOR, _FORWARD_REPLACEMENT, "forward selection")


def patch_backward(source):
    source = _replace_once(source, _COPY_ANCHOR, _COPY_REPLACEMENT, "backward H3 copy")
    if source.count(_BACKWARD_BEGIN) != 1 or source.count(_BACKWARD_END) != 1:
        raise RuntimeError("Learned route patch does not match the pinned BF16 backward")
    begin = source.index(_BACKWARD_BEGIN)
    end = source.index(_BACKWARD_END, begin)
    return source[:begin] + _DIRECT_BACKWARD + textwrap.indent(source[begin:end], "    ") + source[end:]


def select_route(postprocess, norm, x, expert_ids, scores, config, group_size, *,
                 norm_type, copy_activations):
    if type(norm) is not norm_type or x.shape != (TOKENS, WIDTH):
        return postprocess
    if (
        x.dtype != torch.bfloat16 or scores.dtype != torch.float32
        or expert_ids.shape != (TOKENS, TOPK) or scores.shape != expert_ids.shape
        or not expert_ids.is_contiguous() or not scores.is_contiguous()
        or config.num_experts != EXPERTS or group_size != EP_SIZE
        or config.activation != "swiglu_clamped"
        or norm.gain_center != 1.0 or norm.eps != 1e-8
        or norm.weight is None or not norm.weight.requires_grad
    ):
        raise RuntimeError("Learned route requires the compact CP1 BF16 benchmark")
    return LearnedRoute(norm, expert_ids, copy_activations)


def install():
    from dist_moe import api
    from dist_moe._triton_ops import conditional_copy_activations
    from torchtitan.models.muse_spark.model import MuseSparkRMSNormWithInputScale

    patch_api(api, MuseSparkRMSNormWithInputScale, conditional_copy_activations)


def patch_api(api, norm_type, copy_activations):
    from functools import partial

    cls = api._Bf16Autograd
    if getattr(cls, "_exp_learned_route_patched", False):
        return
    methods = {}
    for name, patch in (("forward", patch_forward), ("_backward_impl", patch_backward)):
        original = cls.__dict__[name]
        if not isinstance(original, staticmethod):
            raise TypeError(f"Dist-MoE {name} must be a staticmethod")
        function = original.__func__
        source = inspect.getsource(function)
        # SwiGLU's installed backward is recorded at module indentation.
        # Both sets of anchors use the original class indentation.
        source = textwrap.indent(textwrap.dedent(source), "    ")
        patched = textwrap.dedent(patch(source))
        filename = f"<learned_route:{name}>"
        name_space = {}
        exec(compile(patched, filename, "exec"), api.__dict__, name_space)
        compiled = name_space[name]
        if not isinstance(compiled, staticmethod):
            raise TypeError(f"Learned route {name} must stay a staticmethod")
        compiled.__func__.__qualname__ = function.__qualname__
        linecache.cache[filename] = (len(patched), None, patched.splitlines(True), filename)
        methods[name] = compiled
    api._LearnedRoute = LearnedRoute
    api._select_learned_route = partial(select_route, norm_type=norm_type, copy_activations=copy_activations)
    for name, method in methods.items():
        setattr(cls, name, method)
    cls._exp_learned_route_patched = True
