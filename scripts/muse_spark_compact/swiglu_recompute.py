"""Fuse SwiGLU into the BF16 backward FC13 recompute (Dist-MoE).

Forward already runs FC13 dispatch through the native emit-SwiGLU epilogue
(``swiglu_h2=`` in ``_Bf16Autograd.forward``), which writes both H1 and H2.
The BF16 backward recompute instead runs the plain FC13 dispatch (H1 only,
skipped on device when ``need_recompute`` is 0) followed by an unconditional
standalone Triton ``swiglu_fwd`` (H1 -> H2). This patch makes the recompute
dispatch emit H2 too, and predicates the standalone ``swiglu_fwd`` on
``not need_recompute`` so it only runs for layers whose H1 was saved (where
the dispatch is skipped and H2 must still be rebuilt from the saved H1).

Exactly one function is replaced: ``dist_moe.api._Bf16Autograd._backward_impl``.
Its upstream source is read with ``inspect``, a fixed list of exact-match
edits is applied (each edit must match exactly once, otherwise the patch
refuses to apply and raises), every changed line carries ``# CHANGED:``, and
the result is compiled against ``dist_moe.api``'s globals. Nothing else in
dist_moe is touched; the block-scaled (MXFP8/NVFP4) autograd lives in
``dist_moe._blockscaled._BlockScaledAutograd`` and is unaffected.

Applied on import when EXP_SWIGLU_RECOMPUTE=1; otherwise a no-op.
Set EXP_SWIGLU_RECOMPUTE_DUMP=<path> to write the patched source for review.
"""

import inspect
import linecache
import os
import textwrap

ENABLED = os.environ.get("EXP_SWIGLU_RECOMPUTE", "0") == "1"

# (old, new) exact-text edits against the upstream _backward_impl source, in
# api.py's own indentation (applied before dedent).
_EDITS = (
    # 1. Import the same eligibility predicate the forward uses (api.py:1977).
    (
        "            dist_grouped_gemm_fprop_dispatch,\n"
        "        )\n"
        "        from dist_moe.kernels.grouped_gemm import (\n",
        "            dist_grouped_gemm_fprop_dispatch,\n"
        "            _native_swiglu_fwd_config_is_compatible,  # CHANGED: forward's fusion predicate\n"
        "        )\n"
        "        from dist_moe.kernels.grouped_gemm import (\n",
    ),
    # 2. Same eligibility as forward (api.py:2086-2091) minus the clip-stats
    #    term: backward's standalone swiglu_fwd never collects clip counters.
    (
        "        assert fprop_config is not None and wgrad_config is not None\n",
        "        assert fprop_config is not None and wgrad_config is not None\n"
        "        fused_swiglu_recompute = (  # CHANGED: emit H2 from the recompute dispatch\n"
        '            config.activation in ("swiglu", "swiglu_clamped")  # CHANGED\n'
        "            and _native_swiglu_fwd_config_is_compatible(fprop_config, w2_EDF.shape[2])  # CHANGED\n"
        "        )  # CHANGED\n",
    ),
    # 3. Recompute dispatch: same swiglu_* arguments as the forward call
    #    (api.py:2124-2127). Still predicated on need_recompute.
    (
        "                y=backward_plan.h1_offset,\n"
        "                x_gathered=backward_plan.x_gathered_offset,\n"
        "                conditional_execution=need_recompute,\n"
        "                config=fprop_config,\n"
        "                estimate_recv_num_tokens=estimated_num_recv_rows,\n"
        "            )\n",
        "                y=backward_plan.h1_offset,\n"
        "                x_gathered=backward_plan.x_gathered_offset,\n"
        "                conditional_execution=need_recompute,\n"
        "                config=fprop_config,\n"
        "                estimate_recv_num_tokens=estimated_num_recv_rows,\n"
        "                swiglu_h2=backward_plan.h2_offset if fused_swiglu_recompute else None,  # CHANGED\n"
        '                swiglu_clamped=config.activation == "swiglu_clamped",  # CHANGED\n'
        "                swiglu_alpha=config.swiglu_alpha,  # CHANGED\n"
        "                swiglu_limit=config.swiglu_limit,  # CHANGED\n"
        "            )\n",
    ),
    # 4. Standalone SwiGLU now only rebuilds H2 for layers whose H1 was saved
    #    (dispatch skipped). Launch topology is unchanged: both kernels are
    #    always launched and gate on device.
    (
        "            swiglu_fwd(\n"
        "                x=backward_plan.h1_offset,\n"
        "                z=backward_plan.h2_offset,\n",
        "            swiglu_fwd(\n"
        "                x=backward_plan.h1_offset,\n"
        "                z=backward_plan.h2_offset,\n"
        "                conditional_execution=(  # CHANGED: skip when dispatch emitted H2\n"
        "                    torch.logical_not(need_recompute) if fused_swiglu_recompute else None  # CHANGED\n"
        "                ),  # CHANGED\n",
    ),
)

_FILENAME = "<swiglu_recompute: dist_moe.api._Bf16Autograd._backward_impl>"


def patch_source(src: str) -> str:
    """Apply _EDITS to the upstream source; raise if any anchor drifted."""
    for i, (old, new) in enumerate(_EDITS, 1):
        n = src.count(old)
        if n != 1:
            raise RuntimeError(
                f"swiglu_recompute: edit {i} anchor matched {n} times; "
                "dist_moe.api._backward_impl changed upstream, refusing to patch"
            )
        src = src.replace(old, new)
    return src


def apply() -> None:
    from dist_moe import api

    cls = api._Bf16Autograd
    if getattr(cls, "_exp_swiglu_recompute_patched", False):
        return
    orig = cls.__dict__["_backward_impl"]
    if not isinstance(orig, staticmethod):
        raise RuntimeError("swiglu_recompute: _backward_impl is not a staticmethod")
    func = orig.__func__
    src = inspect.getsource(func)
    if not src.lstrip().startswith("@staticmethod"):
        raise RuntimeError("swiglu_recompute: unexpected _backward_impl source layout")
    new_src = textwrap.dedent(patch_source(src))

    dump = os.environ.get("EXP_SWIGLU_RECOMPUTE_DUMP")
    if dump:
        with open(dump, "w") as f:
            f.write(new_src)
    linecache.cache[_FILENAME] = (
        len(new_src),
        None,
        new_src.splitlines(True),
        _FILENAME,
    )
    namespace: dict = {}
    # Globals are dist_moe.api's own module dict, so every name the body
    # resolves (_get_context, _ep_barrier, torch, ...) is the upstream object.
    exec(compile(new_src, _FILENAME, "exec"), api.__dict__, namespace)
    patched = namespace["_backward_impl"]
    if not isinstance(patched, staticmethod):
        raise RuntimeError("swiglu_recompute: patched _backward_impl lost @staticmethod")
    patched.__func__.__doc__ = func.__doc__
    patched.__func__.__qualname__ = func.__qualname__
    patched.__func__.__module__ = func.__module__
    cls._exp_swiglu_recompute_orig_backward_impl = orig
    cls._backward_impl = patched
    cls._exp_swiglu_recompute_patched = True
    print("[swiglu_recompute] patched dist_moe BF16 backward: fused emit-SwiGLU recompute", flush=True)


if ENABLED:
    apply()
