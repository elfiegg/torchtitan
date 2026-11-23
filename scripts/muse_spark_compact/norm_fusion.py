"""Fuse the Muse Spark residual-stream norm chain into fewer compiled regions.

Today every block runs (forward, and again in the remat recompute):

    attention_norm(x)            compiled region, writes fp32  -> eager .to(bf16)
    post_attention_norm(attn)    compiled region
    attention_residual(x, br)    compiled region
    ffn_norm(stream)             compiled region, writes fp32  -> eager .to(bf16)
    post_ffn_norm(ffn_out)       compiled region
    ffn_residual(stream, br)     compiled region

The two eager ``.to(self.branch_dtype)`` casts (and their backward bf16->fp32
casts) are separate full-size kernels on the compute stream, and the
region boundaries force the fp32 stream and its gradient through HBM between
kernels (plus one eager fp32 ``add`` for the stream gradient).

EXP_NORM_FUSION selects how much is fused (default "0": no-op):

  "cast"  attention_norm(x).to(bf16) and ffn_norm(stream).to(bf16) each become
          ONE compiled graph (same body + the final cast). Removes 4 fwd f32->bf16
          casts and 2 bwd bf16->f32 casts per layer-microbatch and lets the norm
          kernels write/read bf16 instead of fp32. Same math, same graph prefix:
          bitwise-identical up to Inductor's autotuned reduction block size.
  "full"  "cast" plus: post_attention_norm + attention_residual + ffn_norm(+cast)
          in one compiled graph, and post_ffn_norm + ffn_residual in one
          compiled graph. Also removes one eager fp32 stream-gradient add per
          layer-microbatch. Inductor may elide the bf16 rounding of the
          post-norm branch inside a fused graph (it keeps fp32), so results are
          within bf16 rounding, not bitwise.

How: the fused helpers call the ORIGINAL, undecorated module bodies
(``inspect.unwrap`` of the ``@local_compile`` methods), so the math is exactly
upstream's; only the compile-region boundaries move. They are registered
under the existing "muse_spark_rmsnorm" local_compile region, so they are
compiled exactly when the recipe compiles its norms (no config change). The
block forward is rewritten from its upstream source with exact-match edits
(each must match exactly once, otherwise the patch raises), every changed
line carries ``# CHANGED``, compiled against the model module's globals.

Apply by importing this module before the model is built (launch.py import
list). Set EXP_NORM_FUSION_DUMP=<path> to write the patched forward source.
"""

import __future__
import functools
import inspect
import linecache
import os
import textwrap

MODE = os.environ.get("EXP_NORM_FUSION", "0")
if MODE not in ("0", "cast", "full"):
    raise ValueError(f"EXP_NORM_FUSION={MODE!r}; expected 0, cast or full")

# --- exact-text edits against MuseSparkTransformerBlock.forward (class indentation) ---

_EDIT_ATTN_NORM = (
    "        attention_input_TD = self.attention_norm(x).to(self.branch_dtype)\n",
    "        attention_input_TD = _exp_norm_fusion.rmsnorm_to(  # CHANGED: cast inside the norm graph\n"
    "            self.attention_norm, x, self.branch_dtype  # CHANGED\n"
    "        )  # CHANGED\n",
)
_EDIT_FFN_NORM = (
    "        ffn_input_TD = self.ffn_norm(stream_TD).to(self.branch_dtype)\n",
    "        ffn_input_TD = _exp_norm_fusion.rmsnorm_to(  # CHANGED: cast inside the norm graph\n"
    "            self.ffn_norm, stream_TD, self.branch_dtype  # CHANGED\n"
    "        )  # CHANGED\n",
)
_EDIT_ATTN_TAIL = (
    "        attention_branch_TD = self.post_attention_norm(attention_output_TD)\n"
    "        stream_TD = self.attention_residual(x, attention_branch_TD)\n"
    "\n"
    "        ffn_input_TD = self.ffn_norm(stream_TD).to(self.branch_dtype)\n",
    "        stream_TD, ffn_input_TD = _exp_norm_fusion.attn_tail(  # CHANGED: one graph\n"
    "            self.post_attention_norm,  # CHANGED\n"
    "            self.attention_residual,  # CHANGED\n"
    "            self.ffn_norm,  # CHANGED\n"
    "            x,  # CHANGED\n"
    "            attention_output_TD,  # CHANGED\n"
    "            self.branch_dtype,  # CHANGED\n"
    "        )  # CHANGED\n",
)
_EDIT_FFN_TAIL = (
    "        ffn_branch_TD = self.post_ffn_norm(ffn_output_TD)\n"
    "        return self.ffn_residual(stream_TD, ffn_branch_TD)\n",
    "        return _exp_norm_fusion.ffn_tail(  # CHANGED: one graph\n"
    "            self.post_ffn_norm, self.ffn_residual, stream_TD, ffn_output_TD  # CHANGED\n"
    "        )  # CHANGED\n",
)
_EDITS = {
    "cast": (_EDIT_ATTN_NORM, _EDIT_FFN_NORM),
    "full": (_EDIT_ATTN_NORM, _EDIT_ATTN_TAIL, _EDIT_FFN_TAIL),
}

# Lines the fused helpers rely on in the module bodies they call. They are
# called verbatim, so this only guards the assumptions (dtype flow, arguments).
_BODY_ANCHORS = {
    "MuseSparkRMSNorm.forward": (
        "effective_weight_D = self._get_effective_weight()",
        "output_TD = F.rms_norm(",
        "return output_TD.to(input_dtype)",
    ),
    "MuseSparkRMSNormWithInputScale.forward": (
        "input = effective_weight_D * input",
        "output_TD = F.rms_norm(input, self.normalized_shape, None, self.eps)",
        "return output_TD.to(input_dtype)",
    ),
    "MuseSparkNormPreservingResidualConnection.forward": (
        "def forward(\n        self,\n        stream_TD: torch.Tensor,\n        branch_TD: torch.Tensor,\n",
        "output_TD = alpha_1D * stream_TD + beta_1D * branch_TD.to(stream_TD.dtype)",
        "return output_TD.to(stream_TD.dtype)",
    ),
}

_FILENAME = "<norm_fusion: torchtitan.models.muse_spark.model.MuseSparkTransformerBlock.forward>"


def patch_source(src: str, mode: str) -> str:
    """Apply the mode's edits to the upstream block forward; raise on drift."""
    for i, (old, new) in enumerate(_EDITS[mode], 1):
        n = src.count(old)
        if n != 1:
            raise RuntimeError(
                f"norm_fusion: edit {i} ({mode}) anchor matched {n} times; "
                "MuseSparkTransformerBlock.forward changed upstream, refusing to patch"
            )
        src = src.replace(old, new)
    return src


def _check_body_anchors(m) -> None:
    for qualname, needles in _BODY_ANCHORS.items():
        cls_name, attr = qualname.split(".")
        src = inspect.getsource(inspect.unwrap(getattr(getattr(m, cls_name), attr)))
        for needle in needles:
            if src.count(needle) != 1:
                raise RuntimeError(
                    f"norm_fusion: {qualname} no longer contains {needle!r} exactly once; "
                    "refusing to patch"
                )


@functools.lru_cache(maxsize=None)
def helpers():
    """Build (once) and return the fused helpers.

    Registration with ``local_compile`` must happen before
    ``LocalCompileConfig.apply()`` binds the regions, i.e. before the model is
    built. ``apply()`` calls this at import; the unit test calls it directly.
    """
    import types

    from torchtitan.distributed.local_compile import local_compile
    from torchtitan.models.muse_spark import model as m

    _check_body_anchors(m)
    rms_cls = m.MuseSparkRMSNorm
    scaled_cls = m.MuseSparkRMSNormWithInputScale
    res_cls = m.MuseSparkNormPreservingResidualConnection
    # The @local_compile-decorated methods; their __wrapped__ is the plain body.
    rms_fwd = rms_cls.forward
    scaled_fwd = scaled_cls.forward
    res_fwd = res_cls.forward
    rms_body = inspect.unwrap(rms_fwd)
    scaled_body = inspect.unwrap(scaled_fwd)
    res_body = inspect.unwrap(res_fwd)
    for wrapped, body in ((rms_fwd, rms_body), (scaled_fwd, scaled_body), (res_fwd, res_body)):
        if wrapped is body:
            raise RuntimeError(f"norm_fusion: {wrapped.__qualname__} is no longer @local_compile-wrapped")

    # Same region name, batch_invariant flag and recompile budget as the norms
    # these graphs contain: compiled exactly when "muse_spark_rmsnorm" is.
    @local_compile("muse_spark_rmsnorm", batch_invariant=False, recompile_limit=32)
    def _rmsnorm_to(norm, x, dtype):
        return rms_body(norm, x).to(dtype)

    @local_compile("muse_spark_rmsnorm", batch_invariant=False, recompile_limit=16)
    def _attn_tail(post_norm, residual, ffn_norm, x, attn_out, dtype):
        branch = rms_body(post_norm, attn_out)
        stream = res_body(residual, x, branch)
        return stream, rms_body(ffn_norm, stream).to(dtype)

    @local_compile("muse_spark_rmsnorm", batch_invariant=False, recompile_limit=16)
    def _ffn_tail(post_norm, residual, stream, ffn_out):
        return res_body(residual, stream, scaled_body(post_norm, ffn_out))

    # Eager guards: anything that is not exactly the upstream module type runs
    # the original module calls (keeps subclasses/LoRA wrappers correct).
    def rmsnorm_to(norm, x, dtype):
        if type(norm).forward is not rms_fwd:
            return norm(x).to(dtype)
        return _rmsnorm_to(norm, x, dtype)

    def attn_tail(post_norm, residual, ffn_norm, x, attn_out, dtype):
        if (
            type(post_norm).forward is not rms_fwd
            or type(residual).forward is not res_fwd
            or type(ffn_norm).forward is not rms_fwd
        ):
            stream = residual(x, post_norm(attn_out))
            return stream, ffn_norm(stream).to(dtype)
        return _attn_tail(post_norm, residual, ffn_norm, x, attn_out, dtype)

    def ffn_tail(post_norm, residual, stream, ffn_out):
        if type(post_norm).forward is not scaled_fwd or type(residual).forward is not res_fwd:
            return residual(stream, post_norm(ffn_out))
        return _ffn_tail(post_norm, residual, stream, ffn_out)

    return types.SimpleNamespace(rmsnorm_to=rmsnorm_to, attn_tail=attn_tail, ffn_tail=ffn_tail)


def build_forward(mode: str):
    """Return the patched MuseSparkTransformerBlock.forward for ``mode``."""
    from torchtitan.models.muse_spark import model as m

    h = helpers()
    func = m.MuseSparkTransformerBlock.__dict__["forward"]
    src = inspect.getsource(func)
    if not src.startswith("    def forward("):
        raise RuntimeError("norm_fusion: unexpected MuseSparkTransformerBlock.forward layout")
    new_src = textwrap.dedent(patch_source(src, mode))
    dump = os.environ.get("EXP_NORM_FUSION_DUMP")
    if dump:
        with open(dump, "w") as f:
            f.write(new_src)
    linecache.cache[_FILENAME] = (len(new_src), None, new_src.splitlines(True), _FILENAME)
    # Globals: a copy of the model module's dict plus the helper namespace, so
    # every other name (remat, MoE, ...) resolves to the upstream object.
    g = dict(m.__dict__)
    g["_exp_norm_fusion"] = h
    ns: dict = {}
    code = compile(
        new_src,
        _FILENAME,
        "exec",
        flags=__future__.annotations.compiler_flag,  # model.py uses postponed annotations
        dont_inherit=True,
    )
    exec(code, g, ns)
    patched = ns["forward"]
    patched.__qualname__ = func.__qualname__
    patched.__module__ = func.__module__
    patched.__doc__ = func.__doc__
    return patched


def apply(mode: str = MODE) -> None:
    from torchtitan.models.muse_spark import model as m

    cls = m.MuseSparkTransformerBlock
    if getattr(cls, "_exp_norm_fusion_mode", None) is not None:
        if cls._exp_norm_fusion_mode != mode:
            raise RuntimeError("norm_fusion: already applied with a different mode")
        return
    patched = build_forward(mode)
    cls._exp_norm_fusion_orig_forward = cls.__dict__["forward"]
    cls.forward = patched
    cls._exp_norm_fusion_mode = mode
    print(f"[norm_fusion] patched MuseSparkTransformerBlock.forward (mode={mode})", flush=True)


if MODE != "0":
    apply()
