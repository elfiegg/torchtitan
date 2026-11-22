# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-GEMM tile configs for the Dist-MoE BF16 grouped GEMMs (EXP_MOE_TILES).

Upstream picks one BF16 training config per *config variable*, not per GEMM:
``fprop_config`` (``grouped_gemm_training_config()``) feeds the FC13 dispatch
fprop (forward and recompute), the FC2 dgrad dispatch and the FC13 dgrad +
SwiGLU-bwd combine; ``combine_config`` feeds the FC2 combine fprop (forward and
recompute). ``Config.bf16_grouped_gemm_preset`` is a single global override for
all of them, and only accepts 7 registered names (no NUM_C_STAGES, no
EPILOGUE_SUBTILE / NUM_SMEM_BUFFERS choice).

This module wraps the five distributed grouped-GEMM launchers in
``dist_moe.kernels.dist_grouped_gemm`` and swaps only the ``config=`` keyword
for the GEMMs named in the spec. ``dist_moe.api`` imports those launchers at
call time inside ``_Bf16Autograd.forward`` / ``_backward_impl``, so the
wrappers see every call, including the source-patched bodies installed by
swiglu_recompute and h3_offload (any import order). Nothing in
dist_moe's source is rewritten. Custom configs are registered in
``GROUPED_GEMM_CONFIGS`` under ``exp_tiles_<label>`` names; the kernel and
compile caches key on the name, so they never collide with upstream entries.

Drift protection (same spirit as swiglu_recompute): on apply, a fixed list
of exact-text anchors must each match ``dist_moe.api``'s source the expected
number of times (call sites, the weight each launcher receives, which config
variable it receives). Any mismatch raises and nothing is patched. At call
time, if the spec carries ``"expect"`` (tile_sweep.py always writes it) the
incoming upstream config name must equal it, otherwise the wrapper raises
instead of overriding a baseline the sweep never measured.

Spec: EXP_MOE_TILES=<path to JSON file> or inline JSON (first char ``{``).
Unset or empty = no-op. ``apply(spec)`` applies it explicitly.

    {
      "fc13_fprop": {"config": {"NUM_CTAS": 2, "NUM_MMAS": 2, "BLOCK_SIZE_M": 512,
                                "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                                "NUM_SMEM_BUFFERS": 4, "NUM_TMEM_BUFFERS": 1,
                                "NUM_TILE_BUFFERS": 2, "EPILOGUE_SUBTILE": 4,
                                "NUM_C_STAGES": 2},
                     "expect": "training_2cta2mma_bm512_bn256"},
      "fc2_dgrad":  {"config": "training_2cta1mma_bm256_bn256",
                     "expect": "training_2cta2mma_bm512_bn256"}
    }

Keys (missing keys keep upstream's choice; "_meta" is ignored):
  fc13_fprop  FC13 (gate/up) dispatch fprop, forward and recompute
              (launcher dist_grouped_gemm_fprop_dispatch)
  fc2_fprop   FC2 (down) combine fprop, forward and recompute
              (dist_grouped_gemm_fprop_combine)
  fc2_dgrad   FC2 dgrad dispatch (dist_grouped_gemm_dgrad_dispatch)
  fc13_dgrad  FC13 dgrad + fused SwiGLU-bwd combine
              (dist_grouped_gemm_dgrad_swiglu_bwd_combine, and the unfused
              dist_grouped_gemm_dgrad_combine fallback)
"config" is a registered GROUPED_GEMM_CONFIGS name or a full config dict
(NUM_C_STAGES optional, defaults to NUM_TMEM_BUFFERS like upstream).

Offline check (no torch/dist_moe needed):
    python -m torchtitan.experiments.dist_moe_patches.moe_tiles <spec.json|inline>

TODO: per-GEMM BF16 configs belong in meta-pytorch/dist_moe (``dist_moe/api.py``
``_Bf16Autograd`` config selection); delete this module once they land there.
"""

import functools
import inspect
import json
import os

ENV = "EXP_MOE_TILES"

GEMMS = ("fc13_fprop", "fc2_fprop", "fc2_dgrad", "fc13_dgrad")

# launcher name in dist_moe.kernels.dist_grouped_gemm -> GEMM key
LAUNCHERS = {
    "dist_grouped_gemm_fprop_dispatch": "fc13_fprop",
    "dist_grouped_gemm_fprop_combine": "fc2_fprop",
    "dist_grouped_gemm_dgrad_dispatch": "fc2_dgrad",
    "dist_grouped_gemm_dgrad_swiglu_bwd_combine": "fc13_dgrad",
    "dist_grouped_gemm_dgrad_combine": "fc13_dgrad",
}

# How each GEMM's epilogue stages C in shared memory (dist_grouped_gemm_kernel.py
# DistGroupedGemmKernel._setup_attributes):
#   emit_h1   DISPATCH_SWIGLU_FWD + swiglu_emit_h1: NUM_C_STAGES TMA-store stages
#             of the h1 sub-tile, which must be exactly 64 columns wide.
#   tma_store plain DISPATCH: NUM_C_STAGES swizzled TMA-store stages.
#   combine   COMBINE / COMBINE_SWIGLU_BWD: ONE row-major scatter stage with an
#             8-element row skew when NUM_TMEM_BUFFERS == 1 or NUM_C_STAGES > 1;
#             NUM_C_STAGES has no other effect.
EPILOGUE_MODE = {
    "fc13_fprop": "emit_h1",
    "fc2_fprop": "combine",
    "fc2_dgrad": "tma_store",
    "fc13_dgrad": "combine",
}

REQUIRED_KEYS = (
    "NUM_CTAS",
    "NUM_MMAS",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "NUM_SMEM_BUFFERS",
    "NUM_TMEM_BUFFERS",
    "NUM_TILE_BUFFERS",
    "EPILOGUE_SUBTILE",
)
OPTIONAL_KEYS = ("NUM_C_STAGES",)

# Mirrors dist_moe.kernels.config: BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES (the
# per-block dynamic SMEM cap dist_moe derives every config against) and
# SM100_TMEM_COLS.
SMEM_LIMIT_BYTES = 232448
TMEM_COLS = 512
# Fixed per-CTA SMEM outside A/B/C: 3 TMA descriptors (384 B), mbarriers,
# tile-id ring, split sizes, gather slot, TMEM holding word, alignment slack.
# Chosen so the hardware-proven wgrad config (S4/C2 at bm512 bn256, 229376 B of
# A/B/C) passes with ~1 KB to spare.
SMEM_OVERHEAD_BYTES = 2048
NATIVE_SWIGLU_EMIT_H1_COLS = 64  # 2 * NATIVE_SWIGLU_EMIT_H1_GROUP_SIZE

# Exact-text anchors against dist_moe/api.py, with expected match counts.
# They pin: which launcher each GEMM uses, which weight it receives, and which
# config variable it receives. If upstream rewires any of this, apply() raises.
_API_ANCHORS = (
    # Launcher call-site counts: no new, unmapped call sites.
    ("dist_grouped_gemm_fprop_dispatch(", 2),
    ("dist_grouped_gemm_fprop_combine(", 2),
    ("dist_grouped_gemm_dgrad_dispatch(", 1),
    ("dist_grouped_gemm_dgrad_swiglu_bwd_combine(", 1),
    ("dist_grouped_gemm_dgrad_combine(", 1),
    ("config=fprop_config,", 5),
    ("config=combine_config,", 2),
    # Config derivation (forward and backward).
    (
        "        fprop_config = _bf16_compute_config(\n"
        "            config, routed_rows=estimated_num_recv_rows, num_groups=w13_EFD.shape[0]\n"
        "        )\n",
        2,
    ),
    # Forward FC13 dispatch (w13, fprop_config, fused SwiGLU emit).
    (
        "            dist_grouped_gemm_fprop_dispatch(\n"
        "                x=x_TD,\n"
        "                w=w13_EFD,\n",
        1,
    ),
    (
        "                config=fprop_config,\n"
        "                estimate_recv_num_tokens=estimated_num_recv_rows,\n"
        "                swiglu_h2=forward_plan.h2_offset if fused_swiglu_forward else None,\n",
        1,
    ),
    # Forward FC2 combine (w2, combine_config).
    (
        "            h3_MD = dist_grouped_gemm_fprop_combine(\n"
        "                x=forward_plan.h2_offset,\n"
        "                w=w2_EDF,\n",
        1,
    ),
    (
        "                activation_buffer=activation_buffer.buffer,\n"
        "                config=combine_config,\n"
        "                estimate_recv_num_tokens=estimated_num_recv_rows,\n"
        "            )\n",
        1,
    ),
    # Backward recompute FC13 dispatch and FC2 combine.
    (
        "            dist_grouped_gemm_fprop_dispatch(\n"
        "                x=x_offset,\n"
        "                w=w13_EFD,\n",
        1,
    ),
    (
        "            h3_saved_MD = dist_grouped_gemm_fprop_combine(\n"
        "                x=backward_plan.h2_offset,\n"
        "                w=w2_EDF,\n",
        1,
    ),
    (
        "                conditional_execution=need_recompute,\n"
        "                config=combine_config,\n",
        1,
    ),
    # FC2 dgrad dispatch (w2, fprop_config).
    (
        "            dist_grouped_gemm_dgrad_dispatch(\n"
        "                dy=grad_h3_MD,\n"
        "                w=w2_EDF,\n",
        1,
    ),
    (
        "                dy_gathered=backward_plan.grad_h3_gathered_offset,\n"
        "                config=fprop_config,\n"
        "            )\n",
        1,
    ),
    # FC13 dgrad + SwiGLU-bwd combine (w13, fprop_config) and its fallback.
    (
        "                grad_x_TKD, _ = dist_grouped_gemm_dgrad_swiglu_bwd_combine(\n"
        "                    grad_h2=backward_plan.grad_h2_offset,\n"
        "                    h1=backward_plan.h1_offset,\n"
        "                    w=w13_EFD,\n",
        1,
    ),
    (
        "                grad_x_TKD = dist_grouped_gemm_dgrad_combine(\n"
        "                    dy=backward_plan.grad_h1_offset,\n"
        "                    w=w13_EFD,\n",
        1,
    ),
)

# Anchors against dist_moe.kernels.config / grouped_gemm_kernel: configs are
# resolved by name from GROUPED_GEMM_CONFIGS, and NUM_C_STAGES is optional
# with the TMEM-buffer default this module assumes.
_CONFIG_ANCHORS = (("    config = GROUPED_GEMM_CONFIGS[base_name]\n", 1),)
_KERNEL_ANCHORS = (
    (
        '        self.NUM_C_STAGES: int = c.get("NUM_C_STAGES", self.NUM_TMEM_BUFFERS)\n',
        1,
    ),
)


# --------------------------------------------------------------------------
# Pure-python config helpers (shared with tile_sweep.py; no torch import).
# --------------------------------------------------------------------------


def make_config(*, ctas, mmas, bn, smem, tmem, c_stages, epi, bk=64):
    """Full native config dict; BLOCK_SIZE_M = 128 rows per CTA per MMA atom."""
    return {
        "NUM_CTAS": ctas,
        "NUM_MMAS": mmas,
        "BLOCK_SIZE_M": 128 * ctas * mmas,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "NUM_SMEM_BUFFERS": smem,
        "NUM_TMEM_BUFFERS": tmem,
        "NUM_TILE_BUFFERS": tmem + 1,
        "EPILOGUE_SUBTILE": epi,
        "NUM_C_STAGES": c_stages,
    }


def normalized(cfg):
    """Copy with NUM_C_STAGES made explicit (upstream default: NUM_TMEM_BUFFERS)."""
    out = {k: int(cfg[k]) for k in REQUIRED_KEYS}
    out["NUM_C_STAGES"] = int(cfg.get("NUM_C_STAGES", cfg["NUM_TMEM_BUFFERS"]))
    return out


def config_label(cfg):
    c = normalized(cfg)
    return (
        f"c{c['NUM_CTAS']}m{c['NUM_MMAS']}_bm{c['BLOCK_SIZE_M']}_bn{c['BLOCK_SIZE_N']}"
        f"_s{c['NUM_SMEM_BUFFERS']}_t{c['NUM_TMEM_BUFFERS']}"
        f"_c{c['NUM_C_STAGES']}_e{c['EPILOGUE_SUBTILE']}"
        + ("" if c["BLOCK_SIZE_K"] == 64 else f"_bk{c['BLOCK_SIZE_K']}")
    )


def registered_name(cfg):
    return "exp_tiles_" + config_label(cfg)


def _align(x, a):
    return (x + a - 1) // a * a


def smem_breakdown(cfg, mode):
    """Per-CTA dynamic SMEM estimate (bytes) for one epilogue mode."""
    c = normalized(cfg)
    ctas, mmas = c["NUM_CTAS"], c["NUM_MMAS"]
    bm, bn, bk = c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"]
    s, t, cs, e = (
        c["NUM_SMEM_BUFFERS"],
        c["NUM_TMEM_BUFFERS"],
        c["NUM_C_STAGES"],
        c["EPILOGUE_SUBTILE"],
    )
    cta_m = bm // ctas
    # 2-CTA splits both A (M) and B (N) across the CTA pair.
    ab_stage = _align((cta_m * bk + (bn // ctas) * bk) * 2, 1024)
    tile_m = min(cta_m, 128)
    tile_n = bn // e
    split_atom = t == 1 and mmas == 2
    if mode == "combine":
        skew = 8 if (t == 1 or cs > 1) else 0
        c_bytes = _align(tile_m * (tile_n + skew) * 2, 1024)
        extra = _align(tile_m * 8 * (mmas if split_atom else 1), 128)  # scatter ptrs
    elif mode in ("tma_store", "emit_h1"):
        c_bytes = cs * _align(tile_m * tile_n * 2, 1024)
        extra = 0
    else:
        raise ValueError(f"unknown epilogue mode {mode!r}")
    total = s * ab_stage + c_bytes + extra + SMEM_OVERHEAD_BYTES
    return {
        "ab_stage": ab_stage,
        "ab": s * ab_stage,
        "c": c_bytes,
        "extra": extra,
        "total": total,
    }


def tmem_cols(cfg):
    c = normalized(cfg)
    return c["BLOCK_SIZE_N"] * c["NUM_MMAS"] * c["NUM_TMEM_BUFFERS"]


def smem_flop_per_byte(cfg):
    """MMA FLOPs per byte of A/B staged into each CTA's SMEM."""
    c = normalized(cfg)
    cta_m = c["BLOCK_SIZE_M"] // c["NUM_CTAS"]
    flops = 2 * cta_m * c["BLOCK_SIZE_N"] * c["BLOCK_SIZE_K"]
    return flops / smem_breakdown(cfg, "tma_store")["ab_stage"]


def check_config(cfg, gemm, *, n_out=None):
    """Return a list of reasons the config is invalid for this GEMM ([] = OK).

    n_out: output feature width of the GEMM, to reject ragged N tiles.
    """
    problems = []
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    unknown = [k for k in cfg if k not in REQUIRED_KEYS + OPTIONAL_KEYS]
    if missing:
        return [f"missing keys {missing}"]
    if unknown:
        problems.append(f"unknown keys {unknown}")
    try:
        c = normalized(cfg)
    except (TypeError, ValueError) as exc:
        return [f"non-integer field: {exc}"]
    if gemm not in EPILOGUE_MODE:
        return [f"unknown GEMM key {gemm!r}; expected one of {GEMMS}"]
    ctas, mmas = c["NUM_CTAS"], c["NUM_MMAS"]
    bm, bn, bk = c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"]
    s, t, cs, e = (
        c["NUM_SMEM_BUFFERS"],
        c["NUM_TMEM_BUFFERS"],
        c["NUM_C_STAGES"],
        c["EPILOGUE_SUBTILE"],
    )
    if ctas not in (1, 2) or mmas not in (1, 2):
        problems.append("NUM_CTAS and NUM_MMAS must be 1 or 2")
    if bm != 128 * ctas * mmas:
        problems.append(
            f"BLOCK_SIZE_M={bm} != 128*NUM_CTAS*NUM_MMAS (128-row MMA atom per CTA)"
        )
    if bk != 64:
        problems.append("BLOCK_SIZE_K must be 64 (TMA-store/scatter configs)")
    if bn % 64 != 0 or not 64 <= bn <= 256:
        problems.append(f"BLOCK_SIZE_N={bn} must be 64, 128, 192 or 256")
    if n_out is not None and n_out % bn != 0:
        problems.append(f"BLOCK_SIZE_N={bn} does not divide N={n_out} (ragged tiles)")
    if t < 1 or tmem_cols(c) > TMEM_COLS:
        problems.append(f"TMEM: {bn}*{mmas}*{t}={tmem_cols(c)} cols > {TMEM_COLS}")
    if c["NUM_TILE_BUFFERS"] != t + 1:
        problems.append("NUM_TILE_BUFFERS must be NUM_TMEM_BUFFERS + 1")
    if e not in (1, 2, 4, 8) or bn % e != 0 or (bn // e) % 8 != 0 or bn // e < 32:
        problems.append(
            f"EPILOGUE_SUBTILE={e}: sub-tile {bn}/{e} must be a multiple of 8 and >= 32"
        )
    if not 1 <= cs <= 8:
        problems.append(f"NUM_C_STAGES={cs} not in [1, 8]")
    if s < 2:
        problems.append("NUM_SMEM_BUFFERS must be >= 2")
    mode = EPILOGUE_MODE[gemm]
    if mode == "emit_h1" and bn // max(e, 1) != NATIVE_SWIGLU_EMIT_H1_COLS:
        problems.append(
            f"fused SwiGLU emit needs a {NATIVE_SWIGLU_EMIT_H1_COLS}-column epilogue sub-tile; "
            f"got {bn}/{e}"
        )
    if mode == "combine" and t > 1 and cs == 1:
        problems.append(
            "combine epilogue with NUM_TMEM_BUFFERS>1 and NUM_C_STAGES=1 drops the "
            "8-element row skew (32-way SMEM bank conflicts); use NUM_C_STAGES>=2"
        )
    if not problems:
        total = smem_breakdown(c, mode)["total"]
        if total > SMEM_LIMIT_BYTES:
            problems.append(f"SMEM estimate {total} B > {SMEM_LIMIT_BYTES} B")
    return problems


def load_spec(text):
    """Parse EXP_MOE_TILES (path or inline JSON) into {gemm: (config, expect)}."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        raw = json.loads(text)
    else:
        with open(text) as f:
            raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{ENV}: spec must be a JSON object")
    spec = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key not in GEMMS:
            raise ValueError(f"{ENV}: unknown GEMM key {key!r}; expected {GEMMS}")
        if isinstance(value, (str, dict)) and not (
            isinstance(value, dict) and "config" in value
        ):
            value = {"config": value}
        if not isinstance(value, dict) or "config" not in value:
            raise ValueError(f"{ENV}: {key}: expected {{'config': ..., 'expect': ...}}")
        extra = set(value) - {"config", "expect", "label", "note"}
        if extra:
            raise ValueError(f"{ENV}: {key}: unknown fields {sorted(extra)}")
        config = value["config"]
        if not isinstance(config, (str, dict)):
            raise ValueError(f"{ENV}: {key}: config must be a name or a dict")
        expect = value.get("expect")
        if expect is not None and not isinstance(expect, str):
            raise ValueError(f"{ENV}: {key}: expect must be a config name or null")
        spec[key] = (config, expect)
    return spec


# --------------------------------------------------------------------------
# dist_moe integration.
# --------------------------------------------------------------------------


def _check_anchors(source, anchors, where):
    for text, want in anchors:
        got = source.count(text)
        if got != want:
            first = text.strip().splitlines()[0]
            raise RuntimeError(
                f"moe_tiles: anchor {first!r} matched {got} times in {where}, expected "
                f"{want}; dist_moe changed upstream, refusing to patch"
            )


def register_config(cfg, gemm=None):
    """Register a config dict in GROUPED_GEMM_CONFIGS; return its name."""
    from dist_moe.kernels import config as dm_config

    c = normalized(cfg)
    if gemm is not None:
        problems = check_config(c, gemm)
        if problems:
            raise ValueError(
                f"moe_tiles: {gemm} {config_label(c)}: " + "; ".join(problems)
            )
    name = registered_name(c)
    existing = dm_config.GROUPED_GEMM_CONFIGS.get(name)
    if existing is not None and normalized(existing) != c:
        raise RuntimeError(f"moe_tiles: {name} already registered with {existing!r}")
    dm_config.GROUPED_GEMM_CONFIGS[name] = dict(c)
    return name


def resolve_config(config, gemm):
    """Config name for a spec entry (registering dicts); validates either form."""
    from dist_moe.kernels import config as dm_config

    if isinstance(config, str):
        if config not in dm_config.GROUPED_GEMM_CONFIGS:
            raise ValueError(f"moe_tiles: {gemm}: unknown config name {config!r}")
        problems = check_config(dm_config.GROUPED_GEMM_CONFIGS[config], gemm)
        if problems:
            raise ValueError(f"moe_tiles: {gemm}: {config}: " + "; ".join(problems))
        return config
    return register_config(config, gemm)


_state = {"originals": {}, "logged": set(), "spec": {}}


def _make_wrapper(launcher, orig, gemm, new_name, expect):
    from dist_moe.kernels import dist_grouped_gemm as dgg

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if "config" not in kwargs:
            raise RuntimeError(
                f"moe_tiles: {launcher} called without config=; the Dist-MoE call "
                "convention changed, refusing to override"
            )
        current = kwargs["config"]
        if expect is not None and current != expect and current != new_name:
            raise RuntimeError(
                f"moe_tiles: {gemm}: upstream passed config={current!r} but the spec was "
                f"measured against {expect!r} (batch size or upstream tuning changed); "
                f"re-run tile_sweep.py or drop {gemm} from {ENV}"
            )
        if gemm == "fc13_fprop" and kwargs.get("swiglu_h2") is not None:
            w = kwargs.get("w", args[1] if len(args) > 1 else None)
            if not dgg._native_swiglu_fwd_config_is_compatible(
                new_name, int(w.shape[1]) // 2
            ):
                raise RuntimeError(
                    f"moe_tiles: {new_name} cannot emit the fused SwiGLU forward"
                )
        kwargs["config"] = new_name
        if (gemm, launcher) not in _state["logged"]:
            _state["logged"].add((gemm, launcher))
            print(
                f"[moe_tiles] {gemm} ({launcher}): {current} -> {new_name}", flush=True
            )
        return orig(*args, **kwargs)

    wrapper._moe_tiles_original = orig
    return wrapper


def apply(spec=None):
    """Install the per-GEMM overrides. ``spec``: dict from load_spec (default: env)."""
    if _state["originals"]:
        return
    if spec is None:
        spec = load_spec(os.environ.get(ENV, ""))
    if not spec:
        return
    from dist_moe import api
    from dist_moe.kernels import (
        config as dm_config,
        dist_grouped_gemm as dgg,
        grouped_gemm_kernel as ggk,
    )

    _check_anchors(inspect.getsource(api), _API_ANCHORS, "dist_moe/api.py")
    _check_anchors(inspect.getsource(dm_config), _CONFIG_ANCHORS, "kernels/config.py")
    _check_anchors(
        inspect.getsource(ggk), _KERNEL_ANCHORS, "kernels/grouped_gemm_kernel.py"
    )
    for launcher in LAUNCHERS:
        fn = getattr(dgg, launcher)
        if hasattr(fn, "_moe_tiles_original"):
            raise RuntimeError(f"moe_tiles: {launcher} already wrapped")
        if "config" not in inspect.signature(fn).parameters:
            raise RuntimeError(f"moe_tiles: {launcher} has no config parameter")

    names = {
        gemm: (resolve_config(config, gemm), expect)
        for gemm, (config, expect) in spec.items()
    }
    for expect_name in (e for _, e in names.values() if e is not None):
        if expect_name not in dm_config.GROUPED_GEMM_CONFIGS:
            raise ValueError(
                f"moe_tiles: expect={expect_name!r} is not a registered config"
            )

    originals = {}
    for launcher, gemm in LAUNCHERS.items():
        if gemm not in names:
            continue
        new_name, expect = names[gemm]
        orig = getattr(dgg, launcher)
        originals[launcher] = orig
        setattr(dgg, launcher, _make_wrapper(launcher, orig, gemm, new_name, expect))
    _state["originals"] = originals
    _state["spec"] = names
    summary = ", ".join(f"{g}={n}" for g, (n, _) in names.items())
    print(f"[moe_tiles] per-GEMM Dist-MoE BF16 tiles: {summary}", flush=True)


def remove():
    """Restore the original launchers (tests)."""
    from dist_moe.kernels import dist_grouped_gemm as dgg

    for launcher, orig in _state["originals"].items():
        setattr(dgg, launcher, orig)
    _state["originals"] = {}
    _state["logged"] = set()
    _state["spec"] = {}


def _offline_check(text):
    spec = load_spec(text)
    ok = True
    for gemm, (config, expect) in spec.items():
        if isinstance(config, str):
            print(
                f"{gemm}: registered name {config!r} (resolved at apply), expect={expect}"
            )
            continue
        problems = check_config(config, gemm)
        mode = EPILOGUE_MODE[gemm]
        smem = smem_breakdown(config, mode)["total"] if not problems else None
        status = "OK" if not problems else "INVALID: " + "; ".join(problems)
        print(f"{gemm}: {config_label(config)} smem~{smem} B expect={expect}: {status}")
        ok &= not problems
    return ok


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m torchtitan.experiments.dist_moe_patches.moe_tiles <spec.json | inline JSON>"
        )
    raise SystemExit(0 if _offline_check(sys.argv[1]) else 1)

if os.environ.get(ENV, "").strip():
    apply()
