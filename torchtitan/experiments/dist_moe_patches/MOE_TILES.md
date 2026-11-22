# Per-GEMM tile configs for the Dist-MoE BF16 grouped GEMMs

| File | Purpose |
|---|---|
| `moe_tiles.py` | `EXP_MOE_TILES=<spec.json or inline JSON>`: per-GEMM config override. `python -m torchtitan.experiments.dist_moe_patches.moe_tiles <spec>` checks a spec offline |
| `moe_tiles_muse_spark_816b.json` | Measured spec for Muse Spark 816B BF16 at EP64 (see below) |
| `tile_sweep.py` | One-node sweep (4 GPUs) that measures candidates and writes a spec. `--list` prints the candidates without a GPU |

Line numbers refer to dist_moe at commit `b76e0e0`.

## Problem

Upstream picks one BF16 training config per config *variable*, not per GEMM:

- `fprop_config` (`grouped_gemm_training_config()`) feeds the FC13 dispatch
  fprop, the FC2 dgrad dispatch and the FC13 dgrad + SwiGLU-backward combine
  (api.py:2077, 2363, 2485, 2540).
- `combine_config` feeds the FC2 combine fprop.
- `Config.bf16_grouped_gemm_preset` is a single global override for all of
  them, accepts only 7 registered names, and cannot set `NUM_C_STAGES` or choose
  `EPILOGUE_SUBTILE` / `NUM_SMEM_BUFFERS`.

The four GEMMs have different epilogues and shapes, so the best tile differs.

## Mechanism

`moe_tiles` wraps the five distributed grouped-GEMM launchers in
`dist_moe.kernels.dist_grouped_gemm` and swaps only the `config=` keyword for the
GEMMs named in the spec. `dist_moe.api` imports those launchers at call time
inside `_Bf16Autograd.forward` / `_backward_impl`, so the wrappers see every
call, including the source-patched bodies that `swiglu_recompute` and
`h3_offload` install (any import order). Nothing in dist_moe's source is
rewritten. Custom configs are registered in `GROUPED_GEMM_CONFIGS` under
`exp_tiles_<label>` names; the kernel and compile caches key on the name, so they
never collide with upstream entries.

| Spec key | Launcher (epilogue mode) | N x K at the shapes below | Upstream config |
|---|---|---|---|
| `fc13_fprop` | `dist_grouped_gemm_fprop_dispatch` (DISPATCH_SWIGLU_FWD, emits h1 and h2), forward and recompute | 8192 x 4096 | `training_2cta2mma_bm512_bn256` (S4 T1 C1 E4) |
| `fc2_fprop` | `dist_grouped_gemm_fprop_combine` (COMBINE), forward and recompute | 4096 x 4096 | `training_2cta1mma_bm256_bn256` (S6 T2 C2 E4) |
| `fc2_dgrad` | `dist_grouped_gemm_dgrad_dispatch` (DISPATCH) | 4096 x 4096 | `training_2cta2mma_bm512_bn256` |
| `fc13_dgrad` | `dist_grouped_gemm_dgrad_swiglu_bwd_combine` (COMBINE_SWIGLU_BWD), and the unfused `dist_grouped_gemm_dgrad_combine` fallback | 4096 x 8192 | `training_2cta2mma_bm512_bn256` |

Spec format (missing keys keep upstream's choice; keys starting with `_` are
ignored). `config` is a registered name or a full config dict; `NUM_C_STAGES` is
optional and defaults to `NUM_TMEM_BUFFERS`, as upstream:

```json
{
  "fc13_fprop": {"config": {"NUM_CTAS": 2, "NUM_MMAS": 2, "BLOCK_SIZE_M": 512,
                            "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                            "NUM_SMEM_BUFFERS": 4, "NUM_TMEM_BUFFERS": 1,
                            "NUM_TILE_BUFFERS": 2, "EPILOGUE_SUBTILE": 4,
                            "NUM_C_STAGES": 2},
                 "expect": "training_2cta2mma_bm512_bn256"}
}
```

### Drift protection

- **At apply.** 19 exact-text anchors in `api.py` (launcher call-site counts,
  the weight and the config variable each launcher receives, the config
  derivation) and one each in `kernels/config.py` and
  `kernels/grouped_gemm_kernel.py` must match the expected number of times, and
  every launcher must take `config=`. Any mismatch raises and nothing is
  patched. Every config is validated against the SMEM/TMEM/epilogue rules below.
- **At run time.** Each spec entry carries `"expect"`, the upstream config it
  replaces. If training passes a different upstream config (another batch size
  or EP degree, or an upstream retune), the wrapper raises instead of overriding
  a baseline the sweep never measured. If the dispatch emits the fused SwiGLU
  forward, the override must still be emit-compatible.

## What the tile knobs can do

1. **`NUM_C_STAGES` only matters for TMA-store epilogues**: FC2 dgrad (plain
   DISPATCH) and FC13 fprop (emit-h1, `NUM_C_STAGES` stages of the 16 KB h1
   sub-tile). The two combine kernels (FC2 fprop, FC13 dgrad) allocate one
   row-major scatter stage whatever the config says; there `NUM_C_STAGES` only
   switches the 8-element row skew. `NUM_C_STAGES=1` with `NUM_TMEM_BUFFERS=2`
   drops the skew (32-way bank conflicts), so the validator rejects it.
2. **FC13 fprop's epilogue sub-tile is pinned.** The fused SwiGLU emit needs
   `BLOCK_N / EPILOGUE_SUBTILE == 64` and `BLOCK_M == 128 * NUM_CTAS * NUM_MMAS`
   (`_native_swiglu_fwd_config_is_compatible`): E=4 at BN=256, E=2 at BN=128.
3. **On bm512, a second accumulator buffer only fits at BN=128.** TMEM columns
   = BN x NUM_MMAS x NUM_TMEM_BUFFERS <= 512, so bm512/bn256 is single-buffered.

Validity rules (`moe_tiles.check_config`): BM = 128 x CTAs x MMAs; BK = 64;
BN in {64, 128, 192, 256} and dividing N; TMEM <= 512 columns;
`NUM_TILE_BUFFERS = NUM_TMEM_BUFFERS + 1`; epilogue sub-tile >= 32 columns and a
multiple of 8; per-mode SMEM estimate <= 232,448 B (dist_moe's own budget),
calibrated so the hardware-proven wgrad config (S4/C2 at bm512 bn256) passes.

## The sweep

`tile_sweep.py` prunes about 500-2,000 legal configs per GEMM to 7-12 candidates
on paper (operand intensity per staged SMEM byte, ragged N tiles, dominated
S/C/E combinations, combine-epilogue equivalences; see the comments in
`CANDIDATES`), then measures them on one node:

- **Same kernels, same arguments.** An EP4 Dist-MoE context runs one real
  `routed_experts` forward and backward with upstream configs while the
  launchers record their keyword arguments (activation-buffer offsets,
  gather/scatter pointer tables, symmetric buffers, SwiGLU flags). Each GEMM is
  then replayed with only `config=` changed. Per-GPU M, N, K and expert count
  match the 64-GPU shapes; EP4 sends 3/4 of the rows over NVLink against 63/64
  at EP64.
- **No cross-GPU interference.** The combine GEMMs' scatter-pointer tables are
  remapped into private sink symmetric buffers, so concurrent GEMMs on different
  GPUs never write what another GPU reads; the NVLink write pattern is
  unchanged.
- **Phase 1 (split).** Each GPU owns some GEMMs. Every candidate is compiled and
  checked against the upstream output (`bitwise`, `bf16-rounding` within
  `--ulp-tol`, or FAIL, which aborts). All configs are then timed back-to-back
  under sustained CUDA-graph replays, upstream first and last as a drift check.
- **Phase 2 (lockstep).** All GPUs run the same GEMM and config at the same time,
  as in training: upstream, phase-1 winner, upstream. A config is accepted only
  if it beats the mean of its two upstream runs by at least `--min-gain-pct`
  (default 1%).
- **`--verify-spec`** (about 1 minute more) runs one real forward and backward,
  including recompute and wgrad, with `moe_tiles.apply(spec)`. The result must
  match upstream bitwise (or within 2x upstream run-to-run noise) and every spec
  GEMM must have been overridden; otherwise no spec is written.
- A wall-clock budget (`--budget-s`, default 540 s) drops the lowest-priority
  candidates instead of overrunning. Expected runtime is about 5-6 minutes.

```bash
python -m torchtitan.experiments.dist_moe_patches.tile_sweep --list
torchrun --standalone --nproc_per_node=4 \
    -m torchtitan.experiments.dist_moe_patches.tile_sweep \
    --out tile_sweep.json --verify-spec
# writes tile_sweep.json and the EXP_MOE_TILES spec tile_sweep.moe_tiles.json
```

`EXP_MOE_TILES` must be unset while sweeping.

## Measured spec

`moe_tiles_muse_spark_816b.json`, from a one-node sweep at Muse Spark 816B BF16
per-rank shapes (16384 tokens, D = I = 4096, top-8, 4 local experts):

| GEMM | Upstream | Override | Gain (one node) |
|---|---|---|---:|
| `fc13_fprop` | `training_2cta2mma_bm512_bn256` (C1) | `c2m2_bm512_bn256_s4_t1_c2_e4`: the same tile with a second C stage | +1.3% |
| `fc2_fprop` | `training_2cta1mma_bm256_bn256` | `c2m2_bm512_bn256_s4_t1_c1_e4`: the bm512 2-CTA 2-MMA tile | +4.4% |

`fc2_dgrad` and `fc13_dgrad` are not in the spec and keep upstream's configs.

**Numerics:** bitwise identical end to end to the upstream configs. This holds
by construction: BLOCK_K stays 64, each output element accumulates its K in the
same order in fp32 TMEM whatever the M/N tiling or CTA pairing, and the
epilogues are elementwise.

**Expected end-to-end effect:** if the one-node gains carry over, roughly
8 ms/step on `fc13_fprop` (about 0.6 s/step of forward and recompute) plus
7 ms/step on `fc2_fprop` (about 0.15 s/step) at the 64-GPU shapes. This spec has
not been part of a 64-GPU run yet.

The configs JIT-compile during the warmup step, before CUDA-graph capture, like
the existing conditional variants: expect a few extra compiles at startup.

## Tests

```bash
pytest torchtitan/experiments/dist_moe_patches/tests/test_moe_tiles_anchors.py
```

`test_measured_spec_is_valid` is pure Python. `test_apply_and_remove` needs
dist_moe (CPU is enough; it skips otherwise): it applies the measured spec,
which runs every anchor check, then restores the launchers. On GPU, `tile_sweep
--verify-spec` is the end-to-end parity check.
