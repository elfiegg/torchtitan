# Dist-MoE runtime patches

Opt-in runtime patches for the BF16 training path of
[`dist_moe`](https://github.com/meta-pytorch/dist_moe), written against commit
`b76e0e0`. They change dist_moe internals, which torchtitan does not own, so
they live here as patches applied at startup rather than as dist_moe changes.
Their native home is meta-pytorch/dist_moe (`dist_moe/api.py`,
`_Bf16Autograd`); each module is deleted once its change lands there.

| Patch | Module | Enable | Design doc |
|---|---|---|---|
| Fuse SwiGLU into the backward FC13 recompute | `swiglu_recompute.py` | `EXP_SWIGLU_RECOMPUTE=1` | `SWIGLU_RECOMPUTE.md` |
| Keep h3 in pinned host memory, skip the backward FC2 recompute | `h3_offload.py` | `EXP_MOE_H3_OFFLOAD=1` | `H3_OFFLOAD.md` |
| Per-GEMM tile configs for the BF16 grouped GEMMs | `moe_tiles.py` | `EXP_MOE_TILES=<spec.json or inline JSON>` | `MOE_TILES.md` |

A checkout may contain any subset of these modules.

## Guarantees

- **Off by default.** Each module is a no-op on import unless its env var is
  set. Every rank must use the same settings.
- **No silent drift.** `swiglu_recompute` and `h3_offload` read the upstream
  function source with `inspect`, apply a fixed list of exact-text edits, and
  compile the result against `dist_moe.api`'s globals. `moe_tiles` checks
  exact-text anchors in `dist_moe/api.py` and two kernel files before it wraps
  the grouped-GEMM launchers. If any anchor does not match exactly the expected
  number of times, the patch raises at startup and leaves dist_moe untouched.
- **Import order.** `swiglu_recompute` must be applied before `h3_offload`.
  `h3_offload` edits the already patched source; the reverse order fails loudly
  because the SwiGLU anchors no longer match. `moe_tiles` composes in any order.

## Usage

Through the launcher, which imports the patch modules in the required order
and then runs `torchtitan.train`:

```bash
EXP_SWIGLU_RECOMPUTE=1 \
EXP_MOE_H3_OFFLOAD=1 \
EXP_MOE_TILES=torchtitan/experiments/dist_moe_patches/moe_tiles_muse_spark_816b.json \
ENTRYPOINT=torchtitan.experiments.dist_moe_patches.train \
MODULE=torchtitan_recipes.models.muse_spark \
CONFIG=muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark \
./run_train.sh
```

Or explicitly, before the first Dist-MoE call:

```python
from torchtitan.experiments.dist_moe_patches import h3_offload, moe_tiles, swiglu_recompute

swiglu_recompute.apply()  # before h3_offload
h3_offload.apply()
moe_tiles.apply(moe_tiles.load_spec("path/to/spec.json"))
```

## Measured results

On the 64-GPU Muse Spark 816B BF16 benchmark recipe, `swiglu_recompute` and
`h3_offload` together with two norm fusions (four kernel patches in total) gave
**+5.0% end-to-end throughput**, with loss and grad_norm within run-to-run
noise.

- `swiglu_recompute`: bitwise identical by construction, verified in that run.
- `h3_offload`: engaged in that run, with 0 synchronous restores.
- `moe_tiles`: on one node, +1.3% on `fc13_fprop` and +4.4% on `fc2_fprop`,
  bitwise identical end to end. It has not been part of a 64-GPU run yet.

## Tests

Anchor checks, CPU only (they skip when `dist_moe` is not importable):

```bash
pytest torchtitan/experiments/dist_moe_patches/tests
```

GPU tests are torchrun scripts for one node; each design doc lists its own.
