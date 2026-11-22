# Muse Spark 816B-A42B BF16: 64-GPU optimization benchmark

This folder reproduces the BF16 throughput work on
`muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark`:

- 64 GPUs (16 nodes x 4), DP-shard 32 x CP 2, EP 64;
- 32K sequences, 2 microbatches of 16K tokens per GPU per step;
- Dist-MoE BF16 experts, forced balanced routing.

One Slurm allocation runs several variants back to back on the same nodes, then
summarizes throughput and numerics. Compare variants only within one allocation:
EP synchronizes on every MoE layer, so the slowest GPU of the allocation sets the
pace, and its clock varies between node sets.

## Optimizations

| Knob | What it does | Where | Numerics | Measured |
|---|---|---|---|---|
| `EXP_SWIGLU_RECOMPUTE=1` | Backward recompute emits SwiGLU from the FC13 GEMM epilogue, like the forward; the standalone SwiGLU kernel runs only for saved layers | [dist_moe_patches](../dist_moe_patches/SWIGLU_RECOMPUTE.md) | bitwise | in `kern` |
| `EXP_MOE_H3_OFFLOAD=1` | Keeps the FC2 combine output (h3, 1 GiB per layer-microbatch) in pinned host memory and skips the backward FC2 combine recompute | [dist_moe_patches](../dist_moe_patches/H3_OFFLOAD.md) | bitwise | in `kern`; 0 synchronous restores |
| `EXP_NORM_FUSION=cast` | Pre-norm bf16 cast inside the compiled RMSNorm region (`MuseSparkRMSNorm.Config.output_dtype`) | [muse_spark README](../../models/muse_spark/README.md) | bitwise | in `kern`; ~61 ms/step on one GPU |
| `EXP_MOE_POSTNORM=1` | Fused Triton forward/backward for the Dist-MoE post-expert norm (`FusedDistMoeInputScaledRMSNorm`) | [muse_spark README](../../models/muse_spark/README.md) | y, dx within 2 bf16 ulp; dw closer to fp64 | in `kern`; ~50-95 ms/step from trace analysis |
| `EXP_PAD_SEGMENT=128` | Splits each row's tail padding into <= 16 varlen segments (`Decoder.Config.padding_segments`) instead of one causal document on the last CP rank | `torchtitan/models/common/attention/attention.py` | real tokens bitwise | not yet on 64 GPUs |
| `EXP_MOE_TILES=<json>` | Per-GEMM tile configs for the Dist-MoE grouped GEMMs | [dist_moe_patches](../dist_moe_patches/MOE_TILES.md) | bitwise | +1.3% fc13 fprop, +4.4% fc2 fprop on one node; not yet on 64 GPUs |

Memory knobs: `EXP_RESHARD=always`, `DISTMOE_SLOT_GIB=48` and
`EXP_SAVE_MOE_PROJ=1` trade FSDP reshard for a larger Dist-MoE saved-activation
slot (fewer MoE recomputes) and save the MoE projections instead of recomputing
them. `EXP_CP_LB=headtail` selects the head-tail CP load balancer.

## Results

Same 16 nodes, one allocation, 24 steps per variant, mean over steps 9-24:

| Variant | Knobs | Tokens/s vs `base` | Loss / grad_norm vs `base` |
|---|---|---:|---|
| `base` | none | 0% | |
| `kern` | SwiGLU recompute, h3 offload, norm cast, fused post-expert norm | **+5.0%** | within run-to-run noise (about 1e-5) over 24 steps |

`all` (`kern` plus padding split and tile configs) and `all_mem48proj` are not
measured on 64 GPUs yet.

## Runtime

The image must provide these (the pins the results were measured with):

| Package | Version |
|---|---|
| torch | 2.16.0.dev20261119+cu134 (CUDA 13.4, Python 3.12) |
| triton | 3.9.0+gitb7ddcf86 |
| torchao | 0.19.0.dev20261103+cu134 |
| [dist_moe](https://github.com/meta-pytorch/dist_moe) | b76e0e042b4e (the runtime patches check their anchors against it) |
| [remat](https://github.com/meta-pytorch/remat) (torch-remat) | 71905a6e6012 |
| [attention-gym](https://github.com/meta-pytorch/attention-gym) | 7adfbadb06c3 plus PR 651 |
| flash-attn-4 | 4.0.0b31 |
| nvidia-cutlass-dsl | 4.8.0 |

The scripts put this checkout first on `PYTHONPATH`, so torchtitan comes from
the checkout, not from the image.

## Run

Correctness tests, one node:

```bash
IMAGE=/path/runtime.sqsh sbatch -A <account> -p <partition> \
  torchtitan/experiments/muse_spark_bf16_opt/run_unit_tests.sbatch
```

Benchmark, 16 nodes. Pass the account, partition and any node constraint on
the sbatch command line:

```bash
IMAGE=/path/runtime.sqsh OUT_ROOT=/shared/fs/muse-bf16 VARIANTS="base kern all" \
  sbatch -A <account> -p <partition> --time=00:30:00 \
  torchtitan/experiments/muse_spark_bf16_opt/run_bench.sbatch
```

Each variant takes about 4 minutes at 24 steps (`STEPS`). The variant table is
in `run_bench.sbatch`. `EXP_PROFILE_VARIANTS=all` profiles that variant on ranks
0/1/4/63 at steps 13-14.

Results land in `$OUT_ROOT/job-<id>/<variant>/`:

- `metrics-rank*.jsonl` -- every logged metric of every rank;
- `node-*-gpu.csv` -- per-GPU clocks and power every 500 ms;
- `effective-config.json`;
- `status.json`.

`summarize.py` prints mean and median tokens/s per GPU over steps >= 9, the
change against `base`, peak device memory, and the slowest and fastest GPU by
median clock:

```bash
python3 torchtitan/experiments/muse_spark_bf16_opt/summarize.py $OUT_ROOT/job-<id>
```

To check the recipe without GPUs (FakePG):

```bash
EXP_FAKE_BACKEND=1 EXP_NORM_FUSION=cast EXP_MOE_POSTNORM=1 EXP_PAD_SEGMENT=128 \
  python -c "from torchtitan.experiments.muse_spark_bf16_opt.recipes import bf16_opt; print(bf16_opt().model.padding_segments)"
```

## Files

| File | Purpose |
|---|---|
| `recipes.py` | `bf16_opt`: the benchmark recipe plus the `EXP_*` options |
| `bench_train.py` | `torchtitan.train` that imports the Dist-MoE patches, stops after `REPRO_STOP_AFTER` steps and records exact per-rank metrics |
| `audit.py` | After step 1, checks FP32 master weights and gradients and BF16 Adam moments on device |
| `run_bench.sbatch` | 16-node variant sweep |
| `run_unit_tests.sbatch` | One-node correctness tests for every optimization |
| `summarize.py` | Throughput, memory and GPU-clock summary per variant |
