# Muse Spark Open 816B compact benchmark

This experiment replays the same 100 C4 updates as the 64-GPU, 32K CP2 benchmark
at 29,184 physical slots per GPU. It uses DP64, EP64, CP1, TP1 and one
microbatch per update. The replay preserves document boundaries and the
modulo-32 phase of forced balanced routing. Use this setup
for throughput comparisons. Forced routing is unsuitable for training quality
measurements.

## Run

Use the pinned CUDA 13.4 PyTorch build
`2.16.0.dev20261119+cu134` at `daf1a40a5de981f7cab47fed4dd9883b1581cd63`,
Dist-MoE at `b76e0e042b4e97731fec981e974d375eb8eeed8b`,
torch-remat at `71905a6e60123a9adadd4d0cd5acae4e3cb8ba88`, and
[attention-gym PR #651](https://github.com/meta-pytorch/attention-gym/pull/651)
at `d08fdce601ef0ea6c925e970442b3dda26e79752`. Stage the Muse Spark Open tokenizer
in `assets/hf/Muse-Spark-Open-816B-A42B` in this checkout. The branch includes
the activation-offload snapshot for the benchmark base.

Stage the archived `indexed-source` and `indexed-phase-plan` directories on all
nodes, or make them visible from shared storage. Their manifests and NumPy
arrays must stay together. `indexed-source` contains the source token and
document arrays, and `indexed-phase-plan` contains the ordered placement
for all 64 ranks. The tokenized C4 replay and tokenizer are external inputs.
Verify them once on CPU before launching.

```bash
export MUSE_REPACK_SOURCE=/shared/replays/indexed-source
export MUSE_REPACK_PLAN=/shared/replays/indexed-phase-plan
export REPRO_OUTPUT=/shared/results/muse-spark-run
python3 -m scripts.muse_spark_compact.verify_inputs "$MUSE_REPACK_SOURCE" "$MUSE_REPACK_PLAN"
```

## Native BF16 recipe

The native config uses the 29,184-slot replay, a 49 GiB Dist-MoE slot, and the
BF16 benchmark's projection save and offload policy. It uses the installed
TorchTitan and Dist-MoE kernels. The two replay environment variables above
select the input. Inspect the resolved recipe on a CUDA node before launching
on 64 GPUs.

```bash
python3 -m torchtitan.train --module torchtitan_recipes.models.muse_spark \
  --config muse_spark_816b_a42b_dist_moe_bf16_spmd_64_compact_benchmark --print-config
```

For training, run this on each of 16 nodes. Substitute the node's zero-based
rank and one common rendezvous address and port.

```bash
NCCL_NVLS_ENABLE=1 PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  torchrun --nnodes=16 --nproc-per-node=4 --node-rank=NODE_RANK \
  --master-addr=MASTER_ADDR --master-port=MASTER_PORT -m torchtitan.train \
  --module torchtitan_recipes.models.muse_spark \
  --config muse_spark_816b_a42b_dist_moe_bf16_spmd_64_compact_benchmark \
  --output-dir "$REPRO_OUTPUT/trainer"
```

The archived pinned runtime can hang at NCCL teardown after successful training.

## Reproduce the measured experiments

Use a fresh shared output directory for each run. Launch the following on
each of 16 nodes with four CUDA-visible GPUs, substituting the node's
zero-based rank and one common rendezvous endpoint. The script sets
`NCCL_NVLS_ENABLE=1` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
Use an allocation in one NVLink domain. The BF16 launcher adds the measured
kernel and host-offload patches to the native configuration.

```bash
scripts/muse_spark_compact/run_node.sh rubin bf16 NODE_RANK MASTER_ADDR MASTER_PORT
```

After training, verify all 64 completion and dtype receipts and summarize
updates 71 through 90.

```bash
python3 -m scripts.muse_spark_compact.summarize "$REPRO_OUTPUT"
```

The summary shows rank 0 plus all rank results. Nominal tokens/s/GPU is the
mean of `32,768 / update_seconds`. Native tokens/s/GPU and model TFLOPS/GPU
are the means of TorchTitan's logged metrics. The TFLOPS estimate uses the
physical workload represented by the compact recipe.

Use `rubin mxfp8` to run native staged MXFP8 experts and projections with a
42 GiB expert slot. FlashAttention and the language-model head stay in BF16.

## GB300 counterparts

Use `gb300 bf16` or `gb300 mxfp8` in the same node command and use a
separate shared output directory for each. Both GB300 presets remove unpooler
outputs from the activation save and offload lists. The BF16 preset uses a
45 GiB slot and snapshots H3 in eight layers. The MXFP8 preset keeps the 42 GiB
native slot and applies exact-size host pooling. These presets target nodes
with four GB300 GPUs each, and retain the 100-update replay.

| Platform | Precision | Expert slot | Host H3 snapshots |
| --- | --- | ---: | ---: |
| Rubin | BF16 | 49 GiB | 33 |
| Rubin | MXFP8 | 42 GiB | 0 |
| GB300 | BF16 | 45 GiB | 8 |
| GB300 | MXFP8 | 42 GiB | 0 |
