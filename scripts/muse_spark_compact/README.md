# Muse Spark Open 816B compact benchmark

This experiment replays the same 100 C4 updates as the 64-GPU, 32K CP2 benchmark
at 29,184 physical slots per GPU. It uses DP64, EP64, CP1, TP1 and one
microbatch per update with a 49 GiB Dist-MoE activation slot and 33 selected host H3 snapshots. It keeps document
boundaries and the modulo-32 phase of forced balanced routing. Use this setup
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
python3 scripts/muse_spark_compact/verify_inputs.py "$MUSE_REPACK_SOURCE" "$MUSE_REPACK_PLAN"
```

Use a fresh shared output directory for each run. Launch the following on
each of 16 nodes with four CUDA-visible GPUs, substituting the node's
zero-based rank and one common rendezvous endpoint. The script sets
`NCCL_NVLS_ENABLE=1` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
Use an allocation in one NVLink domain.

```bash
scripts/muse_spark_compact/run_node.sh bf16 NODE_RANK MASTER_ADDR MASTER_PORT
```

After training, verify all 64 completion and dtype receipts and summarize
updates 71 at 90 inclusive.

```bash
python3 scripts/muse_spark_compact/summarize.py "$REPRO_OUTPUT"
```

The summary shows rank 0 plus all rank results. Nominal tokens/s/GPU is the
mean of `32,768 / update_seconds`. Native tokens/s/GPU and model TFLOPS/GPU
are the means of TorchTitan's logged metrics. The TFLOPS estimate uses the
physical workload represented by the compact recipe.

Use `mxfp8` to run the compact native staged MXFP8 recipe with a 42 GiB
expert slot. FlashAttention and the language-model head stay in BF16.
