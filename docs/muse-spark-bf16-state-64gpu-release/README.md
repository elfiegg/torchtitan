<!-- TODO before publishing: replace {{HARDWARE}}. -->

# Muse Spark Open 816B-A42B: BF16 weights and optimizer state on 64 GPUs

This release measures TorchTitan training throughput for Muse Spark Open
816B-A42B on 64 {{HARDWARE}} GPUs with everything in BF16: compute, master
weights, gradients, Adam moments and FSDP reductions. A 112 GiB Dist-MoE
activation slot keeps more MoE activations on the GPU, so fewer MoE layers
recompute in backward.

It is a performance benchmark, not a training recipe. Routing is forced to a
balanced round-robin assignment and weights are randomly initialized.

## Results

### Throughput and memory

TPS is tokens/s/GPU. Each update processes 64 rows of 32,768 token slots, and
TPS counts all slots, including padding. Global TPS is TPS/GPU x 64. TFLOPs/GPU
is TorchTitan's analytical model-FLOP estimate; MFU is not reported. The
baseline is the 64-GPU recipe `muse_spark_816b_a42b_dist_moe_bf16_spmd_64`
with the same forced routing; [What changed](#what-changed-from-the-baseline)
lists the differences.

| Configuration | TPS/GPU | Global TPS | vs baseline | TFLOPs/GPU | Peak active GiB | Peak reserved GiB | Peak device used GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 state, DP32 x CP2 | 6,554.21 | 419,469 | +50.63% | 2,145.94 | 241.00 | 250.83 | 266.01 |
| Baseline, DP8 x CP8 | 4,351.26 | 278,481 | -- | 1,424.67 | 209.38 | 257.61 | 266.42 |

Key observations:

- **Throughput:** BF16 state is 50.6% faster than the baseline: 5.000 s
  instead of 7.531 s per update.
- **Memory:** both peak within 0.5 GiB of each other in device memory, with
  about 12 GiB of device memory free.

The two runs were measured differently:

| | BF16 state | Baseline |
|---|---|---|
| Measured updates | 71..90 of 100 | 13..22 of 22 |
| TPS | Mean of the per-update logged values on rank 0 | Longest synchronized rank-local window, divided by 10 updates |
| Peak active / reserved | PyTorch allocator high-water on rank 0 | Maximum over all 64 GPUs |
| Peak device used | Sampled every 200 ms on all GPUs | Sampled every 100 ms on all GPUs |

Allocation retries and OOM counters stayed zero in the BF16-state run. Each
configuration was run once; differences are observed values, not confidence
intervals.

<details>
<summary>Per-update TPS distribution, BF16 state</summary>

| Mean TPS/GPU | Median | Min..max | Global TPS | Mean step s |
|---:|---:|---:|---:|---:|
| 6,554.21 | 6,554.58 | 6,512.03..6,608.31 | 419,469 | 4.9996 |

Updates 71..90, rank 0.

</details>

### What changed from the baseline

| Area | Baseline | BF16 state |
|---|---|---|
| Parallelism | FSDP 8 x CP 8 x EP 64; 8 microbatches per DP rank | FSDP 32 x CP 2 x EP 64; 2 microbatches per DP rank |
| Attention CP | K/V all-gather | K/V all-gather, with selective K/V gather in the sliding-window layers |
| Saved activations | Attention and MoE projection regions kept on the GPU | Attention regions and block inputs offloaded to pinned host memory, except in the last block |
| Dist-MoE | About 17 GiB activation slot (capacity factor 0.25) | 112 GiB activation slot |
| Weights and optimizer state | FP32 master weights, gradients and Adam moments; FP32 reductions | All BF16 |
| FSDP | NCCL collectives | Symmetric-memory collectives |
| Tokenizer | Test tokenizer | Muse Spark Open tokenizer |

Both use the same model, 32K sequences, global batch 64, `c4_test`, seed 42,
forced round-robin routing, Dist-MoE scratch factor 1 and GPU-resident Adam
moments.

### Numerics

| Loss, update 1 | Loss, update 100 | Mean loss, updates 71..90 |
|---:|---:|---:|
| 12.2302 | 7.6649 | 7.7060 |

All 64 ranks completed 100 updates with finite loss and gradient norm. The
baseline used a different tokenizer, so its losses are not comparable.

## Workload

| Area | Configuration |
|---|---|
| Hardware | 64 {{HARDWARE}} GPUs: 16 hosts x 4 GPUs, all in one NVLink domain (required by Dist-MoE EP64). Each CP group must share an NVLink domain. |
| Model | Muse Spark Open 816B-A42B, text decoder: 62 layers, hidden 8,192, 256 compressed-MoE experts, top-8 |
| Batch | Sequence length 32,768; global batch 64 sequences (2,097,152 tokens per update); 2 microbatches per DP rank |
| Data | `c4_test` with concat-then-split packing and at most 64 documents per row; Muse Spark Open tokenizer (revision `a604fa753b672b5dca3304d8648cf756120e29db`) |
| Initial weights | Random, seed 42; no checkpoints saved |
| Parallelism | FSDP 32 x CP 2 x EP 64; no TP or PP |
| Attention | FA4 varlen; K/V all-gather CP with selective K/V gather in the sliding-window layers |
| Routed experts | Dist-MoE, BF16 grouped GEMMs; forced balanced round-robin routing; scratch capacity factor 1; one 112 GiB activation slot |
| Precision | BF16 master weights, gradients, Adam moments and FSDP reductions (`precision.dtype="bfloat16"`) |
| Optimizer | Fused AdamW; lr 5e-5 (router gate 6.25e-6); betas 0.9 / 0.95; eps 1e-8; weight decay 0; gradient clip 1.0; moments on the GPU |
| LR schedule | 10 warmup steps, cosine decay, minimum factor 0.1 |
| Memory policy | Selective activation remat: attention and Dist-MoE regions saved, everything else recomputed. Attention regions and block inputs offloaded to pinned host memory, except in the last block. |
| FSDP | Default resharding; symmetric-memory collectives |
| CUDA graphs | Training graphs per microbatch group and a CUDA graph for the optimizer step |
| Environment | `NCCL_NVLS_ENABLE=1`; `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` |

## Software

The recipe is `muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark`
in `torchtitan_recipes/models/muse_spark.py` at TorchTitan
`2ff401e9dae1cfdc47982e085a482e08f3a0e015`. It changes three things relative to
`muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark`: BF16 master state,
BF16 reductions, and the 112 GiB Dist-MoE slot with scratch factor 1.

| Component | Version |
|---|---|
| PyTorch | nightly `2.16.0.dev20261119+cu134` (git `daf1a40a5de981f7cab47fed4dd9883b1581cd63`) |
| Dist-MoE | `b76e0e042b4e97731fec981e974d375eb8eeed8b` |
| torch_remat | `71905a6e60123a9adadd4d0cd5acae4e3cb8ba88` |
| attention-gym | v0.0.20 (`7adfbadb06c30eeea82dc3f050ae25512d44088e`) plus the unmodified commit of [attention-gym#651](https://github.com/meta-pytorch/attention-gym/pull/651) (`d08fdce601ef0ea6c925e970442b3dda26e79752`). The PR head alone lacks modules that this TorchTitan revision imports. |

The measured run used TorchTitan [#6252](https://github.com/pytorch/torchtitan/pull/6252)
at `814df1c9e892f0e45bb2af5523c1968c52e7262b` plus the activation-offload change of
[#6020](https://github.com/pytorch/torchtitan/pull/6020), with the configuration
above applied on top. The recipe at
`2ff401e9dae1cfdc47982e085a482e08f3a0e015` encodes the same configuration on
the current stack.

## Reproducing

```bash
export NCCL_NVLS_ENABLE=1
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1
torchrun --nnodes=16 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=static --rdzv-endpoint="$MASTER_ADDR:29500" \
  -m torchtitan.train \
  --module torchtitan_recipes.models.muse_spark \
  --config muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark \
  --output-dir /shared/outputs/muse-spark-bf16-state
```

Stage the tokenizer files at `./assets/hf/Muse-Spark-Open-816B-A42B`. The
recipe runs 100 updates; compute TPS as the mean over updates 71..90.

Known issue: with FSDP symmetric memory and EP, `destroy_process_group` can hang
after the last update. The measured run wrapped `Trainer.close` to call
`os._exit(0)` after a successful close.
