<!-- TODO before publishing: replace {{HARDWARE}}. -->

# Muse Spark Open 816B-A42B: BF16 weights and optimizer state on 64 GPUs

This release measures TorchTitan training throughput for Muse Spark Open
816B-A42B on 64 {{HARDWARE}} GPUs with everything in BF16: compute, master
weights, gradients, Adam moments and FSDP reductions. Dropping the FP32 master
copy frees device memory, which goes to a 112 GiB Dist-MoE activation slot so
that fewer MoE layers recompute in backward.

It is a performance benchmark, not a training recipe. Routing is forced to a
balanced round-robin assignment, weights are randomly initialized, and BF16
master weights train to a visibly higher loss than the FP32-master recipe
(see [Numerics](#numerics)).

## Results

All 64 ranks completed 100 updates with finite loss and gradient norm. Step
time is the arithmetic mean of the per-step logged values over updates 71..90
(20 updates). The reference runs the same workload, parallelism and hardware
with FP32 master weights, gradients and Adam state (BF16 compute).

| Configuration | Step s | Throughput vs reference | Peak reserved GiB | Peak device used GiB |
|---|---:|---:|---:|---:|
| BF16 state, 112 GiB Dist-MoE slot | 5.000 | +4.3% | 250.83 | 266.01 |
| Reference: FP32 master weights, gradients and Adam state | 5.215 | 0.0% | 242.99 | -- |

- **Throughput:** each update processes 2,097,152 token slots, so throughput
  scales with the inverse step time; padding slots are counted.
- **Memory:** peak reserved is the PyTorch allocator high-water on rank 0 over
  all 100 updates. Peak device used is the highest device-wide memory sampled
  every 200 ms across all ranks, so it includes memory outside the allocator.
  About 12.5 GiB of device memory stayed free.

Each value comes from a single run; differences are observed values, not
confidence intervals.

### Numerics

| Configuration | Loss, update 1 | Loss, update 100 | Mean loss, updates 71..90 |
|---|---:|---:|---:|
| BF16 state | 12.2302 | 7.6649 | 7.7060 |
| FP32-master reference | 12.2302 | 6.9234 | 7.1077 |

Both runs start from the same seed and data, so update 1 matches. The loss
separates within 100 updates and is 0.74 higher at update 100, most likely
because BF16 master weights round away small Adam updates. This configuration
is for throughput only; it is not loss parity with the FP32-master recipe.

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
