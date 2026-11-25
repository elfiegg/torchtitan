# Muse Spark Open 816B-A42B: 32K CP2 benchmark with the norm fusions

This branch measures whether two norm fusions speed up the 32K, DP32 x CP2 x
EP64 benchmarks of [#6346](https://github.com/pytorch/torchtitan/pull/6346) on
64 NVIDIA Vera Rubin GPUs. The recipes are unchanged; only the model code
changes.

It is a performance benchmark, not a training recipe. Routing is forced to a
balanced round-robin assignment and weights are randomly initialized.

## Contents

The branch is #6346 (stacked on #6252) plus two cherry-picked commits:

| Commit | Change | Upstream |
|---|---|---|
| `d0f6a11988df8e4955dd0247fa45d97785d5797e` | 64-GPU recipes with selective gather and activation offload | [#6252](https://github.com/pytorch/torchtitan/pull/6252) head |
| `3d87262350be2e948d7dd3144bc0b24b6e6d7c8f` | BF16 Adam moments and reductions; the 32K CP2 candidates | [#6346](https://github.com/pytorch/torchtitan/pull/6346) head |
| `b4f707b3b8218a06be881d52417b57b2ab90a0f0` | The pre-norms write BF16 inside their compiled region | [#6369](https://github.com/pytorch/torchtitan/pull/6369) |
| `965778a98e038e05c36241c16d0a06453e177efc` | The post-expert norm runs as Dist-MoE's fused input-scaled RMSNorm | [#6371](https://github.com/pytorch/torchtitan/pull/6371), needs [dist_moe#18](https://github.com/meta-pytorch/dist_moe/pull/18) |

#6369 and #6371 are based on main, which has moved since #6346. The commits
cherry-picked here are the same changes from `elfiegg/torchtitan` branches
`bf16-native/2-pre-norm-cast` (`b6648ab2`) and `bf16-native/3-fused-expert-norm`
(`465274ac`), which apply to #6346 without conflicts. To rebuild the branch:

```bash
git fetch https://github.com/pytorch/torchtitan.git pull/6346/head
git fetch https://github.com/elfiegg/torchtitan.git \
  bf16-native/2-pre-norm-cast bf16-native/3-fused-expert-norm
git checkout -b 32k-norm-fusions 3d87262350be2e948d7dd3144bc0b24b6e6d7c8f
git cherry-pick b6648ab2 465274ac
```

## Software

Same as the #6346 candidates except Dist-MoE:

| Component | Version |
|---|---|
| TorchTitan | this branch, `965778a98e038e05c36241c16d0a06453e177efc` |
| PyTorch | nightly `2.16.0.dev20261119+cu134` (git `daf1a40a5de981f7cab47fed4dd9883b1581cd63`) |
| Dist-MoE | `bea783f9585afbe1aaf087650acf54465a918af6`, the head of [dist_moe#18](https://github.com/meta-pytorch/dist_moe/pull/18): one commit on the #6346 pin `b76e0e042b4e97731fec981e974d375eb8eeed8b` that adds `InputScaledRMSNormPostprocess` |
| torch_remat | `71905a6e60123a9adadd4d0cd5acae4e3cb8ba88` |
| attention-gym | v0.0.20 (`7adfbadb06c30eeea82dc3f050ae25512d44088e`) plus the unmodified commit of [attention-gym#651](https://github.com/meta-pytorch/attention-gym/pull/651) (`d08fdce601ef0ea6c925e970442b3dda26e79752`) |

Install Dist-MoE the same way as for #6346, from the PR head:

```bash
git clone https://github.com/meta-pytorch/dist_moe.git && cd dist_moe
git fetch origin pull/18/head
git checkout bea783f9585afbe1aaf087650acf54465a918af6
```

Without it, the first Dist-MoE layer fails with a missing
`dist_moe.InputScaledRMSNormPostprocess`.

## Reproducing

### Check the recipe on one GPU

Each recipe has a `_fake` variant that runs with FakePG on one GPU:

```bash
NGPU=1 MODULE=torchtitan_recipes.models.muse_spark \
  CONFIG=muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark_fake \
  ./run_train.sh
```

### Run on 64 GPUs

Stage the Muse Spark Open tokenizer at `./assets/hf/Muse-Spark-Open-816B-A42B`.
Launch on 16 hosts x 4 GPUs in one NVLink domain, with the same environment as
#6346:

```bash
NCCL_NVLS_ENABLE=1 PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
torchrun --nnodes=16 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=static --rdzv-endpoint="$MASTER_ADDR:29500" \
  -m torchtitan.train --module torchtitan_recipes.models.muse_spark \
  --config "$CONFIG" --output-dir "/shared/outputs/$CONFIG"
```

| Precision | `CONFIG` |
|---|---|
| BF16 | `muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark` |
| MXFP8 | `muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_bf16_state_benchmark` |

Known issue (as in #6346): with FSDP symmetric memory and EP,
`destroy_process_group` can hang after the last update. The #6346 runs wrapped
`Trainer.close` to call `os._exit(0)` after a successful close.

### Measure

Each recipe runs 100 updates. As in #6346, report TPS/GPU, TFLOPs/GPU and step
time as the mean over updates 71..90 on rank 0, and peak memory.

Compare against the #6346 numbers, which use the same metric code:

| Precision | TPS/GPU | TFLOPs/GPU | Step s | Peak device used GiB |
|---|---:|---:|---:|---:|
| BF16 | 6,498.03 | 2,127.55 | 5.043 | 271.84 |
| MXFP8 | 8,798.90 | 2,880.89 | 3.724 | 262.47 |

Both metrics count padding as work. TPS counts every token slot. TFLOPs uses
the estimator from before [#6325](https://github.com/pytorch/torchtitan/pull/6325),
which charges each token for attention over all 32,768 positions. They are
comparable between this branch and #6346, but not with recipes of other
sequence lengths or document caps.

## Expected effect

Not yet measured at 32K CP2. On other recipes, on 64 GPUs, one run each:

| Recipe | Change | Measured |
|---|---|---|
| BF16 compact (DP64, CP1, 29,184 tokens) | #6369 | -53 ms/step |
| BF16 compact | #6371 on top | -68 ms/step |
| MXFP8 CP1, 32,768 tokens ([#6352](https://github.com/pytorch/torchtitan/pull/6352)) | #6369 + #6371 | -163 ms/step |

Scaled to the 32,768 tokens per GPU and step here, that suggests about 136 ms
(about +2.8%) for BF16 and about 163 ms (about +4.6%) for MXFP8.

Things to check in the results:

- **Memory:** on the BF16 compact recipe the two changes added about 1.4 GiB of
  peak device memory. The BF16 run here had about 6.7 GiB free.
- **Numerics:** #6369 is bitwise unchanged. #6371 is not: BF16 matched the
  fallback to 5 digits; MXFP8 drifted by up to 0.0165 in loss by update 100.

## Results

Pending.
