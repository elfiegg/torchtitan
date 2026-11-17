# Experimental Muse Spark 816B/A42B on 64 GPUs

This recipe packages the DP32/CP2, scratch10/save0, fused-gradient-packing and
W13-barrier-overlap bundle measured by campaign job 3728115. It keeps the
existing `muse_spark_816b_a42b_dist_moe_bf16_spmd_64` recipe unchanged.

**Experimental:** the campaign observed 10.25% higher throughput in a matched
30-update comparison. Cross-topology numerical equivalence and long-run model
quality remain unresolved. The new recipe/engine integration has not yet been
executed on GPUs; the earlier measurement applies to the archived campaign
runner, not automatically to this integration.

## Launch

Use this checkout with the campaign's pinned ARM64 container on 16 four-GPU
GB300 nodes in one NVLink domain. On each allocated node, launch with the same
rendezvous endpoint and that node's rank:

```bash
export MUSE_SPARK_HF_ASSETS_PATH=/campaign/assets/Muse-Spark-Open-816B-A42B
/opt/muse-runtime/bin/torchrun \
  --nnodes=16 --nproc-per-node=4 --node-rank="$SLURM_NODEID" \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" \
  --numa-binding=node \
  -m torchtitan.train \
  --module torchtitan_recipes.models.muse_spark \
  --config muse_spark_816b_a42b_dist_moe_bf16_spmd_64_experimental
```

Run from this checkout, with its root on `PYTHONPATH` ahead of the container's
baked copy. The factory preserves the original 120-update schedule and
uses a distinct checkpoint folder with `load_only=True`. It does not reproduce
the campaign runner's external stop after update 30. Changing `training.steps`
also changes the learning-rate schedule and is not an equivalent benchmark.

## Included changes

| Setting | Original recipe | Experimental recipe |
|---|---|---|
| Parallelism | DP8 / CP8 / EP64 / TP1 / PP1 | DP32 / CP2 / EP64 / TP1 / PP1 |
| Global input positions per update | 64 x 32768 | Same |
| Optional saved-activation factor | 1 | 0, with backward recomputation |
| Scratch factor | 32 | 10, with the native fatal overflow check |
| FSDP gradient preparation | Native packing | Fused Triton packing where eligible |
| Final input-gradient barrier | After W13 weight gradients | Side stream alongside W13 weight gradients; join before top-k reduction |
| Reduce-scatter input-buffer cap | 1 | 1 |

BF16 compute, FP32 masters/reductions/moments, original pinned-host AdamW,
attention, activation-rematerialization regions and per-group model graphs
remain unchanged. Optimizer execution stays eager. Validation, FakePG,
checkpoint resume and alternate geometries are rejected for this experiment.

The C4 source is pinned to `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`
with seed 42. `data.py` builds each original DP8 stream and takes every fourth
complete microbatch for the corresponding DP32 rank. Across two accumulation
rounds, the 32 ranks consume exactly the original 64 global microbatches per
update. This preserves packing and masked-token counts. It also preserves the
campaign's redundant host-side input preparation cost.

## Dependencies and code ownership

- TorchTitan base: `6b123cbe0cc75dc7a6c2b0d541736882e64b798e`.
- PyTorch: `bf270c3251d8e84723cedbf5b60eaa60cacaa3f2`.
- Campaign container SHA256:
  `a69861d1e72b4d3278ef11282b611d86fc56417e197dea6d7350932f6853c9ca`.
- `w13/RUNTIME_PINS.json` records the exact Dist-MoE and TorchTitan source
  identities required by the overlap implementation. The existing local
  `num_sms` pass-through patch is included, with its default `None` retained.
- `packing.py` contains the measured packing kernel and FSDP callback.
- `w13/build_variant.py`, `w13/overlap.py` and `w13/install.py` contain the
  source-pinned backward transformation, stream/event ownership and installer.
- `trainer.py` attaches both implementations through a recipe-specific engine.
  It installs overlap after checkpoint loading and restores the original
  backward only after native context closure and graph teardown.

The W13 implementation uses a private Dist-MoE backward entry point and derives
a modified function from its exact source. It is an experiment, not a portable
upstream API. Moving the scheduling feature into Dist-MoE and packing into
PyTorch is the appropriate follow-up before promoting this recipe.

## Verification

CPU-only structural and ownership regressions:

```bash
python3 -B -m unittest discover \
  -s torchtitan/experiments/muse_spark_64/tests -v
```

Before quoting performance for this new integration, execute the native
configuration/import check in the pinned environment, exercise eager execution
and graph capture/replay through clean teardown, then repeat the matched
original/candidate/original measurement and the numerical checks. Do not loosen
the existing numerical thresholds or infer convergence from finite losses.
