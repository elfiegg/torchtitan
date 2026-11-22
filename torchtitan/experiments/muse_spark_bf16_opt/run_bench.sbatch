#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --job-name=muse-spark-bf16-opt
#SBATCH --output=muse-spark-bf16-opt-%j.log
# Muse Spark 816B-A42B BF16, 64 GPUs (16 nodes x 4): run several optimization variants back to
# back in one allocation, then summarize throughput and numerics per variant.
#
# Pass account, partition and any node constraint on the sbatch command line, e.g.
#   IMAGE=/path/runtime.sqsh OUT_ROOT=/shared/fs/muse-bf16 VARIANTS="base kern all" \
#     sbatch -A <account> -p <partition> --time=00:40:00 \
#     torchtitan/experiments/muse_spark_bf16_opt/run_bench.sbatch
#
# Required env:
#   IMAGE     enroot/pyxis image with the pinned runtime (see README.md)
#   OUT_ROOT  shared-filesystem directory for results
# Optional env:
#   VARIANTS        space-separated keys of TABLE below (default: "base kern all")
#   STEPS           training steps per variant (default 24; the summary uses steps >= 9)
#   LOCAL_SCRATCH   node-local directory for compile caches and metrics (default /tmp)
#   EXP_PROFILE_VARIANTS  comma-separated variants to profile on ranks 0/1/4/63 (default none)
set -uo pipefail
: "${IMAGE:?set IMAGE to the runtime image}"
: "${OUT_ROOT:?set OUT_ROOT to a shared output directory}"
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel) || exit 1
EXP_DIR=torchtitan/experiments/muse_spark_bf16_opt
STEPS=${STEPS:-24}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/tmp}
export EXP_PROFILE_VARIANTS=${EXP_PROFILE_VARIANTS:-none}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

JOB=$OUT_ROOT/job-$SLURM_JOB_ID
mkdir -p "$JOB"
scontrol show job "$SLURM_JOB_ID" > "$JOB/job.txt"
{ git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short; } > "$JOB/source.txt"

# name : env assignments. EXP_* knobs are read by recipes.py (torchtitan config) and by the
# dist_moe runtime patches (EXP_SWIGLU_RECOMPUTE, EXP_MOE_H3_OFFLOAD, EXP_MOE_TILES).
KERN="EXP_SWIGLU_RECOMPUTE=1 EXP_MOE_H3_OFFLOAD=1 EXP_NORM_FUSION=cast EXP_MOE_POSTNORM=1"
ALL="$KERN EXP_PAD_SEGMENT=128 EXP_MOE_TILES=/workspace/torchtitan/experiments/dist_moe_patches/moe_tiles_muse_spark_816b.json"
MEM48PROJ="EXP_RESHARD=always DISTMOE_SLOT_GIB=48 EXP_SAVE_MOE_PROJ=1"
declare -A TABLE=(
  [base]=""
  [swiglu]="EXP_SWIGLU_RECOMPUTE=1"
  [h3]="EXP_MOE_H3_OFFLOAD=1"
  [normcast]="EXP_NORM_FUSION=cast"
  [postnorm]="EXP_MOE_POSTNORM=1"
  [kern]="$KERN"
  [kern_pad]="$KERN EXP_PAD_SEGMENT=128"
  [all]="$ALL"
  [mem48proj]="$MEM48PROJ"
  [all_mem48proj]="$ALL $MEM48PROJ"
  [all_mem48proj_ht]="$ALL $MEM48PROJ EXP_CP_LB=headtail"
)
VARIANTS=${VARIANTS:-"base kern all"}
for VARIANT in $VARIANTS; do
  [ -n "${TABLE[$VARIANT]+x}" ] || { echo "unknown variant $VARIANT"; exit 1; }
done

PORT=29670
for VARIANT in $VARIANTS; do
  PORT=$((PORT + 1))
  OUT=$JOB/$VARIANT
  mkdir -p "$OUT"
  echo "=== $(date -Is) variant=$VARIANT env=[${TABLE[$VARIANT]}]"
  START=$(date +%s)
  env ${TABLE[$VARIANT]} VARIANT=$VARIANT PORT=$PORT STEPS=$STEPS EXP_DIR=$EXP_DIR \
    LOCAL_SCRATCH=$LOCAL_SCRATCH \
  timeout 900 srun --cpu-bind=none --kill-on-bad-exit=1 --container-image="$IMAGE" \
    --container-mounts="$REPO:/workspace,$OUT_ROOT:/results,$LOCAL_SCRATCH:/local" \
    --container-workdir=/workspace bash -c '
    set -euo pipefail
    export PYTHONPATH=/workspace
    export NCCL_NVLS_ENABLE=1
    export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1
    export OMP_NUM_THREADS=8
    export REPRO_STOP_AFTER=$STEPS
    export REPRO_OUTPUT=/results/job-$SLURM_JOB_ID/$VARIANT
    export REPRO_LOCAL_OUTPUT=/local/muse-bf16-opt-$SLURM_JOB_ID/$VARIANT
    export TORCHINDUCTOR_CACHE_DIR=$REPRO_LOCAL_OUTPUT/inductor
    export TRITON_CACHE_DIR=$REPRO_LOCAL_OUTPUT/triton
    mkdir -p "$REPRO_LOCAL_OUTPUT"
    nvidia-smi --query-gpu=timestamp,index,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
      --format=csv,noheader,nounits -lms 500 > "$REPRO_OUTPUT/node-$SLURM_PROCID-gpu.csv" &
    GPUMON_PID=$!
    trap "kill $GPUMON_PID 2>/dev/null || true" EXIT
    python3 -m torch.distributed.run --nnodes=16 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
      --master-addr="$MASTER_ADDR" --master-port=$PORT /workspace/$EXP_DIR/bench_train.py \
      --module torchtitan.experiments.muse_spark_bf16_opt.recipes --config bf16_opt --output-dir "$REPRO_OUTPUT/trainer"
  '
  RC=$?
  echo "{\"variant\": \"$VARIANT\", \"env\": \"${TABLE[$VARIANT]}\", \"rc\": $RC, \"seconds\": $(( $(date +%s) - START ))}" > "$OUT/status.json"
  echo "=== $(date -Is) variant=$VARIANT rc=$RC"
done
python3 "$REPO/$EXP_DIR/summarize.py" "$JOB" | tee "$JOB/summary.txt"
