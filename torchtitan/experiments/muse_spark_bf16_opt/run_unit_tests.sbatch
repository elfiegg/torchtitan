#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=muse-spark-bf16-opt-tests
#SBATCH --output=muse-spark-bf16-opt-tests-%j.log
# One node (4 GPUs): the correctness tests for every optimization in this experiment.
#   IMAGE=/path/runtime.sqsh sbatch -A <account> -p <partition> \
#     torchtitan/experiments/muse_spark_bf16_opt/run_unit_tests.sbatch
# Optional: LOCAL_SCRATCH (node-local dir for compile caches, default /tmp).
set -uo pipefail
: "${IMAGE:?set IMAGE to the runtime image}"
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel) || exit 1
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/tmp}
srun --cpu-bind=none --container-image="$IMAGE" \
  --container-mounts="$REPO:/workspace,$LOCAL_SCRATCH:/local" \
  --container-workdir=/workspace bash -c '
  export PYTHONPATH=/workspace NCCL_NVLS_ENABLE=1 OMP_NUM_THREADS=8
  export TORCHINDUCTOR_CACHE_DIR=/local/muse-bf16-tests-$SLURM_JOB_ID/inductor
  export TRITON_CACHE_DIR=/local/muse-bf16-tests-$SLURM_JOB_ID/triton
  FAILED=0
  run() { name=$1; shift; echo "== $name"; s=$(date +%s)
    timeout 600 "$@"; rc=$?; echo "== $name rc=$rc seconds=$(( $(date +%s) - s ))"
    [ $rc = 0 ] || FAILED=1; }
  run cpu_recipes pytest -q tests/unit_tests/cpu/recipes/test_muse_spark.py -k fused
  run pad_segments pytest -q tests/unit_tests/cpu/models/common/test_varlen_attention.py tests/unit_tests/cpu/components/data/test_grain_data.py
  run norm_cast pytest -q tests/unit_tests/gpu/models/muse_spark/test_attention_compile.py -k fused_pre_norm_cast
  run postnorm pytest -q tests/unit_tests/gpu/models/muse_spark/test_input_scaled_rmsnorm.py
  run dist_moe_anchors pytest -q torchtitan/experiments/dist_moe_patches/tests
  run h3_offload python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    torchtitan/experiments/dist_moe_patches/tests/test_h3_offload.py
  run h3_offload_swiglu python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    torchtitan/experiments/dist_moe_patches/tests/test_h3_offload.py --with-swiglu-recompute
  echo "== ALL rc=$FAILED"; exit $FAILED
'
