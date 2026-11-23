#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one node of the fixed 64-GPU Muse Spark compact benchmark.

  run_node.sh NODE_RANK MASTER_ADDR MASTER_PORT

Set MUSE_REPACK_SOURCE, MUSE_REPACK_PLAN and REPRO_OUTPUT to absolute shared paths.
Set PYTHON if the pinned environment's Python is not on PATH.
EOF
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# != 3 ]]; then
  usage >&2
  exit 2
fi
node_rank=$1
master_addr=$2
master_port=$3
if [[ ! "$node_rank" =~ ^([0-9]|1[0-5])$ ]]; then
  printf 'NODE_RANK must be an integer from 0 to 15\n' >&2
  exit 2
fi
for name in MUSE_REPACK_SOURCE MUSE_REPACK_PLAN REPRO_OUTPUT; do
  if [[ "${!name:-}" != /* ]]; then
    printf '%s must be an absolute path\n' "$name" >&2
    exit 2
  fi
done
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
cd "$repo_root"

mkdir -p "$REPRO_OUTPUT"
export REPRO_LOCAL_OUTPUT
REPRO_LOCAL_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/muse-spark-compact.XXXXXXXX")
trap 'rm -rf -- "$REPRO_LOCAL_OUTPUT"' EXIT
export PYTHONPATH="$script_dir:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=1 PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1
export OMP_NUM_THREADS=8
unset NCCL_LAUNCH_ORDER_IMPLICIT
export MUSE_SLOT_GIB=49
export EXP_NORM_FUSION=cast

"${PYTHON:-python3}" -m torch.distributed.run \
  --nnodes=16 --nproc-per-node=4 --node-rank="$node_rank" \
  --master-addr="$master_addr" --master-port="$master_port" "$script_dir/launch.py" \
  --module repacked_recipe --config bf16_repacked --output-dir "$REPRO_OUTPUT/trainer"
