#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one node of the fixed 64-GPU Muse Spark compact benchmark.

  run_node.sh {rubin|gb300} {bf16|mxfp8} NODE_RANK MASTER_ADDR MASTER_PORT

Set MUSE_REPACK_SOURCE, MUSE_REPACK_PLAN and REPRO_OUTPUT to absolute shared paths.
Set PYTHON if the pinned environment's Python is not on PATH.
EOF
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# != 5 ]]; then
  usage >&2
  exit 2
fi
export MUSE_PLATFORM=$1
precision=$2
node_rank=$3
master_addr=$4
master_port=$5
if [[ "$MUSE_PLATFORM" != rubin && "$MUSE_PLATFORM" != gb300 ]]; then
  printf 'Expected rubin or gb300\n' >&2
  exit 2
fi
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
export EXP_NORM_FUSION=cast EXP_MOE_H3_OFFLOAD=0 REPRO_STOP_AFTER=100
export MUSE_CP=1 MUSE_TP=1 MUSE_EP=64 MUSE_LOSS_CHUNKS=16 MUSE_SAVE_PROJ=1
export MUSE_PROFILE=0 MUSE_SEPARATE_RS=0 MUSE_UNSHARD_CURRENT=0 MUSE_EXPERT_PRESET=''
export MUSE_SCRATCH_FACTOR=1
case "$MUSE_PLATFORM:$precision" in
  rubin:bf16|gb300:bf16)
    launcher=launch.py recipe=repacked_recipe entry=bf16_repacked
    export MUSE_SLOT_GIB=49 VARIANT=repacked29184routeh333full
    if [[ "$MUSE_PLATFORM" == gb300 ]]; then
      export MUSE_SLOT_GIB=45 VARIANT=gb300-bf16-compact
    fi
    export EXP_SWIGLU_RECOMPUTE=1 EXP_LEARNED_ROUTE=1 EXP_SELECTIVE_HOST_H3=1
    ;;
  rubin:mxfp8|gb300:mxfp8)
    launcher=mxfp8_launch.py recipe=mxfp8_recipe entry=mxfp8_repacked
    export MUSE_SLOT_GIB=42 VARIANT=compact-mxfp8
    if [[ "$MUSE_PLATFORM" == gb300 ]]; then
      export VARIANT=gb300-mxfp8-compact
    fi
    export EXP_SWIGLU_RECOMPUTE=0 EXP_LEARNED_ROUTE=0 EXP_SELECTIVE_HOST_H3=0
    ;;
  *) printf 'Expected bf16 or mxfp8\n' >&2; exit 2 ;;
esac
export MUSE_PRECISION="$precision"
if [[ "$MUSE_PLATFORM" == gb300 ]]; then launcher=gb300_launch.py; fi

"${PYTHON:-python3}" -m torch.distributed.run \
  --nnodes=16 --nproc-per-node=4 --node-rank="$node_rank" \
  --master-addr="$master_addr" --master-port="$master_port" "$script_dir/$launcher" \
  --module "$recipe" --config "$entry" --output-dir "$REPRO_OUTPUT/trainer"
