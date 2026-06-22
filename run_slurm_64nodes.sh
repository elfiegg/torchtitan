#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=sw_aidot-deepseek.v3-671b
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --segment=16

set -ex

# Create run directory with timestamp for provenance (logs + copied scripts)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}"
# Use absolute path for RUN_DIR to ensure container can access it
TORCHTITAN_HOME=${TORCHTITAN_HOME:-"/lustre/fsw/sw_aidot/elfieg/torchtitan"}
RUN_DIR="${TORCHTITAN_HOME}/logs/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
DEEP_EP_REPO=${DEEP_EP_REPO:-"/lustre/fsw/sw_aidot/elfieg/DeepEP"}

# Helper to capture dependency versions inside the container
cat > "${RUN_DIR}/dump_deps.py" <<'PY'
import argparse, json, sys, os, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--deepep", default="")
parser.add_argument("--output", required=True)
args = parser.parse_args()


def git_rev(path: str) -> str:
    if not path:
        return "not provided"
    if not os.path.isdir(path):
        return "not found"
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


info = {
    "container_image": args.image,
    "python": sys.version.split()[0],
}

try:
    import torch  # noqa: F401
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["triton"] = getattr(torch.version, "triton", None)
except Exception as exc:  # noqa: BLE001
    info["torch"] = f"unavailable ({exc})"

try:
    import torchao  # noqa: F401
    info["torchao"] = getattr(torchao, "__version__", "unknown")
except Exception as exc:  # noqa: BLE001
    info["torchao"] = f"unavailable ({exc})"

info["deepep_git"] = git_rev(args.deepep)

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
PY

# ==========================================
# Configuration
# ==========================================
CONFIG_FILE=${CONFIG_FILE:-"${TORCHTITAN_HOME}/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

# Preserve launch artifacts for reproducibility
cp "$0" "${RUN_DIR}/$(basename "$0")"
cp "$CONFIG_FILE" "${RUN_DIR}/$(basename "$CONFIG_FILE")"

# Log file paths (per run)
LOG_OUT="${RUN_DIR}/deepseek_v3_671b.out"
LOG_ERR="${RUN_DIR}/deepseek_v3_671b.err"
# Profiling output directory
PROFILE_DIR="${RUN_DIR}/profile_trace"
mkdir -p "$PROFILE_DIR"

# Container configuration (adjust as needed)
IMAGE=${IMAGE:-"/lustre/fsw/sw_aidot/elfieg/container/hybrid-te.sqsh"}
# Mount lustre and output directory
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/lustre:/lustre,${RUN_DIR}:${RUN_DIR}"}

# Cluster configuration (GB200: 4 GPUs per node)
NGPU_PER_NODE=4
NNODES=${SLURM_NNODES:-64}
WORLD_SIZE=$((NNODES * NGPU_PER_NODE))

# Parallelism configuration for 671B on 64 nodes (256 GPUs)
# Adjust these based on your model and memory requirements
DATA_PARALLEL_SHARD_DEGREE=${DATA_PARALLEL_SHARD_DEGREE:--1}
# Note: EP degree of 32 with NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32 worked previously
EXPERT_PARALLEL_DEGREE=${EXPERT_PARALLEL_DEGREE:-64}
PIPELINE_PARALLEL_DEGREE=${PIPELINE_PARALLEL_DEGREE:-1}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-8}
TRAINING_STEPS=${TRAINING_STEPS:-50}

# Dataset path
DATASET_PATH=${DATASET_PATH:-"/lustre/fsw/sw_aidot/elfieg/datasets"}

# Logging configuration
LOG_RANK=${LOG_RANK:-"0,1,2,3"}

# MoE Backend ("hybridep" requires /dev/nvidia-caps and /run/nvidia-fabric mounts)
MOE_BACKEND=${MOE_BACKEND:-"hybridep"}

# MoE quant: torchao | te_mxfp8 | te_bf16 | none (same as run_2nodes.sh)
MOE_QUANT=${MOE_QUANT:-"te_mxfp8"}
# Linear mxfp8_cublas requires torchao with MXLinearConfig; set USE_LINEAR_MX=1 to enable
USE_LINEAR_MX=${USE_LINEAR_MX:-0}

# Build converter args (same logic as run_2nodes.sh)
if [[ "$MOE_QUANT" == "torchao" ]]; then
    if [[ "$USE_LINEAR_MX" == "1" ]]; then
        CONVERTER_ARGS="--model.converters=quantize.linear.mx,quantize.grouped_mm.mx --quantize.linear.mx.recipe_name=mxfp8_cublas --quantize.grouped_mm.mx.fqns=experts"
    else
        CONVERTER_ARGS='--model.converters=quantize.grouped_mm.mx --quantize.grouped_mm.mx.fqns=experts'
    fi
elif [[ "$MOE_QUANT" == "te_mxfp8" ]]; then
    if [[ "$USE_LINEAR_MX" == "1" ]]; then
        CONVERTER_ARGS="--model.converters=quantize.linear.mx,quantize.grouped_mm.te --quantize.linear.mx.recipe_name=mxfp8_cublas --quantize.grouped_mm.te.fqns=experts --quantize.grouped_mm.te.mode=mxfp8"
    else
        CONVERTER_ARGS='--model.converters=quantize.grouped_mm.te --quantize.grouped_mm.te.fqns=experts --quantize.grouped_mm.te.mode=mxfp8'
    fi
elif [[ "$MOE_QUANT" == "te_bf16" ]]; then
    CONVERTER_ARGS='--model.converters=quantize.grouped_mm.te --quantize.grouped_mm.te.fqns=experts --quantize.grouped_mm.te.mode=bf16'
else
    if [[ "$USE_LINEAR_MX" == "1" ]]; then
        CONVERTER_ARGS='--model.converters=quantize.linear.mx --quantize.linear.mx.recipe_name=mxfp8_cublas'
    else
        CONVERTER_ARGS=''
    fi
fi

# ==========================================
# Print Configuration
# ==========================================
echo "=========================================="
echo "SLURM Job Configuration"
echo "=========================================="
echo "Timestamp: $RUN_TIMESTAMP"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NNODES"
echo "GPUs per node: $NGPU_PER_NODE"
echo "Total GPUs: $WORLD_SIZE"
echo "Config: $CONFIG_FILE"
echo "MoE Backend: $MOE_BACKEND"
echo "MoE Quant: $MOE_QUANT"
echo "Use Linear MX (mxfp8_cublas): $USE_LINEAR_MX"
echo "Data Parallel: $DATA_PARALLEL_SHARD_DEGREE"
echo "Expert Parallel: $EXPERT_PARALLEL_DEGREE"
echo "Pipeline Parallel: $PIPELINE_PARALLEL_DEGREE"
echo "Local Batch Size: $LOCAL_BATCH_SIZE"
echo "=========================================="

# ==========================================
# Build Training Command 
# export TORCH_TRACE=$RUN_DIR/traces; \
# mkdir -p \$TORCH_TRACE; \
# ==========================================
TRAIN_CMD="\
cd $TORCHTITAN_HOME; \
ulimit -c 0; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL_DEGREE; \
export USE_MNNVL=1; \
export NCCL_IB_TIMEOUT=22; \
export RDMA_CORE_HOME=\${RDMA_CORE_HOME:-/usr}; \
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0; \
export LD_LIBRARY_PATH=${RDMA_CORE_HOME}/lib:\$LD_LIBRARY_PATH; \
export TRITON_CACHE_DIR=/tmp/triton_cache_\$(whoami)_\$SLURM_PROCID; \
mkdir -p \$TRITON_CACHE_DIR; \
export HYBRIDEP_DEBUG=1; \
export PYTHONPATH=$TORCHTITAN_HOME:\$PYTHONPATH; \
export LOCAL_RANK=\$SLURM_LOCALID; \
python ${RUN_DIR}/dump_deps.py --image $IMAGE --deepep ${DEEP_EP_REPO} --output ${RUN_DIR}/dependency_info.json; \
python -m $TRAIN_FILE \
    --job.config_file $CONFIG_FILE \
    --training.steps=$TRAINING_STEPS \
    --training.dataset_path=$DATASET_PATH \
    --profiling.enable_profiling \
    --comm.init_timeout_seconds=3000 \
    --comm.train_timeout_seconds=2000 \
    --profiling.save_traces_folder $PROFILE_DIR \
    --parallelism.data_parallel_shard_degree=$DATA_PARALLEL_SHARD_DEGREE \
    --parallelism.expert_parallel_degree=$EXPERT_PARALLEL_DEGREE \
    --parallelism.pipeline_parallel_degree=$PIPELINE_PARALLEL_DEGREE \
    --training.local_batch_size=$LOCAL_BATCH_SIZE \
    --activation_checkpoint.mode=full \
    --debug.moe_force_load_balance \
    ${CONVERTER_ARGS} \
    --compile.enable \
    --parallelism.hybridep.enable_non_blocking \
    --parallelism.hybridep.moe_expert_capacity_factor=0.03125 \
    --compile.components=loss \
    --compile.components=model \
    --parallelism.expert_parallel_comm_backend=$MOE_BACKEND"

# Save the fully expanded training command for provenance
printf "%s\n" "$TRAIN_CMD" > "${RUN_DIR}/train_cmd.sh"
chmod +x "${RUN_DIR}/train_cmd.sh"

# ==========================================
# Launch with srun
# ==========================================
srun --container-image="$IMAGE" \
     --container-name=deepseek-v3-671b \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-env="LOG_RANK=${LOG_RANK}" \
     --output="$LOG_OUT" \
     --error="$LOG_ERR" \
     --no-container-mount-home \
     --container-writable \
     bash -c "$TRAIN_CMD"

