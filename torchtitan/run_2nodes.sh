#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Multinode training script for DeepSeek V3 with HybridEP
#
# Usage (automatic with SLURM):
#   srun --nodes=2 --ntasks-per-node=1 ./run_2nodes.sh
#
# Usage (manual):
#   Node 0: MASTER_ADDR=$(hostname) ./run_2nodes.sh
#   Node 1: MASTER_ADDR=<node0_hostname> NODE_RANK=1 ./run_2nodes.sh
#
# Environment variables (all optional, auto-detected from SLURM if available):
#   NGPU          - GPUs per node (default: 4)
#   NNODES        - Number of nodes (default: 2, or from SLURM_NNODES)
#   NODE_RANK     - This node's rank (default: 0, or from SLURM_NODEID)
#   MASTER_ADDR   - Master node address (auto-detected from SLURM_NODELIST or hostname)
#   MASTER_PORT   - Master port (default: 29500)
#   MOE_BACKEND   - "hybridep", "deepep", or "standard" (default: hybridep)
#   CONFIG_FILE   - Path to config TOML file

# =============================================================================
# Auto-detect from SLURM environment (if running under SLURM)
# =============================================================================

NNODES=${NNODES:-2}

# Node rank (SLURM_NODEID is 0-indexed)
if [[ -n "$SLURM_NODEID" ]]; then
    NODE_RANK=${NODE_RANK:-$SLURM_NODEID}
else
    NODE_RANK=${NODE_RANK:-0}
fi

# Master address (first node in SLURM_NODELIST, or current hostname for node 0)
MASTER_ADDR=ptyche0068.ptyche.clusters.nvidia.com

# =============================================================================
# Configuration
# =============================================================================

NGPU=${NGPU:-4}                    # GPUs per node
MASTER_PORT=${MASTER_PORT:-61626}  # Master port
export LOG_RANK=${LOG_RANK:-"0"}   # Which ranks to log from

# Training configuration
CONFIG_FILE=${CONFIG_FILE:-"/lustre/fsw/sw_aidot/elfieg/torchtitan/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

# Communication backend: "hybridep" (GB200), "deepep" (H100), or "standard"
MOE_BACKEND=${MOE_BACKEND:-"hybridep"}

# HybridEP configuration
# - NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: Ranks sharing the same NVLink domain
# - USE_MNNVL: Enable Multi-Node NVLink for NVL72/Grace-Blackwell systems
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4
export USE_MNNVL=${USE_MNNVL:-1}

# Set TRITON_CACHE_DIR to avoid stale file handle errors on shared filesystems
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_$(whoami)"}
mkdir -p "$TRITON_CACHE_DIR"

# Profiling output directory
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${SLURM_JOB_ID:-manual}_${RUN_TIMESTAMP}"
PROFILE_DIR=${PROFILE_DIR:-"/lustre/fsw/sw_aidot/elfieg/torchtitan/logs/profiles/${RUN_ID}"}
mkdir -p "$PROFILE_DIR"

# =============================================================================
# Print configuration
# =============================================================================

echo "=========================================="
echo "Multinode Training Configuration"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "NNODES: $NNODES"
echo "NGPU: $NGPU (per node)"
echo "Total GPUs: $((NNODES * NGPU))"
echo "MoE Backend: $MOE_BACKEND"
echo "MoE Quant: $MOE_QUANT"
echo "Config: $CONFIG_FILE"
echo "Triton Cache: $TRITON_CACHE_DIR"
echo "Profile Dir: $PROFILE_DIR"
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "SLURM Node List: $SLURM_NODELIST"
fi
echo "=========================================="

# =============================================================================
# Launch training
# =============================================================================

# Use local torchtitan instead of installed version
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${TORCHTITAN_ROOT}:${PYTHONPATH}"
echo "Using torchtitan from: $TORCHTITAN_ROOT"

# Change to torchtitan root so relative paths in config (e.g., hf_assets_path) resolve correctly
cd "$TORCHTITAN_ROOT"
echo "Working directory: $(pwd)"

# MoE expert GEMM backend:
#   "torchao"  - torchao's ScaledGroupedMMTensor (MXFP8 via torchao kernels)
#   "te_mxfp8" - TE's TEGroupedMMTensor (MXFP8 via general_grouped_gemm)
#   "te_bf16"  - TE's TEGroupedMMTensor (BF16 via general_grouped_gemm)
#   "none"     - No grouped MM converter (native torch._grouped_mm in BF16)
# All use the original GroupedExperts module — the difference is which tensor
# subclass wraps the weights to intercept torch._grouped_mm.
MOE_QUANT=${MOE_QUANT:-"te_mxfp8"}

# Build converter args based on MOE_QUANT selection
if [[ "$MOE_QUANT" == "torchao" ]]; then
    # Note: grouped_mm.mx only supports recipe_name="mxfp8" (default), not "mxfp8_cublas"
    CONVERTER_ARGS='--model.converters=quantize.grouped_mm.mx --quantize.grouped_mm.mx.fqns=experts'
elif [[ "$MOE_QUANT" == "te_mxfp8" ]]; then
    # TE converter path: wraps weights in TEGroupedMMTensor (MXFP8 mode)
    CONVERTER_ARGS='--model.converters=quantize.grouped_mm.te --quantize.grouped_mm.te.fqns=experts --quantize.grouped_mm.te.mode=mxfp8'
elif [[ "$MOE_QUANT" == "te_bf16" ]]; then
    # TE converter path: wraps weights in TEGroupedMMTensor (BF16 mode)
    CONVERTER_ARGS='--model.converters=quantize.grouped_mm.te --quantize.grouped_mm.te.fqns=experts --quantize.grouped_mm.te.mode=bf16'
else
    CONVERTER_ARGS=''
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_backend=c10d \
    --role rank \
    --tee 3 \
    -m ${TRAIN_FILE} \
    --job.config_file ${CONFIG_FILE} \
    --profiling.no-enable_profiling \
    --training.dataset_path "/lustre/fsw/sw_aidot/elfieg/datasets" \
    --profiling.save_traces_folder "${PROFILE_DIR}" \
    --parallelism.pipeline_parallel_schedule "Interleaved1F1B" \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.pipeline_parallel_first_stage_less_layers 1 \
    --parallelism.pipeline_parallel_last_stage_less_layers 1 \
    --parallelism.tensor_parallel_degree 1 \
    --parallelism.expert_parallel_degree 4 \
    --parallelism.expert_tensor_parallel_degree 1 \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.expert_parallel_comm_backend ${MOE_BACKEND} \
    --parallelism.hybridep.enable-non-blocking \
    --parallelism.hybridep.moe-expert-capacity-factor 1.0 \
    --training.local_batch_size 32 \
    --compile.enable \
    --compile.components=model \
    --compile.components=loss \
    --model.hf_assets_path=/lustre/fsw/sw_aidot/elfieg/torchtitan/assets/hf/deepseek-moe-16b-base \
    ${CONVERTER_ARGS} \
    "$@"
