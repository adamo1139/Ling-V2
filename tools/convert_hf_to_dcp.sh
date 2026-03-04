#!/bin/bash
set -ex

# Convert HF BailingMoeV2 checkpoint to Megatron DCP format.
# Uses the same model config as training, run with torchrun.

export NCCL_P2P_LEVEL=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Tree
export NCCL_PROTO=LL
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
export NCCL_CUMEM_ENABLE=0

HF_PATH="${1:?Usage: $0 <hf-path> <dcp-save-path> [iteration]}"
DCP_SAVE_PATH="${2:?Usage: $0 <hf-path> <dcp-save-path> [iteration]}"
ITERATION="${3:-8000}"

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
MEGATRON_PATH="Megatron-LM-core_v0.13.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZER_PATH="${SCRIPT_DIR}/../resource/tokenizer/apt4"

PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH \
torchrun --nproc_per_node ${GPUS_PER_NODE} \
    tools/load_hf_save_dcp.py \
    --hf-path ${HF_PATH} \
    --save-iteration ${ITERATION} \
    \
    --num-layers 16 \
    --hidden-size 2048 \
    --ffn-hidden-size 2048 \
    --num-attention-heads 16 \
    --num-query-groups 4 \
    --group-query-attention \
    --qk-layernorm \
    --max-position-embeddings 8192 \
    --vocab-size 32000 \
    --make-vocab-size-divisible-by 128 \
    --position-embedding-type rope \
    --rotary-base 84000 \
    --rotary-percent 0.5 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --disable-bias-linear \
    --transformer-impl transformer_engine \
    --bf16 \
    \
    --expert-model-parallel-size 1 \
    --num-experts 128 \
    --moe-ffn-hidden-size 320 \
    --moe-shared-expert-intermediate-size 320 \
    --moe-router-score-function sigmoid \
    --moe-router-topk 16 \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 2 \
    --moe-layer-freq "[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-dtype fp32 \
    \
    --pipeline-model-parallel-size 8 \
    --tensor-model-parallel-size 1 \
    --sequence-parallel \
    \
    --micro-batch-size 1 \
    --global-batch-size 384 \
    --seq-length 6144 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --no-initialization \
    --use-cpu-initialization \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --save ${DCP_SAVE_PATH} \
    --save-interval 1 \
    --ckpt-format torch_dist \
    --no-one-logger \
    --no-masked-softmax-fusion \
    --attention-backend flash \
    --attention-softmax-in-fp32 \
    --mock-data
