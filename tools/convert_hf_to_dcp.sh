#!/usr/bin/env bash
set -euo pipefail

# Convert HF BailingMoeV2 checkpoint to Megatron DCP format.
# Uses the same model config as training, run with torchrun.

# Preserve the user's working patched-P2P/NCCL configuration.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

HF_PATH="${1:?Usage: $0 <hf-path> <dcp-save-path> [iteration]}"
DCP_SAVE_PATH="${2:?Usage: $0 <hf-path> <dcp-save-path> [iteration]}"
ITERATION="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_DIR}/Megatron-LM-core_v0.13.0}"
source "${REPO_DIR}/examples/sft/megatron/poziomka_model_args.sh"
[[ -f "${HF_PATH}/config.json" ]] || { echo "Missing HF config" >&2; exit 1; }
[[ -f "${MEGATRON_PATH}/pretrain_gpt.py" ]] || { echo "Missing patched Megatron" >&2; exit 1; }
[[ ! -e "${DCP_SAVE_PATH}" ]] || { echo "Refusing existing output: ${DCP_SAVE_PATH}" >&2; exit 1; }

PYTHONPATH="${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
torchrun --standalone --nproc_per_node=8 \
    "${SCRIPT_DIR}/load_hf_save_dcp.py" \
    --hf-path "${HF_PATH}" \
    --save-iteration "${ITERATION}" \
    \
    "${POZIOMKA_MODEL_ARGS[@]}" \
    --bf16 \
    \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --seq-length 3072 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${HF_PATH}" \
    --no-initialization \
    --use-cpu-initialization \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --save "${DCP_SAVE_PATH}" \
    --save-interval 1 \
    --ckpt-format torch_dist \
    --no-one-logger \
    --no-masked-softmax-fusion \
    --attention-backend flash \
    --attention-softmax-in-fp32 \
    --mock-data
