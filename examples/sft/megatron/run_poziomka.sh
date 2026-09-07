#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/poziomka_model_args.sh"

: "${SFT_DATA:?Set SFT_DATA to the prepared cache directory}"
: "${LOAD_CHECKPOINT:?Set LOAD_CHECKPOINT to the exact merged checkpoint in Megatron DCP format}"
: "${SAVE_CHECKPOINT:?Set SAVE_CHECKPOINT to a new SFT output directory}"
: "${TRAIN_ITERS:?Set TRAIN_ITERS explicitly (start with a small smoke run)}"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_DIR}/Megatron-LM-core_v0.13.0}"
SEQ_LENGTH="${SEQ_LENGTH:-3072}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"

[[ -f "${MEGATRON_PATH}/pretrain_gpt.py" ]] || { echo "Missing patched Megatron checkout" >&2; exit 1; }
[[ -f "${SFT_DATA}/manifest.json" ]] || { echo "Missing completed SFT manifest" >&2; exit 1; }
[[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || { echo "Missing DCP tracker" >&2; exit 1; }
if [[ "${RESUME:-0}" != 1 && "$(realpath -m "${LOAD_CHECKPOINT}")" == "$(realpath -m "${SAVE_CHECKPOINT}")" ]]; then
    echo "Input and output checkpoints must differ" >&2; exit 1
fi
if [[ "${RESUME:-0}" != 1 && -e "${SAVE_CHECKPOINT}" ]]; then
    echo "Refusing existing output; set RESUME=1 with LOAD_CHECKPOINT pointing to that SFT run" >&2; exit 1
fi
# Resume loads optimizer/RNG/scheduler; initial SFT deliberately resets them.
LOAD_ARGS=(--finetune --no-load-optim --no-load-rng --override-opt_param-scheduler)
if [[ "${RESUME:-0}" == 1 ]]; then
    [[ -f "${SAVE_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || { echo "No SFT checkpoint to resume" >&2; exit 1; }
    [[ "$(realpath -m "${LOAD_CHECKPOINT}")" == "$(realpath -m "${SAVE_CHECKPOINT}")" ]] || { echo "Resume LOAD/SAVE must point to the same SFT run" >&2; exit 1; }
    LOAD_ARGS=(--use-checkpoint-opt_param-scheduler)
fi

# Preserve the user's working patched-P2P/NCCL settings. No forced P2P disable,
# Hopper-only DeepEP, installations, or downloaded dependencies.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export PYTHONPATH="${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --standalone --nproc_per_node=8 "${SCRIPT_DIR}/train_poziomka_sft.py" \
    "${POZIOMKA_MODEL_ARGS[@]}" \
    --sft-manifest "${SFT_DATA}/manifest.json" \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${SFT_DATA}/tokenizer" \
    --seq-length "${SEQ_LENGTH}" --micro-batch-size 1 --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters "${TRAIN_ITERS}" --bf16 --optimizer adam --use-distributed-optimizer \
    --calculate-per-token-loss \
    --lr "${LR:-3e-4}" --min-lr "${LR:-3e-4}" --lr-decay-style constant \
    --lr-warmup-iters "${WARMUP_ITERS:-0}" --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 --seed 42 \
    --moe-router-bias-update-rate "${ROUTER_BIAS_UPDATE_RATE:-0}" \
    --moe-z-loss-coeff 0.0000035 --bias-zero-mean-update \
    --moe-permute-fusion --cross-entropy-loss-fusion --cross-entropy-fusion-impl te \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --dataloader-type single --num-workers "${DATALOADER_WORKERS:-2}" \
    --no-create-attention-mask-in-dataloader --attention-backend flash \
    --attention-softmax-in-fp32 --no-masked-softmax-fusion \
    --load "${LOAD_CHECKPOINT}" --save "${SAVE_CHECKPOINT}" --ckpt-format torch_dist \
    --save-interval "${SAVE_INTERVAL:-100}" --eval-interval "${EVAL_INTERVAL:-100}" \
    --eval-iters "${EVAL_ITERS:-10}" --log-interval 1 --no-one-logger \
    "${LOAD_ARGS[@]}" "$@"
