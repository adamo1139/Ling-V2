#!/usr/bin/env bash
set -euo pipefail

# Run 2: v11 full-conversation SFT at 8192 tokens from the original merged weights.
# Launch with: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run2.sh
# Edit run settings here; no caller environment variables are needed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
export SFT_DATA="${WORK_DIR}/poziomka-sft-cache-v11-8192-all-greedy"
export LOAD_CHECKPOINT="${WORK_DIR}/poziomka-linear-8-9-10-11-sqrt-dcp"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run2_v11_8192"
export RESUME=0

# One pass over the generated cache: ceil(1,318,934 / 768) = 1718.
# Verified greedy cache: 2,720,336,359 tokens; no packing, so steps use records.
# The final batch wraps by 490 samples. Recalculate if cache or batch size changes.
export TRAIN_ITERS=1718
export GLOBAL_BATCH_SIZE=768
export SEQ_LENGTH=8192
export LR=3e-4
export WARMUP_ITERS=0
export SAVE_INTERVAL=100
export EVAL_INTERVAL=100
export EVAL_ITERS=1
export DATALOADER_WORKERS=2
export ROUTER_BIAS_UPDATE_RATE=0

# Megatron enables W&B through CLI flags; entity is read by the W&B SDK.
# Reuse the latest pretraining project and existing training-environment login.
export WANDB_ENTITY="adamo1139"
export WANDB_PROJECT="poziomka_10"
export WANDB_NAME="poziomka_sft_run2_v11_8192"
export WANDB_MODE="online"

# Refuse a run whose checkpoints cannot fit: ~8 GB per save, weights only.
save_parent="$(dirname "${SAVE_CHECKPOINT}")"
[[ -d "${save_parent}" ]] || { echo "No such directory: ${save_parent}" >&2; exit 1; }
free_gb=$(df -BG --output=avail "${save_parent}" | tail -1 | tr -dc '0-9')
needed_gb=$(( (TRAIN_ITERS / SAVE_INTERVAL + 2) * 8 ))
if (( free_gb < needed_gb )); then
    echo "Checkpoints need ~${needed_gb} GB, ${free_gb} GB free on ${save_parent}." >&2
    echo "Raise SAVE_INTERVAL, lower TRAIN_ITERS, or free space." >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/run_poziomka.sh" \
    --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" "$@"
