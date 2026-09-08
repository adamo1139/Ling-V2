#!/usr/bin/env bash
set -euo pipefail

# Poziomka SFT run 1: full-parameter SFT of the linear-8-9-10-11-sqrt merge.
#
# Concrete run configuration on top of run_poziomka.sh. Checkpoints go to the
# nvme; at 7.7 GB each the root filesystem cannot hold a run's worth.
# TRAIN_ITERS=1326 is one full pass over the 1,018,156 cached conversations at
# global batch 768, roughly 47 hours on 8 GPUs.
# Recompute TRAIN_ITERS after rebuilding the v2 full-conversation cache; its
# retained record count can differ from the original role-masked cache.
#
# Batch 768 and LR 3e-4 match Poziomka 11 pretraining, so the LR is used at the
# effective batch it was tuned for rather than at a 6x smaller one.
#
# Every value below can be overridden from the environment, e.g.
#   TRAIN_ITERS=2000 LR=3e-5 bash run_poziomka_sft_run1.sh
# Extra Megatron flags are forwarded: bash run_poziomka_sft_run1.sh --log-interval 10

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# Cache and base checkpoint live beside the checkout, not inside it.
WORK_DIR="${WORK_DIR:-$(cd -- "${REPO_DIR}/.." && pwd)}"

export SFT_DATA="${SFT_DATA:-${WORK_DIR}/poziomka-sft-cache-3072}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${WORK_DIR}/poziomka-linear-8-9-10-11-sqrt-dcp}"
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run1}"

export TRAIN_ITERS="${TRAIN_ITERS:-1326}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
export WARMUP_ITERS="${WARMUP_ITERS:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-1}"

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

exec bash "${SCRIPT_DIR}/run_poziomka.sh" "$@"
