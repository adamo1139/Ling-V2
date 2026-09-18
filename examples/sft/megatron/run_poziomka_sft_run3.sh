#!/usr/bin/env bash
set -euo pipefail

# Run 3: v11 SFT at 16384 tokens, packed, continuing from run 2's weights.
# Launch with: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run3.sh
# Edit run settings here; no caller environment variables are needed.
#
# Differences from run 2, all deliberate:
#   16384 instead of 8192   -- confirmed at 74% of 24 GB by the smoke sweep
#   --long-policy truncate  -- overlong records have ONE reasoning block (99.5%),
#                              so remove-reasoning would strip it entirely and train
#                              a thinking-off answer that needed 25k tokens of thought
#   packing on              -- ~3.2x fewer sequences; parity-verified against unpacked
#   loads run 2, not base   -- further finetune, not a fresh SFT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
export SFT_DATA="${WORK_DIR}/poziomka-sft-cache-v11-16384-truncate"
export LOAD_CHECKPOINT="${WORK_DIR}/poziomka_run2_v11_8192"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run3_v11_16384"
export RESUME=0

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
export PACKING=1
export GLOBAL_BATCH_SIZE=768
export LR=3e-4
export WARMUP_ITERS=0
export SAVE_INTERVAL=100
export EVAL_INTERVAL=100
export EVAL_ITERS=1
export DATALOADER_WORKERS=2
export ROUTER_BIAS_UPDATE_RATE=0

# One pass over the PACKED cache. Packing decides the sequence count at startup,
# so this is an estimate from the length survey (~410,200 sequences / 768).
# The trainer prints the real count as "SFT: ... packed into N sequences" -- check
# it on the first launch and correct TRAIN_ITERS before committing to the full run.
export TRAIN_ITERS=534

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run3_v11_16384_packed"
export WANDB_MODE="online"

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it first:" >&2
    echo "  python3 ${SCRIPT_DIR}/prepare_poziomka_sft.py \\" >&2
    echo "    --input <v11-export> --tokenizer ${WORK_DIR}/poziomka-linear-8-9-10-11-sqrt \\" >&2
    echo "    --output ${SFT_DATA} --seq-length 16384 --long-policy truncate --workers 15" >&2
    exit 1; }

# The cache must match the training length, and must be the truncate cache.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["long_policy"] != "truncate":
    sys.exit(f"Expected a truncate cache, found long_policy={manifest['long_policy']}")
print(f"Cache OK: {manifest['totals']['train']['records']:,} train records, "
      f"{manifest['totals']['train']['tokens']:,} tokens, "
      f"{manifest['totals']['train'].get('truncated_records', 0):,} truncated")
PY

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
