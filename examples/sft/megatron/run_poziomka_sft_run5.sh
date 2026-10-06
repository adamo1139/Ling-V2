#!/usr/bin/env bash
set -euo pipefail

# Run 5: SFT na polskie-sprawy-v3 przy 16384, z pakowaniem, GBS 64, jedna epoka.
# Launch with: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run5.sh
# Edit run settings here; no caller environment variables are needed.
#
# Start z merge'a poziomka-instruct-2026-09-30-7 (wagi po run 3/4, RoPE 640000),
# przekonwertowanego z HF do DCP:
#   ROTARY_BASE=640000 bash Ling-V2/tools/convert_hf_to_dcp.sh \
#     /media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7 \
#     /media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp
# Konwerter odrzuca config.json z rope_theta != ROTARY_BASE.
#
# Dane: polskie-sprawy-v3, 427,970 jednoturowych rozmow, 1% (po id) na
# walidacje. Kazda odpowiedz bez rozumowania ma pusty '<think>\n</think>\n'
# (prepare_polskie_sprawy_v3.py) - juz nie tylko 10% jak w v11/v12.
# Cache: tokenizer i szablon v12, 16384, truncate (zadna rozmowa nie przekracza
# nawet 8192; max ~6,5k tokenow, srednio ~400).
#
# Pakowanie: rozmowy srednio ~400 tokenow, wiec wypelnienie okna ogranicza
# MAX_SUBSEQUENCES=32 (poziomka_data.py), a nie dlugosc: 77,4%. Limit zostaje,
# bo na nim robiona byla weryfikacja thd/parity z run 3.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
export SFT_DATA="${WORK_DIR}/poziomka-sft-cache-polskie-sprawy-v3-16384"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run5_polskie_sprawy_v3_16384"
export RESUME=0

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
# Merge pochodzi z wag trenowanych na 640000 (run 3/4); 16384 wymaga >= 3,1e5.
export ROTARY_BASE=640000
export PACKING=1
export GLOBAL_BATCH_SIZE=64
export LR=3e-4
export WARMUP_ITERS=0
export SAVE_INTERVAL=50
export EVAL_INTERVAL=20
# Walidacja to 136 spakowanych sekwencji; 2 x 64 = 128 to prawie calosc.
export EVAL_ITERS=2
export DATALOADER_WORKERS=2
export ROUTER_BIAS_UPDATE_RATE=0

# Jedna epoka: 13,532 kubelkow po spakowaniu / 64 na krok = 211,4 -> 212,
# policzone kodem treningu: MMapSFTDataset(manifest, "train", pack=True) ->
# len(d.bins). Ostatni batch zawija sie o 16 sekwencji.
export TRAIN_ITERS=212

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run5_polskie_sprawy_v3_16384_packed_212steps"
export WANDB_MODE="online"

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it first:" >&2
    echo "  python3 ${SCRIPT_DIR}/prepare_polskie_sprawy_v3.py \\" >&2
    echo "    --input polskie-sprawy-v3/sft.jsonl --output polskie-sprawy-v3-sft" >&2
    echo "  python3 ${SCRIPT_DIR}/prepare_poziomka_sft.py --input polskie-sprawy-v3-sft \\" >&2
    echo "    --tokenizer poziomka-fun-rp-v12/tokenizer --chat-template poziomka-fun-rp-v12/chat_template.jinja \\" >&2
    echo "    --output ${SFT_DATA} --seq-length 16384 --long-policy truncate --workers 16" >&2
    exit 1; }

# The cache must match the training length and this run's record count.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["totals"]["train"]["records"] != 423718:
    sys.exit(f"Expected 423,718 train records, found {manifest['totals']['train']['records']:,}; "
             "recompute TRAIN_ITERS")
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
