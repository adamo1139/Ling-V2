#!/usr/bin/env bash
set -euo pipefail

# Run 15: SFT (zwykle CE) na polskie-sprawy-v8, przepis run 11 (v6): start z merge'a
# poziomka-instruct-2026-09-30-7, 16384, pakowanie, GBS 16, warmup 20, stale LR z reguly
# pierwiastkowej. (Run 14 to destylacja KD w repo poziomka-destylacja, nie tutaj.)
#
# v8 to te same 1,325,738 rozmow co v6 (te same id, wiec ten sam podzial train/validation
# i te same 107,709 ON / 1,204,847 OFF w train), ale rozmowy OFF sa teraz wieloturowe:
# 1,161,307 z trzema turami uzytkownika, 49,678 z piecioma (kolejne tury napisal instruktor
# Bielik-PL-Minitron-7B po nauka-lotnictwa-2, W8A8). 1 217M tokenow APT4 tresci, ~2x v6.
#
# UWAGA, TRZY WARTOSCI DO UZUPELNIENIA po zbudowaniu cache'u. Skrypt
#   WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v8
# wypisuje na koncu kubelki, tabele GBS -> TRAIN_ITERS / LR i liczbe sekwencji walidacji:
#   TRAIN_ITERS - kubelki / 16 (jedna epoka)
#   LR          - 3e-4 * sqrt(tokeny_na_krok / 2,097,152) dla GBS 16
#   EVAL_ITERS  - sekwencje walidacji / 16 (cala walidacja)
# Dopoki stoja na 0, launcher odmawia startu. Wpisz tez BINS do buildu (zamiast 0).
#
# Start z merge'a poziomka-instruct-2026-09-30-7 (wagi po run 3/4, RoPE 640000), jak
# run 10-13: /media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp
# Eksport HF musi dostac chat_template.jinja z cache (tools/dcp_do_hf.sh robi to sam).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
# Cache na nvme (na / brakuje miejsca); build:
#   WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v8
export SFT_DATA="/media/nvme_2tb/maked/poziomka_train/poziomka-sft-cache-polskie-sprawy-v8-v13-16384"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run15_polskie_sprawy_v8_v13_16384_gbs16"
export RESUME=0

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
# Merge pochodzi z wag trenowanych na 640000 (run 3/4); 16384 wymaga >= 3,1e5.
export ROTARY_BASE=640000
export PACKING=1
# GBS 16 jak run 10/11 (najlepsze zmierzone wyniki Poziomki).
export GLOBAL_BATCH_SIZE=16
# Z buildu v8 (tabela LR przy GBS 16).
export LR=0
export WARMUP_ITERS=20
# Z buildu v8: kubelki / 16.
export TRAIN_ITERS=0
# Z buildu v8: sekwencje walidacji / 16.
export EVAL_ITERS=0
# Zapisy i walidacja: ~co 1/8 epoki i ~1/16 epoki; przelicz po wpisaniu TRAIN_ITERS.
export SAVE_INTERVAL=$(( TRAIN_ITERS > 0 ? (TRAIN_ITERS + 7) / 8 : 1 ))
export EVAL_INTERVAL=$(( TRAIN_ITERS > 0 ? (TRAIN_ITERS + 15) / 16 : 1 ))
export DATALOADER_WORKERS=2
# Zapis synchroniczny: async save run 5 i run 6 (pierwsza proba) zerowal 35/128 wartosci
# q_layernorm w pierwszej warstwie czesci etapow pipeline'u; model w pamieci byl zdrowy.
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run15_polskie_sprawy_v8_v13_16384_packed_gbs16_lr${LR}_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

if [[ "${LR}" == "0" || "${TRAIN_ITERS}" == "0" || "${EVAL_ITERS}" == "0" ]]; then
    echo "Uzupelnij LR, TRAIN_ITERS i EVAL_ITERS w tym skrypcie. Wypisuje je:" >&2
    echo "  WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v8" >&2
    exit 1
fi

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it on this machine first:" >&2
    echo "  WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v8" >&2
    exit 1; }

# The cache must match the training length and this run's record count.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["totals"]["train"]["records"] != 1312556:
    sys.exit(f"Expected 1,312,556 train records (podzial jak v6), found "
             f"{manifest['totals']['train']['records']:,}; recompute TRAIN_ITERS")
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
