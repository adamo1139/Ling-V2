#!/usr/bin/env bash
set -euo pipefail

# Run 11: dane polskie-sprawy-v6, parametry run 10 (start z merge'a -7, 16384,
# pakowanie, zapis synchroniczny, warmup 20), LR z tej samej reguly pierwiastkowej.
#
# UWAGA, DWIE WARTOSCI DO UZUPELNIENIA po zbudowaniu cache'u. Skrypt
# build_polskie_sprawy_v3_cache.sh v6 wypisuje je na koncu:
#   TRAIN_ITERS  - kubelki / GBS
#   LR           - 3e-4 * sqrt(tokeny_na_krok / 2,097,152)
# Dopoki stoja na 0, launcher odmawia startu.
#
# GBS: najlepsze zmierzone wyniki Poziomki szly na GBS 16 - run 10 dal 2,05
# (3 przebiegi, rozrzut 0,04), run 7 1,99 (3 przebiegi). Zmiatanie po partii
# w serii v13 dalo run6 gbs64 1,89, run8 gbs32 1,78, run9 gbs768 0,78, ale po
# jednym przebiegu kazdy, wiec trojki 16/32/64 nie dalo sie uszeregowac.
#
# Dane: cache v4-nc w formacie v13 (poziomka_v13_chat_template.jinja), budowany na
# riggu: bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v4-nc
# Po <think> nastepny token to \n w 107,709 rekordach z rozumowaniem i < w 425,024 bez.
# Eksport HF musi dostac chat_template.jinja z tego cache.
#
# PP8 przy micro-batch 1 daje tylko 16 mikrobatchy na krok, wiec banka
# pipeline'u to 7/23 ~ 30%. 1525 krokow to ~1,8x krokow run 7, kazdy z ~20% wiecej
# tokenow, wiec trening potrwa ~2,2x dluzej niz run 7.
# Launch with: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run10.sh
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
export SFT_DATA="${WORK_DIR}/poziomka-sft-cache-polskie-sprawy-v6-v13-16384"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run11_polskie_sprawy_v6_v13_16384_gbs16"
export RESUME=0

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
# Merge pochodzi z wag trenowanych na 640000 (run 3/4); 16384 wymaga >= 3,1e5.
export ROTARY_BASE=640000
export PACKING=1
export GLOBAL_BATCH_SIZE=16
# LR z reguly pierwiastkowej dla Adama, tej samej co run 7 i run 10:
#   LR = 3e-4 * sqrt(tokeny_SFT_na_krok / 2,097,152)
# gdzie 2,097,152 = 8192 x GBS 256 z pretreningu. Dla v4-nc przy GBS 16 wyszlo
# 1,02e-4 (242,8k tokenow/krok), dla v3 przy GBS 16 - 9,33e-5 (203k).
# WARTOSC Z BUILDU CACHE'U:
export LR=0
# Warmup 20: momenty Adama startuja od zera (--no-load-optim), a w run 5 drugi
# krok mial grad norm 4,55 przy 1,26 w pierwszym.
export WARMUP_ITERS=20
# Co 200 krokow: 8 zapisow (~64 GB).
export SAVE_INTERVAL=200
export EVAL_INTERVAL=150
# Walidacja v4-nc to 235 spakowanych sekwencji; 14 x 16 = 224.
export EVAL_ITERS=14
export DATALOADER_WORKERS=2
# Zapis synchroniczny: async save run 5 i run 6 (pierwsza proba) zerowal 35/128 wartosci
# q_layernorm w pierwszej warstwie czesci etapow pipeline'u; model w pamieci byl zdrowy.
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

# Jedna epoka: kubelki po spakowaniu / GBS. WARTOSC Z BUILDU CACHE'U:
export TRAIN_ITERS=0

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run11_polskie_sprawy_v6_v13_16384_packed_gbs16"
export WANDB_MODE="online"

if [[ "${LR}" == "0" || "${TRAIN_ITERS}" == "0" ]]; then
    echo "Uzupelnij LR i TRAIN_ITERS w tym skrypcie. Wypisuje je:" >&2
    echo "  bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v6" >&2
    exit 1
fi

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it on this machine first:" >&2
    echo "  bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v6" >&2
    exit 1; }

# The cache must match the training length and this run's record count.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
# Liczby rekordow dla v6 nie znamy z gory - raportujemy, zeby bylo w logu runu.
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
