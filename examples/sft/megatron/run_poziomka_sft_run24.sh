#!/usr/bin/env bash
set -euo pipefail

# Run 24: jak run 23 (SFT na samym polskie-sprawy-v8 z jakosc >= 50, start z merge'a
# poziomka-instruct-2026-09-30-7, szablon v13, 16384, pakowanie, warmup 20, stale LR z reguly
# pierwiastkowej, jedna epoka), ale GBS 16 i LR 1,04e-4. Seria 24-27 sprawdza wplyw GBS/LR przy
# tych samych danych; puszczana po kolei przez run_poziomka_sft_runs_24_to_27.sh.
# Zapis i walidacja co 1/3 treningu (3 checkpointy). Przy PP8 i DP1 mikrobatchy = GBS, banka
# pipeline'u 30%.
#
# Cache (build v8-jakosc50): 533,922 rozmow train (39,207 ON, 494,715 OFF), 27,458 kubelkow,
# pakowanie 96,2%, 15,776 prawdziwych tokenow na sekwencje, 5,330 rozmow walidacji (272 sekwencje).
# Eksport HF musi dostac chat_template.jinja z cache (tools/dcp_do_hf.sh robi to sam).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
# Cache na nvme (na / brakuje miejsca); build:
#   WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v8-jakosc50
export SFT_DATA="/media/nvme_2tb/maked/poziomka_train/poziomka-sft-cache-polskie-sprawy-v8-jakosc50-v13-16384"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run24_polskie_sprawy_v8_jakosc50_v13_16384_gbs16"
# Wznowienie: RESUME=1 bash run_poziomka_sft_run24.sh -- wczytuje ostatni checkpoint tego runu
# (latest_checkpointed_iteration.txt) z licznikami iteracji i probek. Optymalizator nie jest
# zapisywany (--no-save-optim), wiec momenty Adama startuja od zera przy pelnym LR; warmup
# liczy sie od kroku 0, wiec po wznowieniu go nie ma.
export RESUME="${RESUME:-0}"
if [[ "${RESUME}" == 1 ]]; then
    export LOAD_CHECKPOINT="${SAVE_CHECKPOINT}"
else
    export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp"
fi

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
# Merge pochodzi z wag trenowanych na 640000 (run 3/4); 16384 wymaga >= 3,1e5.
export ROTARY_BASE=640000
export PACKING=1
export GLOBAL_BATCH_SIZE=16
# Regula pierwiastkowa: GBS 16 x 15,776 = 252,4k tokenow na krok = 0,120 pretreningu -> 3e-4 * sqrt(0,120) = 1,04e-4 (jak run 21/22).
export LR=1.04e-4
export WARMUP_ITERS=20
# Jedna epoka: 27,458 / 16 -> 1717.
export TRAIN_ITERS=1717
# Cala walidacja: 272 sekwencje -> 17 x 16.
export EVAL_ITERS=17
export EXPECTED_TRAIN_RECORDS=533922
# Zapis i walidacja co 1/3 treningu: 573, 1146 plus koncowy 1717 (~24 GB).
export SAVE_INTERVAL=573
export EVAL_INTERVAL=573
export DATALOADER_WORKERS=2
# Zapis synchroniczny: async save run 5 i run 6 (pierwsza proba) zerowal 35/128 wartosci
# q_layernorm w pierwszej warstwie czesci etapow pipeline'u; model w pamieci byl zdrowy.
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run24_polskie_sprawy_v8_jakosc50_v13_16384_packed_gbs16_lr${LR}_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

if [[ "${LR}" == "0" || "${TRAIN_ITERS}" == "0" || "${EVAL_ITERS}" == "0" || "${EXPECTED_TRAIN_RECORDS}" == "0" ]]; then
    echo "Uzupelnij LR, TRAIN_ITERS, EVAL_ITERS i EXPECTED_TRAIN_RECORDS w tym skrypcie. Wypisuje je:" >&2
    echo "  WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v8-jakosc50" >&2
    exit 1
fi

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it on this machine first:" >&2
    echo "  WORK_DIR=/media/nvme_2tb/maked/poziomka_train bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v8-jakosc50" >&2
    exit 1; }

# The cache must match the training length and this run's record count.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" "${EXPECTED_TRAIN_RECORDS}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["totals"]["train"]["records"] != int(sys.argv[3]):
    sys.exit(f"Expected {int(sys.argv[3]):,} train records, found "
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

# --log-throughput: TFLOP/s na GPU w logu i W&B (jak w runach KD 17-20).
exec bash "${SCRIPT_DIR}/run_poziomka.sh" --log-throughput \
    --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" "$@"
