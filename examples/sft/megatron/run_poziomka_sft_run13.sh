#!/usr/bin/env bash
set -euo pipefail

# Run 13: UCZEN do OPD — Poziomka (merge poziomka-instruct-2026-09-30-7) trenowana na
# cpral/nauka-lotnictwa-2 tym samym przepisem co instruktor Bielik-PL-Minitron-7B
# (polskie-dane-instruktazowe, nauczyciel_lotnictwa_1/megatron/run_bielik_pl_minitron_7b_sft.sh),
# zeby oba modele byly jak najblizej siebie przed on-policy distillation.
#
# Zgodnosc z instruktorem:
# - TEN SAM cache co instruktor: bielik-sft-cache-nauka-lotnictwa-2-v13-6144
#   (build_nauka_lotnictwa_1_cache.sh, NAME=nauka-lotnictwa-2, SEQ_LENGTH=6144,
#   UPSAMPLE_MULTITURN=3, --long-policy drop). tokenizer.json Bielika-PL i Poziomki sa
#   identyczne (sprawdzone pole po polu), wiec tokeny, kubelki pakowania, odrzucone
#   rozmowy (1,958 za dlugich) i walidacja sa te same.
# - Loss na wszystkich tokenach poza BOS (prompt systemowy glimmer8, pytania, odpowiedzi),
#   szablon v13, 555,423 rozmowy train (w tym wieloturowe x3).
# - GBS 64, micro-batch 1, LR 3e-5 stale, warmup 20, 1 epoka = 86,378 kubelkow / 64 = 1350
#   krokow — jak instruktor.
# Roznice wynikajace z modelu: architektura Poziomki (MoE, PP8 z poziomka_model_args.sh),
# RoPE 640000 jak w merge'u, recompute/zapisy jak w run 10-12.
#
# Start z merge'a poziomka-instruct-2026-09-30-7 (wagi po run 3/4, RoPE 640000), jak run 10-12.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
# Cache instruktora (Bielik-PL-Minitron-7B), zbudowany na riggu przez polskie-dane-instruktazowe.
export SFT_DATA="${WORK_DIR}/bielik-sft-cache-nauka-lotnictwa-2-v13-6144"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run13_nauka_lotnictwa_2_uczen_v13_6144_gbs64"
export RESUME=0

# Jak instruktor: 6144 (cache instruktora jest na 6144).
export SEQ_LENGTH=6144
export MAX_POSITION_EMBEDDINGS=8192   # poziomka_model_args.sh default; >= SEQ_LENGTH
# Merge pochodzi z wag trenowanych na 640000 (run 3/4); 16384 wymaga >= 3,1e5.
export ROTARY_BASE=640000
export PACKING=1
# Jak instruktor: GBS 64 (64 mikrobatche na krok, banka pipeline'u 7/71 ~ 10%).
export GLOBAL_BATCH_SIZE=64
# Jak instruktor: LR 3e-5 stale (min-lr = lr w run_poziomka.sh), zamiast reguly
# pierwiastkowej z run 7-12 (dla tego cache przy GBS 64 dalaby ~1,6e-4).
export LR=3e-5
# Warmup 20: momenty Adama startuja od zera (--no-load-optim).
export WARMUP_ITERS=20
# Co 270 krokow: zapisy na 270, 540, 810, 1080 plus koncowy na 1350.
export SAVE_INTERVAL=270
# Walidacja co 135 krokow (jak instruktor: co ~1/10 epoki).
export EVAL_INTERVAL=135
# Jak instruktor: 9 x 64 = 576 spakowanych sekwencji walidacyjnych.
export EVAL_ITERS=9
export DATALOADER_WORKERS=2
# Zapis synchroniczny: async save run 5 i run 6 (pierwsza proba) zerowal 35/128 wartosci
# q_layernorm w pierwszej warstwie czesci etapow pipeline'u; model w pamieci byl zdrowy.
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

# Jedna epoka: 86,378 kubelkow / 64 na krok = 1349,7 -> 1350 (tyle samo co instruktor).
export TRAIN_ITERS=1350

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run13_nauka_lotnictwa_2_uczen_v13_6144_packed_gbs64_lr${LR}_const_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

if [[ "${LR}" == "0" || "${TRAIN_ITERS}" == "0" || "${EVAL_ITERS}" == "0" ]]; then
    echo "Uzupelnij LR, TRAIN_ITERS i EVAL_ITERS w tym skrypcie." >&2
    exit 1
fi

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it on this machine first:" >&2
    echo "  polskie-dane-instruktazowe/nauczyciel_lotnictwa_1/megatron/build_nauka_lotnictwa_1_cache.sh" >&2
    echo "  (NAME=nauka-lotnictwa-2 SEQ_LENGTH=6144 UPSAMPLE_MULTITURN=3), jak dla instruktora" >&2
    exit 1; }

# The cache must match the training length and this run's record count.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["totals"]["train"]["records"] != 555423:
    sys.exit(f"Expected 555,423 train records (cache instruktora), found "
             f"{manifest['totals']['train']['records']:,}; recompute TRAIN_ITERS")
if manifest["loss_roles"] != ["all"]:
    sys.exit(f"Expected loss on all tokens like the teacher, got {manifest['loss_roles']}")
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
