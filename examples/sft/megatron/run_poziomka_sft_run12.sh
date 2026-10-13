#!/usr/bin/env bash
set -euo pipefail

# Run 12: polskie-sprawy-v6-reasoning — WYLACZNIE rozmowy z rozumowaniem z v6 (108,770:
# private 59,598, Mistral Small 4 31,688, Qwen-27B 13,511, GLM-4.5-Air 3,973). Pytanie
# eksperymentu: jak trening tylko na rozumowaniu wplywa na wynik wzgledem run 11 (caly v6).
# Parametry run 11 / run 10 (start z merge'a -7, 16384, pakowanie, GBS 16, zapis
# synchroniczny, warmup 20), LR z tej samej reguly pierwiastkowej.
#
# Dane: cache v6-reasoning w formacie v13, budowany na riggu:
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v6-reasoning
# 107,709 rozmow train (wszystkie ON) i 1,061 walidacji, nic nie uciete przy 16384.
#
# Zbior nie ma ani jednej odpowiedzi bez rozumowania (<think></think>): model po tym
# runie moze gorzej trzymac enable_thinking=false — porownywac z run 11 w trybie ON.
# Rozmowy sa dlugie (srednio ~2,2k tokenow APT4, max ~12k), wiec pakowanie powinno byc
# gestsze niz w v6, a krokow znacznie mniej.
#
# Start z merge'a poziomka-instruct-2026-09-30-7 (wagi po run 3/4, RoPE 640000), jak run 10/11.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
export SFT_DATA="${WORK_DIR}/poziomka-sft-cache-polskie-sprawy-v6-reasoning-v13-16384"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka-instruct-2026-09-30-7-dcp"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run12_polskie_sprawy_v6_reasoning_v13_16384_gbs16"
export RESUME=0

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
# Merge pochodzi z wag trenowanych na 640000 (run 3/4); 16384 wymaga >= 3,1e5.
export ROTARY_BASE=640000
export PACKING=1
export GLOBAL_BATCH_SIZE=16
# LR z reguly pierwiastkowej dla Adama, tej samej co run 7, 10 i 11:
#   LR = 3e-4 * sqrt(tokeny_SFT_na_krok / 2,097,152)
# v6-reasoning: 242,701,345 tokenow / 16,341 kubelkow = 14,852 prawdziwych tokenow na
# sekwencje (pakowanie 90,6%), przy GBS 16 to 237,6k tokenow na krok, czyli 0,113
# pretreningu -> 3e-4 * sqrt(0,113) = 1,01e-4 (run 11 na v6: 217,7k/krok, 9,67e-5).
export LR=1.01e-4
# Warmup 20: momenty Adama startuja od zera (--no-load-optim).
export WARMUP_ITERS=20
# Co 256 krokow: zapisy na 256, 512, 768 plus koncowy na 1022 (~4 zapisy, ~48 GB rezerwy).
export SAVE_INTERVAL=256
# Walidacja co 128 krokow: 7 pomiarow w trakcie (128 ... 896) plus koncowy.
export EVAL_INTERVAL=128
# Walidacja v6-reasoning: 1,061 rozmow = 156 spakowanych sekwencji; 9 x 16 = 144 (prawie calosc).
export EVAL_ITERS=9
export DATALOADER_WORKERS=2
# Zapis synchroniczny: async save run 5 i run 6 (pierwsza proba) zerowal 35/128 wartosci
# q_layernorm w pierwszej warstwie czesci etapow pipeline'u; model w pamieci byl zdrowy.
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

# Jedna epoka: 16,341 kubelkow / 16 na krok = 1021,3 -> 1022 (run 11: 2756 przy podobnej
# liczbie tokenow na krok, bo v6-reasoning ma ~40% tokenow v6).
export TRAIN_ITERS=1022

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run12_polskie_sprawy_v6_reasoning_v13_16384_packed_gbs16_lr${LR}_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

if [[ "${LR}" == "0" || "${TRAIN_ITERS}" == "0" || "${EVAL_ITERS}" == "0" ]]; then
    echo "Uzupelnij LR, TRAIN_ITERS i EVAL_ITERS w tym skrypcie. Wypisuje je:" >&2
    echo "  bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v6-reasoning" >&2
    exit 1
fi

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it on this machine first:" >&2
    echo "  bash ${SCRIPT_DIR}/build_polskie_sprawy_v3_cache.sh v6-reasoning" >&2
    exit 1; }

# The cache must match the training length and this run's record count.
python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["totals"]["train"]["records"] != 107709:
    sys.exit(f"Expected 107,709 train records, found {manifest['totals']['train']['records']:,}; "
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
