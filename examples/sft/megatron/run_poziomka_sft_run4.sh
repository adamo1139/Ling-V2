#!/usr/bin/env bash
set -euo pipefail

# Run 4: v12 SFT (loop-cleaned corpus) at 16384, packed. WZNOWIENIE z kroku 100
# do konca jednej epoki (498 krokow); pierwotnie byla to sonda na 100 krokow.
# Launch with: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run4.sh
# Edit run settings here; no caller environment variables are needed.
#
# Differences from run 3, all deliberate:
#   v12 corpus             -- reasoning stripped from 29,101 looped messages and
#                             22,095 records with looped content removed entirely
#                             (see detect_reasoning_loops.py /
#                             strip_looped_reasoning.py). Run 3 trained on the
#                             same corpus WITH the loops; its model degenerated
#                             into repetition at inference while the loss curve
#                             looked healthy. Run 4 re-runs the same setup on
#                             clean data to confirm the data was the cause.
#   498 steps, not 535     -- jedna epoka v12. Mniej niz 535 z run 3, bo
#                             czyszczenie petli zabralo czesc korpusu.
#   EVAL_INTERVAL 20       -- eval every 20 steps for a fine-grained loss curve;
#                             run 3 evaluated every 100.
#
# Everything else is run 3 verbatim: same starting weights (run 2 @ iter 1718),
# same 16384 window, same 640000 RoPE base, same packing, same LR and batch.
# To test un-teaching instead (continue FROM run 3's weights on clean data),
# point LOAD_CHECKPOINT at the run 3 output and set RESUME=0.
#
# RESUME=1 wymaga, zeby LOAD_CHECKPOINT i SAVE_CHECKPOINT wskazywaly ten sam
# katalog - run_poziomka.sh to sprawdza. Zapisy sa tylko wagowe, wiec momenty
# Adama startuja od zera; przy stalym LR bez rozgrzewki nie ma pozycji
# w harmonogramie do stracenia, ale pierwsze kroki po wznowieniu moga miec
# inna dynamike niz ciagly trening.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

export MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
export SFT_DATA="${WORK_DIR}/poziomka-sft-cache-v12-16384-truncate"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run4_v12_16384"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run4_v12_16384"
export RESUME=1

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384   # poziomka_model_args.sh defaults to 8192
# Same 640000 base as run 3: the weights already adapted to it in run 3's
# continuation from run 2, and 16384 needs at least 3.1e5 (Xu et al. Table 2).
export ROTARY_BASE=640000
export PACKING=1
export GLOBAL_BATCH_SIZE=768
export LR=3e-4
export WARMUP_ITERS=0
export SAVE_INTERVAL=100
export EVAL_INTERVAL=20
export EVAL_ITERS=1
export DATALOADER_WORKERS=2
export ROUTER_BIAS_UPDATE_RATE=0

# Wznowienie z kroku 100 do konca jednej epoki. 498 = 382,473 kubelkow po
# spakowaniu / 768 na krok, policzone dokladnie kodem treningu:
#   MMapSFTDataset(manifest, "train", pack=True) -> len(d.bins)
# Nie da sie tego wziac z manifestu: pakowanie to next-fit po potasowanej
# kolejnosci, a jego efektywnosc to 78,9%, wiec z 4,94 mld tokenow robi sie
# 382 tys. kubelkow zamiast 302 tys., ktore wyszlyby przy pakowaniu bez strat.
export TRAIN_ITERS=498

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-sft"
export WANDB_NAME="poziomka_sft_run4_v12_16384_packed_498steps"
export WANDB_MODE="online"

[[ -f "${SFT_DATA}/manifest.json" ]] || {
    echo "No cache at ${SFT_DATA}. Build it first:" >&2
    echo "  python3 ${SCRIPT_DIR}/prepare_poziomka_sft.py \\" >&2
    echo "    --input <v12-export> --tokenizer ${WORK_DIR}/poziomka-linear-8-9-10-11-sqrt \\" >&2
    echo "    --output ${SFT_DATA} --seq-length 16384 --long-policy truncate --workers 10" >&2
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