#!/usr/bin/env bash
# Sweep --seq-length to find what this machine actually fits, before committing to
# a cache or a run. Memory here depends only on seq_length: MMapSFTDataset pads
# every sample to the full window, so a tiny synthetic cache measures the real peak.
#
# Builds throwaway caches, trains 3 iterations per length, records peak reserved
# memory or the OOM, cleans up after itself. Touches no real cache or checkpoint.
#
#   LOAD_CHECKPOINT=/path/to/poziomka-linear-8-9-10-11-sqrt-dcp \
#   bash Ling-V2/examples/sft/megatron/smoke_poziomka_seqlen.sh 8192 12288 16384 24576 32768
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

: "${LOAD_CHECKPOINT:?Set LOAD_CHECKPOINT to the merged base DCP}"
: "${TOKENIZER:=$(cd -- "${REPO_DIR}/.." && pwd)/poziomka-linear-8-9-10-11-sqrt}"
SCRATCH="${SCRATCH:-/tmp/poziomka_seqlen_smoke}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"   # >= pipeline depth, so 1F1B is realistic
ITERS="${ITERS:-3}"
KEEP="${KEEP:-0}"
LENGTHS=("$@")
[[ ${#LENGTHS[@]} -gt 0 ]] || LENGTHS=(8192 12288 16384 24576 32768)

[[ -d "${TOKENIZER}" ]] || { echo "No tokenizer at ${TOKENIZER}; set TOKENIZER" >&2; exit 1; }
[[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || {
    echo "No DCP tracker in ${LOAD_CHECKPOINT}" >&2; exit 1; }

mkdir -p "${SCRATCH}"
CORPUS="${SCRATCH}/corpus"
if [[ ! -d "${CORPUS}" ]]; then
    mkdir -p "${CORPUS}/train" "${CORPUS}/validation"
    # Content is irrelevant to peak memory; only the padded window size matters.
    python3 - "${CORPUS}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def record(i):
    return {"record_id": f"smoke{i}",
            "messages": [{"role": "user", "content": "pytanie " * 200},
                         {"role": "assistant", "content": "odpowiedz " * 200,
                          "reasoning_content": "mysle " * 400}]}
for split, count in (("train", 128), ("validation", 16)):
    with (root / split / "shard.jsonl").open("w", encoding="utf-8") as out:
        for i in range(count):
            out.write(json.dumps(record(i), ensure_ascii=False) + "\n")
PY
    echo "Synthetic corpus: ${CORPUS}"
fi

RESULTS=()
for length in "${LENGTHS[@]}"; do
    echo
    echo "==================== seq-length ${length} ===================="
    cache="${SCRATCH}/cache_${length}"
    save="${SCRATCH}/save_${length}"
    log="${SCRATCH}/train_${length}.log"
    rm -rf "${cache}" "${save}"

    if ! python3 "${SCRIPT_DIR}/prepare_poziomka_sft.py" \
            --input "${CORPUS}" --tokenizer "${TOKENIZER}" --output "${cache}" \
            --seq-length "${length}" --workers 2 --long-policy truncate \
            > "${SCRATCH}/prepare_${length}.log" 2>&1; then
        echo "PREPARE FAILED (see ${SCRATCH}/prepare_${length}.log)"
        RESULTS+=("${length}|prepare failed|-")
        continue
    fi

    # Later flags win in argparse, so this overrides the 8192 in poziomka_model_args.sh.
    # EVAL_ITERS=1 mirrors the known-good run 2 config: vary only seq_length, so a
    # failure here is about length and not about some other setting we introduced.
    SFT_DATA="${cache}" LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" SAVE_CHECKPOINT="${save}" \
    SEQ_LENGTH="${length}" TRAIN_ITERS="${ITERS}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    EVAL_ITERS=1 EVAL_INTERVAL=1000000 SAVE_INTERVAL=1000000 RESUME=0 \
        bash "${SCRIPT_DIR}/run_poziomka.sh" \
            --max-position-embeddings "${length}" \
        > "${log}" 2>&1
    status=$?

    peak=$(grep -o 'max reserved: [0-9.]*' "${log}" | awk '{print $3}' | sort -gr | head -1)
    if [[ ${status} -eq 0 && -n "${peak}" ]]; then
        echo "OK   peak reserved ${peak} MiB"
        RESULTS+=("${length}|OK|${peak}")
    elif grep -qi "out of memory\|CUDA out of memory" "${log}"; then
        echo "OOM  (${log})"
        RESULTS+=("${length}|OOM|${peak:--}")
    else
        echo "FAILED exit ${status} (${log}) -- last lines:"
        # A non-OOM failure is a setup problem, not an answer about this length.
        # Show it immediately instead of repeating an identical failure five times.
        grep -iE "error|Error|Traceback|assert|raise |Exception" "${log}" | tail -15 | sed 's/^/    /'
        echo "    ---"
        tail -20 "${log}" | sed 's/^/    /'
        RESULTS+=("${length}|failed exit ${status}|${peak:--}")
        if [[ "${STOP_ON_FAILURE:-1}" == 1 ]]; then
            echo
            echo "Stopping: this is a setup failure, not a memory limit."
            echo "Fix it, or re-run with STOP_ON_FAILURE=0 to sweep anyway."
            break
        fi
    fi
    [[ "${KEEP}" == 1 ]] || rm -rf "${cache}" "${save}"
done

echo
printf '%-12s %-22s %s\n' "seq-length" "result" "peak reserved (MiB)"
printf '%-12s %-22s %s\n' "----------" "------" "-------------------"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r length result peak <<< "${row}"
    printf '%-12s %-22s %s\n' "${length}" "${result}" "${peak}"
done
echo
echo "Logs in ${SCRATCH}. Peak is the max across ranks, reported after iteration 1."
echo "Re-run with KEEP=1 to retain caches, or GLOBAL_BATCH_SIZE=768 to match run 2."
