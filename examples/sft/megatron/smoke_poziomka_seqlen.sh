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
# Activation recomputation: 'full' (least memory, ~30% slower), 'selective'
# (attention only), 'none' (fastest, most memory). Space-separated to compare.
RECOMPUTE_MODES="${RECOMPUTE_MODES:-${RECOMPUTE:-full}}"
PROJECT_ITERS="${PROJECT_ITERS:-1718}"         # full-corpus run length, for the projection
LENGTHS=("$@")
[[ ${#LENGTHS[@]} -gt 0 ]] || LENGTHS=(8192 12288 16384 24576 32768)

[[ -d "${TOKENIZER}" ]] || { echo "No tokenizer at ${TOKENIZER}; set TOKENIZER" >&2; exit 1; }
[[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || {
    echo "No DCP tracker in ${LOAD_CHECKPOINT}" >&2; exit 1; }

CAPACITY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "Detected ${GPUS:-?} GPUs, ${CAPACITY:-?} MiB each. Peak below is per rank, worst rank wins."

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
    rm -rf "${cache}" "${save}"

    echo "preparing throwaway cache at ${length}..."
    if ! python3 "${SCRIPT_DIR}/prepare_poziomka_sft.py" \
            --input "${CORPUS}" --tokenizer "${TOKENIZER}" --output "${cache}" \
            --seq-length "${length}" --workers 2 --long-policy truncate \
            > "${SCRATCH}/prepare_${length}.log" 2>&1; then
        echo "PREPARE FAILED (see ${SCRATCH}/prepare_${length}.log)"
        RESULTS+=("${length}|-|prepare failed|-|-|-")
        continue
    fi

for mode in ${RECOMPUTE_MODES}; do
    log="${SCRATCH}/train_${length}_${mode}.log"
    rm -rf "${save}"
    echo "-- recompute=${mode}"

    # Later flags win in argparse, so this overrides the 8192 in poziomka_model_args.sh.
    # EVAL_ITERS=1 mirrors the known-good run 2 config: vary only seq_length, so a
    # failure here is about length and not about some other setting we introduced.
    # Full output goes to the log; per-iteration lines stream to the terminal so a
    # long ITERS run is watchable instead of silent. PROGRESS=0 for the old behaviour.
    echo "training ${ITERS} iterations at global batch ${GLOBAL_BATCH_SIZE} (full log: ${log})"
    if [[ "${PROGRESS:-1}" == 1 ]]; then
        SFT_DATA="${cache}" LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" SAVE_CHECKPOINT="${save}" \
        SEQ_LENGTH="${length}" TRAIN_ITERS="${ITERS}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
        EVAL_ITERS=1 EVAL_INTERVAL=1000000 SAVE_INTERVAL=1000000 RESUME=0 \
        RECOMPUTE="${mode}" \
            bash "${SCRIPT_DIR}/run_poziomka.sh" \
                --max-position-embeddings "${length}" 2>&1 \
            | tee "${log}" \
            | stdbuf -oL grep -E --line-buffered \
                'iteration +[0-9]+/|max reserved|out of memory|CUDA error|Traceback|Error:' \
            | stdbuf -oL sed 's/^/    /'
        status=${PIPESTATUS[0]}
    else
        SFT_DATA="${cache}" LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" SAVE_CHECKPOINT="${save}" \
        SEQ_LENGTH="${length}" TRAIN_ITERS="${ITERS}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
        EVAL_ITERS=1 EVAL_INTERVAL=1000000 SAVE_INTERVAL=1000000 RESUME=0 \
        RECOMPUTE="${mode}" \
            bash "${SCRIPT_DIR}/run_poziomka.sh" \
                --max-position-embeddings "${length}" \
            > "${log}" 2>&1
        status=$?
    fi

    # Every rank reports (report_memory guards on data-parallel rank, and DP=1 here).
    # Pipeline stages are not equally sized: the last stage carries the output head
    # and the cross-entropy logits, stage 0 the most in-flight 1F1B microbatches.
    per_rank=$(python3 - "${log}" <<'PY'
import re, sys
peaks = {}
for line in open(sys.argv[1], errors="replace"):
    found = re.search(r"\[Rank (\d+)\].*max reserved: ([0-9.]+)", line)
    if found:
        rank = int(found.group(1))
        peaks[rank] = max(peaks.get(rank, 0.0), float(found.group(2)))
if peaks:
    print(" ".join(f"{rank}:{value:.0f}" for rank, value in sorted(peaks.items())))
PY
)
    peak=$(grep -o 'max reserved: [0-9.]*' "${log}" | awk '{print $3}' | sort -gr | head -1)
    # Iteration 1 carries warmup and autotuning; the last one is the honest rate.
    secs=$(grep -o 'elapsed time per iteration (ms): [0-9.]*' "${log}" \
           | awk '{print $6/1000}' | tail -1)
    if [[ ${status} -eq 0 && -n "${peak}" ]]; then
        hottest=$(tr ' ' '\n' <<< "${per_rank}" | sort -t: -k2 -gr | head -1 | cut -d: -f1)
        echo "OK   peak reserved ${peak} MiB on rank ${hottest} of ${CAPACITY:-?} MiB"
        echo "     per rank: ${per_rank}"
        if [[ -n "${secs}" ]]; then
            echo "     ${secs} s/iteration -> ${PROJECT_ITERS} iters =" \
                 "$(awk -v s="${secs}" -v n="${PROJECT_ITERS}" 'BEGIN{printf "%.1f days", s*n/86400}')"
        fi
        RESULTS+=("${length}|${mode}|OK (rank ${hottest})|${peak}|${secs:--}|-")
    elif grep -qi "out of memory\|CUDA out of memory" "${log}"; then
        echo "OOM  (${log})"
        RESULTS+=("${length}|${mode}|OOM|${peak:--}|-|-")
    else
        echo "FAILED exit ${status} (${log}) -- last lines:"
        # A non-OOM failure is a setup problem, not an answer about this length.
        # Show it immediately instead of repeating an identical failure five times.
        grep -iE "error|Error|Traceback|assert|raise |Exception" "${log}" | tail -15 | sed 's/^/    /'
        echo "    ---"
        tail -20 "${log}" | sed 's/^/    /'
        RESULTS+=("${length}|${mode}|failed exit ${status}|${peak:--}|-|-")
        if [[ "${STOP_ON_FAILURE:-1}" == 1 ]]; then
            echo
            echo "Stopping: this is a setup failure, not a memory limit."
            echo "Fix it, or re-run with STOP_ON_FAILURE=0 to sweep anyway."
            break 2
        fi
    fi
    [[ "${KEEP}" == 1 ]] || rm -rf "${save}"
done
    [[ "${KEEP}" == 1 ]] || rm -rf "${cache}"
done

echo
fmt='%-11s %-10s %-18s %-12s %-10s %-10s %s\n'
# shellcheck disable=SC2059
printf "${fmt}" "seq-length" "recompute" "result" "peak (MiB)" "headroom" "s/iter" "${PROJECT_ITERS} iters"
printf "${fmt}" "----------" "---------" "------" "----------" "--------" "------" "------------"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r length mode result peak secs _ <<< "${row}"
    headroom="-"; projected="-"
    if [[ -n "${CAPACITY}" && "${peak}" != "-" ]]; then
        headroom=$(awk -v p="${peak}" -v c="${CAPACITY}" 'BEGIN{printf "%.0f%%", 100*p/c}')
    fi
    if [[ "${secs}" != "-" && -n "${secs}" ]]; then
        projected=$(awk -v s="${secs}" -v n="${PROJECT_ITERS}" 'BEGIN{printf "%.1f days", s*n/86400}')
    fi
    printf "${fmt}" "${length}" "${mode}" "${result}" "${peak}" "${headroom}" "${secs}" "${projected}"
done
echo
echo "Logs in ${SCRATCH}. Peak is the worst rank, reported after iteration 1."
echo "Treat anything above ~85% used as not viable: fragmentation and the longest"
echo "real batch will exceed this synthetic run."
echo "Re-run with KEEP=1 to retain caches, or GLOBAL_BATCH_SIZE=768 to match run 2."
