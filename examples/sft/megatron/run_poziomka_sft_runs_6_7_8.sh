#!/usr/bin/env bash
set -uo pipefail

# Puszcza run 6, 7 i 8 po kolei (na noc), z 3-minutowa przerwa miedzy nimi, i
# sprawdza q/k layernorm w kazdym zapisanym checkpoincie (tools/check_dcp_qk_norm.py).
# Run in tmux on rigga: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_runs_6_7_8.sh
#
# - Sprawdzany jest tylko checkpoint o numerze <= latest_checkpointed_iteration.txt:
#   katalog iter_* pojawia sie, zanim zapis async sie skonczy.
# - Run uznaje sie za udany, gdy tracker == TRAIN_ITERS. Kod wyjscia nic nie mowi:
#   Megatron konczy kazdy run kodem 1 przez blad W&B w finalizacji ostatniego zapisu.
# - Zera w q/k layernorm sa logowane, a trening leci dalej: model w pamieci jest zdrowy
#   (run 5: walidacja 0,958), a uszkodzone wektory da sie podmienic ze startowego merge'a.
#   STOP_ON_CORRUPTION=1 zatrzymuje biezacy trening i caly lancuch.
# - Run, ktory sie nie udal, nie zatrzymuje kolejnych.
#
# Podsumowanie: ${LOG_DIR}/summary.txt; logi treningow i sprawdzen obok.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

RUNS=(6 7 8)
PAUSE_SECONDS="${PAUSE_SECONDS:-180}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STOP_ON_CORRUPTION="${STOP_ON_CORRUPTION:-0}"
LOG_DIR="${WORK_DIR}/logs/poziomka_runs_6_7_8_$(date +%Y%m%d_%H%M%S)"
CHECKER="${REPO_DIR}/tools/check_dcp_qk_norm.py"
export PYTHONPATH="${REPO_DIR}/Megatron-LM-core_v0.13.0${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/summary.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "${SUMMARY}"; }

launcher_value() {  # the value a launcher exports for a variable, e.g. SAVE_CHECKPOINT
    sed -n "s/^export $2=\"\{0,1\}\([^\"]*\)\"\{0,1\}.*/\1/p" "$1" | head -1
}

# Check every finalized, not yet checked checkpoint of one run. Returns 1 on corruption.
check_saved() {
    local save_dir="$1" checked_file="$2" tracker iter_dir iter status=0
    [[ -f "${save_dir}/latest_checkpointed_iteration.txt" ]] || return 0
    tracker=$(tr -dc '0-9' < "${save_dir}/latest_checkpointed_iteration.txt")
    for iter_dir in "${save_dir}"/iter_*; do
        [[ -d "${iter_dir}" ]] || continue
        iter=$((10#${iter_dir##*iter_}))
        (( iter <= tracker )) || continue
        grep -qxF "${iter_dir}" "${checked_file}" 2>/dev/null && continue
        if python3 "${CHECKER}" "${iter_dir}" >> "${LOG_DIR}/qk_norm.log" 2>&1; then
            say "  qk_norm OK      ${iter_dir##*/}"
        else
            say "  qk_norm CORRUPT ${iter_dir##*/}  (details: ${LOG_DIR}/qk_norm.log)"
            status=1
        fi
        echo "${iter_dir}" >> "${checked_file}"
    done
    return ${status}
}

for launcher in "${RUNS[@]/#/${SCRIPT_DIR}/run_poziomka_sft_run}"; do
    launcher="${launcher}.sh"
    [[ -f "${launcher}" ]] || { echo "Missing ${launcher}" >&2; exit 1; }
done

say "Runs ${RUNS[*]}, logs in ${LOG_DIR}"
for index in "${!RUNS[@]}"; do
    run="${RUNS[$index]}"
    launcher="${SCRIPT_DIR}/run_poziomka_sft_run${run}.sh"
    save_dir=$(launcher_value "${launcher}" SAVE_CHECKPOINT)
    train_iters=$(launcher_value "${launcher}" TRAIN_ITERS)
    checked_file="${LOG_DIR}/run${run}.checked"
    : > "${checked_file}"

    say "run ${run}: start (${train_iters} iters -> ${save_dir})"
    # setsid: own process group, so STOP_ON_CORRUPTION can stop torchrun and all ranks.
    setsid bash "${launcher}" > "${LOG_DIR}/run${run}.log" 2>&1 &
    pid=$!
    corrupt=0
    while kill -0 "${pid}" 2>/dev/null; do
        sleep "${POLL_SECONDS}"
        if ! check_saved "${save_dir}" "${checked_file}"; then
            corrupt=1
            if [[ "${STOP_ON_CORRUPTION}" == 1 ]]; then
                say "run ${run}: STOP_ON_CORRUPTION=1, stopping training and the chain"
                kill -TERM -- "-${pid}" 2>/dev/null
                wait "${pid}" 2>/dev/null
                exit 1
            fi
        fi
    done
    wait "${pid}"; exit_code=$?
    check_saved "${save_dir}" "${checked_file}" || corrupt=1

    final=$({ tr -dc '0-9' < "${save_dir}/latest_checkpointed_iteration.txt"; } 2>/dev/null || true)
    if [[ "${final}" == "${train_iters}" ]]; then
        say "run ${run}: DONE, final checkpoint iter ${final} (exit code ${exit_code}; 1 is the known W&B finalize error)"
    else
        say "run ${run}: FAILED, last checkpoint iter ${final:-none} of ${train_iters}, exit code ${exit_code}"
        say "  last log lines:"; tail -5 "${LOG_DIR}/run${run}.log" | sed 's/^/    /' | tee -a "${SUMMARY}"
    fi
    (( corrupt )) && say "run ${run}: WARNING corrupted q/k layernorm in at least one checkpoint, see qk_norm.log"

    if (( index < ${#RUNS[@]} - 1 )); then
        say "pause ${PAUSE_SECONDS}s"
        sleep "${PAUSE_SECONDS}"
    fi
done
say "All runs finished."
