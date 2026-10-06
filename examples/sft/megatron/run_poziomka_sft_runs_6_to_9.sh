#!/usr/bin/env bash
set -uo pipefail

# Puszcza run 6, 7, 8 i 9 po kolei (na noc). Dla kazdego runu:
#   trening -> 3 min ciszy -> sprawdzenie -> 3 min przerwy -> nastepny run.
# Run in tmux on rigga: bash Ling-V2/examples/sft/megatron/run_poziomka_sft_runs_6_to_9.sh
#
# W trakcie treningu nic nie jest sprawdzane: dodatkowe obciazenie w czasie treningu
# potrafi go rozwalic. Sprawdzenie po treningu:
# - run doszedl do konca: latest_checkpointed_iteration.txt == TRAIN_ITERS. Kod wyjscia
#   nic nie mowi - Megatron konczy kazdy run kodem 1 przez blad W&B przy ostatnim zapisie;
# - kazdy zapisany checkpoint ma niezerowe q/k layernorm (tools/check_dcp_qk_norm.py;
#   run 5 mial zera w q_layernorm pierwszej warstwy 7 z 8 etapow pipeline'u).
# Nastepny run startuje tylko, gdy oba warunki sa spelnione; inaczej lancuch staje.
#
# Podsumowanie: ${LOG_DIR}/summary.txt; logi treningow i sprawdzen obok.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

RUNS=(6 7 8 9)
QUIET_SECONDS="${QUIET_SECONDS:-180}"   # cisza po treningu, przed sprawdzeniem
PAUSE_SECONDS="${PAUSE_SECONDS:-180}"   # przerwa po sprawdzeniu, przed nastepnym runem
LOG_DIR="${WORK_DIR}/logs/poziomka_runs_6_to_9_$(date +%Y%m%d_%H%M%S)"
CHECKER="${REPO_DIR}/tools/check_dcp_qk_norm.py"
export PYTHONPATH="${REPO_DIR}/Megatron-LM-core_v0.13.0${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/summary.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "${SUMMARY}"; }
stop() { say "STOP: $*"; say "Remaining runs not started."; exit 1; }

launcher_value() {  # the value a launcher exports for a variable, e.g. SAVE_CHECKPOINT
    sed -n "s/^export $2=\"\{0,1\}\([^\"]*\)\"\{0,1\}.*/\1/p" "$1" | head -1
}

for run in "${RUNS[@]}"; do
    [[ -f "${SCRIPT_DIR}/run_poziomka_sft_run${run}.sh" ]] || { echo "Missing launcher for run ${run}" >&2; exit 1; }
done
[[ -f "${CHECKER}" ]] || { echo "Missing ${CHECKER}" >&2; exit 1; }

say "Runs ${RUNS[*]}, logs in ${LOG_DIR}"
for index in "${!RUNS[@]}"; do
    run="${RUNS[$index]}"
    launcher="${SCRIPT_DIR}/run_poziomka_sft_run${run}.sh"
    save_dir=$(launcher_value "${launcher}" SAVE_CHECKPOINT)
    train_iters=$(launcher_value "${launcher}" TRAIN_ITERS)

    say "run ${run}: training (${train_iters} iters -> ${save_dir}), log ${LOG_DIR}/run${run}.log"
    # Trening widoczny od razu w konsoli i zapisany do pliku; kod wyjscia z treningu, nie z tee.
    bash "${launcher}" 2>&1 | tee "${LOG_DIR}/run${run}.log"
    exit_code=${PIPESTATUS[0]}
    say "run ${run}: training process exited with code ${exit_code}; quiet ${QUIET_SECONDS}s"
    sleep "${QUIET_SECONDS}"

    final=$({ tr -dc '0-9' < "${save_dir}/latest_checkpointed_iteration.txt"; } 2>/dev/null || true)
    if [[ "${final}" != "${train_iters}" ]]; then
        say "run ${run}: last checkpoint iter ${final:-none} of ${train_iters}; last log lines:"
        tail -5 "${LOG_DIR}/run${run}.log" | sed 's/^/    /' | tee -a "${SUMMARY}"
        stop "run ${run} did not finish"
    fi
    say "run ${run}: reached iter ${final} (exit code 1 is the known W&B finalize error)"

    checkpoints=()
    for iter_dir in "${save_dir}"/iter_*; do
        [[ -d "${iter_dir}" ]] && checkpoints+=("${iter_dir}")
    done
    say "run ${run}: checking q/k layernorm in ${#checkpoints[@]} checkpoints"
    python3 "${CHECKER}" "${checkpoints[@]}" > "${LOG_DIR}/run${run}_qk_norm.log" 2>&1
    checker_status=$?
    grep -E '^(OK|CORRUPT) ' "${LOG_DIR}/run${run}_qk_norm.log" | sed "s|${save_dir}/||; s/^/    /" | tee -a "${SUMMARY}"
    if (( checker_status != 0 )); then
        grep -qE '^(OK|CORRUPT) ' "${LOG_DIR}/run${run}_qk_norm.log" \
            || tail -5 "${LOG_DIR}/run${run}_qk_norm.log" | sed 's/^/    /' | tee -a "${SUMMARY}"
        stop "run ${run}: q/k layernorm check failed (${LOG_DIR}/run${run}_qk_norm.log)"
    fi
    say "run ${run}: DONE, all checkpoints clean"

    if (( index < ${#RUNS[@]} - 1 )); then
        say "pause ${PAUSE_SECONDS}s before run ${RUNS[$((index + 1))]}"
        sleep "${PAUSE_SECONDS}"
    fi
done
say "All runs finished cleanly."
