#!/usr/bin/env bash
set -euo pipefail

# Sprawdza q/k layernorm checkpointu DCP w trakcie treningu, nie dotykajac oryginalu:
# kopiuje go (najnizszy priorytet I/O i CPU) do katalogu tymczasowego, sprawdza kopie
# tools/check_dcp_qk_norm.py i zawsze ja usuwa (takze po bledzie i Ctrl+C).
#
#   bash Ling-V2/tools/check_checkpoint_safe.sh <save_dir>/iter_0000025 [tmp_root]
#
# Odmawia, gdy:
# - zapis jeszcze trwa: numer iteracji > latest_checkpointed_iteration.txt (katalog
#   iter_* pojawia sie, zanim zapis async sie skonczy),
# - jakis proces ma otwarty plik checkpointu (lsof),
# - zrodlo zmienilo sie w trakcie kopiowania (rozmiary i czasy modyfikacji).
# Kod wyjscia: 0 czysty, 1 zera w q/k layernorm, 2 checkpoint niegotowy lub zajety.

CHECKPOINT="$(realpath "${1:?Usage: $0 <iter_dir> [tmp_root]}")"
TMP_ROOT="${2:-/media/nvme_2tb/maked/tmp_qk_check}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}/Megatron-LM-core_v0.13.0${PYTHONPATH:+:${PYTHONPATH}}"

[[ -d "${CHECKPOINT}" && "${CHECKPOINT##*/}" == iter_* ]] || { echo "Not a checkpoint dir: ${CHECKPOINT}" >&2; exit 2; }
tracker_file="$(dirname "${CHECKPOINT}")/latest_checkpointed_iteration.txt"
tracker=$({ tr -dc '0-9' < "${tracker_file}"; } 2>/dev/null || true)
iteration=$((10#${CHECKPOINT##*iter_}))
if [[ -z "${tracker}" ]] || (( iteration > tracker )); then
    echo "Save not finalized yet: iter ${iteration}, tracker ${tracker:-missing}" >&2; exit 2
fi

for tool in lsof ionice; do
    command -v "${tool}" >/dev/null || { echo "Missing ${tool} (sudo apt install ${tool/ionice/util-linux})" >&2; exit 2; }
done
open_files() { lsof -t +D "${CHECKPOINT}" 2>/dev/null || true; }
snapshot() { find "${CHECKPOINT}" -type f -printf '%s %T@ %P\n' | sort; }
if [[ -n "$(open_files)" ]]; then
    echo "Checkpoint files are open by a process (pids: $(open_files | tr '\n' ' ')); try later" >&2; exit 2
fi

needed_kb=$(du -sk "${CHECKPOINT}" | cut -f1)
mkdir -p "${TMP_ROOT}"
free_kb=$(df -k --output=avail "${TMP_ROOT}" | tail -1 | tr -dc '0-9')
(( free_kb > needed_kb + 1048576 )) || { echo "Not enough space in ${TMP_ROOT}" >&2; exit 2; }

TMP_DIR="$(mktemp -d "${TMP_ROOT}/qkcheck_XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT INT TERM

before="$(snapshot)"
echo "Copying ${CHECKPOINT##*/} ($((needed_kb / 1048576)) GB) to ${TMP_DIR} at idle I/O priority..."
ionice -c3 nice -n19 cp -a "${CHECKPOINT}" "${TMP_DIR}/"
if [[ "$(snapshot)" != "${before}" ]]; then
    echo "Checkpoint changed while copying; not checking a torn copy" >&2; exit 2
fi

set +e
# torch warns that it loads in a single process; real errors still reach stderr.
nice -n19 python3 "${SCRIPT_DIR}/check_dcp_qk_norm.py" "${TMP_DIR}/${CHECKPOINT##*/}" \
    2> >(grep -v -e 'UserWarning' -e 'warnings.warn(' >&2) | sed "s|${TMP_DIR}/||"
status=${PIPESTATUS[0]}
set -e
rm -rf "${TMP_DIR}"
echo "Removed temporary copy ${TMP_DIR}"
exit "${status}"
