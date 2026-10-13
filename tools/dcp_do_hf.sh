#!/usr/bin/env bash
set -euo pipefail

# Konwersja checkpointow DCP do HF: sprawdzenie, konwersja, szablon czatu.
#
# Trzy kroki, bo kazdy z nich raz juz nas kosztowal pomiar:
#
# 1. SPRAWDZENIE q/k layernorm. Zapis run 5 wrocil z 35 ze 128 wartosci
#    q_layernorm ustawionymi na dokladnie 0 w pierwszej warstwie 7 z 8 etapow
#    pipeline'u, choc model w pamieci trenowal sie poprawnie. Model po takiej
#    konwersji mielil podslowa nazw rol (`ssi`, `stant`) przy temperaturze 0
#    i nie dalo sie go zmierzyc. RMSNorm nigdy nie jest dokladnie zerem po
#    treningu, wiec kazde zero to uszkodzenie.
#
# 2. KONWERSJA.
#
# 3. SZABLON CZATU z cache'u treningowego. Konwerter kopiuje tokenizer
#    z --override-tokenizer-path, a nie szablon, z ktorym model byl trenowany.
#    Bez tego kroku serwer renderuje prompt innym szablonem niz trening.
#
# Uzycie:
#   bash Ling-V2/tools/dcp_do_hf.sh \
#       /media/nvme_2tb/maked/poziomka_train/poziomka_sft_run11_polskie_sprawy_v6_v13_16384_gbs16 \
#       ~/projects/pretrain/poziomka-sft-cache-polskie-sprawy-v6-v13-16384 \
#       0000700 0001400 0002100 0002756

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MEGATRON="${MEGATRON_PATH:-${REPO_DIR}/Megatron-LM-core_v0.13.0}"
TOKENIZER="${TOKENIZER_PATH:-${REPO_DIR}/resource/tokenizer/apt4}"

(( $# >= 3 )) || { echo "Uzycie: $0 <katalog_checkpointow> <katalog_cache> <iteracja> [<iteracja> ...]" >&2; exit 1; }
SRC="$1"; CACHE="$2"; shift 2
TARGET="${SRC}_hf"
SZABLON="${CACHE}/chat_template.jinja"

[[ -d "${SRC}" ]]        || { echo "Nie ma katalogu ${SRC}" >&2; exit 1; }
[[ -f "${SZABLON}" ]]    || { echo "Nie ma szablonu ${SZABLON}" >&2; exit 1; }

for ITER in "$@"; do
    KROK="${SRC}/iter_${ITER}"
    [[ -d "${KROK}" ]] || { echo "Nie ma ${KROK}" >&2; exit 1; }
done

echo "== 1. q/k layernorm"
PYTHONPATH="${MEGATRON}" python3 "${SCRIPT_DIR}/check_dcp_qk_norm.py" \
    "${@/#/${SRC}/iter_}" || { echo "Przerwane: checkpoint ma wyzerowane wagi normalizacji." >&2; exit 1; }

for ITER in "$@"; do
    echo "== 2. konwersja iter_${ITER}"
    python3 "${SCRIPT_DIR}/convert_dcp_to_safetensors_apt4_poziomka5.py" \
        --checkpoint-path "${SRC}/iter_${ITER}" \
        --target-path "${TARGET}/iter_${ITER}" \
        --megatron-path "${MEGATRON}" \
        --override-tokenizer-path "${TOKENIZER}"

    echo "== 3. szablon czatu -> iter_${ITER}"
    cp "${SZABLON}" "${TARGET}/iter_${ITER}/chat_template.jinja"
done

echo
echo "Gotowe: ${TARGET}"
echo "Sprawdz przed pomiarem, ze model generuje tekst - sonda.py albo jedno zapytanie przy temperaturze 0."
