#!/usr/bin/env bash
set -euo pipefail

# Buduje cache SFT polskie-sprawy (v3 albo v4-nc) w formacie v13 na maszynie treningowej.
# Run on rigga from anywhere:
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh          # v3 (run 6-9)
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v4-nc    # v4-nc (run 10)
#
# v4-nc to rewizja juz naprawiona fix_polskie_sprawy_v4.py (538,079 rekordow): bez 54
# rekordow z tagiem think w rozumowaniu i 424 uszkodzonych odpowiedzi Mistrala.
#
# 1. pobiera cpral/polskie-sprawy-<wersja> (sft.jsonl) z HF,
# 2. dzieli na train/validation (1% po id) bez prefiksu w tresci (--off-prefix none),
# 3. tokenizuje szablonem poziomka_v13_chat_template.jinja i tokenizerem APT4 z repo,
# 4. sprawdza na calym zbiorze, ze po <think> jest \n tylko w rekordach z rozumowaniem,
#    i liczy kubelki pakowania, z ktorych bierze sie TRAIN_ITERS.
#
# Wyniki w ${WORK_DIR} (na riggu ~/projects/pretrain), tam gdzie szuka ich launcher.
# Odmawia nadpisania istniejacych katalogow.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(cd -- "${REPO_DIR}/.." && pwd)}"

VERSION="${1:-v3}"
# Oczekiwane liczby (sprawdzone pelnym buildem): rekordy w sft.jsonl, ON i OFF w train,
# kubelki po spakowaniu przy 16384.
case "${VERSION}" in
    v3)    RECORDS=427970; TRAIN_ON=28003;  TRAIN_OFF=395715; BINS=13528 ;;
    v4-nc) RECORDS=538079; TRAIN_ON=107709; TRAIN_OFF=425024; BINS=24387 ;;
    *) echo "Unknown version ${VERSION}; use v3 or v4-nc" >&2; exit 1 ;;
esac
SEQ_LENGTH=16384
RAW_DIR="${WORK_DIR}/polskie-sprawy-${VERSION}"
SPLIT_DIR="${WORK_DIR}/polskie-sprawy-${VERSION}-sft-v13"
CACHE_DIR="${WORK_DIR}/poziomka-sft-cache-polskie-sprawy-${VERSION}-v13-${SEQ_LENGTH}"
TOKENIZER="${REPO_DIR}/resource/tokenizer/apt4"
TEMPLATE="${SCRIPT_DIR}/poziomka_v13_chat_template.jinja"
WORKERS="${WORKERS:-16}"

for d in "${SPLIT_DIR}" "${CACHE_DIR}"; do
    [[ ! -e "${d}" ]] || { echo "Refusing existing output: ${d}" >&2; exit 1; }
done

if [[ ! -f "${RAW_DIR}/sft.jsonl" ]]; then
    hf download "cpral/polskie-sprawy-${VERSION}" --repo-type dataset --include sft.jsonl --local-dir "${RAW_DIR}"
fi
records=$(wc -l < "${RAW_DIR}/sft.jsonl")
[[ "${records}" == "${RECORDS}" ]] || { echo "Expected ${RECORDS} records in ${RAW_DIR}/sft.jsonl, found ${records}" >&2; exit 1; }

python3 "${SCRIPT_DIR}/prepare_polskie_sprawy_v3.py" \
    --input "${RAW_DIR}/sft.jsonl" --output "${SPLIT_DIR}" --off-prefix none

python3 "${SCRIPT_DIR}/prepare_poziomka_sft.py" \
    --input "${SPLIT_DIR}" --tokenizer "${TOKENIZER}" --chat-template "${TEMPLATE}" \
    --output "${CACHE_DIR}" --seq-length "${SEQ_LENGTH}" --long-policy truncate --workers "${WORKERS}"

PYTHONPATH="${SCRIPT_DIR}" python3 - "${CACHE_DIR}" "${TRAIN_ON}" "${TRAIN_OFF}" "${BINS}" <<'PY'
import collections, glob, math, sys
import numpy as np
from transformers import PreTrainedTokenizerFast
from poziomka_data import MMapSFTDataset

cache = sys.argv[1]
expected_on, expected_off, expected_bins = map(int, sys.argv[2:5])
tok = PreTrainedTokenizerFast.from_pretrained(f"{cache}/tokenizer")
lt, th, ink, gt = tok.convert_tokens_to_ids(["<", "th", "ink", ">"])
newline = tok.convert_tokens_to_ids("<0x0A>")
after = collections.Counter()
for path in sorted(glob.glob(f"{cache}/train/*.tokens.bin")):
    tokens = np.fromfile(path, dtype=np.uint16)
    offsets = np.fromfile(path.replace(".tokens.bin", ".offsets.bin"), dtype=np.uint64)
    for i in range(len(offsets) - 1):
        s = tokens[int(offsets[i]):int(offsets[i + 1])]
        hit = np.where((s[:-4] == lt) & (s[1:-3] == th) & (s[2:-2] == ink) & (s[3:-1] == gt))[0]
        nxt = int(s[hit[0] + 4]) if len(hit) else None
        after["ON (\\n)" if nxt == newline else "OFF (<)" if nxt == lt else f"other {nxt}"] += 1
print("Token after the first <think>:", dict(after))
if after != collections.Counter({"OFF (<)": expected_off, "ON (\\n)": expected_on}):
    sys.exit(f"Expected exactly {expected_off:,} OFF and {expected_on:,} ON records")

train = MMapSFTDataset(f"{cache}/manifest.json", "train", pack=True)
bins = len(train.bins)
print(f"Packed train sequences: {bins:,} (efficiency {train.packing_efficiency():.3f})")
print(f"TRAIN_ITERS at GBS 16: {math.ceil(bins / 16)}, at GBS 32: {math.ceil(bins / 32)}, at GBS 64: {math.ceil(bins / 64)}")
if bins != expected_bins:
    sys.exit(f"Expected {expected_bins:,} packed sequences; launchers' TRAIN_ITERS assume it")
PY

echo "Ready: ${CACHE_DIR}"
