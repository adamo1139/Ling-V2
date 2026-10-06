#!/usr/bin/env bash
set -euo pipefail

# Buduje cache SFT polskie-sprawy-v3 w formacie v13 na maszynie treningowej.
# Run on rigga from anywhere: bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh
#
# 1. pobiera cpral/polskie-sprawy-v3 (sft.jsonl) z HF,
# 2. dzieli na train/validation (1% po id) bez prefiksu w tresci (--off-prefix none),
# 3. tokenizuje szablonem poziomka_v13_chat_template.jinja i tokenizerem APT4 z repo,
# 4. sprawdza na calym zbiorze, ze po <think> jest \n tylko w rekordach z rozumowaniem,
#    i liczy kubelki pakowania, z ktorych bierze sie TRAIN_ITERS run 6.
#
# Wyniki w ${WORK_DIR} (na riggu ~/projects/pretrain), tam gdzie szuka ich launcher.
# Odmawia nadpisania istniejacych katalogow.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(cd -- "${REPO_DIR}/.." && pwd)}"

SEQ_LENGTH=16384
RAW_DIR="${WORK_DIR}/polskie-sprawy-v3"
SPLIT_DIR="${WORK_DIR}/polskie-sprawy-v3-sft-v13"
CACHE_DIR="${WORK_DIR}/poziomka-sft-cache-polskie-sprawy-v3-v13-${SEQ_LENGTH}"
TOKENIZER="${REPO_DIR}/resource/tokenizer/apt4"
TEMPLATE="${SCRIPT_DIR}/poziomka_v13_chat_template.jinja"
WORKERS="${WORKERS:-16}"

for d in "${SPLIT_DIR}" "${CACHE_DIR}"; do
    [[ ! -e "${d}" ]] || { echo "Refusing existing output: ${d}" >&2; exit 1; }
done

if [[ ! -f "${RAW_DIR}/sft.jsonl" ]]; then
    hf download cpral/polskie-sprawy-v3 --repo-type dataset --include sft.jsonl --local-dir "${RAW_DIR}"
fi
records=$(wc -l < "${RAW_DIR}/sft.jsonl")
[[ "${records}" == 427970 ]] || { echo "Expected 427,970 records, found ${records}" >&2; exit 1; }

python3 "${SCRIPT_DIR}/prepare_polskie_sprawy_v3.py" \
    --input "${RAW_DIR}/sft.jsonl" --output "${SPLIT_DIR}" --off-prefix none

python3 "${SCRIPT_DIR}/prepare_poziomka_sft.py" \
    --input "${SPLIT_DIR}" --tokenizer "${TOKENIZER}" --chat-template "${TEMPLATE}" \
    --output "${CACHE_DIR}" --seq-length "${SEQ_LENGTH}" --long-policy truncate --workers "${WORKERS}"

PYTHONPATH="${SCRIPT_DIR}" python3 - "${CACHE_DIR}" <<'PY'
import collections, glob, math, sys
import numpy as np
from transformers import PreTrainedTokenizerFast
from poziomka_data import MMapSFTDataset

cache = sys.argv[1]
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
if after != collections.Counter({"OFF (<)": 395715, "ON (\\n)": 28003}):
    sys.exit("Expected exactly 395,715 OFF and 28,003 ON records")

train = MMapSFTDataset(f"{cache}/manifest.json", "train", pack=True)
bins = len(train.bins)
print(f"Packed train sequences: {bins:,} (efficiency {train.packing_efficiency():.3f})")
print(f"TRAIN_ITERS at GBS 16: {math.ceil(bins / 16)}, at GBS 64: {math.ceil(bins / 64)}")
PY

echo "Ready: ${CACHE_DIR}"
