#!/usr/bin/env bash
set -euo pipefail

# Buduje cache SFT polskie-sprawy (v3 albo v4-nc) w formacie v13 na maszynie treningowej.
# Run on rigga from anywhere:
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh          # v3 (run 6-9)
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v4-nc    # v4-nc (run 10)
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v6-reasoning  # run 12
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v8    # run 15
#   bash Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh v8-nothink  # KD od Bielika
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
# Wersja bez znanych liczb (nowy zbior): 0 znaczy "raportuj, nie sprawdzaj".
# Po pierwszym buildzie przepisz tu wypisane wartosci, zeby kolejne buildy byly
# weryfikowane jak v3 i v4-nc.
case "${VERSION}" in
    v3)    RECORDS=427970; TRAIN_ON=28003;  TRAIN_OFF=395715; BINS=13528 ;;
    v4-nc) RECORDS=538079; TRAIN_ON=107709; TRAIN_OFF=425024; BINS=24387 ;;
    v6)    RECORDS=1325738; TRAIN_ON=107709; TRAIN_OFF=1204847; BINS=44091 ;;
    # Same rozmowy z rozumowaniem z v6 (108,770; run 12): wszystkie ON, zadnego OFF.
    v6-reasoning) RECORDS=108770; TRAIN_ON=107709; TRAIN_OFF=0; BINS=16341 ;;
    # v8: te same rozmowy (i id) co v6, wiec ten sam podzial i te same ON/OFF; rozmowy OFF
    # sa teraz glownie 3-5-turowe (instruktor). Run 15.
    v8)    RECORDS=1325738; TRAIN_ON=107709; TRAIN_OFF=1204847; BINS=83423 ;;
    # v8 bez rozmow z rozumowaniem (meta.thinking == off), do KD od nauczyciela bez rozumowania
    # (Bielik-PL-Minitron-7B-v3.0-Instruct). Filtrowany lokalnie z sft.jsonl v8, nie z HF.
    # Loss tylko na odpowiedziach asystenta (+ ich <|im_end|>): nauczyciel jest trenowany tylko na
    # turach asystenta, wiec jego rozklad na turach uzytkownika nie jest sygnalem do KD.
    v8-nothink) RECORDS=1216968; TRAIN_ON=0; TRAIN_OFF=1204847; BINS=67138; LOSS_ROLES=assistant ;;
    *) echo "Unknown version ${VERSION}; use v3, v4-nc, v6, v6-reasoning, v8 or v8-nothink" >&2; exit 1 ;;
esac
SEQ_LENGTH=16384
# Domyslnie loss na wszystkich tokenach (jak v3-v8); wersja moze to zawezic.
LOSS_ROLES="${LOSS_ROLES:-all}"
RAW_DIR="${WORK_DIR}/polskie-sprawy-${VERSION}"
SPLIT_DIR="${WORK_DIR}/polskie-sprawy-${VERSION}-sft-v13"
CACHE_DIR="${WORK_DIR}/poziomka-sft-cache-polskie-sprawy-${VERSION}-v13-${SEQ_LENGTH}"
TOKENIZER="${REPO_DIR}/resource/tokenizer/apt4"
TEMPLATE="${SCRIPT_DIR}/poziomka_v13_chat_template.jinja"
WORKERS="${WORKERS:-16}"

for d in "${SPLIT_DIR}" "${CACHE_DIR}"; do
    [[ ! -e "${d}" ]] || { echo "Refusing existing output: ${d}" >&2; exit 1; }
done

if [[ "${VERSION}" == v8-nothink && ! -f "${RAW_DIR}/sft.jsonl" ]]; then
    V8_RAW="${WORK_DIR}/polskie-sprawy-v8/sft.jsonl"
    [[ -f "${V8_RAW}" ]] || hf download cpral/polskie-sprawy-v8 --repo-type dataset --include sft.jsonl \
        --local-dir "${WORK_DIR}/polskie-sprawy-v8"
    mkdir -p "${RAW_DIR}"
    python3 -c '
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
kept = dropped = 0
with open(src, encoding="utf-8") as fin, open(dst + ".partial", "x", encoding="utf-8") as fout:
    for line in fin:
        if json.loads(line)["meta"]["thinking"] == "off":
            fout.write(line); kept += 1
        else:
            dropped += 1
os.rename(dst + ".partial", dst)
print(f"v8-nothink: {kept:,} rozmow bez rozumowania, pominieto {dropped:,} z rozumowaniem")
' "${V8_RAW}" "${RAW_DIR}/sft.jsonl"
elif [[ ! -f "${RAW_DIR}/sft.jsonl" ]]; then
    hf download "cpral/polskie-sprawy-${VERSION}" --repo-type dataset --include sft.jsonl --local-dir "${RAW_DIR}"
fi
records=$(wc -l < "${RAW_DIR}/sft.jsonl")
if [[ "${RECORDS}" == "0" ]]; then
    echo "Nowy zbior: ${records} rekordow w ${RAW_DIR}/sft.jsonl (bez weryfikacji)"
else
    [[ "${records}" == "${RECORDS}" ]] || { echo "Expected ${RECORDS} records in ${RAW_DIR}/sft.jsonl, found ${records}" >&2; exit 1; }
fi

python3 "${SCRIPT_DIR}/prepare_polskie_sprawy_v3.py" \
    --input "${RAW_DIR}/sft.jsonl" --output "${SPLIT_DIR}" --off-prefix none

python3 "${SCRIPT_DIR}/prepare_poziomka_sft.py" \
    --input "${SPLIT_DIR}" --tokenizer "${TOKENIZER}" --chat-template "${TEMPLATE}" \
    --output "${CACHE_DIR}" --seq-length "${SEQ_LENGTH}" --long-policy truncate --workers "${WORKERS}" \
    --loss-roles ${LOSS_ROLES}

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
if expected_on or expected_off:
    if after != collections.Counter({"OFF (<)": expected_off, "ON (\\n)": expected_on}):
        sys.exit(f"Expected exactly {expected_off:,} OFF and {expected_on:,} ON records")
else:
    inne = {k: v for k, v in after.items() if not k.startswith(("ON", "OFF"))}
    if inne:
        sys.exit(f"Po <think> pojawiaja sie nieoczekiwane tokeny: {inne}")
    on, off = after["ON (\\n)"], after["OFF (<)"]
    print(f"Nowy zbior: ON={on:,} OFF={off:,} (bez weryfikacji)")

train = MMapSFTDataset(f"{cache}/manifest.json", "train", pack=True)
bins = len(train.bins)
print(f"Packed train sequences: {bins:,} (efficiency {train.packing_efficiency():.3f})")
print(f"TRAIN_ITERS at GBS 16: {math.ceil(bins / 16)}, at GBS 32: {math.ceil(bins / 32)}, at GBS 64: {math.ceil(bins / 64)}")
if expected_bins:
    if bins != expected_bins:
        sys.exit(f"Expected {expected_bins:,} packed sequences; launchers' TRAIN_ITERS assume it")
else:
    # LR z reguly pierwiastkowej dla Adama, tej samej co w run 7 i run 10:
    #   LR = 3e-4 * sqrt(tokeny_SFT_na_krok / 2,097,152)
    # gdzie 2,097,152 = 8192 x GBS 256 z pretreningu, a tokeny_SFT_na_krok to
    # prawdziwe (nie wypelnione) tokeny: tokeny_treningowe / kubelki x GBS.
    import json
    manifest = json.load(open(f"{cache}/manifest.json"))
    tokeny = manifest["totals"]["train"]["tokens"]
    na_sekwencje = tokeny / bins
    print(f"\nPrawdziwych tokenow na spakowana sekwencje: {na_sekwencje:,.0f}")
    print(f"{'GBS':>5} {'TRAIN_ITERS':>12} {'tokeny/krok':>13} {'LR':>10}")
    for gbs in (16, 32, 64):
        krok = na_sekwencje * gbs
        lr = 3e-4 * math.sqrt(krok / 2_097_152)
        print(f"{gbs:>5} {math.ceil(bins / gbs):>12} {krok:>13,.0f} {lr:>10.3g}")
    valid = MMapSFTDataset(f"{cache}/manifest.json", "validation", pack=True, shuffle=False)
    vbins = len(valid.bins)
    print(f"\nWalidacja: {vbins} spakowanych sekwencji -> EVAL_ITERS przy GBS 16: {vbins // 16} "
          f"(calosc), przy GBS 32: {vbins // 32}, przy GBS 64: {vbins // 64}")
    print("\nPrzepisz RECORDS/TRAIN_ON/TRAIN_OFF/BINS do tego skryptu, zeby kolejne "
          "buildy byly weryfikowane.")
PY

echo "Ready: ${CACHE_DIR}"
