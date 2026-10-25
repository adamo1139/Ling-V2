#!/usr/bin/env python3
"""Łączy polskie-sprawy-v8-jakosc50 i poziomka-fun-rp-v12-nothink-jakosc50 w jeden sft.jsonl
w formacie, którego oczekuje prepare_polskie_sprawy_v3.py (id, messages, meta.thinking[, tools]).

- v8: id, messages i meta bez zmian (pola ocena i jakosc odpadają, trening ich nie używa) plus
  meta.zrodlo = "v8"; id to już sha256 w hex.
- fun-rp: id = sha256(record_id) w hex (podział train/validation bierze pierwsze 8 znaków id jako
  liczbę hex), meta.thinking = "off" (wersja nothink nie ma rozumowania).
- fun-rp z niepustym tools (10,815 rozmów, tłumaczony zbiór obsługi klienta w stylu τ-bench)
  jest pomijany: zostały w nim same definicje narzędzi, bez żadnego wywołania i bez wyników,
  więc rozmowy mają dziury, a model uczyłby się nie używać podanych narzędzi.

Najpierw v8, potem fun-rp: prepare_polskie_sprawy_v3.py rozdziela rekordy po shardach po kolei
(każdy kolejny do następnego), a trening tasuje dane, więc kolejność w pliku nie ma znaczenia.

    python3 polacz_jakosc50.py --v8 polskie-sprawy-v8-jakosc50/sft.jsonl \\
        --fun-rp poziomka-fun-rp-v12-nothink-jakosc50/sft.jsonl --output polskie-sprawy-jakosc50/sft.jsonl
"""
import argparse
import hashlib
import json
import os
from collections import Counter


def from_v8(row: dict) -> dict:
    meta = dict(row["meta"])
    meta["zrodlo"] = "v8"
    return {"id": row["id"], "messages": row["messages"], "meta": meta}


def from_fun_rp(row: dict) -> dict | None:
    if row.get("tools"):
        return None
    if any(m.get("reasoning_content") for m in row["messages"]):
        raise ValueError(f"{row['record_id']}: rozumowanie w wersji nothink")
    out = {"id": hashlib.sha256(row["record_id"].encode("utf-8")).hexdigest(),
           "messages": [{k: v for k, v in m.items() if k != "reasoning_content"} for m in row["messages"]],
           "meta": {"thinking": "off", "zrodlo": "fun_rp", "record_id": row["record_id"],
                    "source_dataset": row.get("source_dataset")}}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v8", required=True)
    p.add_argument("--fun-rp", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    stats, seen = Counter(), set()
    tmp = args.output + ".partial"
    with open(tmp, "x", encoding="utf-8") as out:
        for path, convert, name in ((args.v8, from_v8, "v8"), (args.fun_rp, from_fun_rp, "fun_rp")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    record = convert(json.loads(line))
                    if record is None:
                        stats[f"{name}_pominiete_tools"] += 1
                        continue
                    if record["id"] in seen:
                        raise ValueError(f"Powtórzone id {record['id']} ({name})")
                    seen.add(record["id"])
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats[name] += 1
    os.rename(tmp, args.output)
    print(f"jakosc50: {stats['v8']:,} rozmów v8 + {stats['fun_rp']:,} fun-rp = {stats['v8'] + stats['fun_rp']:,} "
          f"(pominięto {stats['fun_rp_pominiete_tools']:,} fun-rp z tools) → {args.output}")


if __name__ == "__main__":
    main()
