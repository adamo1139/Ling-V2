#!/usr/bin/env python3
"""Filtrowanie korpusu wg TSV z detect_reasoning_loops.py --save-ids.

Wejscie: korpus (parquet) + TSV-ki ("record_id<TAB>indeks<TAB>pole"). Wyjscie:
nowy katalog korpusu z tym samym ukladem splitow i chunkow, tym samym schematem
parquet. Dwa tryby dzialania na trafionych rekordach:

  --mode strip   wycina reasoning_content z trafionych wiadomosci (konwencja
                 z prepare_poziomka_sft.py: reasoning_content=None, prefix
                 '\n\n\n' w content, przeliczony reasoning_profile); reszta
                 rekordu nietknieta. Tak powstalo poziomka-fun-rp-v12.
  --mode drop    usuwa CALY rekord, ktorykolwiek z jego tekstow jest
                 zapetlony. Tak powstanie poziomka-fun-rp-v13.

TSV z trzema kolumnami (record_id, indeks, pole) pozwala ograniczyc drop do
jednego pola: --drop-fields content usuwa rekordy trafione w tresci, ignorujac
trafienia w reasoning. Bez tej opcji liczy sie kazde trafienie.

Rekord bez trafien przechodzi bez dotyku (ten sam batch, ta sama kompresja).
UZYCIE:

    # v12: wyciecie reasoning'u z zapetlonych wiadomosci
    python3 strip_looped_reasoning.py --input poziomka-fun-rp-v11 \\
        --output poziomka-fun-rp-v12 --mode strip --workers 8 \\
        --ids wiadomosci_z_petla_train.tsv --ids wiadomosci_z_petla_validation.tsv

    # v13: usuniecie calego rekordu z zapetlona trescia
    python3 strip_looped_reasoning.py --input poziomka-fun-rp-v12 \\
        --output poziomka-fun-rp-v13 --mode drop --drop-fields content \\
        --workers 8 --ids wiadomosci_z_petla_v12_train.tsv \\
        --ids wiadomosci_z_petla_v12_validation.tsv
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Pliki kopiowane do nowego korpusu obok splitow, zeby wyjscie bylo
# samowystarczalne.
AUX_FILES = {"chat_template.jinja"}
AUX_DIRS = {"tokenizer"}


def load_ids(paths):
    """TSV-ki -> {record_id: {indeksy wiadomosci}}."""
    ids = {}
    total = 0
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                record_id = parts[0]
                index = int(parts[1]) if len(parts) > 1 else -1
                field = parts[2] if len(parts) > 2 else ""
                ids.setdefault(record_id, {})[index] = field
                total += 1
    return ids, total


def strip_record(record, indices, stats):
    """Jeden rekord + indeksy wiadomosci -> rekord z wycietym reasoning'em."""
    messages = record.get("messages") or []
    touched = False
    for index in sorted(indices):
        if index >= len(messages):
            stats["out_of_range"] += 1
            continue
        message = messages[index]
        text = message.get("reasoning_content")
        if not text:
            stats["already_empty"] += 1
            continue
        stats["stripped_words"] += len(text.split())
        message["reasoning_content"] = None
        message["content"] = "\n\n\n" + (message["content"] or "")
        touched = True
    if touched:
        stats["stripped_messages"] += len(indices)
        stats["modified_records"] += 1
        flags = [bool(m.get("reasoning_content")) for m in messages
                 if m.get("role") == "assistant"]
        if "reasoning_profile" in record:
            record["reasoning_profile"] = ("mixed" if any(flags) and not all(flags)
                                           else "on" if any(flags) else "off")
    return record


_IDS = {}
_DROP_FIELDS = None


def init_worker(ids, drop_fields):
    global _IDS, _DROP_FIELDS
    _IDS = ids
    _DROP_FIELDS = drop_fields


def process_file(job):
    """Jeden plik parquet: przeczytaj, przetworz, zapisz do wyjscia."""
    src, dst, mode = job
    ids, drop_fields = _IDS, _DROP_FIELDS
    stats = {"records": 0, "matched_records": 0, "modified_records": 0,
             "stripped_messages": 0, "stripped_words": 0,
             "dropped_records": 0, "out_of_range": 0, "already_empty": 0}
    matched = set()
    source = pq.ParquetFile(src)
    schema = source.schema_arrow
    compression = source.metadata.row_group(0).column(0).compression or "snappy"
    with pq.ParquetWriter(dst, schema, compression=compression) as writer:
        for batch in source.iter_batches(batch_size=1024):
            rows = batch.to_pylist()
            changed = False
            kept = []
            for row in rows:
                stats["records"] += 1
                record_id = row.get("record_id")
                hits = ids.get(record_id)
                if not hits:
                    kept.append(row)
                    continue
                stats["matched_records"] += 1
                matched.add(record_id)
                if mode == "drop":
                    fields = {f for f in hits.values() if f}
                    if not fields or (drop_fields and fields & drop_fields):
                        stats["dropped_records"] += 1
                        changed = True
                        continue  # rekord wylatuje
                    kept.append(row)
                else:  # strip
                    strip_record(row, set(hits), stats)
                    changed = True
                    kept.append(row)
            if not kept:
                continue
            if changed:
                table = pa.Table.from_pylist(kept, schema=schema)
                writer.write_table(table)
            else:
                writer.write_batch(batch)
    return stats, matched


def discover_splits(root):
    """Podkatalogi z parquetami, np. train i validation."""
    return sorted(child.name for child in root.iterdir()
                  if child.is_dir() and any(child.glob("*.parquet")))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Korpus zrodlowy, np. poziomka-fun-rp-v12")
    parser.add_argument("--output", required=True, type=Path,
                        help="Korpus docelowy, np. poziomka-fun-rp-v13")
    parser.add_argument("--mode", choices=("strip", "drop"), default="strip",
                        help="strip = wycinaj reasoning, drop = usuwaj rekordy")
    parser.add_argument("--drop-fields", default="",
                        help="Przy --mode drop: tylko te pola powoduja drop "
                             "(koma-rozdzielone, np. content); puste = kazde")
    parser.add_argument("--ids", action="append", required=True, default=[],
                        help="TSV z --save-ids; powtarzalne, np. train + validation")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    drop_fields = ({f.strip() for f in args.drop_fields.split(",") if f.strip()}
                   if args.drop_fields else None)
    if args.mode == "strip" and drop_fields:
        raise SystemExit("--drop-fields dziala tylko z --mode drop")

    ids, id_rows = load_ids(args.ids)
    print(f"wpisow w TSV: {id_rows:,} -> {len(ids):,} unikalnych rekordow")
    init_worker(ids, drop_fields)  # sciezka sekwencyjna czyta globalny; workers dostaja kopie

    splits = discover_splits(args.input)
    if not splits:
        raise SystemExit(f"Nie znaleziono splitow z parquetami w {args.input}")
    print(f"splitly: {', '.join(splits)}")

    for aux in sorted(AUX_FILES):
        source = args.input / aux
        if source.exists():
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / aux).write_bytes(source.read_bytes())
    for aux in sorted(AUX_DIRS):
        if (args.input / aux).is_dir():
            import shutil
            shutil.copytree(args.input / aux, args.output / aux, dirs_exist_ok=True)

    totals = {}
    unmatched = set(ids)
    for split in splits:
        files = sorted((args.input / split).glob("*.parquet"))
        out_dir = args.output / split
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(src, out_dir / src.name, args.mode) for src in files]
        stats = {"records": 0, "matched_records": 0, "modified_records": 0,
                 "stripped_messages": 0, "stripped_words": 0,
                 "dropped_records": 0, "out_of_range": 0, "already_empty": 0}
        done = 0
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers, initializer=init_worker,
                      initargs=(ids, drop_fields)) as pool:
                for file_stats, file_matched in pool.imap_unordered(process_file, jobs):
                    done += 1
                    for key in stats:
                        stats[key] += file_stats[key]
                    unmatched -= file_matched
                    print(f"[{split}] {done}/{len(jobs)} plikow, "
                          f"zmienionych: {stats['modified_records']:,}, "
                          f"dropped: {stats['dropped_records']:,}",
                          file=sys.stderr, flush=True)
        else:
            for job in jobs:
                file_stats, file_matched = process_file(job)
                done += 1
                for key in stats:
                    stats[key] += file_stats[key]
                unmatched -= file_matched
                print(f"[{split}] {done}/{len(jobs)} plikow, "
                      f"zmienionych: {stats['modified_records']:,}, "
                      f"dropped: {stats['dropped_records']:,}",
                      file=sys.stderr, flush=True)
        totals[split] = stats

    print()
    for split, stats in totals.items():
        print(f"{split}:")
        print(f"  rekordow:                 {stats['records']:,}")
        print(f"  trafionych (record_id):   {stats['matched_records']:,}")
        if args.mode == "drop":
            print(f"  usunietych rekordow:      {stats['dropped_records']:,}")
        else:
            print(f"  zmienionych rekordow:     {stats['modified_records']:,}")
            print(f"  wycietych wiadomosci:     {stats['stripped_messages']:,}")
            print(f"  wyciete slowa reasoning:  {stats['stripped_words']:,}")
        if stats["out_of_range"] or stats["already_empty"]:
            print(f"  anomalie: poza zakresem={stats['out_of_range']}, "
                  f"juz puste={stats['already_empty']}")
    if unmatched:
        print(f"\nUWAGA: {len(unmatched):,} rekordow z TSV nie znaleziono w korpusie, "
              f"np. {sorted(unmatched)[:3]}")
    else:
        print("\nwszystkie rekordy z TSV trafione")


if __name__ == "__main__":
    main()