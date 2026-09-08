#!/usr/bin/env python3
"""Stream JSONL/Parquet shards into full-conversation mmap caches, using 15 workers.

Never loads model weights or modifies the input. A completed manifest is published
only after every output shard passes a full scan. Refuses to overwrite an output.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from poziomka_data import (FORMAT, OFFSET_DTYPE, SPECIAL_IDS, encode_record,
                           load_manifest, load_tokenizer, sha256, verify_shard)
import numpy as np

TOKENIZER = None
LOSS_ROLES = ("all",)
EXAMPLES_PER_SHARD = 20  # named drops kept per shard; the count is always exact


def initialize_worker(tokenizer_path, template, loss_roles=("all",)):
    global TOKENIZER, LOSS_ROLES
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = load_tokenizer(tokenizer_path, template)
    LOSS_ROLES = tuple(loss_roles)


def iter_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()
    else:
        raise ValueError(f"Unsupported input: {path}")


def process_shard(job):
    source, root, split, prefix, seq_length, policy, unencodable = job
    root = Path(root)
    stats = Counter()
    dropped = []
    position = 0
    with (root / f"{prefix}.tokens.bin").open("xb") as tokens_out, \
         (root / f"{prefix}.masks.bin").open("xb") as masks_out, \
         (root / f"{prefix}.offsets.bin").open("xb") as offsets_out:
        np.asarray([0], dtype=OFFSET_DTYPE).tofile(offsets_out)
        for row_number, record in enumerate(iter_records(source), 1):
            try:
                ids, mask = encode_record(TOKENIZER, record, LOSS_ROLES)
            except ValueError as exc:
                # Only data/encoding validation failures may be dropped.
                # Unexpected programming errors must abort preparation.
                if unencodable == "error":
                    raise ValueError(f"{source}:{row_number}: {exc}") from exc
                stats["dropped_unencodable_records"] += 1
                if len(dropped) < EXAMPLES_PER_SHARD:
                    dropped.append({"row": row_number, "error": str(exc),
                                    "source_id": record.get("source_id")})
                continue
            try:
                stats["input_records"] += 1
                stats["input_tokens"] += len(ids)
                stats["input_supervised_tokens"] += int(mask.sum())
                if len(ids) > seq_length + 1:
                    stats["overlong_records"] += 1
                    if policy == "error":
                        raise ValueError(f"Conversation exceeds {seq_length + 1} tokens")
                    if policy == "drop":
                        stats["dropped_overlong_records"] += 1
                        continue
                    # Deliberate right truncation: never manufacture a false EOS.
                    ids, mask = ids[:seq_length + 1], mask[:seq_length + 1]
                    stats["truncated_records"] += 1
                if not mask[1:].any():
                    stats["dropped_no_targets_records"] += 1
                    continue
                ids.tofile(tokens_out)
                mask.tofile(masks_out)
                position += len(ids)
                np.asarray([position], dtype=OFFSET_DTYPE).tofile(offsets_out)
                stats["records"] += 1
                stats["tokens"] += len(ids)
                stats["supervised_tokens"] += int(mask.sum())
            except Exception as exc:
                raise ValueError(f"{source}:{row_number}: {exc}") from exc
    shard = dict(split=split, prefix=prefix, source=str(source), stats=dict(stats),
                 records=stats["records"], tokens=stats["tokens"],
                 supervised_tokens=stats["supervised_tokens"])
    if dropped:
        shard["dropped_unencodable"] = dropped
    shard["sha256"] = {suffix: sha256(root / f"{prefix}.{suffix}")
                       for suffix in ("tokens.bin", "masks.bin", "offsets.bin")}
    verify_shard(root, shard, seq_length, check_hashes=False)
    return shard


def discover(root):
    root = Path(root)
    jobs = []
    for split in ("train", "validation"):
        files = sorted((root / split).glob("*.jsonl"))
        parquet = sorted((root / split).glob("*.parquet"))
        if files and parquet:
            raise ValueError("Choose a single input format, not duplicate JSONL and Parquet")
        files = files or parquet
        if not files:
            raise ValueError(f"No JSONL/Parquet files under {root / split}")
        jobs.extend((split, path.resolve()) for path in files)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Directory containing train/ and validation/")
    parser.add_argument("--tokenizer", type=Path, help="Local Poziomka checkpoint/tokenizer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chat-template", type=Path,
                        default=Path(__file__).with_name("poziomka_chatml.jinja"))
    parser.add_argument("--seq-length", type=int, default=3072)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--long-policy", choices=("truncate", "drop", "error"), default="truncate")
    parser.add_argument("--unencodable-policy", choices=("error", "drop"), default="error",
                        help="Rows failing template/mask validation: abort (default) or "
                             "drop them, counted and named in the manifest")
    parser.add_argument("--loss-roles", nargs="+", choices=("all", "user", "assistant"),
                        default=["all"], help="All real tokens (default), or selected message bodies/EOS")
    parser.add_argument("--verify", type=Path, help="Full rescan of an existing manifest; no writes")
    args = parser.parse_args()
    if args.verify:
        manifest = load_manifest(args.verify)
        for shard in manifest["shards"]:
            verify_shard(args.verify.parent, shard, manifest["seq_length"])
            print(f"Verified {shard['prefix']}: {shard['records']} records", flush=True)
        print("Full cache verification passed.")
        return
    if not all((args.input, args.tokenizer, args.output)):
        parser.error("--input, --tokenizer and --output are required for preparation")
    if not 1 <= args.seq_length <= 8192 or args.workers < 1:
        parser.error("seq-length must be 1..8192 and workers must be positive")
    sources = discover(args.input)
    template = args.chat_template.read_text()
    tokenizer_path = args.tokenizer.resolve()
    tokenizer = load_tokenizer(tokenizer_path, template)
    # Validate generation-mask support before spawning workers or creating outputs.
    encode_record(tokenizer, {"messages": [{"role": "user", "content": "Pytanie"},
                                          {"role": "assistant", "content": "Odpowiedź"}]}, args.loss_roles)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer_dir = output / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)  # Correct EOS=4, PAD=2 + actual SFT template.
    (output / "chat_template.jinja").write_text(template)
    jobs = []
    for i, (split, source) in enumerate(sources):
        (output / split).mkdir(exist_ok=True)
        jobs.append((str(source), str(output), split, f"{split}/shard_{i:05d}",
                     args.seq_length, args.long_policy, args.unencodable_policy))
    started = time.monotonic()
    shards = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(str(tokenizer_dir), template, args.loss_roles)) as pool:
        futures = [pool.submit(process_shard, job) for job in jobs]
        for future in as_completed(futures):
            shard = future.result()
            shards.append(shard)
            print(f"{len(shards)}/{len(jobs)} verified: {shard['prefix']} "
                  f"({shard['records']} records)", flush=True)
    totals = {}
    for split in ("train", "validation"):
        stats = Counter()
        for shard in shards:
            if shard["split"] == split:
                stats.update(shard["stats"])
        if not stats["records"]:
            raise ValueError(f"No retained {split} records; no manifest published")
        stats["discarded_supervised_tokens"] = stats["input_supervised_tokens"] - stats["supervised_tokens"]
        totals[split] = dict(stats)
    manifest = dict(format=FORMAT, seq_length=args.seq_length, long_policy=args.long_policy,
                    unencodable_policy=args.unencodable_policy,
                    packing=False, special_ids=SPECIAL_IDS, workers=args.workers,
                    loss_roles=args.loss_roles,
                    tokenizer_sha256=sha256(tokenizer_dir / "tokenizer.json"),
                    template_sha256=sha256(output / "chat_template.jinja"),
                    elapsed_seconds=time.monotonic() - started, verified=True,
                    totals=totals, shards=sorted(shards, key=lambda s: s["prefix"]))
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.rename(output / "manifest.json")
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)
    print(f"Ready: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
