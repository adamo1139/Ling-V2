#!/usr/bin/env python3
"""Measure what each reasoning selector actually retains, on the real corpus.

Read-only. Runs both selectors over the same overlong records and reports the
reasoning tokens each keeps, plus the distribution of reasoning blocks per
record -- because a record with one block gives a selector nothing to choose
between, and the two are then identical by construction.

  python3 compare_poziomka_selectors.py --input <corpus> \
      --tokenizer poziomka-linear-8-9-10-11-sqrt --seq-length 16384 --sample-every 50
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path

from poziomka_data import encode_record, load_tokenizer
import numpy as np

from prepare_poziomka_sft import iter_records, remove_reasoning_to_fit

TOKENIZER = None
SELECTORS = ("smallest-sufficient", "largest-first")


def initialize_worker(tokenizer_path, template):
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = load_tokenizer(tokenizer_path, template)


def reasoning_tokens(record):
    return {i: len(TOKENIZER(m["reasoning_content"], add_special_tokens=False)["input_ids"])
            for i, m in enumerate(record["messages"])
            if m["role"] == "assistant" and m.get("reasoning_content")}


def survey_shard(job):
    source, seq_length, stride = job
    stats = Counter()
    blocks_per_record = Counter()
    for row_number, record in enumerate(iter_records(source)):
        if stride > 1 and row_number % stride:
            continue
        try:
            ids, _ = encode_record(TOKENIZER, record, ("all",))
        except ValueError:
            stats["unencodable"] += 1
            continue
        stats["records"] += 1
        costs = reasoning_tokens(record)
        total_reasoning = sum(costs.values())
        stats["reasoning_tokens_total"] += total_reasoning
        if len(ids) <= seq_length + 1:
            stats["records_fit"] += 1
            # Nothing is removed, so every selector keeps all of it.
            for selector in SELECTORS:
                stats[f"kept_{selector}"] += total_reasoning
            continue
        stats["records_overlong"] += 1
        stats["reasoning_tokens_overlong"] += total_reasoning
        blocks_per_record[min(len(costs), 10)] += 1
        if len(costs) <= 1:
            # One block (or none): both selectors are forced into the same move.
            stats["overlong_with_at_most_one_block"] += 1
        for selector in SELECTORS:
            changed, _, _, removed = remove_reasoning_to_fit(
                TOKENIZER, record, seq_length, ("all",), selector)
            kept = sum(costs[r] for r in costs
                       if changed["messages"][r].get("reasoning_content"))
            stats[f"kept_{selector}"] += kept
            stats[f"kept_overlong_{selector}"] += kept
            stats[f"removed_blocks_{selector}"] += len(removed)
    return dict(stats), dict(blocks_per_record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path,
                        default=Path(__file__).with_name("poziomka_chatml.jinja"))
    parser.add_argument("--seq-length", type=int, default=16384)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--sample-every", type=int, default=50,
                        help="Both selectors re-encode each overlong record, so sample")
    args = parser.parse_args()

    template = args.chat_template.read_text()
    files = (sorted((args.input / args.split).glob("*.parquet"))
             or sorted((args.input / args.split).glob("*.jsonl")))
    if not files:
        raise ValueError(f"No input under {args.input / args.split}")
    jobs = [(str(path.resolve()), args.seq_length, args.sample_every) for path in files]

    stats, blocks = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(str(args.tokenizer.resolve()), template)) as pool:
        futures = [pool.submit(survey_shard, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            shard_stats, shard_blocks = future.result()
            stats.update(shard_stats)
            blocks.update(shard_blocks)
            print(f"{done}/{len(jobs)} shards", flush=True)

    overlong = stats["records_overlong"]
    print(f"\n--seq-length {args.seq_length:,}, split {args.split}, "
          f"1-in-{args.sample_every} sample")
    print(f"records surveyed:      {stats['records']:,}")
    print(f"  fit as-is:           {stats['records_fit']:,}")
    print(f"  overlong:            {overlong:,}")
    if not overlong:
        print("\nNothing is overlong: the selector cannot matter at this length.")
        return

    print(f"\nreasoning blocks per overlong record")
    for count in sorted(blocks):
        label = f"{count}" if count < 10 else "10+"
        print(f"  {label:>3} block(s): {blocks[count]:>9,} "
              f"({100 * blocks[count] / overlong:5.1f}%)")
    forced = stats["overlong_with_at_most_one_block"]
    print(f"\n  {forced:,} of {overlong:,} overlong records ({100 * forced / overlong:.1f}%) "
          f"have at most one block,")
    print("  so both selectors are forced into the same removal there.")

    print(f"\nreasoning tokens retained")
    total = max(stats["reasoning_tokens_total"], 1)
    overlong_total = max(stats["reasoning_tokens_overlong"], 1)
    for selector in SELECTORS:
        print(f"  {selector:<22} corpus-wide {stats[f'kept_{selector}']:>14,} "
              f"({100 * stats[f'kept_{selector}'] / total:5.1f}%)   "
              f"within overlong {100 * stats[f'kept_overlong_{selector}'] / overlong_total:5.1f}%   "
              f"blocks removed {stats[f'removed_blocks_{selector}']:,}")
    gain = stats["kept_smallest-sufficient"] - stats["kept_largest-first"]
    print(f"\n  smallest-sufficient keeps {gain:,} more reasoning tokens "
          f"({100 * gain / total:.1f} points of the corpus total).")
    print("  If that is near zero, the selector was not what cost the reasoning,")
    print("  and block granularity or the window size is the real constraint.")


if __name__ == "__main__":
    main()
