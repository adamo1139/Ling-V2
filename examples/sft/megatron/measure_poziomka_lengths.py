#!/usr/bin/env python3
"""Read-only length survey of an SFT corpus, before any long-policy is applied.

Answers "which --seq-length should this corpus use": encodes every record with
the real template, then reports percentiles, a histogram and per-candidate
coverage both as-is and after full reasoning removal. Writes nothing except the
optional report/npz/png outputs; never touches the source corpus.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
import os
from pathlib import Path
import time

from poziomka_data import encode_record, load_tokenizer
import numpy as np

from prepare_poziomka_sft import iter_records

TOKENIZER = None
LOSS_ROLES = ("all",)
CANDIDATES = (2048, 3072, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536)
PERCENTILES = (50, 75, 90, 95, 99, 99.5, 99.9, 99.99)


def initialize_worker(tokenizer_path, template, loss_roles):
    global TOKENIZER, LOSS_ROLES
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = load_tokenizer(tokenizer_path, template)
    LOSS_ROLES = tuple(loss_roles)


def strip_all_reasoning(record):
    """The end state of the greedy policy: every block gone, thinking-off prefixes set."""
    stripped = copy.deepcopy(record)
    for message in stripped["messages"]:
        if message["role"] == "assistant" and message.get("reasoning_content"):
            message["reasoning_content"] = None
            message["content"] = "<think>\n</think>\n" + (message["content"] or "")
    return stripped


def survey_shard(job):
    source, split, stride, floor = job
    lengths, reasoning, stripped_len, stripped_at = [], [], [], []
    stats = Counter()
    for row_number, record in enumerate(iter_records(source)):
        if stride > 1 and row_number % stride:
            continue
        try:
            ids, _ = encode_record(TOKENIZER, record, LOSS_ROLES)
        except ValueError:
            stats["unencodable"] += 1
            continue
        lengths.append(len(ids))
        reasoning.append(sum(
            len(TOKENIZER(m["reasoning_content"], add_special_tokens=False)["input_ids"])
            for m in record["messages"]
            if m["role"] == "assistant" and m.get("reasoning_content")))
        # The fully-stripped length is the floor the greedy policy can reach. Only
        # worth the second encode for records that might not fit a candidate window.
        if len(ids) > floor and reasoning[-1]:
            try:
                bare, _ = encode_record(TOKENIZER, strip_all_reasoning(record), LOSS_ROLES)
            except ValueError:
                stats["unencodable_stripped"] += 1
            else:
                stripped_len.append(len(bare))
                stripped_at.append(len(lengths) - 1)
    return (split, np.asarray(lengths, dtype=np.int64), np.asarray(reasoning, dtype=np.int64),
            np.asarray(stripped_len, dtype=np.int64), np.asarray(stripped_at, dtype=np.int64),
            dict(stats))


def discover(root, splits=("train", "validation")):
    root = Path(root)
    jobs = []
    for split in splits:
        files = sorted((root / split).glob("*.jsonl")) or sorted((root / split).glob("*.parquet"))
        if not files:
            raise ValueError(f"No JSONL/Parquet files under {root / split}")
        jobs.extend((split, path.resolve()) for path in files)
    return jobs


def histogram(lengths, width=58, bins=24):
    """Log-spaced bars: a corpus with a 25k tail is unreadable on a linear axis."""
    if not len(lengths):
        return []
    edges = np.unique(np.geomspace(max(lengths.min(), 1), lengths.max() + 1, bins + 1).astype(np.int64))
    counts, _ = np.histogram(lengths, bins=edges)
    peak = max(counts.max(), 1)
    rows = []
    for i, count in enumerate(counts):
        share = 100.0 * count / len(lengths)
        bar = "#" * int(round(width * count / peak))
        rows.append(f"  {edges[i]:>7,}-{edges[i + 1] - 1:>7,} | {bar:<{width}} {count:>9,} {share:5.2f}%")
    return rows


def coverage(lengths, reasoning, stripped_len, stripped_at, candidates):
    """Per candidate window: records fitting, token mass kept, reasoning mass kept.

    Both long-policies retain about the same token *count* (a window holds what a
    window holds); they differ in what survives. So the deciding column is how
    much reasoning is left, which remove-reasoning sheds first and by design.
    """
    total_records, total_tokens = len(lengths), int(lengths.sum())
    total_reasoning = max(int(reasoning.sum()), 1)
    floor = lengths.copy()
    floor[stripped_at] = stripped_len  # the shortest the greedy policy can make each record
    body = lengths - reasoning  # everything the greedy policy may never remove
    rows = []
    for limit in candidates:
        window = limit + 1  # encode_record emits one extra next-token target
        fits = int((lengths <= window).sum())
        fits_stripped = int((floor <= window).sum())
        kept_tokens = int(np.minimum(lengths, window).sum())
        # Greedy sheds whole blocks until the record fits, so the reasoning that
        # survives is bounded by the room left once the body is seated. Records
        # already fitting keep all of theirs. Upper bound: blocks are indivisible.
        kept_reasoning = int(np.where(lengths <= window, reasoning,
                                      np.clip(window - body, 0, reasoning)).sum())
        rows.append(dict(limit=limit, fits=fits, fits_pct=100.0 * fits / total_records,
                         fits_stripped_pct=100.0 * fits_stripped / total_records,
                         hopeless_pct=100.0 * (total_records - fits_stripped) / total_records,
                         kept_tokens=kept_tokens,
                         kept_tokens_pct=100.0 * kept_tokens / total_tokens,
                         kept_reasoning=kept_reasoning,
                         kept_reasoning_pct=100.0 * kept_reasoning / total_reasoning,
                         padding_efficiency=100.0 * kept_tokens / (total_records * limit)))
    return rows


def report(split, lengths, reasoning, stripped_len, stripped_at, candidates, out):
    def emit(line=""):
        print(line, flush=True)
        out.append(line)

    total = len(lengths)
    emit(f"\n{'=' * 96}\n{split}: {total:,} records, {int(lengths.sum()):,} tokens\n{'=' * 96}")
    emit(f"mean {lengths.mean():,.0f}   min {lengths.min():,}   max {lengths.max():,}")
    with_reasoning = int((reasoning > 0).sum())
    emit(f"records with reasoning: {with_reasoning:,} ({100.0 * with_reasoning / total:.1f}%)   "
         f"reasoning tokens: {int(reasoning.sum()):,} "
         f"({100.0 * reasoning.sum() / max(lengths.sum(), 1):.1f}% of all tokens)")

    emit("\nfull conversation length percentiles")
    for p in PERCENTILES:
        emit(f"  p{p:<6} {np.percentile(lengths, p):>10,.0f}")

    if len(reasoning[reasoning > 0]):
        emit("\nper-record reasoning token percentiles (records with reasoning)")
        for p in PERCENTILES:
            emit(f"  p{p:<6} {np.percentile(reasoning[reasoning > 0], p):>10,.0f}")

    emit("\nlength histogram (log-spaced bins)")
    for row in histogram(lengths):
        emit(row)

    emit("\ncandidate --seq-length")
    emit("            records    records need   still over even     tokens    reasoning  padding")
    emit("  window    fit as-is   some cutting  with reasoning cut    kept       kept      eff.")
    for row in coverage(lengths, reasoning, stripped_len, stripped_at, candidates):
        emit(f"  {row['limit']:>6,}      {row['fits_pct']:>6.2f}%       {100 - row['fits_pct']:>6.2f}%"
             f"          {row['hopeless_pct']:>6.2f}%       {row['kept_tokens_pct']:>6.2f}%"
             f"    {row['kept_reasoning_pct']:>6.2f}%    {row['padding_efficiency']:>5.1f}%")
    emit("\n  'tokens kept' and 'reasoning kept' are shares of the untouched corpus; reasoning")
    emit("  kept is an upper bound (blocks are indivisible). 'still over' rows must truncate")
    emit("  real content even after every reasoning block is gone. 'padding eff.' is real")
    emit("  tokens over padded positions: the price of one conversation per sequence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory with train/ and validation/")
    parser.add_argument("--split", nargs="+", choices=("train", "validation"), default=["train", "validation"],
                        help="Which splits to survey (default: both)")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path,
                        default=Path(__file__).with_name("poziomka_chatml.jinja"))
    parser.add_argument("--loss-roles", nargs="+", choices=("all", "user", "assistant"), default=["all"])
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Survey every Nth record for a fast estimate (default: all)")
    parser.add_argument("--stripped-floor", type=int, default=4096,
                        help="Only re-encode without reasoning above this length (default: 4096)")
    parser.add_argument("--candidates", type=int, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--save-npz", type=Path, help="Save raw per-record arrays for later analysis")
    parser.add_argument("--save-report", type=Path, help="Save the printed report as text")
    parser.add_argument("--plot", type=Path, help="Write a PNG of the distributions (needs matplotlib)")
    args = parser.parse_args()

    if args.workers < 1 or args.sample_every < 1:
        parser.error("workers and sample-every must be positive")
    template = args.chat_template.read_text()
    tokenizer = load_tokenizer(args.tokenizer.resolve(), template)
    encode_record(tokenizer, {"messages": [{"role": "user", "content": "Pytanie"},
                                           {"role": "assistant", "content": "Odpowiedź"}]},
                  args.loss_roles)
    sources = discover(args.input, args.split)
    jobs = [(str(path), split, args.sample_every, args.stripped_floor) for split, path in sources]

    started = time.monotonic()
    collected = {}
    stats = Counter()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(str(args.tokenizer.resolve()), template, args.loss_roles)) as pool:
        futures = [pool.submit(survey_shard, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            split, lengths, reasoning, stripped_len, stripped_at, shard_stats = future.result()
            bucket = collected.setdefault(split, [[], [], [], []])
            offset = sum(len(chunk) for chunk in bucket[0])
            bucket[0].append(lengths)
            bucket[1].append(reasoning)
            bucket[2].append(stripped_len)
            bucket[3].append(stripped_at + offset)  # rebase into the concatenated array
            stats.update(shard_stats)
            print(f"{done}/{len(jobs)} surveyed ({len(lengths):,} records)", flush=True)

    out = []
    arrays = {}
    for split in args.split:
        if split not in collected:
            continue
        lengths, reasoning, stripped_len, stripped_at = (
            np.concatenate(chunks) if any(len(c) for c in chunks) else np.zeros(0, dtype=np.int64)
            for chunks in collected[split])
        if not len(lengths):
            continue
        arrays[f"{split}_lengths"] = lengths
        arrays[f"{split}_reasoning"] = reasoning
        arrays[f"{split}_stripped_len"] = stripped_len
        arrays[f"{split}_stripped_at"] = stripped_at
        report(split, lengths, reasoning, stripped_len, stripped_at, args.candidates, out)

    if stats:
        line = f"\nskipped rows: {json.dumps(dict(stats))}"
        print(line, flush=True)
        out.append(line)
    if args.sample_every > 1:
        line = f"\nESTIMATE ONLY: surveyed every {args.sample_every}th record."
        print(line, flush=True)
        out.append(line)
    print(f"\nSurveyed in {time.monotonic() - started:,.0f}s", flush=True)

    if args.save_npz:
        np.savez_compressed(args.save_npz, **arrays)
        print(f"Raw arrays: {args.save_npz}", flush=True)
    if args.save_report:
        args.save_report.write_text("\n".join(out) + "\n")
        print(f"Report: {args.save_report}", flush=True)
    if args.plot and "train_lengths" in arrays:
        write_plot(args.plot, arrays["train_lengths"], arrays["train_reasoning"], args.candidates)
        print(f"Plot: {args.plot}", flush=True)


def write_plot(path, lengths, reasoning, candidates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9))
    bins = np.geomspace(max(lengths.min(), 1), lengths.max() + 1, 80)
    top.hist(lengths, bins=bins, color="#4c72b0")
    top.hist(lengths - reasoning, bins=bins, color="#dd8452", alpha=0.65, label="reasoning removed")
    top.hist(lengths, bins=bins, histtype="step", color="#4c72b0", label="as-is")
    top.set_xscale("log")
    top.set_xlabel("conversation length (tokens)")
    top.set_ylabel("records")
    top.set_title("Length distribution, train")
    for limit in candidates:
        if lengths.min() <= limit <= lengths.max():
            top.axvline(limit, color="#555", linewidth=0.8, linestyle="--")
            top.text(limit, top.get_ylim()[1] * 0.95, f"{limit // 1024}k",
                     rotation=90, fontsize=7, va="top", ha="right", color="#555")
    top.legend()

    ordered = np.sort(lengths)
    mass = np.cumsum(ordered) / ordered.sum()
    bottom.plot(ordered, 100 * np.arange(1, len(ordered) + 1) / len(ordered), label="records covered")
    bottom.plot(ordered, 100 * mass, label="token mass covered")
    bottom.set_xscale("log")
    bottom.set_xlabel("--seq-length")
    bottom.set_ylabel("% covered")
    bottom.set_title("Coverage by window")
    bottom.grid(alpha=0.3)
    bottom.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
