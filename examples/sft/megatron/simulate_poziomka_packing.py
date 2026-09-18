#!/usr/bin/env python3
"""Size the payoff of sequence packing from a measured length distribution.

Pure arithmetic on the .npz written by measure_poziomka_lengths.py: no GPU, no
model, no cache. Packing does not change the cost of a step (same seq_length,
same batch); it changes how many steps there are. So the whole benefit is the
drop in sequence count, which bin-packing the real lengths answers exactly.
"""
import argparse
from pathlib import Path

import numpy as np


def next_fit(lengths, window):
    """What a streaming packer does: start a new bin when the current one is full."""
    bins, current = 0, window + 1
    for length in lengths:
        if current + length > window:
            bins += 1
            current = 0
        current += length
    return bins


def first_fit_decreasing(lengths, window, open_bins=64):
    """Near-optimal, but needs the corpus sorted and many bins open at once.

    open_bins caps how many partially-filled bins a real implementation would
    track; unlimited FFD is not something a streaming preparer can do.
    """
    remaining = []
    bins = 0
    for length in sorted(lengths, reverse=True):
        placed = False
        for i, space in enumerate(remaining):
            if space >= length:
                remaining[i] = space - length
                placed = True
                break
        if not placed:
            bins += 1
            remaining.append(window - length)
            # Ascending, so the scan above is best-fit; drop the TIGHTEST bins when
            # over the cap, never the roomiest -- discarding those packs nothing.
            remaining.sort()
            del remaining[:-open_bins]
    return bins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="From measure_poziomka_lengths.py")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-length", type=int, default=16384)
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--seconds-per-iteration", type=float, default=528.0,
                        help="Measured at this seq-length and batch size by the smoke sweep")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Set to the survey's --sample-every so totals scale back up")
    parser.add_argument("--long-policy", choices=("truncate", "drop"), default="truncate",
                        help="What happens to records longer than the window")
    args = parser.parse_args()

    data = np.load(args.npz)
    lengths = data[f"{args.split}_lengths"].astype(np.int64)
    window = args.seq_length + 1  # encode_record emits one extra next-token target
    scale = args.sample_every

    overlong = int((lengths > window).sum())
    if args.long_policy == "truncate":
        effective = np.minimum(lengths, window)
    else:
        effective = lengths[lengths <= window]

    records = len(effective)
    real_tokens = int(effective.sum())
    unpacked_positions = records * args.seq_length

    nf = next_fit(effective, window)
    ffd = first_fit_decreasing(effective, window)

    def line(name, sequences):
        iterations = int(np.ceil(sequences / args.global_batch_size))
        days = iterations * args.seconds_per_iteration / 86400
        efficiency = 100.0 * real_tokens / max(sequences * args.seq_length, 1)
        print(f"  {name:<28} {sequences * scale:>12,} {iterations * scale:>10,} "
              f"{efficiency:>9.1f}% {days * scale:>10.1f}")

    print(f"\n{args.split}: {records * scale:,} records, {real_tokens * scale:,} real tokens "
          f"at --seq-length {args.seq_length:,}")
    if overlong:
        print(f"  {overlong * scale:,} records exceed the window ({args.long_policy})")
    if scale > 1:
        print(f"  scaled up from a 1-in-{scale} sample")
    print(f"\n  {'strategy':<28} {'sequences':>12} {'iterations':>10} {'padding':>10} {'days':>10}")
    print(f"  {'-' * 28} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")
    line("no packing (today)", records)
    line("packed, next-fit", nf)
    line("packed, first-fit-decreasing", ffd)

    print(f"\n  Speedup: {records / max(nf, 1):.2f}x with next-fit, "
          f"{records / max(ffd, 1):.2f}x with first-fit-decreasing.")
    print("  Per-step cost is unchanged by packing, so the speedup is exactly the")
    print("  drop in sequence count. Block-diagonal attention is cheaper than full")
    print("  causal over the same window, so seconds-per-iteration should not rise.")
    print(f"  Unpacked today burns {unpacked_positions * scale:,} padded positions to "
          f"train on {real_tokens * scale:,} real tokens.")


if __name__ == "__main__":
    main()
