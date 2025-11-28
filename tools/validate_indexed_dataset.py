#!/usr/bin/env python3
"""
Sanity-check a Megatron indexed dataset (.bin/.idx pair).

This script ensures:
  * The index/header can be parsed.
  * Sequence pointers are monotonic.
  * Pointer + length never runs past the end of the .bin file.
  * Document indices are consistent with the number of sequences.
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np

from megatron.core.datasets import indexed_dataset


def _validate_paths(prefix: str) -> Tuple[str, str]:
    idx_path = f"{prefix}.idx"
    bin_path = f"{prefix}.bin"

    missing = [p for p in (idx_path, bin_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")

    return idx_path, bin_path


def validate_dataset(prefix: str, max_errors: int) -> bool:
    idx_path, bin_path = _validate_paths(prefix)
    bin_size = os.path.getsize(bin_path)

    # `_IndexReader` exposes the raw arrays without loading the entire dataset.
    index = indexed_dataset._IndexReader(idx_path, multimodal=False)  # pylint: disable=protected-access

    errors = []

    seq_lengths = index.sequence_lengths.astype(np.int64, copy=False)
    seq_pointers = index.sequence_pointers.astype(np.int64, copy=False)
    seq_bytes = seq_lengths * int(index.dtype_size)
    seq_end = seq_pointers + seq_bytes

    # Pointers must start at zero and be monotonically increasing.
    bad_pointer = np.where(seq_pointers[1:] < seq_pointers[:-1])[0]
    if bad_pointer.size:
        for idx_val in bad_pointer[:max_errors]:
            errors.append(f"Pointer decreased at sequence {idx_val} ("
                          f"{seq_pointers[idx_val]} -> {seq_pointers[idx_val + 1]})")

    # Lengths must be positive.
    bad_length = np.where(seq_lengths <= 0)[0]
    if bad_length.size:
        for idx_val in bad_length[:max_errors - len(errors)]:
            errors.append(f"Non-positive length at sequence {idx_val}: {seq_lengths[idx_val]}")

    # Pointers plus lengths must live within the .bin file.
    out_of_range = np.where((seq_pointers < 0) | (seq_end > bin_size))[0]
    if out_of_range.size and len(errors) < max_errors:
        budget = max_errors - len(errors)
        for idx_val in out_of_range[:budget]:
            errors.append(
                f"Sequence {idx_val} spans bytes [{seq_pointers[idx_val]}, {seq_end[idx_val]}) "
                f"outside .bin size {bin_size}"
            )

    # Document indices must be monotonic, start at 0, end at sequence_count.
    doc_idx = index.document_indices.astype(np.int64, copy=False)
    if doc_idx.size == 0 or doc_idx[0] != 0 or doc_idx[-1] != index.sequence_count:
        errors.append(
            "Document indices must start at 0 and end at the number of sequences "
            f"(got start={doc_idx[0] if doc_idx.size else 'N/A'} end={doc_idx[-1] if doc_idx.size else 'N/A'}, "
            f"expected 0 and {index.sequence_count})"
        )
    else:
        bad_doc = np.where(doc_idx[1:] < doc_idx[:-1])[0]
        if bad_doc.size and len(errors) < max_errors:
            budget = max_errors - len(errors)
            for idx_val in bad_doc[:budget]:
                errors.append(
                    f"Document boundary decreased at position {idx_val} "
                    f"({doc_idx[idx_val]} -> {doc_idx[idx_val + 1]})"
                )

    del index  # ensure memmaps are closed before exiting

    if errors:
        print("Dataset validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        if len(errors) < out_of_range.size:
            print(f"  ... {out_of_range.size - len(errors)} additional errors not shown")
        return False

    print("Dataset validation PASSED")
    print(f"  Prefix        : {prefix}")
    print(f"  Documents     : {len(doc_idx) - 1}")
    print(f"  Sequences     : {len(seq_lengths)}")
    print(f"  Binary size   : {bin_size} bytes")
    print(f"  DType         : {str(index.dtype)} (itemsize={index.dtype_size})")
    print(f"  Max seq tokens: {int(seq_lengths.max())}")
    print(f"  Max seq bytes : {int(seq_bytes.max())}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Megatron indexed datasets before launching training runs."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path prefix to dataset files (without .bin/.idx).",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of detailed errors to print.",
    )
    args = parser.parse_args()

    try:
        ok = validate_dataset(args.data_path, max(args.max_errors, 1))
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Dataset validation FAILED: {exc}")
        sys.exit(1)

    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
