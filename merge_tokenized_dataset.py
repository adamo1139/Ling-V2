#!/usr/bin/env python3
import sys, glob, os, argparse

# 1) Point to your Megatron checkout
MEGATRON_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0'))
sys.path.append(MEGATRON_HOME)

from megatron.core.datasets import indexed_dataset
from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args

def main():
    parser = argparse.ArgumentParser("Merge Megatron IndexedDataset shards")
    parser.add_argument("--shard_glob", required=True,
                        help='Glob of shard .idx files, e.g. "/data/shards/*_text_document.idx"')
    parser.add_argument("--out_prefix", required=True,
                        help='Output prefix, e.g. "/data/apt4_merged_text_document"')
    # tokenizer args (so dtype matches original shards)
    parser = _add_tokenizer_args(parser)

    args = parser.parse_args()

    # 2) Provide Megatron runtime defaults that build_tokenizer() expects
    # (these are harmless for this offline script)
    if not hasattr(args, "rank"): args.rank = 0
    if not hasattr(args, "make_vocab_size_divisible_by"): args.make_vocab_size_divisible_by = 128
    if not hasattr(args, "tensor_model_parallel_size"): args.tensor_model_parallel_size = 1
    if not hasattr(args, "vocab_extra_ids"): args.vocab_extra_ids = 0
    if not hasattr(args, "keep_empty"): args.keep_empty = False

    # 3) Collect shards
    idx_paths = sorted(glob.glob(args.shard_glob))
    if not idx_paths:
        raise RuntimeError(f"No shards matched: {args.shard_glob}")

    # 4) Build tokenizer JUST to choose dtype consistent with shards
    tok = build_tokenizer(args)  # uses --tokenizer-type / --tokenizer-model
    dtype = indexed_dataset.DType.optimal_dtype(tok.vocab_size)

    # 5) Build the merged dataset
    out_bin = args.out_prefix + ".bin"
    out_idx = args.out_prefix + ".idx"
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)

    builder = indexed_dataset.IndexedDatasetBuilder(out_bin, dtype=dtype)
    for p in idx_paths:
        prefix = p[:-4]  # drop ".idx"
        bin_path = prefix + ".bin"
        if not os.path.exists(bin_path):
            raise RuntimeError(f"Missing .bin for shard: {prefix}")
        print("Merging:", prefix)
        builder.add_index(prefix)

    builder.finalize(out_idx)
    print(f"Wrote merged files:\n  {out_bin}\n  {out_idx}")

if __name__ == "__main__":
    main()
