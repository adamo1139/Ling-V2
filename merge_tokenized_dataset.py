import sys
# Add Megatron to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0')))


import glob, os, argparse
from megatron.core.datasets import indexed_dataset
from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args

parser = argparse.ArgumentParser()
parser.add_argument("--shard_glob", required=True,
                    help="Glob of shard .idx files, e.g. /data/*_text_document.idx")
parser.add_argument("--out_prefix", required=True,
                    help="Output prefix, e.g. /data/apt4_merged_text_document")
# tokenizer args so we can pick the same dtype as when shards were built
parser = _add_tokenizer_args(parser)
args = parser.parse_args()

# Build tokenizer only to recover vocab size -> dtype matches shards
tok = build_tokenizer(args)
dtype = indexed_dataset.DType.optimal_dtype(tok.vocab_size)

builder = indexed_dataset.IndexedDatasetBuilder(args.out_prefix + ".bin", dtype=dtype)

idx_paths = sorted(glob.glob(args.shard_glob))
assert idx_paths, "No shards found"
for p in idx_paths:
    prefix = p[:-4]  # drop .idx
    print("Merging:", prefix)
    builder.add_index(prefix)

builder.finalize(args.out_prefix + ".idx")
print("Wrote:", args.out_prefix + "_text_document.{bin,idx}")

"""
python3 merge_tokenized_dataset.py \
  --shard_glob "/home/ubuntu/preprocessing/Ling-V2/tokenized_apt4/*_text_document.idx" \
  --out_prefix "/home/ubuntu/preprocessing/Ling-V2/apt4_merged_text_document" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /home/ubuntu/preprocessing/Ling-V2/resource/tokenizer/config_pretrain

"""