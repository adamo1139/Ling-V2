#!/usr/bin/env python3
"""MultiBLiMP-pl accuracy of a Hugging Face causal LM, for the Fabryka model board (PL).

Follows the track's protocol (fabryka-track src/fabryka_track/benchmark_worker.py,
task multiblimp_polish): dataset jumelet/multiblimp, config "pol", split "train", every
pair; a pair is correct when the full grammatical sentence `sen` has a strictly higher
total log-probability than `wrong_sen` (a tie counts as wrong). The track scores its
byte models with an empty context; for a tokenized model the equivalent is the sentence
log-probability given only BOS, which is what this script computes.

  python3 Ling-V2/tools/eval_multiblimp_pl.py cpral/Poziomka-Baza-2026-09-05 \
      --revision d3c131d4dd8ac92b77a470693ce14155739a706c --output multiblimp_pl.json

The JSON output pins the model and dataset revisions and the weights sha256, ready for
`board_pl/multiblimp` (accuracy in %) and the `leaderboard/result` attribute.

Each sentence is scored alone by default (--batch-size 1, no padding). Batching changes
the result for MoE models in bf16: a different batch shape changes rounding, near-tied
router scores flip to another expert, and on Poziomka 8 copies of one sentence in a batch
already differed from the single pass by up to 0.25 nats. Single-sentence scoring makes
the number reproducible; --dtype float32 (needs ~17 GB for Poziomka) also makes routing
less sensitive to rounding.
"""
import argparse
import hashlib
import json
import os
import time

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET, CONFIG, SPLIT = "jumelet/multiblimp", "pol", "train"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def sentence_logprobs(model, sequences, pad_id, device):
    """Total log-probability of each sequence[1:] given its prefix; sequence[0] is BOS."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, :len(seq)] = torch.tensor(seq)
        mask[row, :len(seq)] = 1
    ids, mask = ids.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
    token_lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (token_lp * mask[:, 1:]).sum(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="HF repo id or local directory")
    parser.add_argument("--revision", help="HF commit of the model (pin it for the board)")
    parser.add_argument("--dataset-revision", help="jumelet/multiblimp commit; default: current")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="sentences per forward pass; >1 is faster but not reproducible for MoE")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, help="first N pairs only (smoke test, not for the board)")
    parser.add_argument("--output", help="write the result JSON here")
    args = parser.parse_args()

    local = os.path.isdir(args.model)
    model_dir = args.model if local else snapshot_download(args.model, revision=args.revision)
    model_revision = None if local else HfApi().model_info(args.model, revision=args.revision).sha
    dataset_revision = args.dataset_revision or HfApi().dataset_info(DATASET).sha

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.bos_token_id is None:
        raise SystemExit("Tokenizer has no BOS token; the BOS-context protocol needs one")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, device_map=device,
        dtype=getattr(torch, args.dtype) if device == "cuda" else torch.float32).eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.bos_token_id

    rows = load_dataset(DATASET, CONFIG, split=SPLIT, revision=dataset_revision)
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))
    bos = tokenizer.bos_token_id
    encode = lambda text: [bos] + tokenizer(text, add_special_tokens=False)["input_ids"]

    started = time.time()
    correct = ties = 0
    pairs_per_chunk = 500
    for start in range(0, len(rows), pairs_per_chunk):
        chunk = rows[start:start + pairs_per_chunk]
        sequences = [encode(s) for pair in zip(chunk["sen"], chunk["wrong_sen"]) for s in pair]
        scores = []
        for first in range(0, len(sequences), args.batch_size):
            scores += sentence_logprobs(model, sequences[first:first + args.batch_size], pad_id, device)
        for good, bad in zip(scores[0::2], scores[1::2]):
            correct += int(good > bad)
            ties += int(good == bad)
        done = min(start + pairs_per_chunk, len(rows))
        print(f"{done}/{len(rows)} pairs, accuracy so far {100 * correct / done:.2f}%", flush=True)

    weights = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    result = {
        "task": "multiblimp_polish",
        "accuracy_percent": round(100 * correct / len(rows), 4),
        "correct": correct, "pairs": len(rows), "ties": ties,
        "full_dataset": not args.limit,
        "protocol": "jumelet/multiblimp pol train, sum logprob of full sentence given BOS, "
                    "correct iff sen > wrong_sen (strict)",
        "dataset": DATASET, "dataset_config": CONFIG, "dataset_split": SPLIT,
        "dataset_revision": dataset_revision,
        "model": args.model, "model_revision": model_revision,
        "weights_sha256": {f: sha256(os.path.join(model_dir, f)) for f in weights},
        "dtype": str(model.dtype), "batch_size": args.batch_size,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
