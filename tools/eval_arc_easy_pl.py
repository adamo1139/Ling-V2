#!/usr/bin/env python3
"""ARC-Easy-PL accuracy of a Hugging Face causal LM, for the Fabryka model board (PL).

The track has no Polish ARC task, so this ports the EN board's ARC-Easy protocol (Glint
benchmark.py, as in gollem-v5/ref/glint_parity_eval.py) to the Polish translation:
dataset Eurolingua/arcx (formerly openGPT-X/arcx), file arc_easy_PL_test.jsonl (EU20, DeepL
translation of allenai/ai2_arc ARC-Easy test, same ids), every question, zero-shot. Each choice is scored
as the candidate `question + " " + choice` with LL(candidate) - LL(question); the highest
score wins and raw accuracy is reported (not acc_norm, no length normalization). LL(question)
is the same for every choice of a question, so it does not change the argmax and is not
computed. As in eval_multiblimp_pl.py, a tokenized model scores each candidate given only BOS.
A tie goes to the first choice, as Glint's max() does; the tie count is reported.

  python3 Ling-V2/tools/eval_arc_easy_pl.py cpral/Poziomka-Baza-2026-09-05 \
      --revision d3c131d4dd8ac92b77a470693ce14155739a706c --output arc_easy_pl.json

The JSON output pins the model and dataset revisions and the weights sha256, ready for
`board_pl/arc_easy` (accuracy in %). Chance is about 25 % (almost all questions have 4
choices). Each candidate is scored alone by default (--batch-size 1, no padding) for the
reason given in eval_multiblimp_pl.py: batching changes MoE routing in bf16.
"""
import argparse
import json
import os
import time

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_multiblimp_pl import sentence_logprobs, sha256

DATASET, DATA_FILE = "Eurolingua/arcx", "arc_easy_PL_test.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="HF repo id or local directory")
    parser.add_argument("--revision", help="HF commit of the model (pin it for the board)")
    parser.add_argument("--dataset-revision", help="Eurolingua/arcx commit; default: current")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="candidates per forward pass; >1 is faster but not reproducible for MoE")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, help="first N questions only (smoke test, not for the board)")
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

    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", revision=dataset_revision)
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    bos = tokenizer.bos_token_id
    encode = lambda text: [bos] + tokenizer(text, add_special_tokens=False)["input_ids"]

    started = time.time()
    correct = ties = 0
    for done, row in enumerate(rows, 1):
        sequences = [encode(row["question"] + " " + text) for text in row["choices"]["text"]]
        scores = []
        for first in range(0, len(sequences), args.batch_size):
            scores += sentence_logprobs(model, sequences[first:first + args.batch_size], pad_id, device)
        best = max(range(len(scores)), key=scores.__getitem__)
        correct += int(row["choices"]["label"][best] == row["answerKey"])
        ties += int(scores.count(scores[best]) > 1)
        if done % 250 == 0 or done == len(rows):
            print(f"{done}/{len(rows)} questions, accuracy so far {100 * correct / done:.2f}%", flush=True)

    weights = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    result = {
        "task": "arc_easy_polish",
        "accuracy_percent": round(100 * correct / len(rows), 4),
        "correct": correct, "questions": len(rows), "ties": ties,
        "full_dataset": not args.limit,
        "protocol": "Eurolingua/arcx arc_easy_PL test, zero-shot, Glint ARC-Easy scoring: "
                    "argmax sum logprob of question+' '+choice given BOS, raw accuracy",
        "dataset": DATASET, "dataset_file": DATA_FILE,
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
