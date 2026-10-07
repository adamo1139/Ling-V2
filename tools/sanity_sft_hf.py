#!/usr/bin/env python3
"""Quick health check of an HF Poziomka export against an SFT cache.

1. Per-token loss on the first N validation records of the cache, with the exact
   training tokens and loss masks. A healthy SFT model sits near 1.0-1.5 on
   polskie-sprawy-v3; the broken run 5 export scored 4.46, run 4 iter 100 scored 1.41.
2. Optional greedy samples through the model's own chat template, for enable_thinking
   omitted, False and True. For True it reports whether the model reasoned and closed
   </think> (with the v13 template the True prefill <think>\n precedes only real reasoning).

  python3 Ling-V2/tools/sanity_sft_hf.py <hf-dir> \
      --cache poziomka-sft-cache-polskie-sprawy-v3-16384 [--generate]
"""
import argparse
import glob
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["Jaka jest najlepsza pora na wymianę akumulatora w aucie?",
           "Napisz krótki artykuł na bloga o tym, jak klasa 4a pojechała do pszczelarza."]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--records", type=int, default=60)
    parser.add_argument("--rope-theta", type=float, help="Override config rope_theta")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    overrides = {"rope_theta": args.rope_theta} if args.rope_theta else {}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map="cuda",
                                                 **overrides).eval()

    prefix = sorted(glob.glob(os.path.join(args.cache, "validation", "*.tokens.bin")))[0][:-len(".tokens.bin")]
    tokens = np.fromfile(prefix + ".tokens.bin", dtype=np.uint16)
    masks = np.fromfile(prefix + ".masks.bin", dtype=np.uint8)
    offsets = np.fromfile(prefix + ".offsets.bin", dtype=np.uint64)
    total, count = 0.0, 0.0
    with torch.no_grad():
        for i in range(min(args.records, len(offsets) - 1)):
            start, end = int(offsets[i]), int(offsets[i + 1])
            ids = torch.tensor(tokens[start:end].astype(np.int64), device="cuda")[None]
            mask = torch.tensor(masks[start + 1:end].astype(np.float32), device="cuda")
            logits = model(ids, use_cache=False).logits[0, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none")
            total += float((nll * mask).sum())
            count += float(mask.sum())
    print(f"VAL per-token loss, {args.records} records, {int(count)} tokens: {total / count:.4f}")

    if not args.generate:
        return
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    for prompt in PROMPTS:
        for think in (None, False, True):
            extra = {} if think is None else {"enable_thinking": think}
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, **extra)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=200 if think is not True else 1200, do_sample=False,
                                     eos_token_id=end_id, pad_token_id=2)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
            print(f"\n===== enable_thinking={think} | {prompt}")
            if think is True:
                reasoning, closed, _ = text.partition("</think>")
                print(f"[reasoning chars: {len(reasoning.strip())}, </think> closed: {bool(closed)}]")
            print(text)


if __name__ == "__main__":
    main()
