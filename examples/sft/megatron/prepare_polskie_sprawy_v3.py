#!/usr/bin/env python3
"""Turn polskie-sprawy-v3/sft.jsonl into the train/ + validation/ layout
that prepare_poziomka_sft.py reads.

- Validation: 1% of conversations, chosen by their (sha256) id, so the split
  is deterministic and independent of line order.
- --off-prefix legacy (default, run 5): every assistant message without
  reasoning gets '<think>\n</think>\n' in its content. With the v12 template
  this made ON and OFF identical up to the first reasoning token, so
  enable_thinking=True (prefill '<think>\n') stopped working: the model closed
  the block in 93% of the data and did so at inference too.
- --off-prefix none (run 6+, with poziomka_v13_chat_template.jinja): content is
  left untouched and the v13 template renders OFF as '<think></think>\n'. The
  token after <think> is then \n only before real reasoning.
- meta.thinking must agree with the presence of reasoning_content; any
  disagreement aborts instead of being guessed.

Usage:
  python3 prepare_polskie_sprawy_v3.py --input polskie-sprawy-v3/sft.jsonl \
      --output polskie-sprawy-v3-sft-v13 --off-prefix none
"""

import argparse
import json
from collections import Counter
from pathlib import Path

THINK_OFF_PREFIX = "<think>\n</think>\n"


def convert(record, off_prefix=THINK_OFF_PREFIX):
    thinking = record["meta"]["thinking"]
    messages = []
    for message in record["messages"]:
        message = dict(message)
        if message["role"] == "assistant":
            content = message.get("content") or ""
            if content.lstrip().startswith("<think>"):
                raise ValueError(f"{record['id']}: content already starts with <think>")
            has_reasoning = bool(message.get("reasoning_content"))
            if has_reasoning != (thinking == "on"):
                raise ValueError(f"{record['id']}: meta.thinking={thinking!r} but "
                                 f"reasoning_content present={has_reasoning}")
            if not has_reasoning:
                message["reasoning_content"] = None
                message["content"] = off_prefix + content
        messages.append(message)
    return {"id": record["id"], "messages": messages, "meta": record["meta"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-shards", type=int, default=16)
    parser.add_argument("--validation-percent", type=int, default=1)
    parser.add_argument("--off-prefix", choices=("legacy", "none"), default="legacy",
                        help="legacy: '<think>\\n</think>\\n' in content (run 5, v12 template); "
                             "none: leave content as is (v13 template renders OFF)")
    args = parser.parse_args()
    off_prefix = THINK_OFF_PREFIX if args.off_prefix == "legacy" else ""

    if args.output.exists():
        raise SystemExit(f"Refusing existing output: {args.output}")
    (args.output / "train").mkdir(parents=True)
    (args.output / "validation").mkdir()

    train = [(args.output / "train" / f"shard_{i:05d}.jsonl").open("x", encoding="utf-8")
             for i in range(args.train_shards)]
    validation = (args.output / "validation" / "shard_00000.jsonl").open("x", encoding="utf-8")
    stats = Counter()
    seen = set()
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            record = convert(json.loads(line), off_prefix)
            if record["id"] in seen:
                raise ValueError(f"Duplicate id {record['id']}")
            seen.add(record["id"])
            bucket = int(record["id"][:8], 16)
            text = json.dumps(record, ensure_ascii=False) + "\n"
            split = "validation" if bucket % 100 < args.validation_percent else "train"
            if split == "validation":
                validation.write(text)
            else:
                train[stats["train"] % args.train_shards].write(text)
            stats[split] += 1
            stats[f"{split}_thinking_{record['meta']['thinking']}"] += 1
    for stream in train + [validation]:
        stream.close()
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
