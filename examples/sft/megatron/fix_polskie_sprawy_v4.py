#!/usr/bin/env python3
"""Fix known defects in polskie-sprawy-v4-nc/sft.jsonl before publishing a new revision.

Writes the fixed dataset and, separately, every removed record with its reason, so
removed prompts can be regenerated later. Record order is preserved.

Fixes (counts on the 538,557-record file of 2026-10-09; output 538,079 records):
- think_tag_in_reasoning (removed, 54): reasoning_content contains a literal <think> or
  </think>. Rendered as <think>\\n{reasoning}\\n</think>, it nests or closes the block
  early. Mostly private/Qwen traces quoting the generation prompt ("Użyj <think> z min.
  1500 słów"); with </think> the reasoning also carries a draft answer after the tag.
- thinking_on_without_reasoning (removed, 424): Mistral Small 4 (reasoning_effort=high)
  answers whose reasoning was merged into content: garbage first characters ('zyła',
  'sunshine', 'uée'), content 2.5x longer than other Mistral answers, often cut off
  mid-word at the token limit.
- reasoning_on_thinking_off (fixed in place, 1): meta.thinking=off with a reasoning trace
  ("answer directly, without reasoning shown"); the answer is fine, the trace is dropped.

Reasoning that merely mentions the generation instructions without a think tag is kept.

  python3 fix_polskie_sprawy_v4.py polskie-sprawy-v4-nc/sft.jsonl --output-dir polskie-sprawy-v4-nc-fixed
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

THINK_TAG = re.compile(r"</?think>")


def classify(record):
    """Return (record_or_None, reason). None means: remove."""
    thinking = record["meta"]["thinking"]
    answer = record["messages"][-1]
    if answer["role"] != "assistant":
        raise ValueError(f"{record['id']}: last message is {answer['role']}, expected assistant")
    reasoning = answer.get("reasoning_content") or ""
    content = answer.get("content") or ""
    if THINK_TAG.search(content):
        raise ValueError(f"{record['id']}: think tag in content; not a known defect, inspect it")
    if reasoning and THINK_TAG.search(reasoning):
        return None, "think_tag_in_reasoning"
    if thinking == "on" and not reasoning:
        return None, "thinking_on_without_reasoning"
    if thinking != "on" and reasoning:
        answer["reasoning_content"] = None
        return record, "reasoning_on_thinking_off"
    return record, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    stats = Counter()
    with args.input.open(encoding="utf-8") as source, \
         (args.output_dir / "sft.jsonl").open("x", encoding="utf-8") as kept, \
         (args.output_dir / "removed.jsonl").open("x", encoding="utf-8") as removed:
        for line in source:
            record = json.loads(line)
            stats["input"] += 1
            fixed, reason = classify(record)
            if fixed is None:
                removed.write(json.dumps({"reason": reason, **record}, ensure_ascii=False) + "\n")
                stats[f"removed_{reason}"] += 1
                continue
            if reason:
                stats[f"fixed_{reason}"] += 1
            kept.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            stats["output"] += 1
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
