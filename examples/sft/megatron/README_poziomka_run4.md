# Poziomka SFT run 4: clean corpus probe, 100 steps

Run 4 re-runs run 3's setup on the loop-cleaned v12 corpus for 100 steps, to
confirm that the data -- not packing, learning rate, the RoPE base change or the
DCP->HF conversion -- caused run 3's degenerate looping. `README_poziomka_run3.md`
documents run 3 and stays accurate for it.

```bash
bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run4.sh
```

## What changed, and why

| | run 3 | run 4 |
|---|---|---|
| Corpus | v11 (loops present) | v12 (loops removed) |
| Iterations | 535 | 100 (probe) |
| Eval interval | 100 | 20 |
| Everything else | -- | identical |

Identical means: same starting weights (run 2 @ iter 1718), 16,384 window,
640,000 RoPE base, next-fit packing at batch 768, LR 3e-4 constant, full
recomputation. Run 4 is the controlled version of run 3: one variable changed.

## The v12 corpus

Built from v11 in two passes, both driven by `detect_reasoning_loops.py`
thresholds calibrated on the v11 validation split (see the detector's
docstring for the full calibration history):

1. **strip** (`strip_looped_reasoning.py --mode strip`): reasoning removed from
   29,101 messages whose reasoning was degenerate (28,820 train + 281
   validation). Removed reasoning follows the `prepare_poziomka_sft.py`
   convention: `reasoning_content = None`, a `\n\n\n` content prefix, and a
   recomputed `reasoning_profile`, so the trainer sees them as ordinary
   thinking-off turns.
2. **drop** (`--mode drop --drop-fields content`): 22,095 records whose
   *visible text* looped (21,887 train + 208 validation) deleted entirely --
   1.66% of train, 1.55% of validation. Loops in content sit in assistant
   answers and user prompts alike, so whole records went.

Measurement on the v12 validation split after both passes: 0 looped reasoning
texts, 0 looped content texts. The corpus is clean by the same instrument that
found the problem.

## What to compare against run 3

The first 100 iterations of run 3 are the control. Same starting weights, same
schedule, only the corpus differs.

- **Loss curve.** Run 3's loss looked *good* while the data was broken -- looped
  text is trivially predictable, so loss fell steadily. A healthy run 4 loss is
  expected to be *slightly higher* than run 3's at the same step: clean text is
  harder to predict than repetition. A run 4 loss dramatically *below* run 3's
  would be a red flag, not a win.
- **Validation loss** at iters 50 and 100, against run 3's validation curve.
- **Sampling probe at iter 100** (the real test): greedy generation from the
  run 4 checkpoint on a handful of prompts, checked for repetition loops. Run
  3's model produced "Zbieraja miod i miod, Zbieraja miod i miod, ..." from the
  same weights that showed a clean loss. Loss alone cannot certify this fix;
  only generation can.

## Building the cache

```bash
python3 Ling-V2/examples/sft/megatron/prepare_poziomka_sft.py \
  --input <v12-export> --tokenizer poziomka-linear-8-9-10-11-sqrt \
  --output poziomka-sft-cache-v12-16384-truncate \
  --seq-length 16384 --long-policy truncate --workers 10
```

Truncate stays the right long policy: v12 keeps v11's length shape (the removed
1.66% of records and stripped reasoning barely move the tail), so the overlong
records run 3 truncated are still overlong here, and truncation keeps their
reasoning prefixes. The launcher refuses a cache whose `seq_length` or
`long_policy` does not match.

At startup, confirm the packed sequence count is in the same ballpark as run
3's 410,792 (v12 has ~1.7% fewer records, so expect ~1-2% fewer sequences).
`TRAIN_ITERS=100` needs no epoch arithmetic: 100 x 768 = 76,800 packed
sequences, far below one epoch.

## Open questions

- **100 steps may be too few to show behavioural repair.** Run 3's degeneration
  came from 535 steps over looped data; 100 clean steps may not fully un-teach
  it. If the sampling probe still loops, the next experiment is a full-length
  run 4 (recompute `TRAIN_ITERS` from the built cache like run 3 did), not a
  revert of the data fix.
- **Run 2's thinking-off examples persist in the weights** (inherited from run
  3's starting point; unchanged by this run).
- **Long-context ability is untested** (inherited from run 3; a needle-style
  evaluation is still the first real check).