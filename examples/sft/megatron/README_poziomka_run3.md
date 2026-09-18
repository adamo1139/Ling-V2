# Poziomka SFT run 3: 16384 tokens, packed, truncated

Run 3 continues from run 2's weights at a doubled context length, with sequence
packing and a raised RoPE base. `README_poziomka.md` still describes runs 1 and 2
and remains accurate for them; every behaviour changed here is opt-in, so those
runs replay unchanged.

```bash
bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run3.sh
```

## What changed, and why

| | run 2 | run 3 |
|---|---|---|
| Sequence length | 8,192 | 16,384 |
| RoPE base | 84,000 | 640,000 |
| Long policy | `remove-reasoning` | `truncate` |
| Packing | none | next-fit, 78.6% efficiency |
| Sequences per epoch | 1,318,934 | 410,792 |
| Iterations (batch 768) | 1,718 | 535 |
| Tokens trained on | 2.72 B (37.9% of corpus) | 5.29 B (73.7%) |
| Starting weights | `linear-8-9-10-11-sqrt` | run 2 @ iter 1718 |
| Wall clock | ~10.5 days at 16k unpacked | ~3 days |

Roughly twice the training signal in under a third of the steps. Learning rate,
batch size, optimizer, router settings and full recomputation are unchanged.

## The corpus, measured

`measure_poziomka_lengths.py` over the v11 export (1-in-20 sample, matched the
manifest totals to 0.3%):

```
mean 5,417   p50 1,904   p90 13,388   p95 24,809   p99 55,120   max 108,776
reasoning: 76.7% of all tokens
```

A short-conversation corpus with a very heavy reasoning tail. That shape drives
every decision below: the median record wastes most of a 16k window, while the
tail cannot fit one at all.

## Why truncate, not remove-reasoning

Run 2 trained on 37.9% of its own corpus. `remove-reasoning` deleted
4,430,322,650 tokens -- 62% of the token mass, and 80.6% of all reasoning.

The cause is structural, not a tuning problem. `compare_poziomka_selectors.py`
measured the reasoning blocks per overlong record:

```
--seq-length 8192     4,337 overlong: 99.5% have exactly one reasoning block
--seq-length 16384    2,208 overlong: 100%  have exactly one reasoning block
reasoning retained within overlong records: 0.0%
```

Overlong records are single-reasoning-turn conversations carrying one enormous
trace. Removing whole blocks is therefore all-or-nothing on them: the policy
strips 100% of their reasoning and leaves a thinking-off example whose answer
required tens of thousands of tokens of reasoning the model never sees. That
teaches confident unsupported answers on precisely the hardest questions.

Truncation keeps the first 16,384 tokens of those records: real reasoning, at the
cost of an unclosed `<think>`, no final answer and no `<|im_end|>` on 108,116
records (8.2%). That is a diffuse cost against a sharp one, so run 3 truncates.

`drop` remains available and is the only policy with no misleading signal at all,
at the cost of losing the longest 8% of records entirely.

### Built cache

```
input_records            1,318,934      input_tokens      7,174,679,373
records                  1,318,934      tokens            5,291,364,775   (73.7%)
overlong/truncated         108,116      discarded         1,883,314,598
validation                  13,401      truncated             1,082
```

### The selector is not a lever here

`prepare_poziomka_sft.py` gained `--reasoning-selector smallest-sufficient`
(default), which sheds the smallest block that closes the gap rather than the
largest. On a corpus with multi-block records this retains substantially more
reasoning. On **this** corpus it changes nothing: with one block per overlong
record, both selectors make the identical removal, measured at exactly 0 tokens
difference at both 8,192 and 16,384. Kept because it is strictly better where it
applies; do not expect it to recover anything here.
`--reasoning-selector largest-first` reproduces pre-existing caches.

## Sequence length: 16,384

Measured by `smoke_poziomka_seqlen.sh` on 8x24,564 MiB, micro-batch 1, PP8/TP1/EP1,
full recomputation:

| seq-length | peak reserved | of capacity | result |
|---|---|---|---|
| 8,192 | 11,844 MiB | 48% | ok |
| 12,288 | 15,274 MiB | 62% | ok |
| **16,384** | **18,146 MiB** | **74%** | **ok** |
| 17,408 | 18,868 MiB | 77% | ok |
| 18,432 | 19,590 MiB | 80% | OOM |
| 20,480+ | -- | -- | OOM |

Note 18,432 reports 80% after iteration 1 and still OOMs later: the reported peak
is sampled early and understates fragmentation. Treat it as a floor, not a budget.

17,408 also fits but buys ~0.4% more records for 3% more memory and is not a power
of two, so 16,384 is the choice.

**Activation recomputation cannot be reduced.** Both `selective` and `none` OOM at
16,384, dying in the MoE token dispatcher (`moe_sort_chunks_by_index`) on a 2 GiB
permutation buffer. 128 experts at top-32 leave no room. `run_poziomka.sh` keeps
`--recompute-granularity full` hardcoded; do not re-litigate this.

Pipeline stages are evenly loaded (15.6-18.1 GiB across ranks), so an uneven
pipeline split would not help.

## RoPE base: 640,000

Per Xu et al., *Base of RoPE Bounds Context Length* (NeurIPS 2024), Table 2 gives
the minimum base for a target context:

| context | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|---|
| min base | 4.3e3 | 1.6e4 | 2.7e4 | **8.4e4** | **3.1e5** | **6.4e5** | 2.1e6 | 7.8e6 |

**The pretrained base of 84,000 is exactly the 8k bound.** It was provisioned for
the 8,192 pretrain with zero headroom, so training at 16,384 without raising it
puts the model below the bound, where long-context ability is superficial: low
perplexity, poor long-range retrieval.

Run 3 uses 640,000 -- the 32k row -- so a later 32,768 run needs no second base
change and no second adaptation of the weights.

Expect an elevated loss for the first iterations: a 7.6x base change shifts every
position encoding and the weights must re-adapt. This is normal for context
extension and is not a bug.

**The HF export must use the same base.** `tools/convert_dcp_to_safetensors_apt4.py`
writes `rope_theta` from `args.rotary_base` and `max_position_embeddings` from
`args.seq_length`. Exporting run 3 with the defaults produces a config claiming
84,000 and 8,192, and inference then disagrees with training silently. The same
applies to any GGUF built downstream.

## Packing

With p50 = 1,904 against a 16,384 window, one conversation per sequence wastes
~76% of every step. Packing places several conversations in one window, isolated
by `thd` (variable-length) attention.

Packing is a **training-time view of the existing cache**, not a new on-disk
format: record lengths come from the offsets already stored, so no rebuild is
needed and the same cache serves both modes.

- The bin plan is computed once from lengths alone with a seeded next-fit pass, so
  every rank and dataloader worker derives an identical plan without communication.
  Per-epoch shuffling reorders *bins*, not records, keeping the epoch length fixed.
- `position_ids` restart per conversation. (Megatron derives packed RoPE positions
  from `cu_seqlens` rather than `position_ids`, so this is for consistency.)
- Trailing padding is its own sub-sequence, so `cu_seqlens` covers the whole window
  and there is no padded/unpadded special case.
- `cu_seqlens` has a fixed shape, padded with repeats, so it collates and
  broadcasts; `packed_forward_step` trims the tail before building `PackedSeqParams`.
- Every pipeline stage builds the dataset when packing, because attention runs on
  all of them and middle stages otherwise have no `cu_seqlens`. Safe here because
  DP=1, TP=1 and the dataset is deterministic.
- Requires micro-batch 1: Megatron drops the batch dimension for `thd`
  (`attention.py:665`). Enforced in `datasets_provider`.

**Next-fit, deliberately.** First-fit-decreasing packs ~9 points tighter but sorts
by length, grouping the longest conversations together and correlating batch
composition with length. Not worth it for 0.4 days.

### Verification

| check | result |
|---|---|
| `probe_poziomka_thd.py` | thd works under the launcher's `NVTE_*` settings; zero leak across boundaries |
| `test_poziomka_packing.py` | 13 CPU tests: content identical to the unpacked reader, positions restart, padding masked, plan deterministic |
| `parity_poziomka_packing.py` | packed vs unpacked logits identical (0.000000, 100% top-1) |
| same, `--negative-control` | boundaries removed: 6.3 max diff, 0% top-1 on conversations 1-3 |
| smoke run, `PACKING=1` | real 128-expert MoE steps cleanly, peak unchanged at 18,146 MiB |

The negative control matters: a parity pass proves nothing unless the same harness
fails when packing is wrong. It does.

## Reproducing runs 1 and 2

Every change is opt-in and defaults to the previous behaviour:

| setting | default | run 3 |
|---|---|---|
| `ROTARY_BASE` | 84000 | 640000 |
| `MAX_POSITION_EMBEDDINGS` | 8192 | 16384 |
| `PACKING` | 0 | 1 |
| `--reasoning-selector` | smallest-sufficient | n/a (truncate) |

With `PACKING` unset, `choose_forward_step` calls `pretrain_gpt.forward_step`
unchanged, `datasets_provider` takes the original branch, and the dataset uses the
original read path. A regression test pins the unpacked reader's behaviour.

Caches record `reasoning_selector` in their manifest. To rebuild a pre-existing
cache exactly, pass `--reasoning-selector largest-first`.

## Rebuilding from scratch

```bash
# 1. Cache (truncate, 16384)
python3 Ling-V2/examples/sft/megatron/prepare_poziomka_sft.py \
  --input <v11-export> --tokenizer poziomka-linear-8-9-10-11-sqrt \
  --output poziomka-sft-cache-v11-16384-truncate \
  --seq-length 16384 --long-policy truncate --workers 15

# 2. Exact iteration count, offline: the plan is deterministic from lengths + seed
python3 -c "
import sys, math, os
home = os.path.expanduser('~/projects/pretrain')
sys.path.insert(0, os.path.join(home, 'Ling-V2/examples/sft/megatron'))
from poziomka_data import MMapSFTDataset
d = MMapSFTDataset(os.path.join(home, 'poziomka-sft-cache-v11-16384-truncate/manifest.json'),
                   'train', pack=True, seed=42)
print(d.sample_count, math.ceil(d.sample_count / 768))
"

# 3. Set TRAIN_ITERS in run_poziomka_sft_run3.sh, then launch
bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run3.sh
```

The launcher refuses a cache whose `seq_length` or `long_policy` does not match.

### At startup, confirm

```
SFT: 1318934 train conversations packed into 410792 sequences (78.6% ...)
```

A different sequence count means the plan is not reproducing and `TRAIN_ITERS` is
wrong. Stop rather than spend three days.

Then: loss elevated on the first iterations (RoPE change, recovers), peak reserved
~18,146 MiB on rank 7.

## Analysis tools added

| script | purpose | GPU |
|---|---|---|
| `measure_poziomka_lengths.py` | length percentiles, histogram, per-window coverage | no |
| `simulate_poziomka_packing.py` | bin-packs a measured distribution to size the payoff | no |
| `compare_poziomka_selectors.py` | reasoning retained by each selector, blocks per record | no |
| `smoke_poziomka_seqlen.sh` | sequence-length memory/throughput sweep | 8 |
| `probe_poziomka_thd.py` | is thd attention available and isolating? | 1 |
| `parity_poziomka_packing.py` | packed vs unpacked output parity, with negative control | 1 |

All are read-only with respect to the corpus, caches and checkpoints.

## Open questions

- **Long-context ability is untested.** 640,000 satisfies the bound for 16,384,
  but nothing here measures retrieval or perplexity-versus-position. A
  needle-style evaluation after run 3 would be the first real check.
- **Run 2's thinking-off examples persist in the weights.** Run 2 trained 212,316
  conversations as thinking-off whose answers required long reasoning. Run 3 shows
  those same conversations with reasoning intact, distinguished by the empty
  `<think>` prefix, so the model should end up with both behaviours selectable at
  inference -- but run 3 is partly un-teaching run 2 rather than building on it.
- **8.2% of records end mid-reasoning** with no answer and no EOS. Watch for
  non-termination in evaluation.
- **32,768 does not fit** on 24 GB cards with this configuration. Reaching it needs
  context parallelism (currently rejected by `datasets_provider`), or sharding the
  MoE experts, neither of which is implemented.
