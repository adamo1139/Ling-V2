# Poziomka native Megatron SFT

This is a separate Poziomka entry point. The original `run.sh` and Ling-mini
ModelOpt SFT example remain unchanged. Use the **same working Ling-patched
Megatron core_v0.13.0 / Transformer Engine / FlashAttention environment used for
pretraining**, not an unpatched Megatron installation or the Hopper-only example
Docker build. No HF Trainer, LoRA, DeepSpeed or DeepEP is used here.

## What changed

- `poziomka_model_args.sh`: shared SFT/converter architecture: 16 layers, 128
  total experts, **32 selected per token**, expert width 320, APT4 vocabulary
  32000, RoPE base 84000 / fraction 0.5, PP8 / TP1 / EP1.
- `poziomka_chatml.jinja`: same rendered text as the cleanup template, with
  optional generation blocks for role-specific loss. By default **all real
  conversation tokens** are targets: headers, system/user/assistant/tool content,
  reasoning, tool calls, separators and `<|im_end|>`. Only added padding is masked;
  the initial BOS has no preceding prediction position. No injected system prompt.
  BOS=1, PAD=2, EOS=4; no vocabulary changes.
- `prepare_poziomka_sft.py`: streams all input shards, 15 worker processes by
  default, stores uint16 tokens / uint8 masks / uint64 offsets. Each worker
  performs a complete output scan. The final manifest is written only after
  every shard passes. No corpus-sized Python object lists or rank-local copies.
- `poziomka_data.py`: read-only mmap cache, deterministic per-epoch shuffle,
  exact next-token label/mask alignment and right padding. The Megatron sampler
  handles data parallel sampling; first/last pipeline stages see identical data.
- `train_poziomka_sft.py`: reuses `pretrain_gpt` model provider, forward step,
  pipeline-aware batch transfer and loss. Only the dataset provider changes.
- `tools/convert_hf_to_dcp.sh`: shares top-32 architecture flags, quotes paths,
  refuses existing output and leaves P2P settings alone. The Python importer
  rejects incompatible HF architecture/router/RoPE settings before copying weights.

## Deliberate first-version choices

One conversation per sequence, **no packing**. This wastes some padding compute
but prevents cross-conversation attention without custom attention kernels.
No reset on `im_end`: it is a turn boundary, not a document boundary.

Default sequence length is **3072**, matching the Poziomka 11 pretraining script.
It is configurable up to 8192; cache and training lengths must match. Default
`--long-policy truncate` keeps the prefix, counts every truncated conversation
and discarded supervised token, and never invents an EOS. Prefixes with no
selected targets are dropped. `drop` and `error` are alternatives. This is a
training-view decision only; the cleaned source corpus is not changed.
The corpus has many long conversations: inspect the manifest's truncation
counts before committing to a long run. Increasing the length needs a GPU
memory smoke test; successful pretraining at 3072 does not establish 8192 fit.

BF16 full-parameter SFT, microbatch 1, global batch 128, Adam, constant LR 3e-4
(configurable with `LR`, minimum LR set to the same value), no warmup by default,
full layer recomputation. Loss is normalized by supervised tokens. Native fused cross-entropy is used:
affected older TE cross-entropy kernels apply only the first mask value to all
token gradients, which is incorrect even when only padding is masked.
The LR is the user-selected starting point, matching Poziomka 8–11 pretraining.
The batch default remains 128; use `GLOBAL_BATCH_SIZE=768` to match the Poziomka
11 script's global batch. With microbatch 1 and PP8/TP1/EP1 on eight GPUs,
this increases gradient accumulation, not the per-microbatch sequence count.
Reassess the LR if the effective batch differs substantially.
Router weights remain trainable; the expert-balancing bias **buffer** update
rate defaults to zero to retain the base routing correction during initial SFT.
Set `ROUTER_BIAS_UPDATE_RATE=1e-3` to restore the Poziomka 11 update rate.
No expert capacity cap/token dropping is enabled. These are conservative starting
settings, not measured throughput/quality optima.

## 1. CPU tests (no model weights)

From `dataset-cleanup/`:

```bash
python3 Ling-V2/examples/sft/megatron/test_poziomka_sft.py \
  --tokenizer poziomka-linear-8-9-10-11-sqrt \
  --reference-template templates/poziomka_chatml.jinja -v
```

Preprocessing needs numpy and a recent transformers version supporting
`return_assistant_tokens_mask` (tested locally with 4.57.3). Parquet input also
needs pyarrow. No model implementation is loaded and no remote code executes.

## 2. Prepare the whole corpus (not launched automatically)

Choose **either** the JSONL directory below or `repacked-dataset-100/parquet`,
not both copies. The input directory must contain `train/` and `validation/`.

```bash
python3 Ling-V2/examples/sft/megatron/prepare_poziomka_sft.py \
  --input repacked-dataset-100/jsonl \
  --tokenizer poziomka-linear-8-9-10-11-sqrt \
  --output poziomka-sft-cache-3072 \
  --workers 15 --seq-length 3072 --long-policy truncate \
  --loss-roles all
```

The default `--loss-roles all` supervises every real next-token target. Optional
`--loss-roles assistant` or `--loss-roles user assistant` retain body-only
objectives; they require separate caches. Generation annotations do not affect
rendered text, and full-conversation mode does not use their masks.

**Rebuild old caches:** this preparer writes `poziomka-sft-v2`; training rejects
v1 caches. Prepare from the original corpus into a fresh directory and point
`SFT_DATA` there. Do not just rename the format in an old manifest: its masks and
retained rows reflect the previous objective. Recompute `TRAIN_ITERS` from the
new record count. Restart the diagnostic run from the original base weights
into a fresh output directory, rather than continuing degraded SFT weights.

Output must not already exist. A failure leaves incomplete files for inspection,
without a completed manifest; use a fresh output directory on retry. The cache
is portable: paths used by training are relative to its manifest. Copy the whole
cache, including `tokenizer/` and `chat_template.jinja`, to the training machine.
Token payload is three bytes per retained token plus eight bytes per record
offset, before filesystem overhead (roughly 21 GB upper bound for 7B tokens).

Optional full readback, including SHA-256 checks, after copying:

```bash
python3 Ling-V2/examples/sft/megatron/prepare_poziomka_sft.py \
  --verify poziomka-sft-cache-3072/manifest.json
```

The initial preparation also scans the entire output, not a sample. JSON parse,
tokenization, template or mask failures abort rather than silently reject rows.

`--unencodable-policy drop` optionally skips rows raising encoding/validation
`ValueError`s, recording counts and up to twenty examples per shard. It covers
validation errors beyond literal control tokens; unexpected programming errors
still abort. In full-conversation mode, literal special-token IDs in real text
are supervised normally and are not rejected merely for being special tokens.
Review dropped-row counts before training.

## 3. Import the exact merged checkpoint, if necessary (8 GPUs)

Do not substitute an older Poziomka 11 DCP for the `linear-8-9-10-11-sqrt` merge.
If that exact merged checkpoint already has a validated Megatron DCP, reuse it.
Otherwise, from `dataset-cleanup/`, in the working GPU environment:

```bash
export MEGATRON_PATH=/absolute/path/to/working/Megatron-LM-core_v0.13.0
bash Ling-V2/tools/convert_hf_to_dcp.sh \
  poziomka-linear-8-9-10-11-sqrt poziomka-merged-dcp 1
```

The existing importer loads HF weights one rank at a time to bound host memory.
It supports TP=EP=1 and equal-size pipeline stages. GPU conversion and HF/DCP
numerical parity have **not** been exercised here; verify before a full run.

## 4. Explicit GPU smoke test, then training

Example **not executed**:

```bash
SFT_DATA=/absolute/path/to/poziomka-sft-cache-3072 \
LOAD_CHECKPOINT=/absolute/path/to/poziomka-merged-dcp \
SAVE_CHECKPOINT=/absolute/path/to/poziomka-sft-smoke \
TRAIN_ITERS=2 GLOBAL_BATCH_SIZE=16 EVAL_ITERS=2 SAVE_INTERVAL=2 \
bash Ling-V2/examples/sft/megatron/run_poziomka.sh
```

Check all eight ranks initialize, finite loss/gradients, peak memory, validation,
and checkpoint save/reload. Then start the actual run into a **new** output from
the merged base, choosing `TRAIN_ITERS` deliberately. At global batch 128, one
pass is approximately `ceil(manifest.totals.train.records / 128)` steps; the
final batch wraps if needed. Epoch order reshuffles deterministically. Validation
uses the original held-out split and no invented test split. Each evaluation
uses `EVAL_ITERS * GLOBAL_BATCH_SIZE` samples, advancing through that split and
cycling if required; `EVAL_ITERS=10` is not a full validation-corpus scan.

Optional environment settings: `SEQ_LENGTH`, `GLOBAL_BATCH_SIZE`, `LR`,
`WARMUP_ITERS`, `SAVE_INTERVAL`, `EVAL_INTERVAL`, `EVAL_ITERS`,
`DATALOADER_WORKERS` (default 2 per dataset-building rank).
Additional Megatron options can be appended to the launcher command.

Checkpoints are weight-only (`--no-save-optim --no-save-rng --async-save`), as in
Poziomka 8-11 pretraining. With DP=1 the distributed optimizer shards nothing
across ranks, so saving Adam state means roughly 49 GB of host copies and the OOM
killer takes a rank on a 94 GB machine. Megatron writes `opt_param_scheduler`
inside the same `no_save_optim` guard, so scheduler state is not stored either.

To continue an interrupted run, set `RESUME=1` and point **both** `LOAD_CHECKPOINT`
and `SAVE_CHECKPOINT` to the same SFT output directory; do not use it for starting
SFT from the base checkpoint. Resume restores the weights and the iteration and
sample counters, so the data order continues, but Adam moments restart from zero.
Constant LR with no warmup means no schedule position is lost; expect a short
re-warming of the moment estimates. Keep the same cache, seed, batch size,
sequence length and model configuration. Without `RESUME=1`, SFT refuses an
existing output directory. Original pretraining scripts/checkpoints are untouched.

## Rigga run configuration

After rebuilding the cache, launch from `~/projects/pretrain`:

```bash
bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run1.sh
```

All run-specific settings are assigned in that script: the v2 cache beside the
checkout, original merged DCP, fresh NVMe `poziomka_sft_run2` output, `RESUME=0`,
1326 iterations, batch 768, sequence length 3072, LR 3e-4, and zero warmup.
Edit the script to change these settings; no environment exports are required.
1326 is a fixed budget of 1,018,368 samples, not a promise of exactly one epoch
on the rebuilt cache. The underlying launcher uses native cross-entropy.
