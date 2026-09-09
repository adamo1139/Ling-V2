# Poziomka native Megatron SFT — fun-rp-v11

This is a separate Poziomka entry point. The original `run.sh` and Ling-mini
ModelOpt SFT example remain unchanged. Use the **same working Ling-patched
Megatron core_v0.13.0 / Transformer Engine / FlashAttention environment used for
pretraining**, not an unpatched Megatron installation or the Hopper-only example
Docker build. No HF Trainer, LoRA, DeepSpeed or DeepEP is used here.

## Components

- `poziomka_model_args.sh`: shared SFT/converter architecture: 16 layers, 128
  total experts, **32 selected per token**, expert width 320, APT4 vocabulary
  32000, RoPE base 84000 / fraction 0.5, PP8 / TP1 / EP1.
- `poziomka-fun-rp-v11/chat_template.jinja`: use this explicitly for v11;
  the preparer still defaults to the older local `poziomka_chatml.jinja`.
  The v11 template preserves source reasoning and supports Qwen-style generation
  prefixes, with generation blocks for role-specific loss. By default **all real
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

## v11 data and hybrid reasoning

Use the corrected `poziomka-fun-rp-v11` export: **1,318,934 training conversations**
and **13,401 validation conversations**, in 99 training Parquet files and one
validation file. Both fun v9 and RP are mixed in every file. Do not also load
v9, RP or v10 alongside v11: they are already included.

All **1,027,979 assistant messages with source reasoning** retain it in
`messages[].reasoning_content`. System prompts and user messages are unchanged.
Of all conversations, 1,199,102 retain the v10 format and 133,233 are selected
for the hybrid format. In selected conversations, each assistant message without
reasoning already has a single empty `<think>\n</think>\n` prefix in `content`.
The supplied template renders existing reasoning and leaves these prefixes intact.
Do not strip reasoning, add another empty block, or prepend user commands.

The encoder consumes `messages` and JSON-decoded `tools`; provenance and other
metadata are not appended to the training text. `hybrid_format` records the 10%
selection. `reasoning_profile` describes the source conversation (`on`, `off`,
`mixed`); it is **not a command to rewrite every turn**. A mixed conversation
keeps both its reasoning and non-reasoning answers. No special metadata handling
or model architecture change is needed in the trainer.

Training uses `add_generation_prompt=False` and preserves reasoning in every
historical message. For inference with the **v11 tokenizer/template**:

| `enable_thinking` | Prefix after the assistant header |
|---|---|
| Omitted | No extra prefix, as in v10; the model chooses the continuation. |
| `True` | Open `<think>\n`. |
| `False` | Empty closed `<think>\n</think>\n`. |

This setting controls only the next answer. It does not erase historical reasoning
or insert instructions into user/system messages. Carry the cache's tokenizer and
`chat_template.jinja` into inference/export; selecting a dataset does not update
an independently loaded model tokenizer. Prefix handling has CPU test coverage;
model quality and switching reliability still require evaluation after training.

## Deliberate first-version choices

One conversation per sequence, **no packing**. This wastes some padding compute
but prevents cross-conversation attention without custom attention kernels.
No reset on `im_end`: it is a turn boundary, not a document boundary.

This v11 run uses **8192 tokens**, the default for both preparation and training.
Cache and training lengths must match. Default
`--long-policy truncate` keeps the prefix, counts every truncated conversation
and discarded supervised token, and never invents an EOS. Prefixes with no
selected targets are dropped. `drop` and `error` are alternatives. This is a
training-view decision only; the cleaned source corpus is not changed.
The corpus has many long conversations: inspect the manifest's truncation
counts before committing to a long run. Run the GPU smoke test at 8192 too.

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

The command above tests the original trainer template against its reference.
Also run the v11-specific tests from the cleanup workspace:

```bash
python3 scripts/test_v11.py -v
```

These cover exact v10 rendering for unselected conversations, preservation of all
reasoning, mixed-turn on/off prefixes, actual training-adapter masks, and a small
Parquet export. Neither command prepares the full corpus or launches training.

Preprocessing needs numpy and a recent transformers version supporting
`return_assistant_tokens_mask` (tested locally with 4.57.3). Parquet input also
needs pyarrow; the v11 export tests also need duckdb. No model implementation
is loaded and no remote code executes.

## 2. Prepare v11 (not launched automatically)

Run from `dataset-cleanup/`, with the complete `poziomka-fun-rp-v11/` directory
beside `Ling-V2/`. The input must contain `train/` and `validation/`.

**Length policy is a separate data-retention decision.** The example below retains
the 8192-token prefix truncation policy. It can discard
reasoning and final-answer tokens beyond the limit; v11's lossless construction
does not make this token cache lossless. Use `--long-policy error` instead if no
truncation is acceptable: preparation will abort on an overlong conversation.
8192 is the current maximum supported sequence length; longer conversations
can still exceed it. The current preparer has no lossless windowing or
packing path for arbitrarily long conversations.

```bash
python3 Ling-V2/examples/sft/megatron/prepare_poziomka_sft.py \
  --input poziomka-fun-rp-v11 \
  --tokenizer poziomka-fun-rp-v11/tokenizer \
  --chat-template poziomka-fun-rp-v11/chat_template.jinja \
  --output poziomka-sft-cache-v11-8192-all \
  --workers 15 --seq-length 8192 --long-policy truncate \
  --loss-roles all
```

The default `--loss-roles all` supervises every real next-token target. Optional
`--loss-roles assistant` or `--loss-roles user assistant` retain body-only
objectives; they require separate caches. Generation annotations do not affect
rendered text, and full-conversation mode does not use their masks.

**Build a fresh v11 cache:** do not reuse v9/v10 caches or a cache from the discarded
v11 variant that removed reasoning. The cache format remains `poziomka-sft-v2`;
that version describes the cache layout, not the dataset revision. An older v2
cache can load successfully while containing the wrong data/template. Training
rejects v1 caches. Prepare from the original corpus into a fresh directory and point
`SFT_DATA` there. Do not just rename the format in an old manifest: its masks and
retained rows reflect the previous objective. Recompute `TRAIN_ITERS` from the
new record count. For a fresh v11 diagnostic run, use the original base weights
and a fresh output directory; resuming an existing run is a separate operation
with the constraints described below.

Output must not already exist. A failure leaves incomplete files for inspection,
without a completed manifest; use a fresh output directory on retry. The cache
is portable: paths used by training are relative to its manifest. Copy the whole
cache, including `tokenizer/` and `chat_template.jinja`, to the training machine.
Token payload is three bytes per retained token plus eight bytes per record
offset, before filesystem overhead (roughly 21 GB upper bound for 7B tokens).

Optional full readback, including SHA-256 checks, after copying:

```bash
python3 Ling-V2/examples/sft/megatron/prepare_poziomka_sft.py \
  --verify poziomka-sft-cache-v11-8192-all/manifest.json
```

Before GPU training, inspect `manifest.json`: `totals.train` and
`totals.validation` record `input_records`, retained `records`, `overlong_records`,
`truncated_records`, any dropped-record counts, and `discarded_supervised_tokens`.
With the default encoding error policy, input counts should be 1,318,934 and
13,401. Review token losses explicitly; do not infer them from retained record
counts alone. `template_sha256` identifies the actual cached template.

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
SFT_DATA=/absolute/path/to/poziomka-sft-cache-v11-8192-all \
LOAD_CHECKPOINT=/absolute/path/to/poziomka-merged-dcp \
SAVE_CHECKPOINT=/absolute/path/to/poziomka-sft-smoke \
SEQ_LENGTH=8192 TRAIN_ITERS=2 GLOBAL_BATCH_SIZE=16 EVAL_ITERS=2 SAVE_INTERVAL=2 \
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

Run 1 used `run_poziomka_sft_run1.sh` at 3072 tokens and was stopped manually
after 400 iterations. That script is preserved with its original configuration
and planned iteration budget; 400 is the actual stopping point, not a new limit.
Its historical checkpoint directory is named `poziomka_sft_run2`; that existing
path is preserved despite the script being run 1.

Use the separate script below for the new v11 run 2.

`run_poziomka_sft_run2.sh` points to `poziomka-sft-cache-v11-8192-all`,
uses `SEQ_LENGTH=8192`, and saves to a fresh `poziomka_sft_run2_v11_8192` directory.
It retains the existing fixed budget of 1326 iterations at batch 768, LR 3e-4,
and zero warmup. Review `TRAIN_ITERS` against the completed cache if you want a
full pass rather than that fixed budget. The script assigns its own variables,
so caller environment exports do not override those settings.

After updating those settings, launch from `~/projects/pretrain`:

```bash
bash Ling-V2/examples/sft/megatron/run_poziomka_sft_run2.sh
```

The run starts from the original merged DCP with `RESUME=0` and uses native
cross-entropy. Edit the script to change its paths or run budget.
1326 iterations consume 1,018,368 samples. If all 1,318,934 training records are
retained, one pass at batch 768 needs 1,718 iterations; compute it again from the
completed cache if any records are dropped.
