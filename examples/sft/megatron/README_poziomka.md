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
  generation blocks covering **user and assistant** content, reasoning, tool
  calls and `<|im_end|>`. No injected system prompt. System messages and tool
  results are context-only; role headers and padding are also masked.
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
full layer recomputation. Loss is normalized by supervised tokens.
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
  --loss-roles user assistant
```

The default loss roles are `user assistant`. To opt into assistant-only loss,
prepare a separate cache with `--loss-roles assistant`. The manifest records
the choice; training consumes those exact masks, with no additional filtering.
The Hugging Face API calls these masks `assistant_masks`, but our template's
generation blocks deliberately cover both selected roles by default.

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

For a genuine continuation, set `RESUME=1` and point **both** `LOAD_CHECKPOINT`
and `SAVE_CHECKPOINT` to the same SFT output directory. That loads optimizer,
RNG and scheduler state; do not use it for starting SFT from the base checkpoint.
Keep the same cache, seed, batch size, sequence length and model configuration.
Without `RESUME=1`, SFT resets optimizer/RNG/scheduler and refuses an existing
output directory. Original pretraining scripts/checkpoints are untouched.
