#!/usr/bin/env python3
"""Native Ling-patched Megatron core_v0.13.0 SFT; no HF Trainer or ModelOpt SFT loader."""
from functools import partial
from pathlib import Path

import torch

from poziomka_data import MMapSFTDataset, load_manifest, sha256
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args, pretrain, print_rank_0
# Reuse the exact model, PP-aware batch transfer, forward and loss implementation
# used for Poziomka pretraining. Our dataset supplies already-shifted labels/masks.
from pretrain_gpt import (model_provider, forward_step, loss_func, has_nvidia_modelopt)


def extra_args(parser):
    if has_nvidia_modelopt:
        from pretrain_gpt import add_modelopt_args
        parser = add_modelopt_args(parser)
    parser.add_argument("--sft-manifest", required=True)
    parser.add_argument("--sft-packing", action="store_true",
                        help="Pack several conversations per sequence using thd attention. "
                             "Off by default: runs 1 and 2 must stay reproducible.")
    return parser


def packed_forward_step(data_iterator, model):
    """Forward step for packed batches; only used with --sft-packing.

    Unpacked training keeps using pretrain_gpt.forward_step untouched, so earlier
    runs replay exactly. Every pipeline stage reads its own identical batch (DP=1,
    TP=1, deterministic dataset), which is how cu_seqlens reaches the middle stages
    that have no tokens of their own.
    """
    args = get_args()
    batch = next(data_iterator)
    device = torch.cuda.current_device()
    tokens = batch["tokens"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    loss_mask = batch["loss_mask"].to(device, non_blocking=True)
    position_ids = batch["position_ids"].to(device, non_blocking=True)
    # Megatron squeezes the batch dimension for thd (attention.py:665), so packing
    # is only defined at micro-batch 1. Checked here as well as in the provider.
    cu_seqlens = batch["cu_seqlens"][0].to(device, non_blocking=True).to(torch.int32)
    # The dataset pads cu_seqlens to a fixed shape with repeats of seq_length so the
    # batch collates; thd needs the strictly increasing prefix back.
    kept = int((cu_seqlens < args.seq_length).sum().item()) + 1
    cu_seqlens = cu_seqlens[:kept].contiguous()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens, cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen)
    output_tensor = model(tokens, position_ids, None, labels=labels, loss_mask=loss_mask,
                          packed_seq_params=packed_seq_params)
    return output_tensor, partial(loss_func, loss_mask, model=model)


def datasets_provider(sample_counts):
    args = get_args()
    manifest_path = Path(args.sft_manifest).resolve()
    manifest = load_manifest(manifest_path)
    if args.seq_length != manifest["seq_length"]:
        raise ValueError("Training seq-length must match prepared cache")
    if args.context_parallel_size != 1 or args.tensor_model_parallel_size != 1 or args.expert_model_parallel_size != 1:
        raise ValueError("This initial Poziomka SFT adapter supports TP=CP=EP=1")
    if args.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Virtual/interleaved pipeline parallelism is not supported here")
    if args.reset_position_ids or args.reset_attention_mask or args.eod_mask_loss:
        raise ValueError("Do not reset on im_end: it ends a turn, not a conversation")
    if args.dataloader_type != "single":
        raise ValueError("Use the single sampler; the dataset handles deterministic epoch shuffling")
    tokenizer_dir = Path(args.tokenizer_model).resolve()
    if sha256(tokenizer_dir / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Training tokenizer differs from cache tokenizer")
    if sha256(manifest_path.parent / "chat_template.jinja") != manifest["template_sha256"]:
        raise ValueError("Cache chat template changed after preparation")
    if args.sft_packing and args.micro_batch_size != 1:
        raise ValueError("Packing requires micro-batch 1: thd drops the batch dimension")
    # Unpacked keeps the original behaviour exactly: only the stages that need
    # tokens build datasets. Packed needs cu_seqlens on every stage, because
    # attention runs on all of them, so every stage builds the same dataset.
    if not args.sft_packing and not (parallel_state.is_pipeline_first_stage(ignore_virtual=True) or
                                     parallel_state.is_pipeline_last_stage(ignore_virtual=True)):
        return None, None, None
    train = MMapSFTDataset(manifest_path, "train", sample_counts[0], args.seed,
                           pack=args.sft_packing)
    valid = MMapSFTDataset(manifest_path, "validation", sample_counts[1], args.seed,
                           shuffle=False, pack=args.sft_packing)
    if args.sft_packing:
        print_rank_0(f"SFT: {train.record_count} train conversations packed into "
                     f"{train.sample_count} sequences "
                     f"({100 * train.packing_efficiency():.1f}% of positions are real tokens); "
                     f"{valid.record_count} validation; loss roles={manifest['loss_roles']}")
    else:
        print_rank_0(f"SFT: {train.record_count} train / {valid.record_count} validation "
                     f"conversations; loss roles={manifest['loss_roles']}; "
                     f"no cross-conversation packing")
    return train, valid, None  # Never repurpose validation as an independent test set.


def choose_forward_step(data_iterator, model):
    """Dispatch per step: --sft-packing is the only thing that changes the path."""
    if get_args().sft_packing:
        return packed_forward_step(data_iterator, model)
    return forward_step(data_iterator, model)


if __name__ == "__main__":
    datasets_provider.is_distributed = True
    pretrain(datasets_provider, model_provider, ModelType.encoder_or_decoder, choose_forward_step,
             extra_args_provider=extra_args,
             args_defaults={"tokenizer_type": "HuggingFaceTokenizer"})
