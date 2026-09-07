#!/usr/bin/env python3
"""Native Ling-patched Megatron core_v0.13.0 SFT; no HF Trainer or ModelOpt SFT loader."""
from pathlib import Path

from poziomka_data import MMapSFTDataset, load_manifest, sha256
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0
# Reuse the exact model, PP-aware batch transfer, forward and loss implementation
# used for Poziomka pretraining. Our dataset supplies already-shifted labels/masks.
from pretrain_gpt import (model_provider, forward_step, has_nvidia_modelopt)


def extra_args(parser):
    if has_nvidia_modelopt:
        from pretrain_gpt import add_modelopt_args
        parser = add_modelopt_args(parser)
    parser.add_argument("--sft-manifest", required=True)
    return parser


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
    if not (parallel_state.is_pipeline_first_stage(ignore_virtual=True) or
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)):
        return None, None, None
    train = MMapSFTDataset(manifest_path, "train", sample_counts[0], args.seed)
    valid = MMapSFTDataset(manifest_path, "validation", sample_counts[1], args.seed, shuffle=False)
    print_rank_0(f"SFT: {train.record_count} train / {valid.record_count} validation conversations; "
                 f"loss roles={manifest['loss_roles']}; no cross-conversation packing")
    return train, valid, None  # Never repurpose validation as an independent test set.


if __name__ == "__main__":
    datasets_provider.is_distributed = True
    pretrain(datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step,
             extra_args_provider=extra_args,
             args_defaults={"tokenizer_type": "HuggingFaceTokenizer"})
