#!/usr/bin/env python3
"""Loss/logit parity between packed and unpacked batches, on a single GPU.

Packing bugs do not crash. A wrong cu_seqlens trim, a label off by one, or a
RoPE position that fails to restart all produce a model that trains to a worse
optimum with no error anywhere. The only way to catch that is to run the same
conversations both ways through the same weights and compare outputs.

Small model, real attention geometry: 16 heads, 4 KV groups, head_dim 128,
rotary base 84000 at 50%, matching poziomka_model_args.sh. No checkpoint needed.

  MEGATRON_PATH=/path/to/Megatron-LM-core_v0.13.0 \
    torchrun --standalone --nproc_per_node=1 parity_poziomka_packing.py
"""
import argparse
import os
from pathlib import Path
import sys

import torch

MEGATRON_PATH = os.environ.get("MEGATRON_PATH")
if MEGATRON_PATH:
    sys.path.insert(0, MEGATRON_PATH)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Megatron-LM-core_v0.13.0"))

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

VOCAB = 32000
CONVERSATIONS = [301, 117, 233, 89]  # lengths that leave real padding in the window


def build_model(seq_length, layers):
    config = TransformerConfig(
        num_layers=layers, hidden_size=2048, ffn_hidden_size=2048,
        num_attention_heads=16, num_query_groups=4, kv_channels=128,
        use_cpu_initialization=False, perform_initialization=True,
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        gated_linear_unit=True, add_bias_linear=False, normalization="RMSNorm",
        layernorm_epsilon=1e-6, qk_layernorm=True, attention_dropout=0.0,
        hidden_dropout=0.0, tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    model = GPTModel(
        config=config, transformer_layer_spec=spec, vocab_size=VOCAB,
        max_sequence_length=seq_length, position_embedding_type="rope",
        rotary_base=84000, rotary_percent=0.5,
        share_embeddings_and_output_weights=False, pre_process=True, post_process=True)
    return model.cuda().eval()


def logits_of(output):
    """GPTModel returns [s, b, v] or [b, s, v] depending on version; normalise."""
    if output.shape[0] == 1:
        return output[0]
    return output[:, 0] if output.shape[1] == 1 else output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="Max absolute logit difference allowed (bf16 kernels differ)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--negative-control", action="store_true",
                        help="Deliberately break the boundaries: treat the whole window as "
                             "one sequence. This MUST fail. A passing parity run proves "
                             "nothing unless the same harness fails when packing is wrong.")
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(args.seed)
    torch.manual_seed(args.seed)

    total = sum(CONVERSATIONS)
    if total > args.seq_length:
        raise ValueError("Conversations must fit one window for this comparison")
    model = build_model(args.seq_length, args.layers)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    pieces = [torch.randint(5, VOCAB, (length,), device="cuda", generator=generator)
              for length in CONVERSATIONS]
    for piece in pieces:
        piece[0] = 1  # BOS, as encode_record guarantees

    # --- Unpacked: one conversation per sequence, right padded (today's path).
    unpacked = []
    with torch.no_grad():
        for piece, length in zip(pieces, CONVERSATIONS):
            tokens = torch.full((1, args.seq_length), 2, dtype=torch.long, device="cuda")
            tokens[0, :length] = piece
            position_ids = torch.arange(args.seq_length, device="cuda").unsqueeze(0)
            out = model(tokens, position_ids, None)
            unpacked.append(logits_of(out)[:length].float())

    # --- Packed: all four in one window, boundaries via cu_seqlens, positions restart.
    tokens = torch.full((1, args.seq_length), 2, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(1, args.seq_length, dtype=torch.long, device="cuda")
    boundaries, filled = [0], 0
    for piece, length in zip(pieces, CONVERSATIONS):
        tokens[0, filled:filled + length] = piece
        position_ids[0, filled:filled + length] = torch.arange(length, device="cuda")
        filled += length
        boundaries.append(filled)
    if filled < args.seq_length:
        position_ids[0, filled:] = torch.arange(args.seq_length - filled, device="cuda")
        boundaries.append(args.seq_length)
    if args.negative_control:
        # One sequence spanning the window: conversations can now see their
        # predecessors, which is exactly the bug real packing must not have.
        effective = [0, args.seq_length]
        print("NEGATIVE CONTROL: boundaries removed on purpose; this must FAIL.")
    else:
        effective = boundaries
    cu_seqlens = torch.tensor(effective, dtype=torch.int32, device="cuda")
    longest = max(int(cu_seqlens[i + 1] - cu_seqlens[i]) for i in range(len(effective) - 1))
    packed_seq_params = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens, cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=longest, max_seqlen_kv=longest)
    with torch.no_grad():
        out = model(tokens, position_ids, None, packed_seq_params=packed_seq_params)
    packed = logits_of(out).float()

    print(f"\n{'conversation':<14} {'tokens':>8} {'max |diff|':>12} {'mean |diff|':>13} "
          f"{'top-1 agree':>12}")
    print(f"{'-' * 14} {'-' * 8} {'-' * 12} {'-' * 13} {'-' * 12}")
    worst, disagreements = 0.0, 0
    for index, length in enumerate(CONVERSATIONS):
        start = boundaries[index]
        theirs = packed[start:start + length]
        mine = unpacked[index]
        difference = (theirs - mine).abs()
        agree = (theirs.argmax(-1) == mine.argmax(-1)).float().mean().item()
        disagreements += int(round((1 - agree) * length))
        worst = max(worst, difference.max().item())
        print(f"{index:<14} {length:>8} {difference.max().item():>12.6f} "
              f"{difference.mean().item():>13.6f} {100 * agree:>11.2f}%")

    print(f"\nworst absolute logit difference: {worst:.6f} (tolerance {args.tolerance})")
    print(f"top-1 disagreements: {disagreements} of {total} positions")
    print("\nbf16 thd and sbhd kernels do not produce bit-identical results, so small")
    print("differences are expected. A boundary or label bug looks different: large")
    print("differences concentrated in the conversations after the first.")

    failed = bool(disagreements) or worst > args.tolerance
    if args.negative_control:
        # Inverted: the control is only useful if breaking packing breaks the result.
        if failed:
            print("\nPASS (negative control): removing boundaries changed the outputs, "
                  "so the parity check genuinely detects packing bugs.")
            return 0
        print("\nFAIL (negative control): outputs were identical even with boundaries "
              "removed. The parity check is vacuous -- it cannot detect a real bug.")
        return 1
    if failed:
        print("\nFAIL: packed and unpacked disagree beyond kernel noise.")
        return 1
    print("\nPASS: packing preserves per-token outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
