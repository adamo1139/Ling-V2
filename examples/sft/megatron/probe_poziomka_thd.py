#!/usr/bin/env python3
"""Probe whether packed (thd) attention works in this TE build, on one GPU.

Packing is only worth implementing if TE can do variable-length attention here.
The launcher sets NVTE_FUSED_ATTN=0, and thd normally wants the cuDNN backend,
so this is the assumption most likely to be wrong. Runs in seconds, loads no
model weights, and checks the property that actually matters: a packed sequence
must not attend across its neighbours' boundaries.

  python3 probe_poziomka_thd.py                    # current launcher env
  NVTE_FUSED_ATTN=1 python3 probe_poziomka_thd.py  # if the default fails
"""
import os
import sys

import torch

# Poziomka attention shape: 16 heads, 4 KV groups (GQA), head_dim 128.
# thd accepts only padding/padding_causal masks; Megatron substitutes this
# automatically for packed sequences (extensions/transformer_engine.py:928).
HEADS, KV_GROUPS, HEAD_DIM = 16, 4, 128
SEQUENCES = [4096, 2048, 8192, 2048]  # four conversations packed into one 16384 window


def main():
    if not torch.cuda.is_available():
        print("No CUDA device; run this on the training machine.")
        return 1
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        print(f"{name}={os.environ.get(name, '<unset>')}")

    try:
        from transformer_engine.pytorch import DotProductAttention
        import transformer_engine
        print(f"transformer_engine {getattr(transformer_engine, '__version__', '?')}")
    except Exception as exc:
        print(f"FAIL: cannot import transformer_engine: {exc}")
        return 1

    device, dtype = torch.device("cuda"), torch.bfloat16
    total = sum(SEQUENCES)
    cu_seqlens = torch.tensor([0] + list(torch.tensor(SEQUENCES).cumsum(0)),
                              dtype=torch.int32, device=device)
    torch.manual_seed(0)
    # thd layout: [total_tokens, heads, head_dim] with no batch dimension.
    query = torch.randn(total, HEADS, HEAD_DIM, device=device, dtype=dtype)
    key = torch.randn(total, KV_GROUPS, HEAD_DIM, device=device, dtype=dtype)
    value = torch.randn(total, KV_GROUPS, HEAD_DIM, device=device, dtype=dtype)

    attention = DotProductAttention(
        num_attention_heads=HEADS, kv_channels=HEAD_DIM,
        num_gqa_groups=KV_GROUPS, attn_mask_type="padding_causal",
        qkv_format="thd", attention_dropout=0.0).to(device)

    def run(k, v):
        with torch.no_grad():
            return attention(query, k, v, qkv_format="thd",
                             cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                             max_seqlen_q=max(SEQUENCES), max_seqlen_kv=max(SEQUENCES),
                             attn_mask_type="padding_causal")

    try:
        baseline = run(key, value)
    except Exception as exc:
        print(f"\nFAIL: thd attention raised: {type(exc).__name__}: {exc}")
        print("If this is a backend error rather than a shape/mask error, try")
        print("NVTE_FUSED_ATTN=1: thd prefers the cuDNN kernel.")
        return 1
    print(f"\nthd forward OK: output {tuple(baseline.shape)}, dtype {baseline.dtype}")

    # The real test: perturb the LAST sequence. Under correct packing, every
    # earlier sequence is unchanged. Causal masking alone would also protect
    # earlier positions, so also perturb the FIRST and check the last is unchanged
    # -- that can only hold if the boundaries are honoured.
    first_end = SEQUENCES[0]
    poisoned_key, poisoned_value = key.clone(), value.clone()
    poisoned_key[:first_end].normal_()
    poisoned_value[:first_end].normal_()
    perturbed = run(poisoned_key, poisoned_value)

    later = slice(first_end, total)
    drift = (perturbed[later].float() - baseline[later].float()).abs().max().item()
    changed = (perturbed[:first_end].float() - baseline[:first_end].float()).abs().max().item()

    print(f"max change inside the perturbed sequence: {changed:.6f}  (must be > 0)")
    print(f"max leak into the following sequences:    {drift:.6f}  (must be 0)")

    if changed == 0:
        print("\nFAIL: perturbation had no effect; the probe is not measuring anything.")
        return 1
    if drift != 0:
        print("\nFAIL: sequences are NOT isolated -- packing here would let one "
              "conversation attend to another.")
        return 1
    print("\nPASS: thd attention works and sequences are isolated. Packing is viable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
