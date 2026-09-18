#!/usr/bin/env bash
# Shared by SFT and HF -> DCP conversion. 128 TOTAL experts, 32 ACTIVE per token.
# Matches the local linear-8-9-10-11-sqrt HF config; no rope scaling is enabled.
#
# ROTARY_BASE defaults to the pretrained 84000. Per Xu et al., "Base of RoPE Bounds
# Context Length" (NeurIPS 2024) Table 2, the minimum base for a context length is
# 8k: 8.4e4   16k: 3.1e5   32k: 6.4e5   64k: 2.1e6   128k: 7.8e6
# So 84000 is exactly the 8k bound and has no headroom: training at 16384 without
# raising it leaves the model below the bound, where long context is superficial.
# Any run above 8192 must set ROTARY_BASE, and the HF export must use the same
# value or inference will silently disagree with training.
POZIOMKA_MODEL_ARGS=(
    --num-layers 16 --hidden-size 2048 --ffn-hidden-size 2048
    --num-attention-heads 16 --num-query-groups 4 --group-query-attention --qk-layernorm
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-8192}"
    --vocab-size 32000 --make-vocab-size-divisible-by 128
    --position-embedding-type rope --rotary-base "${ROTARY_BASE:-84000}" --rotary-percent 0.5
    --swiglu --untie-embeddings-and-output-weights --normalization RMSNorm
    --norm-epsilon 1e-6 --disable-bias-linear --transformer-impl transformer_engine
    --attention-dropout 0 --hidden-dropout 0
    --num-experts 128 --moe-router-topk 32
    --moe-ffn-hidden-size 320 --moe-shared-expert-intermediate-size 320
    --moe-router-score-function sigmoid --moe-router-dtype fp32
    --moe-router-enable-expert-bias --moe-router-topk-scaling-factor 2.5
    --moe-router-num-groups 8 --moe-router-group-topk 2
    --moe-layer-freq '[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --moe-grouped-gemm --moe-token-dispatcher-type alltoall
    --pipeline-model-parallel-size 8 --tensor-model-parallel-size 1
)
