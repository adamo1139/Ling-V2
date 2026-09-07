#!/usr/bin/env bash
# Shared by SFT and HF -> DCP conversion. 128 TOTAL experts, 32 ACTIVE per token.
# Matches the local linear-8-9-10-11-sqrt HF config; no rope scaling is enabled.
POZIOMKA_MODEL_ARGS=(
    --num-layers 16 --hidden-size 2048 --ffn-hidden-size 2048
    --num-attention-heads 16 --num-query-groups 4 --group-query-attention --qk-layernorm
    --max-position-embeddings 8192 --vocab-size 32000 --make-vocab-size-divisible-by 128
    --position-embedding-type rope --rotary-base 84000 --rotary-percent 0.5
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
