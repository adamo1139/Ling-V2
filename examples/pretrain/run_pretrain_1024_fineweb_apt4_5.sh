#!/bin/bash
set -ex

MODEL_PATH="" # no checkpoint needed for from-scratch training
JOB_DIR="pretrain_1024_fineweb_apt4_5"
DATA_PATH="apt4_processed_data_text_document"
MEGATRON_PATH="Megatron-LM-core_v0.13.0"


mkdir -p ${JOB_DIR}
CHECKPOINT_PATH=${JOB_DIR}
TENSORBOARD_LOGS_PATH=${JOB_DIR}/runs

if [[ $RANK -eq 0 ]]; then
    cp -r ${0} ${JOB_DIR}
    pip list > ${JOB_DIR}/pip_list.txt
    python -m torch.utils.collect_env > ${JOB_DIR}/collect_env.txt
fi


GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
echo "---> from pytorch runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
LAUNCHER=" \
    torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "

LOG_PATH="${JOB_DIR}/log_${NODE_RANK}.txt"

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0

export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2  # 2 means DEBUG level

export NCCL_DEBUG=OFF

DEVICE_MODEL=$(nvidia-smi -i 0 -q | grep "Product Name" | awk -F: '{ print $2 }')
DEVICE_MODEL=$(echo "$DEVICE_MODEL" | xargs)  # drop white space

if [[ $DEVICE_MODEL == NVIDIA* ]]; then
    DEVICE_MODEL=${DEVICE_MODEL#"NVIDIA"}
    DEVICE_MODEL=$(echo "$DEVICE_MODEL" | sed 's/^ *//')
fi

if [ "$DEVICE_MODEL" = "NVIDIA GeForce RTX 3090 Ti" ] || [ "$DEVICE_MODEL" = "A100-SXM4-80GB" ]; then
    # Ampere GPUs do not support multicast. If `--tp-comm-overlap` is set on Ampere-arch GPUs, this env must be set.
    export UB_SKIPMC=1  
fi

MOE_ARGS=(
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --num-experts 16
    --moe-ffn-hidden-size 512
    --moe-router-score-function sigmoid
    --moe-router-topk 2
    --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-z-loss-coeff 0.0000035
    --moe-router-bias-update-rate 1e-3
    --moe-layer-freq [1,1,1,1,1,1,1,1]
    --bias-zero-mean-update
)

MPT_ARGS=(
    --mtp-num-layers 0
)

GPT_MODEL_ARGS=(
    --num-layers 8
    --hidden-size 4096
    --ffn-hidden-size 1024
    --num-attention-heads 16
    --num-query-groups 8
    --group-query-attention
    --qk-layernorm
    --use-flash-attn
    --max-position-embeddings 8192
    --vocab-size 32000
    --make-vocab-size-divisible-by 128
    --position-embedding-type "rope"
    --rotary-base 10000
    --rotary-percent 0.5
    --rotary-scaling-factor 40
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization "RMSNorm"
    --norm-epsilon "1e-06"
    --disable-bias-linear
    --transformer-impl "transformer_engine"
    --attention-dropout 0
    --hidden-dropout 0
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 512
    --seq-length 8192
    --train-iters 167
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --clip-grad 1.0
    
    --bf16

    --fp8-param-gather
    --fp8-recipe "blockwise"
    --fp8-format "e4m3"

    --optimizer "adamw-bnb-8bit"
    --lr "7.0e-4"
    --lr-decay-style cosine
    --min-lr "3.0e-5"
    --lr-warmup-iters 10
    --seed 42

    --manual-gc
    --manual-gc-interval 200
)

MODEL_PARALLEL_ARGS=(
    --pipeline-model-parallel-size 1
    --tensor-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --recompute-granularity selective
    --recompute-modules moe
    
    --overlap-param-gather
    --overlap-grad-reduce

)

DATA_ARGS=(
    --data-path ${DATA_PATH}
    --tokenizer-type "HuggingFaceTokenizer"
    --tokenizer-model `dirname $(readlink -f "${BASH_SOURCE[0]}")`/../../resource/tokenizer/apt4
    --split 9999,1,0
    --dataloader-type "single"
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
)

EVAL_AND_LOGGING_ARGS=(
    --save-interval 300 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH
    --ckpt-format "torch_dist"
    --async-save
    --eval-iters 2
    --log-interval 1
    --log-throughput
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

KERNEL_ARGS=(
    --attention-backend flash
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32	
    --cross-entropy-loss-fusion
)

CMD="${LAUNCHER} ${MEGATRON_PATH}/pretrain_gpt.py \
    ${MOE_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${KERNEL_ARGS[@]} \
    ${MPT_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
"

echo ${CMD}
PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH ${CMD} 2>&1 | tee ${LOG_PATH}
