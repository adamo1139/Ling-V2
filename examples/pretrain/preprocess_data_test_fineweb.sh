#!/bin/bash
set -ex

TOKENIZER_PATH=`dirname $(readlink -f "${BASH_SOURCE[0]}")`/../../resource/tokenizer/config_pretrain
MEGATRON_PATH="Megatron-LM-core_v0.13.0"

PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH python ${MEGATRON_PATH}/tools/preprocess_data.py \
    --input fineweb2-pol-100k.parquet \
    --output-prefix processed_data \
    --tokenizer-type BailingTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --workers 4 \
    --append-eod
