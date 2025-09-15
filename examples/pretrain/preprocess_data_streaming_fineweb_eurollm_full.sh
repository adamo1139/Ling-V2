#!/bin/bash
set -ex

# Streaming preprocessing script for large parquet files (280GB+)
# This replaces the original script that fails with memory issues on large files

TOKENIZER_PATH=`dirname $(readlink -f "${BASH_SOURCE[0]}")`/../../resource/tokenizer/euroLLM/tokenizer.model
MEGATRON_PATH="Megatron-LM-core_v0.13.0"

PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH python ../../preprocess_large_parquet.py \
    --input fineweb2-pol-full.parquet \
    --output-prefix eurollm_processed_data_streaming \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --batch-size 1000000 \
    --append-eod \
    --log-interval 10

echo "Streaming preprocessing completed!"
echo "Output files created:"
echo "  eurollm_processed_data_streaming_text_document.bin"
echo "  eurollm_processed_data_streaming_text_document.idx"
