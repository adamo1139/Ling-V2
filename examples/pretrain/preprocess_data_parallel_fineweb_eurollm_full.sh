#!/bin/bash
set -ex

# Parallel preprocessing script for large parquet files (280GB+)
# Splits the file into chunks and processes them in parallel for maximum speed

TOKENIZER_PATH="/home/ubuntu/preprocessing/Ling-V2/resource/tokenizer/euroLLM/tokenizer.model"
MEGATRON_PATH="Megatron-LM-core_v0.13.0"

PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH python3 /home/ubuntu/preprocessing/Ling-V2/preprocess_large_parquet_parallel.py \
    --input fineweb2-pol-full.parquet \
    --output-prefix eurollm_processed_data_parallel \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --workers 16 \
    --append-eod \
    --batch-size 10000

echo "Parallel preprocessing completed!"
echo "Output files created:"
echo "  eurollm_processed_data_parallel_text_document.bin"
echo "  eurollm_processed_data_parallel_text_document.idx"
echo ""
echo "Performance should be 16x faster than single-threaded streaming!"
