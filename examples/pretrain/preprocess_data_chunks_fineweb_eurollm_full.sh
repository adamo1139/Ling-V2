#!/bin/bash
set -ex

# Chunk-based preprocessing script for large datasets
# Downloads original chunks and processes them in parallel

TOKENIZER_PATH="/home/ubuntu/preprocessing/Ling-V2/resource/tokenizer/euroLLM/tokenizer.model"
MEGATRON_PATH="Megatron-LM-core_v0.13.0"

# Set dataset chunks directory
CHUNKS_DIR="fineweb2-pol-chunks"

# Step 1: Download chunks if they don't exist
if [ ! -d "$CHUNKS_DIR" ]; then
    echo "Downloading dataset chunks..."
    python3 /home/ubuntu/preprocessing/Ling-V2/examples/pretrain/download_example_data_fineweb_full_chunks.py
else
    echo "Chunks directory already exists: $CHUNKS_DIR"
fi

# Step 2: Process chunks in parallel
echo "Processing chunks with EuroLLM tokenizer..."
PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH python ../../preprocess_chunks_parallel.py \
    --chunks-dir ${CHUNKS_DIR} \
    --output-prefix eurollm_processed_data_chunks \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --workers 16 \
    --append-eod

echo "Chunk-based preprocessing completed!"
echo "Output files created:"
echo "  eurollm_processed_data_chunks_text_document.bin"
echo "  eurollm_processed_data_chunks_text_document.idx"
echo ""
echo "This approach should be much more memory-efficient!"
