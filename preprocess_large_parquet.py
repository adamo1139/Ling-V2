#!/usr/bin/env python3
"""
Streaming preprocessing script for large Parquet files (280GB+)
Handles memory-efficient processing of massive datasets for pretraining.
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
from typing import Iterator, Dict, Any, Tuple
import torch
import numpy as np

# Add Megatron to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0')))

try:
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    datasets_available = True
except ImportError:
    datasets_available = False

try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


class StreamingEncoder:
    """Encoder that processes data in streaming fashion"""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
    
    def initializer(self):
        """Initialize tokenizer for worker processes"""
        self.tokenizer = build_tokenizer(self.args)
    
    def encode_batch(self, batch_data: list) -> list:
        """Encode a batch of documents"""
        if self.tokenizer is None:
            self.initializer()
        
        encoded_batch = []
        for item in batch_data:
            if isinstance(item, dict):
                # Handle dict format from streaming dataset
                text = item.get('text', '')
            else:
                # Handle JSON string format
                try:
                    data = json.loads(item)
                    text = data.get('text', '')
                except:
                    text = str(item)
            
            if text:
                # Tokenize the text
                tokens = self.tokenizer.tokenize(text)
                if self.args.append_eod and tokens:
                    tokens.append(self.tokenizer.eod)
                encoded_batch.append(tokens)
            else:
                encoded_batch.append([])
        
        return encoded_batch


def stream_parquet_file(parquet_path: str, batch_size: int = 1000) -> Iterator[list]:
    """
    Stream a large parquet file in batches to avoid memory issues
    
    Args:
        parquet_path: Path to the parquet file
        batch_size: Number of rows to process at once
    
    Yields:
        Batches of data as lists
    """
    if not datasets_available:
        raise ImportError("datasets library is required for streaming Parquet files")
    
    print(f"Streaming parquet file: {parquet_path}")
    
    # Use streaming dataset loading
    try:
        # First try with streaming=True for very large files
        dataset = load_dataset(
            'parquet', 
            data_files=parquet_path, 
            split='train',
            streaming=True
        )
        
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining items
        if batch:
            yield batch
            
    except Exception as e:
        print(f"Streaming failed, falling back to chunked reading: {e}")
        
        # Fallback: Use pyarrow to read in chunks
        parquet_file = pq.ParquetFile(parquet_path)
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # Convert to list of dicts
            batch_data = []
            for i in range(batch.num_rows):
                row = {}
                for col_name in batch.column_names:
                    row[col_name] = batch[col_name].to_pylist()[i]
                batch_data.append(row)
            yield batch_data


def process_streaming_parquet(args):
    """Process large parquet file using streaming approach"""
    
    print(f"Starting streaming preprocessing of {args.input}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    
    # Initialize encoder and tokenizer
    encoder = StreamingEncoder(args)
    encoder.initializer()
    
    # Setup output files
    level = "document"
    if args.split_sentences:
        level = "sentence"
    
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    
    for key in args.json_keys:
        output_bin_files[key] = f"{args.output_prefix}_{key}_{level}.bin"
        output_idx_files[key] = f"{args.output_prefix}_{key}_{level}.idx"
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(encoder.tokenizer.vocab_size),
        )
    
    # Process the file in streaming fashion
    total_processed = 0
    start_time = time.time()
    last_report_time = start_time
    
    try:
        for batch_idx, batch_data in enumerate(stream_parquet_file(args.input, args.batch_size)):
            # Encode the batch
            encoded_batch = encoder.encode_batch(batch_data)
            
            # Add to builders
            for i, tokens in enumerate(encoded_batch):
                if tokens:  # Only add non-empty documents
                    for key in args.json_keys:
                        builders[key].add_document(tokens, [len(tokens)])
            
            total_processed += len(batch_data)
            
            # Progress reporting
            current_time = time.time()
            if current_time - last_report_time >= args.log_interval:
                elapsed = current_time - start_time
                docs_per_sec = total_processed / elapsed if elapsed > 0 else 0
                print(f"Processed {total_processed} documents ({docs_per_sec:.1f} docs/s)")
                last_report_time = current_time
            
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted after {total_processed} documents")
        return False
    except Exception as e:
        print(f"Error during processing: {e}")
        return False
    
    # Finalize the output files
    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])
    
    total_time = time.time() - start_time
    print(f"Completed processing {total_processed} documents in {total_time:.1f} seconds")
    print(f"Average speed: {total_processed/total_time:.1f} docs/s")
    
    return True


def get_args():
    parser = argparse.ArgumentParser(description='Streaming preprocessing for large Parquet files')
    
    # Add tokenizer arguments from Megatron
    parser = _add_tokenizer_args(parser)
    
    # Input/output arguments
    group = parser.add_argument_group('input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input Parquet file')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='Space-separated list of keys to extract from data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    
    # Processing arguments
    group = parser.add_argument_group('processing')
    group.add_argument('--batch-size', type=int, default=1000000,
                       help='Number of documents to process in each batch (default: 1 000 000)')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (default: 1, streaming uses 1 worker)')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences')
    group.add_argument('--log-interval', type=int, default=10,
                       help='Progress reporting interval in seconds (default: 10)')
    
    # Default values for tokenizer
    args = parser.parse_args()
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.keep_empty = False
    args.lang = 'english'
    args.keep_newlines = False
    
    return args


def main():
    args = get_args()
    
    # Check dependencies
    if not datasets_available:
        print("ERROR: datasets library is required. Install with: pip install datasets pyarrow")
        return 1
    
    # Process the file
    success = process_streaming_parquet(args)
    
    if success:
        print("Preprocessing completed successfully!")
        return 0
    else:
        print("Preprocessing failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
