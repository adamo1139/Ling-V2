#!/usr/bin/env python3
"""
Parallel preprocessing script for large Parquet files (280GB+)
Splits large files into chunks and processes them in parallel for maximum speed.
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
import math
from pathlib import Path
import subprocess
import shutil

# Add Megatron to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0')))

try:
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    datasets_available = True
except ImportError:
    datasets_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


def split_parquet_file(input_file: str, num_partitions: int, temp_dir: str) -> list:
    """
    Split a large parquet file into smaller chunks for parallel processing
    
    Args:
        input_file: Path to the large parquet file
        num_partitions: Number of chunks to create
        temp_dir: Directory to store temporary chunk files
        
    Returns:
        List of chunk file paths
    """
    print(f"Splitting {input_file} into {num_partitions} chunks...")
    
    # Read parquet file
    table = pq.read_table(input_file)
    total_rows = len(table)
    rows_per_chunk = math.ceil(total_rows / num_partitions)
    
    chunk_files = []
    
    for i in range(num_partitions):
        start_idx = i * rows_per_chunk
        end_idx = min((i + 1) * rows_per_chunk, total_rows)
        
        if start_idx >= total_rows:
            break
            
        chunk_table = table.slice(start_idx, end_idx - start_idx)
        chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.parquet")
        
        pq.write_table(chunk_table, chunk_file)
        chunk_files.append(chunk_file)
        
        print(f"Created chunk {i+1}/{num_partitions}: {len(chunk_table)} rows -> {chunk_file}")
    
    return chunk_files


def process_chunk(args_tuple):
    """
    Process a single chunk file
    
    Args:
        args_tuple: (chunk_file, output_prefix, tokenizer_args)
    """
    chunk_file, output_prefix, tokenizer_args = args_tuple
    
    print(f"Processing chunk: {chunk_file}")
    
    # Build command to process this chunk
    cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0', 'tools', 'preprocess_data.py'),
        '--input', chunk_file,
        '--output-prefix', output_prefix,
        '--workers', '1'  # Each chunk uses 1 worker, but we run multiple chunks in parallel
    ]
    
    # Add tokenizer arguments
    for key, value in tokenizer_args.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key.replace("_", "-")}')
        elif not isinstance(value, bool) and value is not None:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # Run the preprocessing
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0')
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(f"Completed chunk: {chunk_file}")
        return True, chunk_file
    except subprocess.CalledProcessError as e:
        print(f"Error processing chunk {chunk_file}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, chunk_file


def merge_chunk_outputs(chunk_prefixes: list, final_output_prefix: str, json_keys: list):
    """
    Merge the output from multiple chunks into final files
    """
    print("Merging chunk outputs...")
    
    level = "document"
    tokenizer = build_tokenizer(get_merge_args())
    
    for key in json_keys:
        output_bin_file = f"{final_output_prefix}_{key}_{level}.bin"
        output_idx_file = f"{final_output_prefix}_{key}_{level}.idx"
        
        builder = indexed_dataset.IndexedDatasetBuilder(
            output_bin_file,
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )
        
        for chunk_prefix in chunk_prefixes:
            chunk_bin_file = f"{chunk_prefix}_{key}_{level}.bin"
            chunk_idx_file = f"{chunk_prefix}_{key}_{level}.idx"
            
            if os.path.exists(chunk_idx_file):
                print(f"Merging chunk: {chunk_prefix}")
                builder.add_index(f"{chunk_prefix}_{key}_{level}")
        
        builder.finalize(output_idx_file)
        print(f"Created final output: {output_bin_file}")


def get_merge_args():
    """Get minimal args for tokenizer during merge"""
    class Args:
        def __init__(self):
            self.rank = 1
            self.make_vocab_size_divisible_by = 128
            self.tensor_model_parallel_size = 1
            self.vocab_extra_ids = 0
            self.tokenizer_type = 'GPT2BPETokenizer'  # Default fallback
            self.tokenizer_model = None
    return Args()


def main():
    parser = argparse.ArgumentParser(description='Parallel preprocessing for large Parquet files')
    
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
    group.add_argument('--workers', type=int, default=16,
                       help='Number of parallel processes (default: 16)')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences')
    group.add_argument('--temp-dir', type=str, default='./temp_chunks',
                       help='Directory for temporary chunk files')
    
    args = parser.parse_args()
    
    # Set required defaults for Megatron
    if not hasattr(args, 'rank'):
        args.rank = 1
    if not hasattr(args, 'make_vocab_size_divisible_by'):
        args.make_vocab_size_divisible_by = 128
    if not hasattr(args, 'tensor_model_parallel_size'):
        args.tensor_model_parallel_size = 1
    if not hasattr(args, 'vocab_extra_ids'):
        args.vocab_extra_ids = 0
    if not hasattr(args, 'keep_empty'):
        args.keep_empty = False
    
    # Check dependencies
    if not datasets_available:
        print("ERROR: datasets library is required. Install with: pip install datasets pyarrow")
        return 1
    
    start_time = time.time()
    
    # Create temp directory
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Split the large file into chunks
        chunk_files = split_parquet_file(args.input, args.workers, temp_dir)
        
        # Prepare arguments for parallel processing
        tokenizer_args = {
            'tokenizer_type': args.tokenizer_type,
            'tokenizer_model': getattr(args, 'tokenizer_model', None),
            'append_eod': args.append_eod,
            'split_sentences': args.split_sentences,
            'json_keys': args.json_keys
        }
        
        # Create processing tasks
        tasks = []
        chunk_prefixes = []
        
        for i, chunk_file in enumerate(chunk_files):
            chunk_prefix = os.path.join(temp_dir, f"chunk_{i:04d}_processed")
            chunk_prefixes.append(chunk_prefix)
            tasks.append((chunk_file, chunk_prefix, tokenizer_args))
        
        # Process chunks in parallel
        print(f"Processing {len(chunk_files)} chunks in parallel with {args.workers} processes...")
        
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(process_chunk, tasks)
        
        # Check if all chunks processed successfully
        failed_chunks = [chunk for success, chunk in results if not success]
        if failed_chunks:
            print(f"Failed to process chunks: {failed_chunks}")
            return 1
        
        # Merge the results
        merge_chunk_outputs(chunk_prefixes, args.output_prefix, args.json_keys)
        
        total_time = time.time() - start_time
        print(f"Parallel preprocessing completed in {total_time:.1f} seconds")
        
        # Cleanup temp files
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        return 0
        
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        # Cleanup on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return 1


if __name__ == '__main__':
    sys.exit(main())
