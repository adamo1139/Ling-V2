#!/usr/bin/env python3
"""
Chunk-based parallel preprocessing for large datasets
Processes original dataset chunks in parallel, then merges results.
"""

import argparse
import os
import sys
import time
import multiprocessing
import glob
import subprocess

# Add Megatron to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0')))

try:
    from datasets import load_dataset
    datasets_available = True
except ImportError:
    datasets_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


def find_chunk_files(chunks_dir: str) -> list:
    """
    Find all parquet files in the chunks directory
    """
    # Look for parquet files in the cache directory structure
    parquet_files = []
    
    # Search recursively for .parquet files
    for root, dirs, files in os.walk(chunks_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    # Sort files for consistent processing order
    parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet chunk files")
    for i, file in enumerate(parquet_files[:5]):  # Show first 5
        print(f"  {i}: {file}")
    
    if len(parquet_files) > 5:
        print(f"  ... and {len(parquet_files) - 5} more files")
    
    return parquet_files


def process_chunk(args_tuple):
    """
    Process a single chunk file using standard Megatron preprocessing
    
    Args:
        args_tuple: (chunk_file, output_prefix, tokenizer_args, chunk_id)
    """
    chunk_file, output_prefix, tokenizer_args, chunk_id = args_tuple
    
    print(f"Processing chunk {chunk_id}: {os.path.basename(chunk_file)}")
    
    # Build command to process this chunk
    megatron_script = os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0', 'tools', 'preprocess_data.py')
    
    cmd = [
        'python3', megatron_script,
        '--input', chunk_file,
        '--output-prefix', f"{output_prefix}_chunk_{chunk_id:04d}",
        '--workers', '1'  # Each chunk uses 1 worker
    ]
    
    # Add tokenizer arguments
    for key, value in tokenizer_args.items():
        if key == 'json_keys':
            cmd.extend(['--json-keys'] + value)
        elif isinstance(value, bool) and value:
            cmd.append(f'--{key.replace("_", "-")}')
        elif not isinstance(value, bool) and value is not None:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'Megatron-LM-core_v0.13.0')
    
    start_time = time.time()
    
    try:
        # Run the preprocessing
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"Completed chunk {chunk_id} in {elapsed:.1f}s: {os.path.basename(chunk_file)}")
        
        return True, chunk_id, f"{output_prefix}_chunk_{chunk_id:04d}"
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing chunk {chunk_id} ({chunk_file}): {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, chunk_id, None


def merge_chunk_outputs(chunk_prefixes: list, final_output_prefix: str, 
                       json_keys: list, tokenizer_args: dict):
    """
    Merge the output from multiple chunks into final files
    """
    print("Merging chunk outputs...")
    
    level = "document"
    if tokenizer_args.get('split_sentences', False):
        level = "sentence"
    
    # Create temporary args for tokenizer
    class MergeArgs:
        def __init__(self, tokenizer_args):
            for key, value in tokenizer_args.items():
                setattr(self, key, value)
            self.rank = 1
            self.make_vocab_size_divisible_by = 128
            self.tensor_model_parallel_size = 1
            self.vocab_extra_ids = 0
    
    merge_args = MergeArgs(tokenizer_args)
    tokenizer = build_tokenizer(merge_args)
    
    for key in json_keys:
        output_bin_file = f"{final_output_prefix}_{key}_{level}.bin"
        output_idx_file = f"{final_output_prefix}_{key}_{level}.idx"
        
        print(f"Merging {key} outputs...")
        
        builder = indexed_dataset.IndexedDatasetBuilder(
            output_bin_file,
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )
        
        merged_count = 0
        for chunk_prefix in chunk_prefixes:
            chunk_idx_file = f"{chunk_prefix}_{key}_{level}.idx"
            
            if os.path.exists(chunk_idx_file):
                print(f"  Merging: {os.path.basename(chunk_prefix)}")
                builder.add_index(f"{chunk_prefix}_{key}_{level}")
                merged_count += 1
            else:
                print(f"  Warning: Missing {chunk_idx_file}")
        
        builder.finalize(output_idx_file)
        print(f"Created final output: {output_bin_file} (merged {merged_count} chunks)")
    
    # Clean up chunk files
    print("Cleaning up chunk output files...")
    cleanup_count = 0
    for chunk_prefix in chunk_prefixes:
        for key in json_keys:
            chunk_bin_file = f"{chunk_prefix}_{key}_{level}.bin"
            chunk_idx_file = f"{chunk_prefix}_{key}_{level}.idx"
            if os.path.exists(chunk_bin_file):
                os.remove(chunk_bin_file)
                cleanup_count += 1
            if os.path.exists(chunk_idx_file):
                os.remove(chunk_idx_file)
                cleanup_count += 1
    
    print(f"Cleaned up {cleanup_count} temporary chunk files")


def main():
    parser = argparse.ArgumentParser(description='Chunk-based parallel preprocessing')
    
    # Add tokenizer arguments from Megatron
    parser = _add_tokenizer_args(parser)
    
    # Input/output arguments
    group = parser.add_argument_group('input data')
    group.add_argument('--chunks-dir', type=str, required=True,
                       help='Directory containing parquet chunk files')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='Space-separated list of keys to extract from data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    
    # Processing arguments
    group = parser.add_argument_group('processing')
    group.add_argument('--workers', type=int, default=16,
                       help='Number of parallel worker processes (default: 16)')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences')
    
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
    
    # Check dependencies
    if not datasets_available:
        print("ERROR: datasets library is required. Install with: pip install datasets pyarrow")
        return 1
    
    print(f"Starting chunk-based parallel preprocessing with {args.workers} workers")
    print(f"Chunks directory: {args.chunks_dir}")
    print(f"Output prefix: {args.output_prefix}")
    
    start_time = time.time()
    
    try:
        # Find all chunk files
        chunk_files = find_chunk_files(args.chunks_dir)
        
        if not chunk_files:
            print("ERROR: No parquet files found in chunks directory!")
            return 1
        
        # Prepare tokenizer arguments
        tokenizer_args = {
            'tokenizer_type': args.tokenizer_type,
            'tokenizer_model': getattr(args, 'tokenizer_model', None),
            'append_eod': args.append_eod,
            'split_sentences': args.split_sentences,
            'json_keys': args.json_keys
        }
        
        # Create processing tasks
        tasks = []
        for i, chunk_file in enumerate(chunk_files):
            tasks.append((chunk_file, args.output_prefix, tokenizer_args, i))
        
        print(f"Processing {len(chunk_files)} chunks with {args.workers} parallel workers...")
        
        # Process chunks in parallel
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(process_chunk, tasks)
        
        # Check results and collect chunk prefixes
        chunk_prefixes = []
        failed_chunks = []
        
        for success, chunk_id, prefix in results:
            if success:
                if prefix:
                    chunk_prefixes.append(prefix)
            else:
                failed_chunks.append(chunk_id)
        
        if failed_chunks:
            print(f"Failed chunks: {failed_chunks}")
            return 1
        
        print(f"Successfully processed {len(chunk_prefixes)} chunks")
        
        # Merge the results
        merge_chunk_outputs(chunk_prefixes, args.output_prefix, 
                           args.json_keys, tokenizer_args)
        
        total_time = time.time() - start_time
        
        print(f"Chunk-based parallel preprocessing completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Processed {len(chunk_files)} chunks")
        print(f"Average time per chunk: {total_time/len(chunk_files):.1f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"Error during chunk-based processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
