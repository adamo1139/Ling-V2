#!/usr/bin/env python3
"""
Indexed parallel preprocessing script for large Parquet files (280GB+)
Uses start/end sample indexes to parallelize streaming without file splitting.
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
import math
from typing import Iterator, Tuple

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


def get_parquet_row_count(parquet_path: str) -> int:
    """
    Get total number of rows in parquet file without loading data into memory
    """
    parquet_file = pq.ParquetFile(parquet_path)
    return parquet_file.metadata.num_rows


def stream_parquet_chunk(parquet_path: str, start_idx: int, end_idx: int, 
                        batch_size: int = 1000) -> Iterator[list]:
    """
    Stream a specific chunk of a parquet file using start/end indexes
    
    Args:
        parquet_path: Path to the parquet file
        start_idx: Starting document index
        end_idx: Ending document index
        batch_size: Batch size for streaming
    
    Yields:
        Batches of data as lists
    """
    print(f"Streaming chunk: docs {start_idx} to {end_idx}")
    
    # Use streaming dataset loading
    dataset = load_dataset(
        'parquet', 
        data_files=parquet_path, 
        split='train',
        streaming=True
    )
    
    batch = []
    current_idx = 0
    
    for item in dataset:
        # Skip until we reach start_idx
        if current_idx < start_idx:
            current_idx += 1
            continue
        
        # Stop when we reach end_idx
        if current_idx >= end_idx:
            break
            
        batch.append(item)
        current_idx += 1
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield remaining items
    if batch:
        yield batch


def process_worker_chunks(args_tuple):
    """
    Process multiple chunks assigned to a single worker using streaming
    
    Args:
        args_tuple: (worker_id, chunk_ranges, input_file, output_prefix, tokenizer_args)
    """
    worker_id, chunk_ranges, input_file, output_prefix, tokenizer_args = args_tuple
    
    print(f"Worker {worker_id} processing {len(chunk_ranges)} chunks")
    
    # Initialize encoder
    # Create a temporary args object for this worker
    class WorkerArgs:
        def __init__(self, tokenizer_args):
            for key, value in tokenizer_args.items():
                setattr(self, key, value)
            # Set required defaults
            self.rank = 1
            self.make_vocab_size_divisible_by = 128
            self.tensor_model_parallel_size = 1
            self.vocab_extra_ids = 0
            self.keep_empty = False
    
    worker_args = WorkerArgs(tokenizer_args)
    encoder = StreamingEncoder(worker_args)
    encoder.initializer()
    
    # Setup output files for this worker
    level = "document"
    if getattr(worker_args, 'split_sentences', False):
        level = "sentence"
    
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    
    json_keys = tokenizer_args.get('json_keys', ['text'])
    for key in json_keys:
        output_bin_files[key] = f"{output_prefix}_worker_{worker_id:04d}_{key}_{level}.bin"
        output_idx_files[key] = f"{output_prefix}_worker_{worker_id:04d}_{key}_{level}.idx"
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(encoder.tokenizer.vocab_size),
        )
    
    total_processed = 0
    start_time = time.time()
    
    try:
        # Process each chunk assigned to this worker
        for chunk_id, (start_idx, end_idx) in enumerate(chunk_ranges):
            print(f"Worker {worker_id} processing chunk {chunk_id+1}/{len(chunk_ranges)}: docs {start_idx}-{end_idx}")
            
            # Stream through this chunk
            for batch_data in stream_parquet_chunk(input_file, start_idx, end_idx):
                # Encode the batch
                encoded_batch = encoder.encode_batch(batch_data)
                
                # Add to builders
                for i, tokens in enumerate(encoded_batch):
                    if tokens:  # Only add non-empty documents
                        for key in json_keys:
                            builders[key].add_document(tokens, [len(tokens)])
                
                total_processed += len(batch_data)
        
        # Finalize the output files
        for key in json_keys:
            builders[key].finalize(output_idx_files[key])
        
        total_time = time.time() - start_time
        print(f"Worker {worker_id} completed: {total_processed} documents in {total_time:.1f}s")
        
        return True, worker_id, [f"{output_prefix}_worker_{worker_id:04d}" for _ in json_keys]
        
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return False, worker_id, None


def merge_worker_outputs(worker_prefixes: list, final_output_prefix: str, 
                        json_keys: list, tokenizer_args: dict):
    """
    Merge the output from multiple workers into final files
    """
    print("Merging worker outputs...")
    
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
        
        builder = indexed_dataset.IndexedDatasetBuilder(
            output_bin_file,
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )
        
        for worker_prefix in worker_prefixes:
            worker_idx_file = f"{worker_prefix}_{key}_{level}.idx"
            
            if os.path.exists(worker_idx_file):
                print(f"Merging worker output: {worker_prefix}")
                builder.add_index(f"{worker_prefix}_{key}_{level}")
        
        builder.finalize(output_idx_file)
        print(f"Created final output: {output_bin_file}")
    
    # Clean up worker files
    print("Cleaning up worker output files...")
    for worker_prefix in worker_prefixes:
        for key in json_keys:
            worker_bin_file = f"{worker_prefix}_{key}_{level}.bin"
            worker_idx_file = f"{worker_prefix}_{key}_{level}.idx"
            if os.path.exists(worker_bin_file):
                os.remove(worker_bin_file)
            if os.path.exists(worker_idx_file):
                os.remove(worker_idx_file)


def main():
    parser = argparse.ArgumentParser(description='Indexed parallel preprocessing for large Parquet files')
    
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
                       help='Number of parallel worker processes (default: 16)')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences')
    group.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for streaming (default: 1000)')
    
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
    
    print(f"Starting indexed parallel preprocessing with {args.workers} workers")
    print(f"Input file: {args.input}")
    print(f"Output prefix: {args.output_prefix}")
    
    start_time = time.time()
    
    try:
        # Get total document count without loading data
        print("Getting document count from parquet metadata...")
        total_docs = get_parquet_row_count(args.input)
        print(f"Total documents: {total_docs:,}")
        
        # Calculate chunk ranges (4x workers for better load balancing)
        num_chunks = args.workers * 4
        docs_per_chunk = math.ceil(total_docs / num_chunks)
        
        print(f"Creating {num_chunks} chunks (~{docs_per_chunk:,} docs each)")
        print(f"Each worker will process 4 chunks sequentially")
        
        # Create chunk ranges
        chunk_ranges = []
        for i in range(num_chunks):
            start_idx = i * docs_per_chunk
            end_idx = min((i + 1) * docs_per_chunk, total_docs)
            if start_idx < total_docs:
                chunk_ranges.append((start_idx, end_idx))
        
        # Assign chunks to workers (each worker gets 4 chunks)
        chunks_per_worker = 4
        worker_tasks = []
        
        for worker_id in range(args.workers):
            worker_chunk_start = worker_id * chunks_per_worker
            worker_chunk_end = min(worker_chunk_start + chunks_per_worker, len(chunk_ranges))
            
            if worker_chunk_start < len(chunk_ranges):
                worker_chunk_ranges = chunk_ranges[worker_chunk_start:worker_chunk_end]
                
                # Prepare tokenizer arguments
                tokenizer_args = {
                    'tokenizer_type': args.tokenizer_type,
                    'tokenizer_model': getattr(args, 'tokenizer_model', None),
                    'append_eod': args.append_eod,
                    'split_sentences': args.split_sentences,
                    'json_keys': args.json_keys,
                    'batch_size': args.batch_size
                }
                
                worker_tasks.append((
                    worker_id,
                    worker_chunk_ranges,
                    args.input,
                    args.output_prefix,
                    tokenizer_args
                ))
        
        print(f"Starting {len(worker_tasks)} workers for parallel processing...")
        
        # Process chunks in parallel
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(process_worker_chunks, worker_tasks)
        
        # Check results and collect worker prefixes
        worker_prefixes = []
        failed_workers = []
        
        for success, worker_id, prefixes in results:
            if success:
                if prefixes:
                    worker_prefixes.extend(prefixes)
            else:
                failed_workers.append(worker_id)
        
        if failed_workers:
            print(f"Failed workers: {failed_workers}")
            return 1
        
        # Remove duplicates from worker_prefixes
        worker_prefixes = list(set(worker_prefixes))
        
        # Merge the results
        tokenizer_args = {
            'tokenizer_type': args.tokenizer_type,
            'tokenizer_model': getattr(args, 'tokenizer_model', None),
            'split_sentences': args.split_sentences
        }
        
        merge_worker_outputs(worker_prefixes, args.output_prefix, 
                           args.json_keys, tokenizer_args)
        
        total_time = time.time() - start_time
        avg_speed = total_docs / total_time if total_time > 0 else 0
        
        print(f"Indexed parallel preprocessing completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Total documents: {total_docs:,}")
        print(f"Average speed: {avg_speed:.1f} docs/s")
        print(f"Speedup vs streaming (~400 docs/s): {avg_speed/400:.1f}x")
        
        return 0
        
    except Exception as e:
        print(f"Error during indexed parallel processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
