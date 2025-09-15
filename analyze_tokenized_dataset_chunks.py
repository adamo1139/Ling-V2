#!/usr/bin/env python3
"""
Analyze tokenized datasets in Megatron's indexed format (.bin/.idx files)
Provides statistics like sample count, token length distribution, etc.
Processes multiple datasets in a folder and outputs results to CSV.
"""

import os
import sys
import argparse
import struct
import numpy as np
import csv
from typing import List, Tuple, Optional

class IndexedDatasetReader:
    """
    Reader for Megatron's indexed dataset format
    Compatible with the binary format used by IndexedDatasetBuilder
    """
    
    def __init__(self, path_prefix: str):
        self.path_prefix = path_prefix
        self.idx_path = f"{path_prefix}.idx"
        self.bin_path = f"{path_prefix}.bin"
        
        # Verify files exist
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"Binary file not found: {self.bin_path}")
            
        self._load_index()
    
    def _load_index(self):
        """Load the index file to get document boundaries"""
        _INDEX_HEADER = b"MMIDIDX\x00\x00"
        
        with open(self.idx_path, 'rb') as f:
            # Read header
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header, cannot read: {self.idx_path}"
            
            # Read version
            version = struct.unpack('<Q', f.read(8))[0]
            assert version == 1, f"Bad version, cannot read: {self.idx_path}"
            
            # Read dtype info
            dtype_code = struct.unpack('<B', f.read(1))[0]
            
            # Read sequence count
            self.sequence_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read document count  
            self.document_count = struct.unpack('<Q', f.read(8))[0]
            
            # Current offset for reading arrays
            offset = f.tell()
        
        # Use memory mapping for efficient reading of arrays
        self.bin_buffer_mmap = np.memmap(self.idx_path, mode='r', order='C')
        self.bin_buffer = memoryview(self.bin_buffer_mmap)
        
        # Read sequence lengths
        self.sequence_lengths = np.frombuffer(
            self.bin_buffer, dtype=np.int32, count=self.sequence_count, offset=offset
        )
        
        # Read sequence pointers
        self.sequence_pointers = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        
        # Read document indices  
        self.document_indices = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
    
    def get_document_lengths(self) -> List[int]:
        """Get list of document lengths (number of tokens per document)"""
        doc_lengths = []
        doc_start_idx = 0
        
        # Document indices mark the end of each document
        for doc_end_idx in self.document_indices[1:]:  # Skip the first 0
            # Sum sequence lengths for this document
            doc_length = int(np.sum(self.sequence_lengths[doc_start_idx:doc_end_idx]))
            doc_lengths.append(doc_length)
            doc_start_idx = doc_end_idx
            
        return doc_lengths
    
    def get_num_documents(self) -> int:
        """Get total number of documents"""
        return self.document_count - 1  # -1 because document_indices includes the initial 0
    
    def get_total_tokens(self) -> int:
        """Get total number of tokens across all documents"""
        return int(np.sum(self.sequence_lengths))
    
    def __del__(self):
        """Clean up memory mapped files"""
        if hasattr(self, 'bin_buffer_mmap'):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

def analyze_dataset(data_path: str) -> Tuple[dict, List[int]]:
    """
    Analyze a tokenized dataset and return statistics
    
    Args:
        data_path: Path prefix to the dataset (without .bin/.idx extension)
        
    Returns:
        Tuple of (statistics dictionary, document lengths list)
    """
    try:
        reader = IndexedDatasetReader(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None, None
    
    doc_lengths = reader.get_document_lengths()
    num_docs = reader.get_num_documents()
    total_tokens = reader.get_total_tokens()
    
    # Calculate statistics
    doc_lengths_array = np.array(doc_lengths)
    
    stats = {
        'data_path': data_path,
        'num_documents': num_docs,
        'total_tokens': total_tokens,
        'max_length': int(np.max(doc_lengths_array)),
        'min_length': int(np.min(doc_lengths_array)),
        'mean_length': float(np.mean(doc_lengths_array)),
        'median_length': float(np.median(doc_lengths_array)),
        'std_length': float(np.std(doc_lengths_array)),
        'percentile_95': float(np.percentile(doc_lengths_array, 95)),
        'percentile_99': float(np.percentile(doc_lengths_array, 99)),
        'percentile_5': float(np.percentile(doc_lengths_array, 5)),
    }
    
    return stats, doc_lengths

def find_dataset_pairs(folder_path: str) -> List[str]:
    """
    Find valid .bin/.idx file pairs in the specified folder
    
    Args:
        folder_path: Path to the folder containing dataset files
        
    Returns:
        List of data_path prefixes (without extension) for valid pairs
    """
    valid_pairs = []
    bin_files = []
    
    # Find all .bin files
    for filename in os.listdir(folder_path):
        if filename.endswith('.bin'):
            bin_files.append(filename[:-4])  # Remove .bin extension
    
    # Check for corresponding .idx files
    for base_name in bin_files:
        idx_path = os.path.join(folder_path, f"{base_name}.idx")
        bin_path = os.path.join(folder_path, f"{base_name}.bin")
        
        if os.path.exists(idx_path) and os.path.exists(bin_path):
            valid_pairs.append(os.path.join(folder_path, base_name))
        else:
            if not os.path.exists(idx_path):
                print(f"Warning: Missing .idx file for {base_name}.bin", file=sys.stderr)
            if not os.path.exists(bin_path):
                print(f"Warning: Missing .bin file for {base_name}.idx", file=sys.stderr)
    
    return valid_pairs

def calculate_overall_stats(all_stats: List[dict]) -> dict:
    """
    Calculate overall statistics across all datasets
    
    Args:
        all_stats: List of statistics dictionaries from individual datasets
        
    Returns:
        Dictionary with overall statistics
    """
    if not all_stats:
        return {}
    
    overall_stats = {
        'data_path': 'OVERALL',
        'num_documents': sum(s['num_documents'] for s in all_stats),
        'total_tokens': sum(s['total_tokens'] for s in all_stats),
        'max_length': max(s['max_length'] for s in all_stats),
        'min_length': min(s['min_length'] for s in all_stats),
        'mean_length': np.mean([s['mean_length'] for s in all_stats]),
        'median_length': np.mean([s['median_length'] for s in all_stats]),
        'std_length': np.mean([s['std_length'] for s in all_stats]),
        'percentile_95': np.mean([s['percentile_95'] for s in all_stats]),
        'percentile_99': np.mean([s['percentile_99'] for s in all_stats]),
        'percentile_5': np.mean([s['percentile_5'] for s in all_stats]),
    }
    
    # Calculate document counts above thresholds
    total_docs_gt_512 = 0
    total_docs_gt_1024 = 0
    total_docs_gt_2048 = 0
    total_docs_all = 0
    
    for stats in all_stats:
        if stats['doc_lengths'] is not None:
            doc_array = np.array(stats['doc_lengths'])
            total_docs_gt_512 += np.sum(doc_array > 512)
            total_docs_gt_1024 += np.sum(doc_array > 1024)
            total_docs_gt_2048 += np.sum(doc_array > 2048)
            total_docs_all += len(doc_array)
    
    overall_stats.update({
        'num_docs_gt_512': int(total_docs_gt_512),
        'num_docs_gt_1024': int(total_docs_gt_1024),
        'num_docs_gt_2048': int(total_docs_gt_2048),
        'pct_docs_gt_512': (total_docs_gt_512 / total_docs_all * 100) if total_docs_all > 0 else 0,
        'pct_docs_gt_1024': (total_docs_gt_1024 / total_docs_all * 100) if total_docs_all > 0 else 0,
        'pct_docs_gt_2048': (total_docs_gt_2048 / total_docs_all * 100) if total_docs_all > 0 else 0,
    })
    
    return overall_stats

def write_results_to_csv(results: List[dict], output_csv: str):
    """
    Write analysis results to CSV file
    
    Args:
        results: List of statistics dictionaries
        output_csv: Path to output CSV file
    """
    fieldnames = [
        'data_path', 'num_documents', 'total_tokens', 'min_length', 'max_length',
        'mean_length', 'median_length', 'std_length', 'percentile_5', 'percentile_95',
        'percentile_99', 'num_docs_gt_512', 'num_docs_gt_1024', 'num_docs_gt_2048',
        'pct_docs_gt_512', 'pct_docs_gt_1024', 'pct_docs_gt_2048'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Ensure all required fields are present
            row = {field: result.get(field, 0) for field in fieldnames}
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze tokenized datasets in Megatron's indexed format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_tokenized_dataset_chunks.py --folder /path/to/datasets
  python analyze_tokenized_dataset_chunks.py --folder /path/to/datasets --output-csv results.csv
        """
    )
    
    parser.add_argument(
        '--folder', 
        type=str, 
        required=True,
        help='Path to folder containing dataset files (.bin and .idx pairs)'
    )
    
    parser.add_argument(
        '--output-csv', 
        type=str, 
        default='analysis_results.csv',
        help='Output CSV file path (default: analysis_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Find valid dataset pairs in the folder
    print(f"Scanning folder: {args.folder}", file=sys.stderr)
    dataset_pairs = find_dataset_pairs(args.folder)
    
    if not dataset_pairs:
        print("No valid dataset pairs found in the folder.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(dataset_pairs)} valid dataset pairs", file=sys.stderr)
    
    # Analyze each dataset
    all_results = []
    for i, data_path in enumerate(dataset_pairs, 1):
        print(f"Processing {i}/{len(dataset_pairs)}: {os.path.basename(data_path)}", file=sys.stderr)
        
        stats, doc_lengths = analyze_dataset(data_path)
        
        if stats is not None:
            # Add document length information for threshold calculations
            stats['doc_lengths'] = doc_lengths
            
            # Calculate document counts above thresholds
            if doc_lengths:
                doc_array = np.array(doc_lengths)
                stats.update({
                    'num_docs_gt_512': int(np.sum(doc_array > 512)),
                    'num_docs_gt_1024': int(np.sum(doc_array > 1024)),
                    'num_docs_gt_2048': int(np.sum(doc_array > 2048)),
                    'pct_docs_gt_512': float(np.mean(doc_array > 512) * 100),
                    'pct_docs_gt_1024': float(np.mean(doc_array > 1024) * 100),
                    'pct_docs_gt_2048': float(np.mean(doc_array > 2048) * 100),
                })
            else:
                stats.update({
                    'num_docs_gt_512': 0,
                    'num_docs_gt_1024': 0,
                    'num_docs_gt_2048': 0,
                    'pct_docs_gt_512': 0.0,
                    'pct_docs_gt_1024': 0.0,
                    'pct_docs_gt_2048': 0.0,
                })
            
            all_results.append(stats)
    
    # Calculate overall statistics
    overall_stats = calculate_overall_stats(all_results)
    
    # Add overall stats to results
    if overall_stats:
        all_results.append(overall_stats)
    
    # Write results to CSV
    write_results_to_csv(all_results, args.output_csv)
    print(f"Results written to: {args.output_csv}", file=sys.stderr)

if __name__ == "__main__":
    main()
