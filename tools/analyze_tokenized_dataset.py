#!/usr/bin/env python3
"""
Analyze tokenized datasets in Megatron's indexed format (.bin/.idx files)
Provides statistics like sample count, token length distribution, etc.
"""

import os
import sys
import argparse
import struct
import numpy as np
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
        print(f"Error: {e}")
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

def print_analysis(stats: dict, doc_lengths: List[int] = None):
    """Print analysis results in a formatted way"""
    if stats is None:
        return
        
    print("="*60)
    print("TOKENIZED DATASET ANALYSIS")
    print("="*60)
    print(f"Dataset Path: {stats['data_path']}")
    print(f"Index File: {stats['data_path']}.idx")
    print(f"Binary File: {stats['data_path']}.bin")
    print()
    
    print("DOCUMENT STATISTICS:")
    print("-" * 30)
    print(f"Number of Documents: {stats['num_documents']:,}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print()
    
    print("TOKEN LENGTH DISTRIBUTION:")
    print("-" * 30)
    print(f"Minimum Length: {stats['min_length']:,} tokens")
    print(f"Maximum Length: {stats['max_length']:,} tokens")
    print(f"Mean Length: {stats['mean_length']:.1f} tokens")
    print(f"Median Length: {stats['median_length']:.1f} tokens")
    print(f"Standard Deviation: {stats['std_length']:.1f} tokens")
    print()
    
    print("PERCENTILES:")
    print("-" * 30)
    print(f"5th Percentile: {stats['percentile_5']:.1f} tokens")
    print(f"95th Percentile: {stats['percentile_95']:.1f} tokens")
    print(f"99th Percentile: {stats['percentile_99']:.1f} tokens")
    print()
    
    # Additional insights
    avg_tokens_per_doc = stats['mean_length']
    if avg_tokens_per_doc < 100:
        length_category = "Very Short"
    elif avg_tokens_per_doc < 300:
        length_category = "Short"
    elif avg_tokens_per_doc < 600:
        length_category = "Medium"
    elif avg_tokens_per_doc < 1000:
        length_category = "Long"
    else:
        length_category = "Very Long"
    
    print("INSIGHTS:")
    print("-" * 30)
    print(f"Average document category: {length_category}")
    
    # Calculate percentages if we have document lengths
    if doc_lengths:
        doc_array = np.array(doc_lengths)
        pct_512 = np.mean(doc_array > 512) * 100
        pct_1024 = np.mean(doc_array > 1024) * 100
        print(f"Documents longer than 512 tokens: {pct_512:.1f}%")
        print(f"Documents longer than 1024 tokens: {pct_1024:.1f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze tokenized datasets in Megatron's indexed format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_tokenized_dataset.py --data-path processed_data_text_document
  python analyze_tokenized_dataset.py --data-path fineweb-100k/processed_data_text_document
        """
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        required=True,
        help='Path prefix to the dataset files (without .bin/.idx extension)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Only print essential statistics'
    )
    
    args = parser.parse_args()
    
    # Analyze the dataset
    stats, doc_lengths = analyze_dataset(args.data_path)
    
    if stats is None:
        sys.exit(1)
    
    # Print results
    if args.quiet:
        print(f"Documents: {stats['num_documents']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Avg Length: {stats['mean_length']:.1f}")
        print(f"Max Length: {stats['max_length']:,}")
        print(f"95th %ile: {stats['percentile_95']:.1f}")
    else:
        print_analysis(stats, doc_lengths)

if __name__ == "__main__":
    main()
