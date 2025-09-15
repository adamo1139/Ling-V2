"""Download pretraining data as original chunks from https://huggingface.co/datasets/adamo1139/fineweb2-pol"""
import os
import certifi
from datasets import load_dataset

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

dataset_name = "adamo1139/fineweb2-pol"
name = dataset_name.split('/')[-1]

# Create chunks directory
chunks_dir = f"{name}-chunks"
print(f"Creating directory: {chunks_dir}")

print("Downloading dataset to local directory (keeps original chunks)...")
ds = load_dataset(dataset_name, local_dir=chunks_dir)

print(f"Dataset downloaded successfully to: {chunks_dir}")
print("Original parquet files are preserved in their chunk structure.")
print("You can now process these chunks in parallel!")
