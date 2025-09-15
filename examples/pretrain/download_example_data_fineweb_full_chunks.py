"""Download pretraining data as original chunks from https://huggingface.co/datasets/adamo1139/fineweb2-pol"""
import os
import certifi
from huggingface_hub import snapshot_download

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

dataset_name = "adamo1139/fineweb2-pol"
name = dataset_name.split('/')[-1]

# Create chunks directory
chunks_dir = f"{name}-chunks"
print(f"Creating directory: {chunks_dir}")

print("Downloading dataset repository to local directory (keeps original chunks)...")
snapshot_download(
    repo_id=dataset_name,
    repo_type="dataset",
    local_dir=chunks_dir
)

print(f"Dataset downloaded successfully to: {chunks_dir}")
print("Original parquet files are preserved in their chunk structure.")

# Count the parquet files
parquet_count = 0
for root, dirs, files in os.walk(chunks_dir):
    for file in files:
        if file.endswith('.parquet'):
            parquet_count += 1

print(f"Found {parquet_count} parquet chunk files ready for parallel processing!")
