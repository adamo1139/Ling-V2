"""Download pretraining data from https://huggingface.co/datasets/adamo1139/fineweb2-pol"""
import os
import certifi
from datasets import load_dataset

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

dataset_name = "adamo1139/fineweb2-pol"
name = dataset_name.split('/')[-1]

# Load the full dataset directly (no streaming)
print("Loading full dataset...")
ds = load_dataset(dataset_name, split='train')

# Filter to text column only
print("Filtering columns...")
ds = ds.select_columns(['text'])

# Save as Parquet
print("Saving to Parquet...")
ds.to_parquet(f"{name}-full.parquet")
print(f"Saved {len(ds)} samples to {name}-full.parquet")
