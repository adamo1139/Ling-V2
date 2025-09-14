"""Download pretraining data from https://huggingface.co/datasets/adamo1139/fineweb2-pol"""
import os
import certifi
from datasets import load_dataset, Dataset

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

dataset_name = "adamo1139/fineweb2-pol"
name = dataset_name.split('/')[-1]

# Stream the dataset to avoid downloading everything
print("Loading dataset in streaming mode...")
ds_stream = load_dataset(dataset_name, split='train', streaming=True)

# Take only first 100k samples
print("Taking first 100k samples...")
ds_stream = ds_stream.take(100000)

# Convert to regular dataset and filter to text column only
print("Converting to dataset and filtering columns...")
ds = Dataset.from_generator(lambda: ds_stream).select_columns(['text'])

# Save as Parquet
print("Saving to Parquet...")
ds.to_parquet(f"{name}-100k.parquet")
print(f"Saved {len(ds)} samples to {name}-100k.parquet")
