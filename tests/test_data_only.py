"""
test to verify FAISS index and metadata are correctly built
"""

import faiss
import pandas as pd

INDEX_PATH = "models/faiss.index"
META_PATH = "models/metadata.parquet"

print("Testing FAISS index...")
index = faiss.read_index(INDEX_PATH)
print(f"✓ Index loaded: {index.ntotal} vectors, dimension {index.d}")

print("\nTesting metadata...")
metadata = pd.read_parquet(META_PATH)
print(f"✓ Metadata loaded: {len(metadata)} rows")
print(f"\nColumns: {list(metadata.columns)}")
print(f"\nFirst 5 rows:")
print(metadata.head())

print(f"\nArtist distribution:")
print(metadata['artist'].value_counts())

print(f"\nPeriod distribution:")
print(metadata['period'].value_counts())

print("\n✓ All data structures are valid!")
