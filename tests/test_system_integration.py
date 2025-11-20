"""
tests script to verify the system works with real FAISS index.
whats more - it tests both embedding generation and vector search functionality.
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuration
INDEX_PATH = "models/faiss.index"
META_PATH = "models/metadata.parquet"
TEST_IMAGE = "data/sample_images/Monet_artwork_1.jpg"

def test_index_loading():
    """Test loading FAISS index and metadata."""
    print("Testing index loading...")
    
    if not os.path.exists(INDEX_PATH):
        print(f"FAISS index not found at {INDEX_PATH}")
        return False
    
    if not os.path.exists(META_PATH):
        print(f"Metadata not found at {META_PATH}")
        return False
    
    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_parquet(META_PATH)
    
    print(f"FAISS index loaded: {index.ntotal} vectors")
    print(f"Metadata loaded: {len(metadata)} artworks")
    print(f"Artists: {metadata['artist'].nunique()}")
    
    return True


def test_embedding_generation():
    """Test CLIP embedding generation."""
    print("\nTesting embedding generation...")
    
    if not os.path.exists(TEST_IMAGE):
        print(f"Test image not found at {TEST_IMAGE}")
        return False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    img = Image.open(TEST_IMAGE).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    
    embedding = embedding.cpu().numpy().astype("float32")
    embedding /= np.linalg.norm(embedding)
    
    print(f"Embedding generated: shape {embedding.shape}")
    print(f"L2 norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
    
    return True


def test_vector_search():
    """Test end-to-end vector search."""
    print("\nTesting vector search...")
    
    # Load index
    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_parquet(META_PATH)
    
    # Generate embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    img = Image.open(TEST_IMAGE).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    
    embedding = embedding.cpu().numpy().astype("float32")
    embedding /= np.linalg.norm(embedding)
    
    # Search
    k = 3
    distances, indices = index.search(embedding, k)
    
    print(f"Search completed, found {len(indices[0])} results")
    print("\nTop 3 matches:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        row = metadata.iloc[idx]
        confidence = np.exp(-dist)
        print(f"{i+1}. {row['artist']} - {row['title']}")
        print(f"Period: {row['period']} | Distance: {dist:.4f} | Confidence: {confidence:.2%}")
    
    return True


def main():
    print("Art Guide - System Verification Test")
    
    success = True
    
    # Test 1: Index loading
    if not test_index_loading():
        success = False
    
    # Test 2: Embedding generation
    if not test_embedding_generation():
        success = False
    
    # Test 3: Vector search
    if not test_vector_search():
        success = False
    
    if success:
        print("All tests passed! System is ready.")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
