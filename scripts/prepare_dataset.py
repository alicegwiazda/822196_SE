"""
Dataset Preparation Script for Art Guide System
Processes artwork images from data/artworks/ and creates:
1. FAISS vector index with CLIP embeddings
2. Metadata parquet file with artwork information

Based on Task 2 requirements: Using Kaggle art datasets (Best Artworks of All Time)
curated into a subset for exhibition scenario.

Authors: AlBeSa Team
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DATA_DIR = "data/artworks"
OUTPUT_DIR = "models"
INDEX_FILE = f"{OUTPUT_DIR}/faiss.index"
METADATA_FILE = f"{OUTPUT_DIR}/metadata.parquet"

# Artist metadata (periods can be refined based on actual data)
ARTIST_INFO = {
    "Da_Vinci": {"full_name": "Leonardo da Vinci", "period": "Renaissance", "years": "1452-1519"},
    "Monet": {"full_name": "Claude Monet", "period": "Impressionism", "years": "1840-1926"},
    "Picasso": {"full_name": "Pablo Picasso", "period": "Cubism/Modern", "years": "1881-1973"},
    "Rembrandt": {"full_name": "Rembrandt van Rijn", "period": "Dutch Golden Age", "years": "1606-1669"},
    "Van_Gogh": {"full_name": "Vincent van Gogh", "period": "Post-Impressionism", "years": "1853-1890"}
}

def load_clip_model():
    """Load CLIP model for generating embeddings."""
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return model, processor, device


def generate_embedding(image_path, model, processor, device):
    """
    Generate CLIP embedding for an image.
    
    Args:
        image_path: Path to image file
        model: CLIP model
        processor: CLIP processor
        device: torch device (cuda/cpu)
        
    Returns:
        numpy array of shape (512,) normalized to unit length
    """
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        
        # Convert to numpy and normalize (L2 norm)
        embedding = embedding.cpu().numpy().astype("float32").flatten()
        embedding /= np.linalg.norm(embedding)
        
        return embedding
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def collect_artworks(data_dir):
    """
    Scan data/artworks directory and collect all image files.
    
    Returns:
        List of dicts with artwork information
    """
    artworks = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    print(f"\nScanning {data_dir} for artwork images...")
    
    for artist_dir in sorted(Path(data_dir).iterdir()):
        if not artist_dir.is_dir():
            continue
        
        artist_key = artist_dir.name
        if artist_key not in ARTIST_INFO:
            print(f"Warning: Unknown artist {artist_key}, skipping...")
            continue
        
        artist_info = ARTIST_INFO[artist_key]
        
        # Find all images in artist directory
        image_files = [
            f for f in artist_dir.iterdir() 
            if f.suffix.lower() in supported_formats
        ]
        
        print(f"  {artist_info['full_name']}: {len(image_files)} images")
        
        for img_file in image_files:
            artworks.append({
                'artist': artist_info['full_name'],
                'artist_key': artist_key,
                'period': artist_info['period'],
                'years': artist_info['years'],
                'title': img_file.stem.replace('_', ' ').title(),
                'image_path': str(img_file),
                'filename': img_file.name
            })
    
    print(f"\nTotal artworks found: {len(artworks)}")
    return artworks


def build_index(artworks, model, processor, device):
    """
    Build FAISS index from artwork images.
    
    Args:
        artworks: List of artwork dicts
        model: CLIP model
        processor: CLIP processor
        device: torch device
        
    Returns:
        tuple: (faiss_index, metadata_df)
    """
    print("\nGenerating CLIP embeddings...")
    
    embeddings = []
    valid_artworks = []
    
    for i, artwork in enumerate(artworks):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(artworks)} images...")
        
        embedding = generate_embedding(artwork['image_path'], model, processor, device)
        
        if embedding is not None:
            embeddings.append(embedding)
            valid_artworks.append(artwork)
    
    print(f"Successfully generated {len(embeddings)} embeddings")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype='float32')
    
    # Build FAISS index (L2 distance, but embeddings are normalized so it's equivalent to cosine)
    print("\nBuilding FAISS index...")
    dimension = embeddings_array.shape[1]  # Should be 512 for CLIP
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    print(f"  Index built with {index.ntotal} vectors of dimension {dimension}")
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(valid_artworks)
    
    return index, metadata_df


def save_outputs(index, metadata_df, index_file, metadata_file):
    """Save FAISS index and metadata to disk."""
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    
    print(f"\nSaving FAISS index to {index_file}...")
    faiss.write_index(index, index_file)
    
    print(f"Saving metadata to {metadata_file}...")
    metadata_df.to_parquet(metadata_file, index=False)
    
    print("\n✓ Dataset preparation complete!")
    print(f"  - FAISS index: {index.ntotal} vectors")
    print(f"  - Metadata: {len(metadata_df)} artworks")
    print(f"  - Artists: {metadata_df['artist'].nunique()}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Art Guide - Dataset Preparation")
    print("=" * 70)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found!")
        print("Please ensure artwork images are in data/artworks/<artist_name>/")
        return 1
    
    # Load CLIP model
    model, processor, device = load_clip_model()
    
    # Collect artwork files
    artworks = collect_artworks(DATA_DIR)
    
    if len(artworks) == 0:
        print("Error: No artworks found!")
        return 1
    
    # Build FAISS index
    index, metadata_df = build_index(artworks, model, processor, device)
    
    # Save outputs
    save_outputs(index, metadata_df, INDEX_FILE, METADATA_FILE)
    
    # Display summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print("=" * 70)
    print(metadata_df.groupby('artist').size().to_string())
    print("\nPeriods represented:")
    print(metadata_df.groupby('period').size().to_string())
    
    return 0


if __name__ == '__main__':
    exit(main())
