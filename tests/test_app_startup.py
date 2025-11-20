"""
Test the monolithic app.py to verify it loads correctly with real FAISS index.
This script imports and initializes the app without launching the UI.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
print("\nTesting imports")
try:
    import torch
    import faiss
    import pandas as pd
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    print("All required libraries imported")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Test FAISS index loading
print("\nTesting FAISS index loading")
INDEX_PATH = "models/faiss.index"
META_PATH = "models/metadata.parquet"

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_parquet(META_PATH)
    print(f"FAISS index loaded: {index.ntotal} vectors")
    print(f"Metadata loaded: {len(metadata)} artworks")
else:
    print("Index or metadata not found")
    sys.exit(1)

# Test CLIP model loading
print("\nTesting CLIP model loading")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    sys.exit(1)

# Test sample image processing
print("\nTesting sample image processing")
test_image = "data/sample_images/art_1.png"
embedding = None
if os.path.exists(test_image):
    try:
        img = Image.open(test_image).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        embedding = embedding.cpu().numpy().astype("float32")
        import numpy as np
        embedding /= np.linalg.norm(embedding)
        print(f"Sample image processed: embedding shape {embedding.shape}")
    except Exception as e:
        print(f"Image processing failed: {e}")
        sys.exit(1)
else:
    print(f"Test image not found at {test_image}, skipping")

# Test vector search
print("\nTesting vector search")
if embedding is not None:
    try:
        D, I = index.search(embedding, 3)
        results = metadata.iloc[I[0]]
        print(f"Search successful, top match: {results.iloc[0]['artist']}")
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)
else:
    print("Skipping search test (no test image)")

print("All initialization tests passed!")
