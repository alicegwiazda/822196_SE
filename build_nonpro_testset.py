import os
import random
import csv
import json
import math
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(42)

import os
import random
import csv
import json
import math
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(42)

SRC_DIR = Path("data/full_dataset_kaggle/images/images")
OUT_DIR = Path("data")
ARTISTS_CSV = Path("data/full_dataset_kaggle/artists.csv")

N = 50                # number of artworks to sample
K_VARIANTS = 5        # number of "non-professional" photos per artwork

# Create output directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR/"clean").mkdir(parents=True, exist_ok=True)
(OUT_DIR/"nonpro").mkdir(parents=True, exist_ok=True)

print(f"Searching for images in: {SRC_DIR}")
print(f"Results will be saved to: {OUT_DIR}")
SRC_DIR = Path("data/full_dataset_kaggle/images/images")
OUT_DIR = Path("data")
ARTISTS_CSV = Path("data/full_dataset_kaggle/artists.csv")

N = 50                # number of artworks to sample
K_VARIANTS = 5        # number of "non-professional" photos per artwork

# Create output directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR/"clean").mkdir(parents=True, exist_ok=True)
(OUT_DIR/"nonpro").mkdir(parents=True, exist_ok=True)

print(f"Looking for images in: {SRC_DIR}")
print(f"Results will be saved to: {OUT_DIR}")

# 1) Collect all images from artist subdirectories
images = []
for artist_dir in SRC_DIR.iterdir():
    if artist_dir.is_dir():
        artist_images = [p for p in artist_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        images.extend(artist_images)

print(f"Found {len(images)} images")
assert len(images) >= N, f"Not enough images in {SRC_DIR} (min {N})"

# Randomly select N images
sample = random.sample(images, N)

# 2) Load artist metadata
artists_df = pd.read_csv(ARTISTS_CSV)
# Create mapping artist name -> metadata
artist_lookup = {}
for _, row in artists_df.iterrows():
    # Normalize artist name for better matching
    artist_name = row['name'].replace(' ', '_').replace('-', '_')
    artist_lookup[artist_name] = {
        'name': row['name'],
        'genre': row.get('genre', ''),
        'nationality': row.get('nationality', ''),
        'years': row.get('years', ''),
        'bio': row.get('bio', '')[:200] + '...' if len(str(row.get('bio', ''))) > 200 else row.get('bio', '')
    }

# Add variations of names to improve matching
artist_variations = {}
for _, row in artists_df.iterrows():
    original_name = row['name']
    variations = [
        original_name.replace(' ', '_'),
        original_name.replace(' ', '_').replace('-', '_'),
        original_name.replace('ü', 'u').replace('é', 'e').replace('ç', 'c'),
        original_name.replace(' ', '_').replace('ü', 'u').replace('é', 'e').replace('ç', 'c')
    ]
    
    for variation in variations:
        if variation not in artist_variations:
            artist_variations[variation] = {
                'name': original_name,
                'genre': row.get('genre', ''),
                'nationality': row.get('nationality', ''),
                'years': row.get('years', ''),
                'bio': row.get('bio', '')[:200] + '...' if len(str(row.get('bio', ''))) > 200 else row.get('bio', '')
            }

print(f"Loaded metadata for {len(artists_df)} artists with {len(artist_variations)} name variations")

# 3) Process selected images
rows = []
for p in tqdm(sample, desc="Processing images"):
    # Extract artist name from path (e.g., Claude_Monet from Claude_Monet/Claude_Monet_1.jpg)
    artist_folder_name = p.parent.name
    img_id = p.stem  # e.g., Claude_Monet_1
    
    # Find artist metadata with improved matching
    artist_info = artist_variations.get(artist_folder_name, {})
    if not artist_info:
        # Try some common variations
        for variant in [artist_folder_name.replace('_', ' '), 
                       artist_folder_name.replace('╠ê', 'ü'),
                       artist_folder_name.replace('Du╠êrer', 'Dürer')]:
            if variant in artist_lookup:
                artist_info = artist_lookup[variant]
                break
    
    try:
        # Open and copy as "clean"
        im = Image.open(p).convert("RGB")
        clean_path = OUT_DIR/"clean"/f"{img_id}.jpg"
        im.save(clean_path, quality=95)
        
        rows.append({
            'id': img_id,
            'artist': artist_info.get('name', artist_folder_name),
            'genre': artist_info.get('genre', ''),
            'nationality': artist_info.get('nationality', ''),
            'years': artist_info.get('years', ''),
            'bio': artist_info.get('bio', ''),
            'original_path': str(p),
            'clean_image_path': str(clean_path)
        })
        
    except Exception as e:
        print(f"Error processing {p}: {e}")

print(f"Successfully processed {len(rows)} images")

# Save metadata
metadata_df = pd.DataFrame(rows)
metadata_df.to_csv(OUT_DIR/"metadata.csv", index=False)
print(f"Saved metadata to: {OUT_DIR}/metadata.csv")

# 4) "Non-professional" augmentation functions
def rand_perspective(img):
    """Simulates slight tilt/perspective"""
    w, h = img.size
    dx, dy = int(0.08*w), int(0.08*h)
    
    # Simple rotation instead of complex perspective transformation
    angle = random.uniform(-8, 8)  # degrees
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

def add_glare(img):
    """Adds glare/reflection"""
    w, h = img.size
    overlay = Image.new("RGB", (w, h), (255, 255, 255))
    
    # Glare position
    cx = random.randint(int(0.2*w), int(0.8*w))
    cy = random.randint(int(0.1*h), int(0.4*h))
    
    # Simple implementation - white circle with gradient
    for x in range(w):
        for y in range(h):
            d = math.hypot(x-cx, y-cy)
            max_radius = 0.3 * max(w, h)
            if d < max_radius:
                alpha = max(0, 1.0 - d/max_radius) * 0.3
                # Blend pixel
                orig_pixel = img.getpixel((x, y))
                new_pixel = tuple(int(orig_pixel[i] * (1-alpha) + 255 * alpha) for i in range(3))
                img.putpixel((x, y), new_pixel)
    
    return img

def nonpro_variants(img):
    """Generates 5 non-professional photo variants"""
    variants = []
    
    # 1: tilt + slight blur
    v1 = rand_perspective(img.copy())
    v1 = v1.filter(ImageFilter.GaussianBlur(radius=0.8))
    variants.append(("tilt+blur", v1))
    
    # 2: compression + resize (simulates poor quality)
    v2 = img.copy()
    v2 = v2.resize((int(img.width*0.7), int(img.height*0.7)), Image.LANCZOS)
    v2 = v2.resize(img.size, Image.LANCZOS)
    variants.append(("lowres+compress", v2))
    
    # 3: poor lighting + bad color balance
    v3 = img.copy()
    v3 = ImageEnhance.Brightness(v3).enhance(0.8)
    v3 = ImageEnhance.Color(v3).enhance(0.85)
    v3 = ImageEnhance.Contrast(v3).enhance(0.9)
    variants.append(("lowlight+wb", v3))
    
    # 4: glare/reflection
    v4 = img.copy()
    v4 = add_glare(v4)
    variants.append(("glare", v4))
    
    # 5: motion blur
    v5 = img.copy()
    v5 = v5.filter(ImageFilter.GaussianBlur(radius=1.5))
    variants.append(("motion_blur", v5))
    
    return variants

# 5) Generate non-professional variants
print("Generating non-professional variants...")
variant_records = []

for row in tqdm(rows, desc="Augmentation"):
    try:
        base_img = Image.open(row["clean_image_path"]).convert("RGB")
        target_dir = OUT_DIR/"nonpro"/row["id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        variants = nonpro_variants(base_img)
        
        for i, (tag, variant_img) in enumerate(variants, start=1):
            output_path = target_dir / f'{row["id"]}_v{i}.jpg'
            variant_img.save(output_path, quality=80)
            
            variant_records.append({
                'variant_path': str(output_path),
                'id': row["id"],
                'artist': row["artist"],
                'transforms_applied': tag,
                'variant_number': i,
                'notes': f"Generated from {row['clean_image_path']}"
            })
            
    except Exception as e:
        print(f"Augmentation error for {row['id']}: {e}")

# Save variant information
variants_df = pd.DataFrame(variant_records)
variants_df.to_csv(OUT_DIR/"nonpro_variants.csv", index=False)

print(f"\n✅ Done!")
print(f"📁 Metadata: {OUT_DIR}/metadata.csv ({len(rows)} images)")
print(f"📁 Variants: {OUT_DIR}/nonpro_variants.csv ({len(variant_records)} variants)")
print(f"📁 Clean images: {OUT_DIR}/clean/")
print(f"📁 Non-professional variants: {OUT_DIR}/nonpro/")
print(f"🎯 Average {len(variant_records)/len(rows):.1f} variants per image")