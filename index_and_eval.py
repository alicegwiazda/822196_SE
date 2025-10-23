from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# Paths adapted to your structure
DATA_DIR = Path("data")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("Loading CLIP model...")

# Use model compatible with sentence-transformers
model = SentenceTransformer("clip-ViT-B-32", device=device)

# Load metadata
print("Loading metadata...")
meta = pd.read_csv(DATA_DIR/"metadata.csv")
nonpro = pd.read_csv(DATA_DIR/"nonpro_variants.csv")

print(f"Found {len(meta)} clean images and {len(nonpro)} non-professional variants")

def emb_image(path):
    """Generates embedding for image"""
    try:
        img = Image.open(path).convert("RGB")
        # sentence-transformers automatically handles preprocessing
        return model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros(512)  # fallback

def short_caption(row):
    """Creates short textual description for image"""
    artist = str(row.get("artist", "")).strip()
    genre = str(row.get("genre", "")).strip()
    nationality = str(row.get("nationality", "")).strip()
    years = str(row.get("years", "")).strip()
    
    # Create description like: "Claude Monet, French Impressionist painter (1840-1926)"
    parts = []
    if artist:
        parts.append(artist)
    if nationality and genre:
        parts.append(f"{nationality} {genre} painter")
    elif nationality:
        parts.append(f"{nationality} painter")
    elif genre:
        parts.append(f"{genre} painter")
    if years:
        parts.append(f"({years})")
    
    return ", ".join(parts) if parts else "Unknown artwork"

# 1) Generate embeddings for clean images
print("Generating embeddings for clean images...")
clean_paths = meta["clean_image_path"].tolist()
clean_ids = meta["id"].tolist()

E_clean = []
for path in tqdm(clean_paths, desc="Embedding clean images"):
    emb = emb_image(path)
    E_clean.append(emb)

E_clean = np.vstack(E_clean)
print(f"Clean image embeddings shape: {E_clean.shape}")

# 2) Generate text embeddings
print("Generating text embeddings...")
captions = [short_caption(row) for _, row in meta.iterrows()]
print("Sample descriptions:")
for i, cap in enumerate(captions[:3]):
    print(f"  {i+1}. {cap}")

E_text = model.encode(captions, convert_to_numpy=True, normalize_embeddings=True)
print(f"Text embeddings shape: {E_text.shape}")

# 3) Image+text fusion
alpha = 0.8  # image weight vs text weight
E_index = alpha * E_clean + (1 - alpha) * E_text
print(f"Final index shape: {E_index.shape}")

# 4) Evaluation on non-professional variants
print("Evaluating on non-professional variants...")
topk = 5
records = []

for _, r in tqdm(nonpro.iterrows(), total=len(nonpro), desc="Non-pro queries"):
    try:
        # Query embedding
        q = emb_image(r["variant_path"])
        
        # Compute similarities
        sims = cosine_similarity([q], E_index)[0]
        
        # Find top-k
        order = np.argsort(-sims)[:topk]
        top_ids = [clean_ids[i] for i in order]
        top_similarities = [float(sims[i]) for i in order]
        
        # Check hits
        top1_id = top_ids[0]
        hit1 = int(top1_id == r["id"])
        hit5 = int(r["id"] in top_ids)
        
        # Find position of true image
        true_position = top_ids.index(r["id"]) + 1 if r["id"] in top_ids else -1
        
        records.append({
            'variant_path': r["variant_path"],
            'id': r["id"],
            'artist': r["artist"],
            'transforms_applied': r["transforms_applied"],
            'top1_id': top1_id,
            'top5_ids': str(top_ids),
            'hit@1': hit1,
            'hit@5': hit5,
            'true_position': true_position,
            'similarity_top1': top_similarities[0],
            'similarities_top5': str(top_similarities)
        })
        
    except Exception as e:
        print(f"Error processing variant {r.get('variant_path', 'unknown')}: {e}")

# Save results
df = pd.DataFrame(records)
df.to_csv(DATA_DIR/"retrieval_eval.csv", index=False)

# 5) Results analysis
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

rec1 = df["hit@1"].mean()
rec5 = df["hit@5"].mean()
print(f"📊 Recall@1: {rec1:.3f} ({rec1*100:.1f}%)")
print(f"📊 Recall@5: {rec5:.3f} ({rec5*100:.1f}%)")

# Analysis per transformation type
print(f"\n📈 Results per transformation type:")
transform_stats = df.groupby('transforms_applied').agg({
    'hit@1': ['mean', 'count'],
    'hit@5': 'mean',
    'similarity_top1': 'mean'
}).round(3)

transform_stats.columns = ['Recall@1', 'Count', 'Recall@5', 'Avg_Similarity']
transform_stats = transform_stats.sort_values('Recall@1')
print(transform_stats)

# Most difficult cases
print(f"\n❌ Most difficult cases (Recall@1 = 0):")
failed_cases = df[df['hit@1'] == 0].copy()
if len(failed_cases) > 0:
    failed_stats = failed_cases['transforms_applied'].value_counts()
    print(failed_stats.head())
    
    print(f"\nFailure examples:")
    for i, (_, row) in enumerate(failed_cases.head(3).iterrows()):
        print(f"  {i+1}. {row['id']} ({row['transforms_applied']}) -> found: {row['top1_id']}")

# Average similarity
avg_sim = df['similarity_top1'].mean()
print(f"\n🎯 Average top-1 similarity: {avg_sim:.3f}")

print(f"\n💾 Detailed results saved to: {DATA_DIR}/retrieval_eval.csv")
print(f"📈 Total analyzed {len(df)} variants from {len(df['id'].unique())} unique images")

# Additional analysis - which transformation is most difficult
print(f"\n🔍 Transformation difficulty ranking (easiest to hardest):")
difficulty_ranking = df.groupby('transforms_applied')['hit@1'].mean().sort_values(ascending=False)
for transform, recall in difficulty_ranking.items():
    status = "✅" if recall > 0.8 else "⚠️" if recall > 0.5 else "❌"
    print(f"  {status} {transform}: {recall:.3f}")