import os
import sys
import json
import time
import base64
import io

from dotenv import load_dotenv
load_dotenv()

import redis
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# LLM Integration (Gemini API)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Using placeholder descriptions.")

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REQUEST_QUEUE = "artguide:requests"
RESPONSE_PREFIX = "artguide:response:"
INDEX_PATH = os.getenv('INDEX_PATH', 'models/faiss.index')
META_PATH = os.getenv('META_PATH', 'models/metadata.parquet')

# Initialize Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"CLIP model loaded on {device}")

# Load FAISS index and metadata
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    print(f"Loading FAISS index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_parquet(META_PATH)
    print(f"Loaded {len(metadata)} artworks from index")
else:
    print("Warning: FAISS index or metadata not found. Running without index.")
    index = None
    metadata = pd.DataFrame(columns=["artist", "title", "period", "image_path"])

# Initialize Gemini API client (if API key available)
gemini_client = None
if GEMINI_AVAILABLE:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            gemini_client = genai.Client(api_key=api_key)
            print("✓ Gemini API client initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini client: {e}")
            gemini_client = None
    else:
        print("Info: GOOGLE_API_KEY not set. Using placeholder descriptions.")


def embed_image(img: Image.Image) -> np.ndarray:
    """
    Generate CLIP embedding for an image.
    
    Args:
        img: PIL Image
        
    Returns:
        Normalized 512-dimensional embedding
    """
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb.cpu().numpy().astype("float32")
    emb /= np.linalg.norm(emb)  # L2 normalization
    return emb


def search_index(img: Image.Image, k: int = 5):
    """
    Search vector database for similar artworks.
    
    Args:
        img: PIL Image
        k: Number of top results to return
        
    Returns:
        tuple: (results DataFrame, embedding)
    """
    if index is None or len(metadata) == 0:
        return None, None

    emb = embed_image(img)
    D, I = index.search(emb, k)
    
    results = metadata.iloc[I[0]].copy()
    results["distance"] = D[0]
    
    return results, emb


def generate_description(artist: str, title: str, period: str) -> str:
    """
    Generate artwork description using Gemini LLM or placeholder.
    
    Args:
        artist: Artist name
        title: Artwork title
        period: Historical period/style
        
    Returns:
        Generated tour-guide style description (150-250 words)
    """
    if gemini_client is not None:
        try:
            prompt = f"""You are an enthusiastic and knowledgeable art museum tour guide. Generate a comprehensive, engaging description for this artwork.

**Artwork Details:**
- Artist: {artist}
- Title: {title}
- Period/Style: {period}

**Instructions:**
Create a detailed 300-400 word description structured in the following sections:

1. **Introduction** (2-3 sentences): Welcome visitors and introduce the artwork with enthusiasm
2. **Artist Background** (3-4 sentences): Discuss the artist's significance, life, and contribution to art history
3. **Artistic Analysis** (4-5 sentences): Analyze the techniques, style, colors, composition, and visual elements
4. **Historical Context** (3-4 sentences): Explain the period, movement, and cultural significance
5. **Legacy & Impact** (2-3 sentences): Describe the artwork's influence and importance today

Write in a warm, conversational tone that educates and inspires museum visitors. Use vivid language and make art history accessible to everyone."""
            
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            return response.text
        except Exception as e:
            print(f"Warning: Gemini API call failed: {e}. Using placeholder.")
            # Fall through to placeholder
    
    # Placeholder fallback
    description = (
        f"This artwork, titled '{title}', was created by {artist} during the {period} period. "
        f"{artist} is recognized as a significant figure in the {period} movement, "
        f"known for distinctive style and innovative techniques. "
        f"This piece exemplifies the characteristics of {period} art through its composition, "
        f"color palette, and thematic elements. "
        f"The work reflects the cultural and artistic values of its time and continues to "
        f"influence contemporary art appreciation."
    )
    return description


def process_request(request_data):
    """
    Process recognition request from orchestrator.
    
    Args:
        request_data: Dictionary with request_id, image (base64), timestamp
        
    Returns:
        Response dictionary with recognition results
    """
    try:
        request_id = request_data['request_id']
        image_b64 = request_data['image']
        show_context = request_data.get('show_context', False)
        
        # Decode image
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Search index
        results, emb = search_index(img, k=5)
        
        if results is None:
            return {
                'request_id': request_id,
                'status': 'error',
                'message': 'No index loaded',
                'artist': 'Unknown',
                'title': 'Unknown',
                'period': 'Unknown',
                'confidence': 0.0,
                'description': 'Recognition service is not available. Please try again later.'
            }
        
        # Get top result
        top1 = results.iloc[0]
        artist = top1["artist"]
        title = top1.get("title", "Unknown")
        period = top1.get("period", "Unknown")
        distance = float(top1["distance"])
        
        # Convert distance to confidence (lower distance = higher confidence)
        # Using inverse exponential: confidence = exp(-distance)
        confidence = np.exp(-distance)
        
        # Generate description
        description = generate_description(artist, title, period)
        
        # Add context if requested
        if show_context and len(results) > 1:
            context_items = []
            for idx, row in results[1:4].iterrows():  # Top 2-4
                context_items.append(
                    f"{row['artist']} - {row.get('title', 'Unknown')} (similarity: {np.exp(-row['distance']):.2f})"
                )
            description += "\n\nSimilar artworks: " + "; ".join(context_items)
        
        return {
            'request_id': request_id,
            'status': 'success',
            'artist': artist,
            'title': title,
            'period': period,
            'confidence': float(confidence),
            'description': description
        }
    
    except Exception as e:
        return {
            'request_id': request_data.get('request_id', 'unknown'),
            'status': 'error',
            'message': f'AI processing error: {str(e)}',
            'artist': 'Unknown',
            'title': 'Unknown',
            'period': 'Unknown',
            'confidence': 0.0,
            'description': 'An error occurred during recognition.'
        }


def main():
    """Main loop: listen to orchestrator queue and process requests."""
    print(f"AI Server started. Listening to queue: {REQUEST_QUEUE}")
    print(f"Orchestrator (Redis): {REDIS_HOST}:{REDIS_PORT}")
    
    while True:
        try:
            # Blocking pop from queue (timeout 1 second)
            result = redis_client.blpop(REQUEST_QUEUE, timeout=1)
            
            if result:
                _, request_json = result
                request_data = json.loads(request_json)
                
                print(f"Processing request: {request_data['request_id']}")
                
                # Process request
                response = process_request(request_data)
                
                # Send response back via Redis
                response_key = f"{RESPONSE_PREFIX}{response['request_id']}"
                redis_client.setex(
                    response_key,
                    60,  # Expire after 60 seconds
                    json.dumps(response)
                )
                
                print(f"Completed request: {response['request_id']} - Status: {response['status']}")
        
        except KeyboardInterrupt:
            print("\nShutting down AI Server...")
            break
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)


if __name__ == '__main__':
    main()
