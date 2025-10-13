import os
import time
import csv
import gradio as gr
import faiss
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model for embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Paths (adjust as needed)
INDEX_PATH = "models/faiss.index"
META_PATH = "models/metadata.parquet"
LOG_PATH = "app/logs/telemetry.csv"
SAMPLE_IMAGES_DIR = "data/sample_images/"

# Load FAISS index + metadata
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_parquet(META_PATH)
else:
    index = None
    metadata = pd.DataFrame(columns=["artist", "title", "period", "image_path"])

# Ensure log directory
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "artist", "confidence", "response_time"])


def embed_image(img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb.cpu().numpy().astype("float32")
    emb /= np.linalg.norm(emb)
    return emb


def search_index(img: Image.Image, k: int = 5):
    if index is None or len(metadata) == 0:
        return None, None

    emb = embed_image(img)
    D, I = index.search(emb, k)
    results = metadata.iloc[I[0]].copy()
    results["distance"] = D[0]
    return results, emb


def generate_description(artist: str, title: str, period: str) -> str:
    # Placeholder for LLM
    return f"This is '{title}' by {artist}, created in the {period} period. More contextual details will be generated here."


def recognize(img, show_context):
    start_time = time.time()
    results, emb = search_index(img, k=5)

    if results is None:
        return "No index loaded.", None, "N/A"

    # Top-1 recognition
    top1 = results.iloc[0]
    artist, title, period, conf = (
        top1["artist"],
        top1.get("title", "Unknown"),
        top1.get("period", "Unknown"),
        float(top1["distance"]),
    )

    description = generate_description(artist, title, period)
    response_time = round(time.time() - start_time, 2)

    # Log telemetry
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), artist, conf, response_time])

    if show_context:
        neighbors = results[["artist", "title", "period", "distance"]].to_dict(orient="records")
        return f"Recognized: {artist} (confidence {conf:.4f})", img, description + "\nContext: " + str(neighbors)
    else:
        return f"Recognized: {artist} (confidence {conf:.4f})", img, description


# Collect sample images for quick demo
sample_images = []
if os.path.exists(SAMPLE_IMAGES_DIR):
    for file in os.listdir(SAMPLE_IMAGES_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_images.append(os.path.join(SAMPLE_IMAGES_DIR, file))

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Art Guide – Demo App")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload artwork photo")
            sample = gr.Dropdown(sample_images, label="Or choose a sample image", type="value")
            show_context = gr.Checkbox(label="Show retrieved context", value=False)
            run_btn = gr.Button("Recognize Artwork")
        with gr.Column():
            label_output = gr.Textbox(label="Recognition Result")
            img_output = gr.Image(label="Preview")
            desc_output = gr.Textbox(label="Description")

    def run_pipeline(uploaded, sample_path, show_context):
        if uploaded is None and sample_path:
            uploaded = Image.open(sample_path)
        if uploaded is None:
            return "No image provided", None, "Please upload or select a sample."
        return recognize(uploaded, show_context)

    run_btn.click(run_pipeline, inputs=[img_input, sample, show_context], outputs=[label_output, img_output, desc_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
