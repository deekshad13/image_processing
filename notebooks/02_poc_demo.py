"""
PoC demo — compare a single image against a specific symptom.

Usage: python notebooks/02_poc_demo.py <symptom_id> <image_path>
Example: python notebooks/02_poc_demo.py galls data/raw/Galls/img_001.jpg
"""
import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import get_config
from src.data.preprocessing import get_transform
from src.data.symptom_registry import resolve_symptom_id, get_symptom_ids
from src.data.gallery_builder import load_gallery, search_symptom
from src.model.encoder import load_model


def compare_image(symptom_id, image_path):
    info = resolve_symptom_id(symptom_id)
    if info is None:
        print(f"Unknown symptom_id: '{symptom_id}'")
        print(f"Valid IDs: {', '.join(get_symptom_ids())}")
        return

    print(f"\nQuery image: {image_path}")
    print(f"Verifying against: {info['display_name']}")
    print("=" * 55)

    cfg = get_config()
    model, model_type = load_model()
    print(f"Model: {model_type}")

    gallery = load_gallery()
    if symptom_id not in gallery:
        print(f"No gallery embeddings found for {symptom_id}")
        return

    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        query_emb = model(tensor).squeeze().numpy()

    top_n = min(cfg["similarity"]["top_n_mean"], gallery[symptom_id]["count"])
    sims, _ = search_symptom(gallery, symptom_id, query_emb, top_k=top_n)
    mean_sim = float(np.mean(sims[:top_n])) * 100

    thresholds = cfg["similarity"]["thresholds"]
    if mean_sim >= thresholds["high"]:
        label = "Take Action"
    elif mean_sim >= thresholds["medium"]:
        label = "Monitor"
    else:
        label = "Low Concern"

    bar = "█" * int(mean_sim / 10) + "░" * (10 - int(mean_sim / 10))
    print(f"\nResult: {mean_sim:.1f}%  [{bar}]  {label}")
    print(f"Gallery size: {gallery[symptom_id]['count']} reference images")
    print(f"Averaged top {top_n} matches\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python notebooks/02_poc_demo.py <symptom_id> <image_path>")
        print(f"Valid symptom IDs: {', '.join(get_symptom_ids())}")
    else:
        compare_image(sys.argv[1], sys.argv[2])
