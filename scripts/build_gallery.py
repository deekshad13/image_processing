"""
Rebuild the embedding gallery.
Uses fine-tuned model if weights exist, otherwise baseline DINOv2.

Usage: python scripts/build_gallery.py
"""
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import get_config
from src.data.preprocessing import get_transform
from src.data.symptom_registry import get_symptom_map
from src.model.encoder import load_model


def build():
    cfg = get_config()
    symptom_map = get_symptom_map()
    raw_dir = cfg["data"]["raw_dir"]
    embeddings_dir = cfg["data"]["embeddings_dir"]
    transform = get_transform()

    model, model_type = load_model()
    print(f"Using {model_type} model")

    os.makedirs(embeddings_dir, exist_ok=True)

    total_images = 0

    for symptom_id, info in symptom_map.items():
        raw_folder = info["raw_folder"]
        emb_folder = info["embedding_folder"]
        folder_path = os.path.join(raw_dir, raw_folder)

        if not os.path.exists(folder_path):
            print(f"Folder not found, skipping: {raw_folder}")
            continue

        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not images:
            print(f"No images found in: {raw_folder}")
            continue

        print(f"\nProcessing: {info['display_name']} ({len(images)} images)")

        embeddings = []
        labels = []

        for image_file in tqdm(images):
            image_path = os.path.join(folder_path, image_file)
            try:
                img = Image.open(image_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    emb = model(tensor).squeeze().numpy()
                embeddings.append(emb)
                labels.append(image_file)
            except Exception as e:
                print(f"  Skipping {image_file}: {e}")
                continue

        if not embeddings:
            continue

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Save to disk
        save_dir = os.path.join(embeddings_dir, emb_folder)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings_array)
        with open(os.path.join(save_dir, "labels.json"), "w") as f:
            json.dump(labels, f)

        total_images += len(embeddings)
        dim = embeddings_array.shape[1]
        print(f"  Saved {len(embeddings)} embeddings ({dim}-dim)")

    print(f"\nGallery build complete. {total_images} total embeddings across {len(symptom_map)} symptoms.")


if __name__ == "__main__":
    build()
