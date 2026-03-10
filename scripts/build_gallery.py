import os
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model.encoder import load_dinov2, get_embedding

SYMPTOM_FOLDERS = [
    "Burnt appearance",
    "Fruit_bored_holes",
    "Fruit_cracking",
    "Galls",
    "Leaf chewed portion",
    "Leaf_color_change",
    "Leaf_Healthy",
    "Leaf_shape_change",
    "Stem_bored_holes",
    "Stem_cracking"
]

RAW_DATA_DIR   = "data/raw"
EMBEDDINGS_DIR = "data/embeddings"

def build_gallery():
    print("Loading DINOv2...")
    model = load_dinov2()
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    for symptom in SYMPTOM_FOLDERS:
        folder_path = os.path.join(RAW_DATA_DIR, symptom)

        if not os.path.exists(folder_path):
            print(f"Folder not found, skipping: {symptom}")
            continue

        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(images) == 0:
            print(f"No images found in: {symptom}")
            continue

        print(f"\nProcessing: {symptom} ({len(images)} images)")

        embeddings = []
        labels     = []

        for image_file in tqdm(images):
            image_path = os.path.join(folder_path, image_file)
            try:
                embedding = get_embedding(image_path, model)
                embeddings.append(embedding)
                labels.append(symptom)
            except Exception as e:
                print(f"  Skipping {image_file}: {e}")
                continue

        symptom_safe = symptom.replace(" ", "_")
        save_dir     = os.path.join(EMBEDDINGS_DIR, symptom_safe)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "embeddings.npy"), np.array(embeddings))
        with open(os.path.join(save_dir, "labels.json"), "w") as f:
            json.dump(labels, f)

        print(f"  Saved {len(embeddings)} embeddings for {symptom}")

    print("\nGallery build complete.")

if __name__ == "__main__":
    build_gallery()