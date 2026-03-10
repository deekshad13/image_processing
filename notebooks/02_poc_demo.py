import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.encoder import load_dinov2, get_embedding
from model.similarity import find_top_matches

EMBEDDINGS_DIR = "data/embeddings"

def load_full_gallery():
    all_embeddings = []
    all_labels = []

    for symptom_folder in os.listdir(EMBEDDINGS_DIR):
        folder_path = os.path.join(EMBEDDINGS_DIR, symptom_folder)
        emb_path    = os.path.join(folder_path, "embeddings.npy")
        lab_path    = os.path.join(folder_path, "labels.json")

        if not os.path.exists(emb_path):
            continue

        embeddings = np.load(emb_path)
        with open(lab_path, "r") as f:
            labels = json.load(f)

        all_embeddings.append(embeddings)
        all_labels.extend(labels)

    return np.vstack(all_embeddings), all_labels


def compare_image(image_path: str, top_k: int = 5):
    print(f"\nQuery image: {image_path}")
    print("=" * 55)

    print("Loading DINOv2...")
    model = load_dinov2()

    print("Loading gallery...")
    ref_embeddings, ref_labels = load_full_gallery()

    print("Extracting embedding...")
    query_embedding = get_embedding(image_path, model)

    results = find_top_matches(query_embedding, ref_embeddings,
                                ref_labels, top_k=top_k)

    print(f"\nTop {top_k} matches:\n")
    for r in results:
        bar = "█" * int(r["similarity"] / 10) + "░" * (10 - int(r["similarity"] / 10))
        print(f"  {r['rank']}. {r['symptom']:<25} "
              f"{r['similarity']:>5.1f}%  [{bar}]  {r['action']}")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python notebooks/02_poc_demo.py <image_path>")
        print("Example: python notebooks/02_poc_demo.py data/raw/Leaf_Healthy/img_001.jpg")
    else:
        compare_image(sys.argv[1])