"""
Verification-based evaluation for the Rootstalk system.
Tests: "Does a symptom image score high against its own gallery and low against others?"

Usage: python scripts/evaluate.py
"""
import os
import sys
import random
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import get_config
from src.data.preprocessing import get_transform
from src.data.symptom_registry import get_symptom_map
from src.data.gallery_builder import load_gallery, search_symptom
from src.model.encoder import load_model
from src.utils.metrics import verification_accuracy, compute_roc_curve, compute_auc


def evaluate(n_positive=5, n_negative=5):
    cfg = get_config()
    symptom_map = get_symptom_map()
    raw_dir = cfg["data"]["raw_dir"]
    top_n = cfg["similarity"]["top_n_mean"]
    transform = get_transform()

    model, model_type = load_model()
    print(f"Using {model_type} model\n")

    gallery = load_gallery()

    all_similarities = []
    all_labels = []

    for symptom_id, info in symptom_map.items():
        if symptom_id not in gallery:
            continue

        raw_folder = info["raw_folder"]
        folder_path = os.path.join(raw_dir, raw_folder)
        if not os.path.isdir(folder_path):
            continue

        images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Positive samples: images from this symptom
        pos_samples = random.sample(images, min(n_positive, len(images)))

        # Negative samples: images from other symptoms
        neg_images = []
        for other_id, other_info in symptom_map.items():
            if other_id == symptom_id:
                continue
            other_path = os.path.join(raw_dir, other_info["raw_folder"])
            if os.path.isdir(other_path):
                neg_images.extend([
                    os.path.join(other_path, f)
                    for f in os.listdir(other_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
        neg_samples = random.sample(neg_images, min(n_negative, len(neg_images)))

        symptom_sims = []
        symptom_labels = []

        for img_path in pos_samples:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    emb = model(tensor).squeeze().numpy()
                gallery_count = gallery[symptom_id]["count"]
                k = min(top_n, gallery_count)
                sims, _ = search_symptom(gallery, symptom_id, emb, top_k=k)
                mean_sim = float(np.mean(sims[:k])) * 100
                symptom_sims.append(mean_sim)
                symptom_labels.append(True)
            except Exception:
                continue

        for img_path in neg_samples:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    emb = model(tensor).squeeze().numpy()
                gallery_count = gallery[symptom_id]["count"]
                k = min(top_n, gallery_count)
                sims, _ = search_symptom(gallery, symptom_id, emb, top_k=k)
                mean_sim = float(np.mean(sims[:k])) * 100
                symptom_sims.append(mean_sim)
                symptom_labels.append(False)
            except Exception:
                continue

        all_similarities.extend(symptom_sims)
        all_labels.extend(symptom_labels)

        result = verification_accuracy(
            symptom_sims, symptom_labels,
            threshold=cfg["similarity"]["thresholds"]["medium"],
        )
        print(f"  {info['display_name']:<25} Acc={result['accuracy']:.2f}  "
              f"P={result['precision']:.2f}  R={result['recall']:.2f}  F1={result['f1']:.2f}")

    print(f"\n{'=' * 70}")
    print("Overall Verification Metrics:")

    for thresh in [60, 70, 80]:
        result = verification_accuracy(all_similarities, all_labels, threshold=thresh)
        print(f"  @{thresh}% threshold: Acc={result['accuracy']:.4f}  "
              f"P={result['precision']:.4f}  R={result['recall']:.4f}  F1={result['f1']:.4f}")

    fpr, tpr, _ = compute_roc_curve(all_similarities, all_labels)
    auc = compute_auc(fpr, tpr)
    print(f"\n  AUC-ROC: {auc:.4f}")


if __name__ == "__main__":
    random.seed(42)
    evaluate(n_positive=5, n_negative=5)
