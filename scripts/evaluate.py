import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model.fine_tune import CropSimilarityModel
from src.model.encoder import load_dinov2, get_embedding
from src.model.similarity import cosine_similarity


# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Load baseline embeddings from .npy files ──────────────────────────────────
def load_baseline_gallery(embeddings_dir):
    gallery = {}
    for symptom in os.listdir(embeddings_dir):
        symptom_path = os.path.join(embeddings_dir, symptom)
        if not os.path.isdir(symptom_path):
            continue
        emb_file   = os.path.join(symptom_path, "embeddings.npy")
        label_file = os.path.join(symptom_path, "labels.json")
        if not os.path.exists(emb_file):
            continue
        embeddings = np.load(emb_file)
        with open(label_file) as f:
            labels = json.load(f)
        gallery[symptom] = {"embeddings": embeddings, "labels": labels}
    print(f"Baseline gallery loaded: {len(gallery)} symptoms")
    return gallery


# ── Build fine-tuned gallery on the fly ───────────────────────────────────────
def build_finetuned_gallery(data_dir, model):
    gallery = defaultdict(lambda: {"embeddings": [], "labels": []})
    for symptom in os.listdir(data_dir):
        symptom_dir = os.path.join(data_dir, symptom)
        if not os.path.isdir(symptom_dir):
            continue
        for img_file in os.listdir(symptom_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(symptom_dir, img_file)
            try:
                img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    emb = model(img).squeeze().numpy()
                gallery[symptom]["embeddings"].append(emb)
                gallery[symptom]["labels"].append(symptom)
            except Exception:
                continue
    for symptom in gallery:
        gallery[symptom]["embeddings"] = np.array(gallery[symptom]["embeddings"])
    print(f"Fine-tuned gallery built: {len(gallery)} symptoms")
    return dict(gallery)


# ── Precision@K ───────────────────────────────────────────────────────────────
def precision_at_k(gallery, query_symptom, query_embedding, k=3):
    scores = []
    for symptom, data in gallery.items():
        for ref_emb in data["embeddings"]:
            sim = cosine_similarity(query_embedding, ref_emb)
            scores.append((sim, symptom))
    scores.sort(reverse=True)
    top_k = scores[:k]
    correct = sum(1 for _, s in top_k if s == query_symptom)
    return correct / k


# ── Recall@K ──────────────────────────────────────────────────────────────────
def recall_at_k(gallery, query_symptom, query_embedding, k=3):
    scores = []
    for symptom, data in gallery.items():
        for ref_emb in data["embeddings"]:
            sim = cosine_similarity(query_embedding, ref_emb)
            scores.append((sim, symptom))
    scores.sort(reverse=True)
    top_k = scores[:k]
    correct_in_top_k = sum(1 for _, s in top_k if s == query_symptom)
    total_relevant = len(gallery[query_symptom]["embeddings"])
    return correct_in_top_k / total_relevant if total_relevant > 0 else 0.0


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(gallery, model=None, use_finetuned=False, k=3, n_queries=5):
    all_precisions = []
    all_recalls    = []
    data_dir       = "data/raw"

    for symptom in os.listdir(data_dir):
        symptom_dir = os.path.join(data_dir, symptom)
        if not os.path.isdir(symptom_dir):
            continue

        images = [
            os.path.join(symptom_dir, f)
            for f in os.listdir(symptom_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:n_queries]

        symptom_precisions = []
        symptom_recalls    = []

        for img_path in images:
            try:
                if use_finetuned:
                    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
                    with torch.no_grad():
                        query_emb = model(img).squeeze().numpy()
                else:
                    query_emb = get_embedding(img_path, model)

                matched_key = next(
                    (key for key in gallery
                     if key.lower().replace(" ", "_") == symptom.lower().replace(" ", "_")),
                    None
                )
                if matched_key is None:
                    continue

                p = precision_at_k(gallery, matched_key, query_emb, k=k)
                r = recall_at_k(gallery, matched_key, query_emb, k=k)
                symptom_precisions.append(p)
                symptom_recalls.append(r)

            except Exception:
                continue

        if symptom_precisions:
            avg_p = np.mean(symptom_precisions)
            avg_r = np.mean(symptom_recalls)
            all_precisions.append(avg_p)
            all_recalls.append(avg_r)
            print(f"  {symptom:<30} Precision@{k} = {avg_p:.2f}  Recall@{k} = {avg_r:.2f}")

    return np.mean(all_precisions), np.mean(all_recalls)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    embeddings_dir = "data/embeddings"
    data_dir       = "data/raw"

    # Baseline
    print("\n── Baseline DINOv2 ──────────────────────────────")
    baseline_gallery  = load_baseline_gallery(embeddings_dir)
    baseline_model    = load_dinov2()
    baseline_p3, baseline_r3 = evaluate(
        baseline_gallery, model=baseline_model, use_finetuned=False, k=3, n_queries=5
    )
    print(f"\nBaseline  Precision@3: {baseline_p3:.4f}  Recall@3: {baseline_r3:.4f}")

    # Fine-tuned
    print("\n── Fine-tuned Model ─────────────────────────────")
    finetuned_model = CropSimilarityModel()
    finetuned_model.load_state_dict(
        torch.load("data/embeddings/fine_tuned_model.pt", map_location="cpu")
    )
    finetuned_model.eval()
    finetuned_gallery = build_finetuned_gallery(data_dir, finetuned_model)
    finetuned_p3, finetuned_r3 = evaluate(
        finetuned_gallery, model=finetuned_model, use_finetuned=True, k=3, n_queries=5
    )
    print(f"\nFine-tuned Precision@3: {finetuned_p3:.4f}  Recall@3: {finetuned_r3:.4f}")

    # Comparison
    p_improvement = (finetuned_p3 - baseline_p3) / baseline_p3 * 100
    r_improvement = (finetuned_r3 - baseline_r3) / baseline_r3 * 100
    print(f"\n{'='*50}")
    print(f"Precision Improvement: {p_improvement:.1f}%")
    print(f"Recall Improvement:    {r_improvement:.1f}%")