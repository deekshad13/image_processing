"""
compute_prototypes.py  —  Build symptom prototype vectors from trained model

Run this after training to generate / refresh prototypes.pt:
    python scripts/compute_prototypes.py

Or with custom paths:
    python scripts/compute_prototypes.py \
        --model   data/embeddings/fine_tuned_model.pt \
        --data    data/raw \
        --output  data/embeddings/prototypes.pt

What this produces:
    prototypes.pt  —  a dict with:
        "prototypes"  : Tensor [C, 128]   — one 128-d unit vector per symptom
        "class_names" : list[str]         — index → symptom name
        "label_map"   : dict[str, int]    — symptom name → index

At inference in routes.py (see Inference Usage below), this replaces the
entire gallery bin search with 10 cosine distance computations.

Inference Usage (drop-in replacement for your current similarity search):
─────────────────────────────────────────────────────────────────────────
    import torch, torch.nn.functional as F

    # load once at server startup
    data        = torch.load("data/embeddings/prototypes.pt", map_location="cpu")
    prototypes  = data["prototypes"]   # [10, 128]
    class_names = data["class_names"]  # ["Burnt_appearance", ...]

    # per-query inference
    def predict(query_embedding: torch.Tensor, top_k: int = 3, threshold: float = 0.5):
        # query_embedding : [128]  — output of model.forward(image)
        query  = F.normalize(query_embedding, dim=0)           # ensure unit norm
        sims   = (prototypes @ query).tolist()                  # cosine similarity
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])  # descending

        results = [
            {"symptom": class_names[i], "confidence": round(s, 4)}
            for i, s in ranked[:top_k]
        ]

        # if top confidence is below threshold → flag as uncertain
        top_conf = ranked[0][1]
        if top_conf < threshold:
            results[0]["uncertain"] = True

        return results
─────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset   import SymptomDataset, get_val_transform
from src.model.fine_tune import CropSimilarityModel


def compute_prototypes(
    model_path : str,
    data_dir   : str,
    output_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Model path : {model_path}")
    print(f"Data dir   : {data_dir}")
    print(f"Output     : {output_path}\n")

    # ── load model ────────────────────────────────────────────────────────────
    model = CropSimilarityModel()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # ── dataset (no augmentation — we want clean embeddings) ─────────────────
    dataset = SymptomDataset(data_dir, transform=get_val_transform())
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # ── compute embeddings ────────────────────────────────────────────────────
    C                 = len(dataset.class_names)
    class_embeddings  = {i: [] for i in range(C)}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Embedding images"):
            images = images.to(device)
            embs   = model(images)                   # [B, 128]  L2-normalised
            for emb, lbl in zip(embs.cpu(), labels):
                class_embeddings[lbl.item()].append(emb)

    # ── build prototypes ──────────────────────────────────────────────────────
    prototypes = []
    print("\nPrototype summary:")
    print("-" * 55)

    for i, name in enumerate(dataset.class_names):
        embs  = torch.stack(class_embeddings[i])   # [N_i, 128]
        proto = embs.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes.append(proto)

        # intra-class cosine similarity: how tightly clustered is this symptom?
        sims   = (embs @ proto).tolist()
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        print(f"  [{i:2d}] {name:<30s}  "
              f"n={len(embs):4d}  "
              f"avg_cos={avg_sim:.3f}  "
              f"min_cos={min_sim:.3f}")

    proto_tensor = torch.stack(prototypes)  # [C, 128]

    # ── inter-class similarity matrix ─────────────────────────────────────────
    print("\nInter-class cosine similarity matrix (lower = better separation):")
    sim_matrix = (proto_tensor @ proto_tensor.T)
    header     = "".join(f"{i:>6}" for i in range(C))
    print(f"{'':>4}" + header)
    for i in range(C):
        row = "".join(f"{sim_matrix[i,j].item():>6.2f}" for j in range(C))
        print(f"{i:>4}" + row)
    print("\n(Diagonal = 1.0 is expected — each prototype is identical to itself)")

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "prototypes":  proto_tensor,
        "class_names": dataset.class_names,
        "label_map":   dataset.label_map,
    }, output_path)

    print(f"\nSaved → {output_path}")
    print(f"Shape  : {proto_tensor.shape}")
    return proto_tensor, dataset.class_names


def main():
    parser = argparse.ArgumentParser(
        description="Compute symptom prototype vectors from trained model"
    )
    parser.add_argument(
        "--model",
        default="data/embeddings/fine_tuned_model.pt",
        help="Path to trained model .pt file"
    )
    parser.add_argument(
        "--data",
        default="data/raw",
        help="Path to symptom data directory (same layout as data/raw/)"
    )
    parser.add_argument(
        "--output",
        default="data/embeddings/prototypes.pt",
        help="Where to save prototypes.pt"
    )
    args = parser.parse_args()

    compute_prototypes(
        model_path  = args.model,
        data_dir    = args.data,
        output_path = args.output,
    )


if __name__ == "__main__":
    main()