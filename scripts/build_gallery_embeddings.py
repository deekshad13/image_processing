import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset    import get_val_transform
from src.model.fine_tune import CropSimilarityModel
 
 
def build_gallery_embeddings(
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
 
    transform = get_val_transform()
 
    # ── embed each image individually ─────────────────────────────────────────
    gallery = {}
 
    symptom_folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ])
 
    for symptom in symptom_folders:
        folder = os.path.join(data_dir, symptom)
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        if not images:
            continue
 
        gallery[symptom] = {}
        print(f"Embedding {symptom} ({len(images)} images)...")
 
        for fname in tqdm(images, desc=f"  {symptom}", leave=False):
            path = os.path.join(folder, fname)
            try:
                img    = Image.open(path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor).squeeze().cpu()  # [128]
                gallery[symptom][fname] = emb
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
 
        print(f"  ✓ {len(gallery[symptom])} embeddings")
 
    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(gallery, output_path)
 
    total = sum(len(v) for v in gallery.values())
    print(f"\nSaved → {output_path}")
    print(f"Total  : {total} image embeddings across {len(gallery)} symptoms")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Build per-image gallery embeddings"
    )
    parser.add_argument(
        "--model",
        default="data/embeddings/fine_tuned_model.pt",
    )
    parser.add_argument(
        "--data",
        default="data/raw",
    )
    parser.add_argument(
        "--output",
        default="data/embeddings/gallery_embeddings.pt",
    )
    args = parser.parse_args()
 
    build_gallery_embeddings(
        model_path  = args.model,
        data_dir    = args.data,
        output_path = args.output,
    )
 
 
if __name__ == "__main__":
    main()