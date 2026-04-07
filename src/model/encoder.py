import os
import torch
import numpy as np
from PIL import Image
from src.config import get_config
from src.data.preprocessing import get_transform


def load_model():
    """Load the best available model: fine-tuned if weights exist, otherwise baseline DINOv2."""
    cfg = get_config()
    weights_path = cfg["model"]["fine_tuned_weights"]

    if os.path.exists(weights_path):
        from src.model.fine_tune import CropSimilarityModel
        model = CropSimilarityModel(
            embed_dim=cfg["model"]["backbone_dim"],
            project_dim=cfg["model"]["projection_dim"],
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded fine-tuned model from {weights_path}")
        return model, "fine-tuned"
    else:
        model = torch.hub.load(cfg["model"]["backbone"], cfg["model"]["variant"])
        model.eval()
        print("Fine-tuned weights not found — loaded baseline DINOv2")
        return model, "baseline"


def load_baseline_dinov2():
    """Load raw DINOv2 backbone (384-dim). For baseline evaluation only."""
    cfg = get_config()
    model = torch.hub.load(cfg["model"]["backbone"], cfg["model"]["variant"])
    model.eval()
    return model


def get_embedding(image_path, model):
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.squeeze().numpy()
