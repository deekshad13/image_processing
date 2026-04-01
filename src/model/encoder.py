import torch
from src.model.fine_tune import CropSimilarityModel

def load_dinov2():
    model = CropSimilarityModel()
    state = torch.load("data/embeddings/fine_tuned_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model