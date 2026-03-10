import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def load_dinov2():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def get_embedding(image_path: str, model) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.squeeze().numpy()