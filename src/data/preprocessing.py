from torchvision import transforms
from src.config import get_config

_TRANSFORM = None

def get_transform():
    global _TRANSFORM
    if _TRANSFORM is not None:
        return _TRANSFORM
    cfg = get_config()
    t = cfg["transform"]
    _TRANSFORM = transforms.Compose([
        transforms.Resize(t["resize"]),
        transforms.CenterCrop(t["crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=t["mean"], std=t["std"]),
    ])
    return _TRANSFORM
