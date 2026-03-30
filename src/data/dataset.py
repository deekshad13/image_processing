"""
dataset.py  —  SupCon-ready dataset for symptom-semantic learning

Returns (image, label_index) pairs instead of triplets.
The training loop handles batch construction; no triplet sampling needed.

Augmentation strategy:
  - Training : heavy augmentation so the model learns symptom patterns,
               not crop identity or lighting conditions.
  - Inference: deterministic centre-crop only (same as before).
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transform():
    """
    Heavy augmentation for training.
    Color jitter handles lighting variation across different fields/cameras.
    Horizontal + vertical flips handle orientation variation.
    RandomResizedCrop forces the model to recognise symptoms at different scales.
    GaussianBlur + RandomGrayscale add robustness to image quality variation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transform():
    """Deterministic — identical to inference transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class SymptomDataset(Dataset):
    """
    Flat symptom-labelled dataset.

    data_dir layout (same as your current data/raw/):
        data/raw/
            Fruit_cracking/   ← symptom folder name = class label
                img001.jpg
                img002.jpg
                ...
            Leaf_Healthy/
                ...

    Returns:
        image  : torch.Tensor [3, 224, 224]
        label  : int  (symptom index, 0-based, consistent with self.class_names)
    """

    def __init__(self, data_dir: str, transform=None, min_images: int = 2):
        self.data_dir  = data_dir
        self.transform = transform

        # ── collect symptom folders ───────────────────────────────────────────
        self.classes: dict[str, list[str]] = {}
        for folder in sorted(os.listdir(data_dir)):          # sorted = stable ordering
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]
            if len(images) >= min_images:
                self.classes[folder] = images

        self.class_names = list(self.classes.keys())   # index → symptom name
        self.label_map   = {name: i for i, name in enumerate(self.class_names)}

        # ── flat index → (symptom_name, image_path) ──────────────────────────
        self.samples: list[tuple[str, str]] = []
        for name, paths in self.classes.items():
            for path in paths:
                self.samples.append((name, path))

        # ── class weights for WeightedRandomSampler ───────────────────────────
        # Inverse-frequency weighting so rare symptoms aren't starved.
        counts = {name: len(paths) for name, paths in self.classes.items()}
        total  = len(self.samples)
        self.sample_weights = [
            total / counts[name] for name, _ in self.samples
        ]

        self._print_summary()

    def _print_summary(self):
        print(f"\nSymptomDataset loaded from: {self.data_dir}")
        print(f"  {len(self.class_names)} symptom classes, {len(self.samples)} total images")
        print("  Per-class counts:")
        for name, paths in self.classes.items():
            idx = self.label_map[name]
            print(f"    [{idx:2d}] {name:<30s} {len(paths):4d} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, path = self.samples[idx]
        label      = self.label_map[name]

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label