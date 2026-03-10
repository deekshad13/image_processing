import os
import random
from torch.utils.data import Dataset
from PIL import Image


class TripletDataset(Dataset):
    """
    Creates triplets from the folder structure.
    Each folder = one symptom = one class.
    
    For each triplet:
      Anchor   → image from class A
      Positive → different image from same class A
      Negative → image from any other class
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir  = data_dir
        self.transform = transform

        # Build a dictionary: {symptom_name: [list of image paths]}
        self.classes = {}
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(images) >= 2:
                self.classes[folder] = images

        self.class_names = list(self.classes.keys())
        print(f"Found {len(self.class_names)} symptom classes")
        for name, imgs in self.classes.items():
            print(f"  {name}: {len(imgs)} images")

    def __len__(self):
        return sum(len(imgs) for imgs in self.classes.values())

    def __getitem__(self, idx):
        # Pick a random anchor class
        anchor_class = random.choice(self.class_names)

        # Pick anchor and positive from same class
        anchor_path, positive_path = random.sample(
            self.classes[anchor_class], 2
        )

        # Pick negative from a different class
        negative_class = random.choice(
            [c for c in self.class_names if c != anchor_class]
        )
        negative_path = random.choice(self.classes[negative_class])

        # Load images
        anchor   = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor   = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative