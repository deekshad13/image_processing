import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import TripletDataset


# ── Model ─────────────────────────────────────────────────────────────────────

class CropSimilarityModel(nn.Module):
    """
    DINOv2 backbone + a small projection head.
    The projection head pushes same-symptom embeddings closer
    and different-symptom embeddings further apart.
    """
    def __init__(self, embed_dim=384, project_dim=128):
        super().__init__()
        self.backbone  = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14'
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, project_dim),
            nn.BatchNorm1d(project_dim)
        )

    def forward(self, x):
        features  = self.backbone(x)           # [B, 384]
        projected = self.projector(features)   # [B, 128]
        return nn.functional.normalize(projected, dim=1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir: str = "data/raw",
          epochs: int = 10,
          batch_size: int = 16,
          save_path: str = "data/embeddings/fine_tuned_model.pt"):

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Dataset and dataloader
    dataset    = TripletDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    # Model
    model     = CropSimilarityModel()
    criterion = nn.TripletMarginLoss(margin=0.3)

    # Phase 1 — freeze backbone, only train projector
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.projector.parameters(), lr=1e-3)

    print("\nPhase 1 — Training projector only (backbone frozen)")
    print("=" * 55)

    for epoch in range(epochs // 2):
        total_loss = 0
        for anchor, positive, negative in tqdm(dataloader,
                                                desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}")

    # Phase 2 — unfreeze last N layers, fine tune entire model
    print("\nPhase 2 — Fine-tuning backbone (last layers unfrozen)")
    print("=" * 55)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.projector.parameters(), "lr": 1e-4}
    ])

    for epoch in range(epochs // 2, epochs):
        total_loss = 0
        for anchor, positive, negative in tqdm(dataloader,
                                                desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()