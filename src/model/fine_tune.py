import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import TripletDataset
from src.data.preprocessing import get_transform
from src.config import get_config


class CropSimilarityModel(nn.Module):
    
    def __init__(self, embed_dim=384, project_dim=128):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, project_dim),
            nn.BatchNorm1d(project_dim)
        )

    def forward(self, x):
        features = self.backbone(x)           # [B, 384]
        projected = self.projector(features)  # [B, 128]
        return nn.functional.normalize(projected, dim=1)



def save_checkpoint(model, optimizer, epoch, phase, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_phase{phase}_epoch{epoch+1}.pt")
    torch.save({
        "epoch":       epoch,
        "phase":       phase,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "loss":        loss,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    if not files:
        return None

    def _key(name):
        parts = name.replace(".pt", "").split("_")
        phase = int(parts[1].replace("phase", ""))
        epoch = int(parts[2].replace("epoch", ""))
        return (phase, epoch)

    files.sort(key=_key)
    path = os.path.join(checkpoint_dir, files[-1])
    ckpt = torch.load(path, map_location="cpu")
    print(f"  Found checkpoint: phase={ckpt['phase']}, epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}")
    return ckpt


def train(data_dir="data/raw",
          epochs=10,
          batch_size=16,
          save_path="data/embeddings/fine_tuned_model.pt",
          checkpoint_dir="data/embeddings/checkpoints"):

    transform = get_transform()

    dataset    = TripletDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    model     = CropSimilarityModel()
    criterion = nn.TripletMarginLoss(margin=0.3)

    ckpt           = find_latest_checkpoint(checkpoint_dir)
    phase1_epochs  = epochs // 2   
    phase2_epochs  = epochs // 2   

    phase1_done = (
        ckpt is not None and (
            ckpt["phase"] == 2 or
            (ckpt["phase"] == 1 and ckpt["epoch"] + 1 >= phase1_epochs)
        )
    )

    # Phase 1 training
    if not phase1_done:
        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer   = torch.optim.Adam(model.projector.parameters(), lr=1e-3)
        start_epoch = 0

        if ckpt is not None and ckpt["phase"] == 1:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            start_epoch = ckpt["epoch"] + 1

        print("\nPhase 1 - Training projector only (backbone frozen)")
        print("=" * 55)

        for epoch in range(start_epoch, phase1_epochs):
            total_loss = 0
            for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                loss = criterion(model(anchor), model(positive), model(negative))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, 1, avg_loss, checkpoint_dir)

    else:
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
        print("  Phase 1 complete - skipping to Phase 2")

    # Phase 2 training
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(),  "lr": 1e-5},
        {"params": model.projector.parameters(), "lr": 1e-4},
    ])

    start_epoch = 0
    if ckpt is not None and ckpt["phase"] == 2:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1

    print("\nPhase 2 - Fine-tuning backbone (last layers unfrozen)")
    print("=" * 55)

    for epoch in range(start_epoch, phase2_epochs):
        total_loss = 0
        for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, 2, avg_loss, checkpoint_dir)

    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved -> {save_path}")


if __name__ == "__main__":
    train()