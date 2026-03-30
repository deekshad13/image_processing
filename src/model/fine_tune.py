"""
fine_tune.py  —  Symptom-semantic training with Supervised Contrastive Loss

Architecture:
    DINOv2 ViT-S/14 backbone  →  projector head  →  L2-normalised 128-d embedding

Training strategy:
    Phase 1 : backbone frozen,  train projector only          (lr=1e-3, ~5 epochs)
    Phase 2 : backbone unfrozen (last 2 blocks + head only),  fine-tune end-to-end
              (backbone lr=1e-5, projector lr=1e-4, ~5 epochs)

Loss:
    SupConLoss — Supervised Contrastive Loss (Khosla et al., 2020).
    Unlike TripletLoss which sees 3 images at a time, SupCon operates on
    the entire batch: every (anchor, positive) pair from the same symptom
    class is pulled together, and every (anchor, negative) pair from a
    different class is pushed apart.  This means a batch of 32 images gives
    hundreds of training signal pairs per step vs. just 32 with TripletLoss.

Hard negative mining:
    Within each SupCon batch, the loss naturally up-weights hard negatives
    (negatives that are close in embedding space) via the softmax denominator.
    No explicit mining loop needed — SupCon handles it analytically.

Post-training:
    compute_prototypes() builds one prototype vector per symptom class
    (mean of all embeddings for that class).  Inference uses cosine distance
    to these 10 prototype vectors — no gallery bin required at all.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import SymptomDataset, get_train_transform, get_val_transform


# ── Model ─────────────────────────────────────────────────────────────────────

class CropSimilarityModel(nn.Module):
    """
    DINOv2 ViT-S/14 backbone + MLP projector head.
    Output: L2-normalised 128-d embedding.

    Unchanged from your original — keeping this identical means your existing
    encoder.py / routes.py that load the model state dict require zero changes.
    """

    def __init__(self, embed_dim: int = 384, project_dim: int = 128):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, project_dim),
            nn.BatchNorm1d(project_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.backbone(x)           # [B, 384]
        projected = self.projector(features)   # [B, 128]
        return F.normalize(projected, dim=1)   # unit sphere


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — Khosla et al. (NeurIPS 2020).
    https://arxiv.org/abs/2004.11362

    Args:
        temperature : softmax temperature τ.  Lower = sharper separation.
                      0.07 is the value used in the original paper.
                      0.1–0.2 works well when the backbone is partly frozen.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : [B, D]  L2-normalised embeddings
            labels   : [B]     integer symptom class labels

        Returns:
            scalar loss
        """
        device = features.device
        B      = features.shape[0]

        # pairwise cosine similarity (dot product of unit vectors)
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # mask: 1 where same class, 0 elsewhere — excluding self-pairs
        labels   = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)           # [B, B]
        self_mask = torch.eye(B, device=device)
        pos_mask  = pos_mask - self_mask                              # remove diagonal

        # for numerical stability, subtract row max before exp
        sim = sim - self_mask * 1e9   # mask self-similarity to -inf
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)                                      # [B, B]

        # sum of exp similarities excluding self-pairs
        denom = exp_sim.sum(dim=1, keepdim=True) - exp_sim * self_mask  # [B, B]

        # log probability for each positive pair
        log_prob = sim - torch.log(denom + 1e-8)                     # [B, B]

        # mean over positives per anchor; mean over all anchors
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss  = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, phase, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(
        checkpoint_dir, f"checkpoint_phase{phase}_epoch{epoch+1}.pt"
    )
    torch.save({
        "epoch":       epoch,
        "phase":       phase,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "loss":        loss,
    }, path)
    print(f"  Checkpoint saved → {path}")


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
    print(f"  Resumed from checkpoint: phase={ckpt['phase']}, "
          f"epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}")
    return ckpt


# ── Prototype computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_prototypes(
    model: CropSimilarityModel,
    data_dir: str,
    save_path: str,
    device: torch.device,
):
    """
    After training, compute one prototype vector per symptom class.
    Prototype = mean of all L2-normalised embeddings for that class,
                then L2-normalised again.

    Saves a dict:
        {
          "prototypes" : Tensor [num_classes, 128],
          "class_names": list[str],           # index → symptom name
          "label_map"  : dict[str, int],      # symptom name → index
        }
    to save_path.

    At inference:
        query_emb    = model(image)               # [1, 128]
        dists        = 1 - query_emb @ protos.T   # cosine distance [1, C]
        predicted    = dists.argmin().item()
        confidence   = softmax(-dists / T)
    """
    model.eval()
    dataset = SymptomDataset(data_dir, transform=get_val_transform())
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # accumulate embeddings per class
    class_embeddings: dict[int, list[torch.Tensor]] = {
        i: [] for i in range(len(dataset.class_names))
    }

    for images, labels in tqdm(loader, desc="Computing embeddings"):
        images = images.to(device)
        embs   = model(images)               # [B, 128]  (already L2-normed)
        for emb, lbl in zip(embs.cpu(), labels):
            class_embeddings[lbl.item()].append(emb)

    prototypes = []
    for i in range(len(dataset.class_names)):
        class_embs = torch.stack(class_embeddings[i])  # [N_i, 128]
        proto      = class_embs.mean(dim=0)
        proto      = F.normalize(proto, dim=0)          # re-normalise
        prototypes.append(proto)

    proto_tensor = torch.stack(prototypes)  # [C, 128]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "prototypes":  proto_tensor,
        "class_names": dataset.class_names,
        "label_map":   dataset.label_map,
    }, save_path)

    print(f"\nPrototypes saved → {save_path}")
    print(f"  Shape: {proto_tensor.shape}  "
          f"({len(dataset.class_names)} symptoms × 128 dims)")
    print("  Classes:")
    for i, name in enumerate(dataset.class_names):
        n = len(class_embeddings[i])
        print(f"    [{i:2d}] {name:<30s} ({n} images averaged)")

    return proto_tensor, dataset.class_names


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_dir       : str   = "data/raw",
    epochs         : int   = 10,
    batch_size     : int   = 32,
    save_path      : str   = "data/embeddings/fine_tuned_model.pt",
    proto_path     : str   = "data/embeddings/prototypes.pt",
    checkpoint_dir : str   = "data/embeddings/checkpoints",
    temperature    : float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = SymptomDataset(data_dir, transform=get_train_transform())

    # WeightedRandomSampler: ensures each symptom class gets equal representation
    # per epoch regardless of how many images each class has (fixes your uneven bin)
    sampler = WeightedRandomSampler(
        weights     = dataset.sample_weights,
        num_samples = len(dataset),
        replacement = True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,
        drop_last   = True,    # SupCon needs full batches for stable gradients
    )

    # ── model + loss ──────────────────────────────────────────────────────────
    model     = CropSimilarityModel().to(device)
    criterion = SupConLoss(temperature=temperature)

    phase1_epochs = epochs // 2
    phase2_epochs = epochs - phase1_epochs

    ckpt       = find_latest_checkpoint(checkpoint_dir)
    phase1_done = (
        ckpt is not None and (
            ckpt["phase"] == 2 or
            (ckpt["phase"] == 1 and ckpt["epoch"] + 1 >= phase1_epochs)
        )
    )

    # ── Phase 1: projector only ───────────────────────────────────────────────
    if not phase1_done:
        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer   = torch.optim.Adam(model.projector.parameters(), lr=1e-3)
        start_epoch = 0

        if ckpt is not None and ckpt["phase"] == 1:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            start_epoch = ckpt["epoch"] + 1

        print("\nPhase 1 — projector only (backbone frozen)")
        print("=" * 55)

        for epoch in range(start_epoch, phase1_epochs):
            model.train()
            total_loss = 0.0

            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{phase1_epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                embs = model(images)
                loss = criterion(embs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} — SupConLoss: {avg:.4f}")
            save_checkpoint(model, optimizer, epoch, 1, avg, checkpoint_dir)

    else:
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
        print("Phase 1 complete — resuming from Phase 2")

    # ── Phase 2: unfreeze last 2 transformer blocks + head ───────────────────
    # Unfreezing the entire backbone on small datasets causes catastrophic
    # forgetting of DINOv2's pretrained features. We only unfreeze:
    #   - the last 2 transformer blocks (blocks[-2], blocks[-1])
    #   - the norm + head layers
    # This preserves low/mid-level features while adapting high-level semantics.

    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable_layers = []
    if hasattr(model.backbone, "blocks"):
        for block in model.backbone.blocks[-2:]:           # last 2 blocks
            trainable_layers.append(block)
    for attr in ["norm", "head"]:
        if hasattr(model.backbone, attr):
            trainable_layers.append(getattr(model.backbone, attr))

    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 2 — fine-tuning last 2 blocks + head ({n_trainable:,} params)")
    print("=" * 55)

    optimizer = torch.optim.Adam([
        {"params": [p for l in trainable_layers for p in l.parameters()], "lr": 1e-5},
        {"params": model.projector.parameters(),                          "lr": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase2_epochs
    )

    start_epoch = 0
    if ckpt is not None and ckpt["phase"] == 2:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, phase2_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{phase2_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embs = model(images)
            loss = criterion(embs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(dataloader)
        lr  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} — SupConLoss: {avg:.4f}  lr: {lr:.2e}")
        save_checkpoint(model, optimizer, epoch, 2, avg, checkpoint_dir)

    # ── save final model ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved → {save_path}")

    # ── build prototypes immediately after training ───────────────────────────
    print("\nBuilding symptom prototypes...")
    compute_prototypes(model, data_dir, proto_path, device)


if __name__ == "__main__":
    train()