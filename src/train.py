
import os
import csv
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image


BACKBONE       = "convnext_base"
PRETRAINED     = True
FREEZE_BACKBONE= False

IMG_SIZE       = 256
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LR              = 3e-5
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.15
SEED           = 42

CHECKPOINT_DIR = "../checkpoints/"
SAVE_EVERY_N   = 5              # save a checkpoint every N epochs
LOG_EVERY_N    = 10             # print every N batches

DEVICE = "cuda"


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class RatingDataset(Dataset):
    def __init__(self, samples: list[tuple[str, float]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, score = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(score, dtype=torch.float32)


def load_samples_from_csv(csv_path: str, image_root: str) -> list[tuple[str, float]]:
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = os.path.join(image_root, row["filename"]) if image_root else row["filename"]
            s = float(row["score"])
            samples.append((p, s))
    return samples


def build_model(backbone_name: str, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, backbone_name)(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )

    return model

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (imgs, scores) in enumerate(loader):
        imgs   = imgs.to(device)
        scores = scores.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % LOG_EVERY_N == 0:
            print(f"  Epoch {epoch} | Step {i+1}/{len(loader)} | Loss {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, scores in loader:
        imgs   = imgs.to(device)
        scores = scores.to(device).unsqueeze(1)
        preds  = model(imgs)
        loss   = criterion(preds, scores)
        total_loss += loss.item()
        all_preds.extend(preds.squeeze(1).cpu().tolist())
        all_targets.extend(scores.squeeze(1).cpu().tolist())

    avg_loss = total_loss / len(loader)

    # Mean Absolute Error
    mae = sum(abs(p - t) for p, t in zip(all_preds, all_targets)) / len(all_preds)

    return avg_loss, mae


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    samples = load_samples_from_csv("../images.csv", "")

    print(f"Loaded {len(samples)} samples  |  score range: "
          f"{min(s for _, s in samples):.3f} – {max(s for _, s in samples):.3f}")

    n_val   = max(1, int(len(samples) * VAL_SPLIT))
    n_train = len(samples) - n_val
    train_tf, val_tf = get_transforms()

    random.shuffle(samples)
    train_ds = RatingDataset(samples[:n_train], transform=train_tf)
    val_ds   = RatingDataset(samples[n_train:], transform=val_tf)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = build_model(BACKBONE, PRETRAINED).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {BACKBONE}  |  Trainable params: {trainable:,}")

    criterion = lambda p, t: 0.8 * nn.MSELoss()(p, t) + 0.2 * nn.L1Loss()(p, t)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    best_path     = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, DEVICE, epoch)
        val_loss, val_mae = evaluate(model, val_dl, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"Train MSE {train_loss:.4f} | "
              f"Val MSE {val_loss:.4f} | "
              f"Val MAE {val_mae:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "val_mae": val_mae}, best_path)
            print(f"    Best model saved (val_loss={val_loss:.4f})")

        if epoch % SAVE_EVERY_N == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt)

    print(f"\nDone. Best val MSE: {best_val_loss:.4f}  →  {best_path}")


if __name__ == "__main__":
    main()