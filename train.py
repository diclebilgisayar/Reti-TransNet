import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.loss import LabelSmoothingCrossEntropy

from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

def render_bar(epoch, total_epochs, batch_idx, total_batches, bar_len=25):
    progress = (batch_idx + 1) / total_batches
    filled = int(bar_len * progress)
    bar = "█" * filled + "-" * (bar_len - filled)
    percent = int(progress * 100)

    return (
        f"\rEpoch {epoch}/{total_epochs}: "
        f"[{bar}] {percent}%"
    )


def main():
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============================
    # Dataset
    # ===============================
    csv_path = None
    for root, _, files in os.walk("dataset"):
        for f in files:
            if f.endswith(".csv") and "train" in f:
                csv_path = os.path.join(root, f)

    if not csv_path:
        print("❌ Dataset not found.")
        return

    df = pd.read_csv(csv_path)
    if "id_code" not in df.columns:
        df.rename(
            columns={df.columns[0]: "id_code", df.columns[1]: "diagnosis"},
            inplace=True,
        )

    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        stratify=df["diagnosis"],
        random_state=42,
    )

    # ===============================
    # Augmentations
    # ===============================
    train_tfms = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    # ===============================
    # Weighted Sampler
    # ===============================
    targets = train_df["diagnosis"].values
    class_weights = 1.0 / np.bincount(targets)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        RetinopathyDataset(train_df, "dataset", transform=train_tfms),
        batch_size=16,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # ===============================
    # Model / Optim / Loss
    # ===============================
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs("weights", exist_ok=True)

    TOTAL_EPOCHS = 25
    print("🔥 Starting Training (25 Epochs)...")

    # ===============================
    # Training Loop
    # ===============================
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()

        if epoch == 16:
            print("\n🔄 Switching to Fine-Tuning Phase (Lower LR)...")
            for g in optimizer.param_groups:
                g["lr"] = 5e-5

        running_loss = 0.0
        correct = 0
        total = 0

        total_batches = len(train_loader)

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 🔥 AYNI SATIRDA BAR GÜNCELLE
            sys.stdout.write(
                render_bar(epoch, TOTAL_EPOCHS, batch_idx, total_batches)
            )
            sys.stdout.flush()

        epoch_loss = running_loss / total_batches
        epoch_acc = 100.0 * correct / total

        sys.stdout.write(
            f" | Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_acc:.2f}%\n"
        )
        sys.stdout.flush()

        torch.save(model.state_dict(), "weights/retitransnet_best.pth")


if __name__ == "__main__":
    main()
