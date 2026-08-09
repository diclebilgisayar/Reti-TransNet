import os
import argparse
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
  
    args = parser.parse_args()
    
    seed_everything(args.seed) 

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

# Custom Modules
from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

def render_bar(epoch, total_epochs, batch_idx, total_batches, bar_len=25):
    """
    Utility function to render a progress bar in the console.
    """
    progress = (batch_idx + 1) / total_batches
    filled = int(bar_len * progress)
    bar = "█" * filled + "-" * (bar_len - filled)
    percent = int(progress * 100)
    return (
        f"\rEpoch {epoch}/{total_epochs}: "
        f"[{bar}] {percent}%"
    )

def freeze_backbones(model, freeze=True):
    """
    Freezes or unfreezes the backbone layers (CNN and Swin Transformer).
    The Fusion Gate, Projection layers, and Classifier Head remain trainable at all times.
    
    Args:
        model (nn.Module): The RetiTransNet instance.
        freeze (bool): If True, requires_grad is set to False for backbones.
    """
    # Freeze/Unfreeze CNN Backbone (EfficientNet-B0)
    for param in model.cnn.parameters():
        param.requires_grad = not freeze
    
    # Freeze/Unfreeze Transformer Backbone (Swin Tiny)
    for param in model.swin.parameters():
        param.requires_grad = not freeze
        
    status = "❄️ FROZEN" if freeze else "🔥 UNFROZEN"
    print(f"\n[INFO] Backbones (CNN & Swin) are now {status}.")

def main():
    # 1. Reproducibility
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============================
    # 2. Dataset Preparation
    # ===============================
    csv_path = None
    # Locate the training CSV file dynamically
    for root, _, files in os.walk("dataset"):
        for f in files:
            if f.endswith(".csv") and "train" in f:
                csv_path = os.path.join(root, f)

    if not csv_path:
        print("❌ Error: Dataset CSV not found.")
        return

    df = pd.read_csv(csv_path)
    # Standardize column names
    if "id_code" not in df.columns:
        df.rename(columns={df.columns[0]: "id_code", df.columns[1]: "diagnosis"}, inplace=True)

    # Stratified Split (80% Train, 20% Validation implicitly via subsequent splits if needed)
    train_df, _ = train_test_split(
        df, test_size=0.2, stratify=df["diagnosis"], random_state=42
    )

    # ===============================
    # 3. Augmentation & DataLoader
    # ===============================
    # Ben Graham's preprocessing is assumed to be handled within the Dataset class or offline.
    train_tfms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30),
        A.Normalize(), # ImageNet statistics
        ToTensorV2(),
    ])

    # Handle Class Imbalance via Weighted Random Sampling
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
        pin_memory=True
    )

    # ===============================
    # 4. Model Initialization
    # ===============================
    model = RetiTransNet(num_classes=5).to(DEVICE)
    
    # --- PHASE 1 SETUP: FREEZE BACKBONES ---
    # As stated in the manuscript, we initially freeze the heavy backbones 
    # to allow the randomized fusion weights to converge first.
    freeze_backbones(model, freeze=True)
    
    # Optimizer initialization: Only pass parameters that require gradients (Head & Fusion)
    # Learning Rate: 1e-4 for the initial phase
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") # Mixed Precision Scaler

    os.makedirs("weights", exist_ok=True)
    TOTAL_EPOCHS = 25
    
    print(f"🚀 Training Started on {DEVICE} (Total Epochs: {TOTAL_EPOCHS})")
    print("👉 Phase 1: Training Fusion Head Only (Epochs 1-15)")

    # ===============================
    # 5. Training Loop
    # ===============================
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()

        # --- PHASE 2 SWITCH: UNFREEZE (Epoch 16) ---
        if epoch == 16:
            print("\n" + "="*50)
            print("👉 Phase 2: Fine-Tuning Entire Network (Epochs 16-25)")
            print("="*50)
            
            # 1. Unfreeze all backbone layers
            freeze_backbones(model, freeze=False)
            
            # 2. Re-initialize optimizer to include all parameters (Backbones + Head)
            # 3. Reduce Learning Rate to 5e-5 to prevent catastrophic forgetting
            optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Context
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            sys.stdout.write(render_bar(epoch, TOTAL_EPOCHS, batch_idx, total_batches))
            sys.stdout.flush()

        # Epoch Summary
        epoch_loss = running_loss / total_batches
        epoch_acc = 100.0 * correct / total

        sys.stdout.write(f" | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%\n")
        sys.stdout.flush()

        # Save Checkpoint
        # Note: In a production setting, this should save based on Validation Kappa/Loss.
        torch.save(model.state_dict(), "weights/retitransnet_last.pth")

if __name__ == "__main__":
    main()
