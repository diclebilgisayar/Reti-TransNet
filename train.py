import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

from model import RetiTransNet
from dataset import RetinopathyDataset
from utils import seed_everything

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

def main():
    print("🚀 Initializing Reti-TransNet Training Pipeline...")
    seed_everything(42)
    
    # 1. Locate Data
    # Automatically finds train.csv in dataset folder
    csv_path = None
    if os.path.exists('dataset/train.csv'):
        csv_path = 'dataset/train.csv'
    else:
        # Check subfolders just in case
        for root, _, files in os.walk('dataset'):
            if 'train.csv' in files:
                csv_path = os.path.join(root, 'train.csv')
                break
    
    if not csv_path:
        print("❌ Error: 'train.csv' not found. Please run 'python download_data.py' first.")
        return

    # 2. Prepare Data
    df = pd.read_csv(csv_path)
    # Fix column names if needed
    if 'id_code' not in df.columns: 
        df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
    
    # Stratified Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    # Augmentations (Albumentations)
    train_aug = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Sampler for Class Imbalance
    targets = train_df['diagnosis'].values
    class_counts = np.bincount(targets)
    weights = 1. / class_counts
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weights), len(samples_weights))
    
    # Data Loaders
    # Note: 'dataset' is passed as root, the smart loader will find images inside.
    train_ds = RetinopathyDataset(train_df, 'dataset', transform=train_aug)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

    # 3. Model Setup
    model = RetiTransNet(num_classes=5).to(DEVICE)
    
    # Optimizer & Loss (Label Smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision for speed

    # 4. Training Loop
    os.makedirs("weights", exist_ok=True)
    print(f"🔥 Starting training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        # Two-Stage Logic: Lower LR after epoch 15 (Fine-tuning)
        if epoch == 16:
            print("🔄 Switching to Fine-Tuning Phase (Lower Learning Rate)...")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5

        for img, lbl in loop:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            
            # AMP Forward
            with torch.amp.autocast('cuda'):
                outputs = model(img)
                loss = criterion(outputs, lbl)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"weights/retitransnet_epoch_{epoch}.pth")

    # Save Final Model
    torch.save(model.state_dict(), "weights/retitransnet_best.pth")
    print("✅ Training Completed. Model saved to 'weights/retitransnet_best.pth'.")

if __name__ == "__main__":
    main()
