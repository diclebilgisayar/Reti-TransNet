import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

# Modular imports
from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

def main():
    # 1. Reproducibility
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Configuration
    BATCH_SIZE = 16
    EPOCHS = 25
    LR = 1e-4
    
    print(f"🔥 Starting Training on {DEVICE} ({EPOCHS} Epochs)...")
    
    # 3. Data Preparation
    # Automatically locate the CSV file
    csv_path = None
    for root, _, files in os.walk('dataset'):
        for f in files:
            if f.endswith('.csv') and 'train' in f: 
                csv_path = os.path.join(root, f)
                break
    
    if not csv_path:
        print("❌ Error: Dataset not found. Please run 'python download_data.py' first.")
        return

    df = pd.read_csv(csv_path)
    # Standardize column names
    if 'id_code' not in df.columns: 
        df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
    
    # Stratified Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    # Augmentation Pipeline
    train_aug = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), 
        A.Rotate(limit=30, p=0.5), 
        A.Normalize(), 
        ToTensorV2()
    ])
    
    # Handling Class Imbalance with Weighted Random Sampler
    targets = train_df['diagnosis'].values
    weights = 1. / np.bincount(targets)
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weights), len(samples_weights))
    
    # Data Loader
    train_loader = DataLoader(
        RetinopathyDataset(train_df, 'dataset', transform=train_aug), 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=2
    )

    # 4. Model, Optimizer, Loss
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs("weights", exist_ok=True)

    # 5. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # Fine-Tuning Phase: Reduce Learning Rate at Epoch 16
        if epoch == 16:
            # We print a small notification above the bar
            tqdm.write("\nℹ️  Info: Switching to Fine-Tuning Phase (Lower LR)...")
            for g in optimizer.param_groups: 
                g['lr'] = 5e-5

        # Metrics initialization
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # TQDM Progress Bar
        # leave=True ensures the bar remains on screen after completion (stacking effect)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

        for img, lbl in loop:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with torch.amp.autocast('cuda'):
                out = model(img)
                loss = criterion(out, lbl)
            
            # Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate Batch Statistics
            running_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            
            # Store for Kappa calculation later
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            
            # Update Progress Bar with Real-Time Loss & Acc
            # Kappa is not calculated per batch to save time, only at end
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")
            
        # --- End of Epoch ---
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        # Final update to the progress bar to show the final Kappa score
        loop.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.2f}%", kappa=f"{epoch_kappa:.4f}")
            
        # Save Model Checkpoint
        torch.save(model.state_dict(), f"weights/retitransnet_epoch_{epoch}.pth")

    # Save Final Best Model
    torch.save(model.state_dict(), "weights/retitransnet_best.pth")
    print("\n✅ Training Completed. Model saved to 'weights/' folder.")

if __name__ == "__main__":
    main()
