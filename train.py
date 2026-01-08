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

# Modüler Yapı
from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

def main():
    # 1. Reproducibility
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Config
    BATCH_SIZE = 16
    TOTAL_EPOCHS = 25
    INITIAL_LR = 1e-4
    FINETUNE_LR = 5e-5
    
    print(f"🚀 Initializing Reti-TransNet Training on {DEVICE}...")

    # 3. Data Setup (Otomatik Yol Bulma)
    csv_path = None
    # Dataset klasörünü tara
    for root, _, files in os.walk('dataset'):
        for f in files:
            if f.endswith('.csv') and 'train' in f: 
                csv_path = os.path.join(root, f)
                break
    
    if not csv_path:
        print("❌ Error: 'train.csv' not found. Run 'python download_data.py' first.")
        return

    df = pd.read_csv(csv_path)
    if 'id_code' not in df.columns: 
        df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    # Augmentation
    train_aug = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        A.Rotate(limit=30, p=0.5), 
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(), 
        ToTensorV2()
    ])
    
    # Weighted Sampler (Dengesizlik Çözümü)
    targets = train_df['diagnosis'].values
    class_counts = np.bincount(targets)
    weights = 1. / class_counts
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weights), len(samples_weights))
    
    # Loaders
    train_loader = DataLoader(
        RetinopathyDataset(train_df, 'dataset', transform=train_aug), 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=2
    )

    # 4. Model & Optimizer
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-2)
    
    # Label Smoothing (Overfitting önleyici)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # 5. Training Loop
    os.makedirs("weights", exist_ok=True)
    print(f"🔥 Starting Training ({TOTAL_EPOCHS} Epochs)...")
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        
        # --- FINE TUNING GEÇİŞİ (Epoch 16) ---
        if epoch == 16:
            print("\\n🔄 Switching to Fine-Tuning Phase (Lower LR)...")
            for param_group in optimizer.param_groups:
                param_group['lr'] = FINETUNE_LR
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # TQDM Barı
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")
        
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
            
            # İstatistikler
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += lbl.size(0)
            correct += (preds == lbl).sum().item()
            
            # Anlık Gösterim (Loss ve Acc)
            loop.set_postfix(loss=loss.item(), acc=f"{100*correct/total:.2f}%")
        
        # Epoch Sonu Özeti
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch}/{TOTAL_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
            
        # Kaydet
        torch.save(model.state_dict(), f"weights/retitransnet_epoch_{epoch}.pth")

    # Final Modeli Kaydet
    torch.save(model.state_dict(), "weights/retitransnet_best.pth")
    print("✅ Training Completed. Model saved.")

if __name__ == "__main__":
    main()
