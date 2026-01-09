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
import time # Süre ölçümü için

# Modüler Importlar
from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

def main():
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Veri Hazırlığı
    csv_path = None
    for root, _, files in os.walk('dataset'):
        for f in files:
            if f.endswith('.csv') and 'train' in f: csv_path = os.path.join(root, f)
    
    if not csv_path:
        print("❌ Error: Dataset not found. Run 'python download_data.py'.")
        return

    df = pd.read_csv(csv_path)
    if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    aug = A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=0.5), A.Rotate(limit=30), A.Normalize(), ToTensorV2()])
    
    targets = train_df['diagnosis'].values
    weights = 1. / np.bincount(targets)
    sampler = WeightedRandomSampler(torch.from_numpy(weights[targets]), len(targets))
    
    loader = DataLoader(RetinopathyDataset(train_df, 'dataset', transform=aug), 
                        batch_size=16, sampler=sampler, num_workers=2)
    
    # 2. Model Kurulumu
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"🔥 Starting Training on {DEVICE} (25 Epochs)...")
    print("-" * 65) # Ayırıcı çizgi
    print(f"{'Epoch':<10} | {'Loss':<10} | {'Accuracy':<10} | {'Kappa':<10}")
    print("-" * 65)

    os.makedirs("weights", exist_ok=True)
    
    # 3. Eğitim Döngüsü (Clean Log Modu)
    for epoch in range(1, 26):
        model.train()
        
        # Fine-tuning Sessiz Geçiş
        if epoch == 16:
            for g in optimizer.param_groups: g['lr'] = 5e-5

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # TQDM YOK - Sessiz Döngü
        for batch_idx, (img, lbl) in enumerate(loader):
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(img)
                loss = criterion(out, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # İstatistik
            running_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            
            # Opsiyonel: Çok uzun sürüyorsa kullanıcı dondu sanmasın diye her %25'te bir nokta koyabiliriz
            # if batch_idx % (len(loader)//4) == 0: print(".", end="", flush=True)
            
        # Epoch Bitti - Hesapla ve Yazdır
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        # O istediğiniz temiz format:
        print(f"Epoch {epoch}/{25:<4} | {epoch_loss:.4f}     | {epoch_acc:.2f}%      | {epoch_kappa:.4f}")
            
        torch.save(model.state_dict(), "weights/retitransnet_best.pth")

    print("-" * 65)
    print("✅ Training Completed.")

if __name__ == "__main__":
    main()
