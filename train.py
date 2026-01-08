
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

from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

def main():
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    csv_path = None
    for root, _, files in os.walk('dataset'):
        for f in files:
            if f.endswith('.csv') and 'train' in f: csv_path = os.path.join(root, f)
    
    if not csv_path:
        print("❌ Error: 'train.csv' not found. Run 'python download_data.py' first.")
        return

    df = pd.read_csv(csv_path)
    if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    aug = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=30), A.Normalize(), ToTensorV2()])
    
    targets = train_df['diagnosis'].values
    weights = 1. / np.bincount(targets)
    sampler = WeightedRandomSampler(torch.from_numpy(weights[targets]), len(targets))
    
    loader = DataLoader(RetinopathyDataset(train_df, 'dataset', transform=aug), 
                        batch_size=16, sampler=sampler, num_workers=2)
    
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    print("🔥 Starting Training...")
    os.makedirs("weights", exist_ok=True)
    
    for epoch in range(1, 26):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch}/25")
        
        if epoch == 16: 
            for g in optimizer.param_groups: g['lr'] = 5e-5

        for img, lbl in loop:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(img), lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
            
        torch.save(model.state_dict(), "weights/retitransnet_best.pth")

if __name__ == "__main__":
    main()
