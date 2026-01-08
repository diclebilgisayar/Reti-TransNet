import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

from model import RetiTransNet
from utils import seed_everything, ben_graham_preprocessing

# --- ROBUST DATASET CLASS ---
class RetinopathyDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_map = {}
        print(f"🔍 Indexing images in '{root_dir}' directory...")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    key = os.path.splitext(file)[0]
                    self.image_map[key] = os.path.join(root, file)
        
        print(f"✅ Indexed {len(self.image_map)} images found.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id_code'])
        img_path = self.image_map.get(img_id, None)
        
        image = None
        if img_path:
            try:
                image = ben_graham_preprocessing(img_path)
            except: pass
        
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']
            
        label = torch.tensor(int(row['diagnosis']), dtype=torch.long)
        return image, label

def main():
    seed_everything(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔌 Using device: {DEVICE}")

    BATCH_SIZE = 16
    EPOCHS = 25
    LR = 1e-4
    
    print("🚀 Loading Data...")
    
    csv_path = None
    if os.path.exists('dataset/train.csv'):
        csv_path = 'dataset/train.csv'
    else:
        for root, _, files in os.walk('dataset'):
            for f in files:
                if f.endswith('.csv') and 'train' in f:
                    csv_path = os.path.join(root, f)
                    break
    
    if not csv_path:
        print("❌ ERROR: 'train.csv' not found! Please upload it manually.")
        return

    df = pd.read_csv(csv_path)
    if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

    train_aug = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), 
        A.Rotate(limit=30, p=0.5), 
        A.Normalize(), 
        ToTensorV2()
    ])
    
    train_ds = RetinopathyDataset(train_df, 'dataset', transform=train_aug)
    
    targets = train_df['diagnosis'].values
    class_counts = np.bincount(targets)
    weights = 1. / class_counts
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weights), len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print("🔥 Starting Training Loop (25 Epochs)...")
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # --- DÜZELTME BURADA ---
        # LR değişikliğini loop başlamadan ÖNCE yapıyoruz.
        # Böylece tqdm barı bozulmuyor.
        if epoch == 16:
            print("\n🔄 Switching to Fine-Tuning Phase (Lower Learning Rate)...")
            for param_group in optimizer.param_groups: param_group['lr'] = 5e-5
        
        # tqdm barı şimdi başlıyor
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for img, lbl in loop:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(img)
                loss = criterion(out, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())
        
        torch.save(model.state_dict(), f"weights/retitransnet_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "weights/retitransnet_best.pth")
    print("✅ Training Completed. Weights saved to 'weights/' folder.")

if __name__ == "__main__":
    main()
