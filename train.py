import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import os

from model import RetiTransNet
from utils import seed_everything, ben_graham_preprocessing

# --- Dataset Class (Embedded for Simplicity) ---
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['id_code']) + '.png')
        if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
        
        image = ben_graham_preprocessing(img_path)
        if self.transform: image = self.transform(image=image)['image']
        return image, torch.tensor(int(row['diagnosis']), dtype=torch.long)

def main():
    seed_everything()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Data Setup
    print("Loading Data...")
    if not os.path.exists('dataset/train.csv'):
        print("❌ Dataset not found! Run 'python download_data.py' first.")
        return

    df = pd.read_csv('dataset/train.csv')
    # Column fix
    if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

    train_aug = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=30), A.Normalize(), ToTensorV2()])
    
    # Sampler
    targets = train_df['diagnosis'].values
    class_weights = 1. / np.bincount(targets)
    samples_weights = class_weights[targets]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weights), len(samples_weights))

    train_loader = DataLoader(RetinopathyDataset(train_df, 'dataset', transform=train_aug), 
                              batch_size=16, sampler=sampler, num_workers=2)

    # 2. Model
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop (25 Epochs)
    print("🚀 Starting Training (25 Epochs)...")
    for epoch in range(1, 26):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/25")
        
        # Fine-tuning switch
        if epoch == 16:
            print("🔄 Switching to Fine-Tuning Phase (Lower LR)...")
            for param_group in optimizer.param_groups: param_group['lr'] = 5e-5

        for img, lbl in loop:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(img), lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())

        # Save weights
        torch.save(model.state_dict(), "weights/retitransnet_last.pth")

if __name__ == "__main__":
    main()
