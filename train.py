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
import cv2

from model import RetiTransNet
from utils import seed_everything, ben_graham_preprocessing

# --- ROBUST DATASET CLASS (Dosyaları Otomatik Bulur) ---
class RetinopathyDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. Klasördeki TÜM resimlerin yollarını hafızaya al (Haritalama)
        self.image_map = {}
        print(f"🔍 Indexing images in {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    # Dosya adını (uzantısız) anahtar yap
                    key = os.path.splitext(file)[0]
                    self.image_map[key] = os.path.join(root, file)
        
        print(f"✅ Indexed {len(self.image_map)} images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id_code'])
        
        # 2. Haritadan dosya yolunu çek
        img_path = self.image_map.get(img_id, None)
        
        image = None
        if img_path:
            try:
                # Ben Graham Preprocessing (utils.py içinden)
                image = ben_graham_preprocessing(img_path)
            except:
                pass
        
        # Eğer resim bulunamazsa veya bozuksa siyah ekran ver (Eğitimi durdurma)
        if image is None:
            # print(f"⚠️ Warning: Image not found for ID {img_id}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']
            
        label = torch.tensor(int(row['diagnosis']), dtype=torch.long)
        return image, label

def main():
    seed_everything()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    EPOCHS = 25
    LR = 1e-4
    
    # 1. Veri Hazırlığı
    print("🚀 Loading Data...")
    
    # CSV Dosyasını Bul
    csv_path = None
    if os.path.exists('dataset/train.csv'):
        csv_path = 'dataset/train.csv'
    else:
        # Alt klasörleri ara
        for root, _, files in os.walk('dataset'):
            for f in files:
                if f.endswith('.csv') and 'train' in f:
                    csv_path = os.path.join(root, f)
                    break
    
    if not csv_path:
        print("❌ ERROR: 'train.csv' not found! Please upload it manually.")
        return

    df = pd.read_csv(csv_path)
    # Sütun düzeltme
    if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

    train_aug = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), 
        A.Rotate(limit=30, p=0.5), 
        A.Normalize(), 
        ToTensorV2()
    ])
    
    # Dataset ('dataset' klasörünü kök olarak veriyoruz, içini kendi tarayacak)
    train_ds = RetinopathyDataset(train_df, 'dataset', transform=train_aug)
    
    # Sampler
    targets = train_df['diagnosis'].values
    class_counts = np.bincount(targets)
    weights = 1. / class_counts
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weights), len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

    # 2. Model
    model = RetiTransNet(num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # 3. Eğitim Döngüsü
    print("🔥 Starting Training Loop (25 Epochs)...")
    
    # Klasör yoksa oluştur
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        # Fine-tuning switch (16. Epochta LR düşür)
        if epoch == 16:
            print("🔄 Switching to Fine-Tuning Phase (Lower LR)...")
            for param_group in optimizer.param_groups: param_group['lr'] = 5e-5

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
        
        # Save weights
        torch.save(model.state_dict(), f"weights/retitransnet_epoch_{epoch}.pth")

    # En son modeli 'best' olarak da kaydet
    torch.save(model.state_dict(), "weights/retitransnet_best.pth")
    print("✅ Training Completed. Weights saved to 'weights/' folder.")

if __name__ == "__main__":
    main()
