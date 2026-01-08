import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import ben_graham_preprocessing

class RetinopathyDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
        # --- SMART INDEXING ---
        # Resimleri klasör yapısına bakmaksızın bulur
        self.image_map = {}
        # print(f"🔍 Indexing images in '{root_dir}'...") 
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    key = os.path.splitext(file)[0]
                    self.image_map[key] = os.path.join(root, file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        img_id = str(row["id_code"])
        label = torch.tensor(row["diagnosis"], dtype=torch.long)

        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
