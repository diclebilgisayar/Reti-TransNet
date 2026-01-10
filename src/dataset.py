import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.utils import ben_graham_preprocessing

class RetinopathyDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_map = {}
        # print(f"🔍 Indexing images in '{root_dir}'...")
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    key = os.path.splitext(file)[0]
                    self.image_map[key] = os.path.join(root, file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        img_id = str(row["id_code"])
        img_path = self.image_map.get(img_id, None)
    
        image = None
        if img_path:
            image = ben_graham_preprocessing(img_path)
    
        if image is None: 
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']
    
        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return image, label
