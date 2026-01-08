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
        
        # Smart Indexing (Alt klasörleri bul)
        self.image_map = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
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
