import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils import ben_graham_preprocessing

class RetinopathyDataset(Dataset):
    """
    Custom Dataset class for Diabetic Retinopathy.

    Features:
    1. Robust Image Indexing: Automatically finds images in sub-directories.
    2. Integration with Ben Graham's preprocessing.
    3. Handles Albumentations transforms.
    """
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image IDs and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # --- SMART INDEXING ---
        # Map image_id -> full image path (recursive)
        self.image_map = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    key = os.path.splitext(file)[0]
                    self.image_map[key] = os.path.join(root, file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get row safely (position-based)
        row = self.df.iloc[idx]

        # ✅ FIX: use iloc instead of row[0], row[1]
        img_id = str(row.iloc[0])
        label = torch.tensor(int(row.iloc[1]), dtype=torch.long)

        # 2. Retrieve image path
        img_path = self.image_map.get(img_id)

        # 3. Load & preprocess image
        if img_path is not None:
            image = ben_graham_preprocessing(img_path)
        else:
            # Fallback for missing images
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # 4. Apply Albumentations transforms
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
