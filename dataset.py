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
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
        # --- SMART INDEXING ---
        # Scan the root_dir recursively to map 'image_id' to 'full_path'.
        # This solves the issue where images are scattered in subfolders.
        self.image_map = {}
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    # Key: Filename without extension (e.g., '0024cdb')
                    key = os.path.splitext(file)[0]
                    self.image_map[key] = os.path.join(root, file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Image ID and Label
        # Assuming the first column is ID and second is Label in the DataFrame
        row = self.df.iloc[idx]
        img_id = str(row[0]) 
        
        # 2. Retrieve Path from Map
        img_path = self.image_map.get(img_id, None)
        
        # 3. Load & Preprocess
        image = None
        if img_path:
            # Apply Ben Graham Method (imported from utils.py)
            image = ben_graham_preprocessing(img_path)
        
        # Fallback for missing/corrupt images (Black image to prevent crash)
        if image is None: 
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # 4. Apply Augmentations (Albumentations)
        if self.transform:
            image = self.transform(image=image)['image']
            
        # 5. Return Tensor
        # Convert label to long tensor (for CrossEntropy)
        label = torch.tensor(int(row[1]), dtype=torch.long)
        
        return image, label