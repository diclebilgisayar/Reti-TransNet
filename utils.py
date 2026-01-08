import os
import random
import numpy as np
import torch
import cv2

def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    This is a requirement for Q1 academic papers to guarantee consistent results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Global seed set to {seed}")

def ben_graham_preprocessing(image_path, img_size=224):
    """
    Applies Ben Graham's method to improve lighting and emphasize blood vessels.
    
    Steps:
    1. Read image.
    2. Auto-crop to remove black borders.
    3. Resize to target size.
    4. Apply Gaussian Blur and weighted addition.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Auto-crop black borders
        # Check if image is grayscale or color
        if img.ndim == 2: 
            mask = img > 7
            img = img[np.ix_(mask.any(1), mask.any(0))]
        else: 
            mask = img > 7
            img = img[np.ix_(mask.any(1), mask.any(0))]
        
        # 2. Resize
        img = cv2.resize(img, (img_size, img_size))
        
        # 3. Gaussian Blur (Local Average Color Subtraction)
        # alpha=4, beta=-4, gamma=128 are standard values for this method
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
        
        return img
        
    except Exception as e:
        # Return a black image in case of error to prevent crashing
        # print(f"⚠️ Warning: Could not process image {image_path}: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
