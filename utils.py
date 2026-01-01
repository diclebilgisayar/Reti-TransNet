import cv2
import numpy as np
import torch
import random
import os

def seed_everything(seed=42):
    """Q1 standardı: Tekrarlanabilirlik için tohum sabitleme."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ben_graham_preprocessing(image_path):
    """
    Retina görüntüleri için standart ön işleme (Ben Graham's Method).
    Damar yapılarını belirginleştirir ve ışık farklarını giderir.
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Siyah kenarları kırp
        if img.ndim == 2: mask = img > 7
        else: mask = img[:,:,0] > 7
        img = img[np.ix_(mask.any(1),mask.any(0))]
        
        # 2. Boyutlandır
        img = cv2.resize(img, (224, 224))
        
        # 3. Gaussian Blur ile yerel renk ortalamasını çıkar
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
        return img
    except Exception as e:
        print(f"Error loading image: {image_path}")
        return np.zeros((224, 224, 3), dtype=np.uint8)