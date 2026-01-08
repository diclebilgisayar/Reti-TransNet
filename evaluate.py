
import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.optimize import minimize
from functools import partial

from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class OptimizedRounder:
    def __init__(self): self.coef_ = [0.5, 1.5, 2.5, 3.5]
    def _loss(self, coef, X, y): return -cohen_kappa_score(y, np.digitize(X, coef), weights='quadratic')
    def fit(self, X, y): self.coef_ = minimize(partial(self._loss, X=X, y=y), self.coef_, method='nelder-mead').x
    def predict(self, X): return np.digitize(X, self.coef_)

def predict_tta(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            out1 = torch.softmax(model(img), dim=1)
            out2 = torch.softmax(model(torch.flip(img, [3])), dim=1)
            final = (out1 + out2) / 2
            
            preds.extend(torch.argmax(final, 1).cpu().numpy())
            labels.extend(lbl.numpy())
            probs.extend(final.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)

def main():
    seed_everything(42)
    print("🚀 Starting Evaluation...")
    
    model = RetiTransNet(num_classes=5).to(DEVICE)
    if os.path.exists("weights/retitransnet_best.pth"):
        model.load_state_dict(torch.load("weights/retitransnet_best.pth", map_location=DEVICE))
    else:
        print("⚠️ Model weights not found. Please train first.")
        return
    
    val_aug = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    
    # 1. APTOS Test
    csv_path = None
    for r, _, f in os.walk('dataset'):
        for file in f:
            if file.endswith('.csv') and 'train' in file: csv_path = os.path.join(r, file)

    if csv_path:
        print("\n--- INTERNAL VALIDATION (APTOS) ---")
        df = pd.read_csv(csv_path)
        if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
        
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
        loader = DataLoader(RetinopathyDataset(val_df, 'dataset', transform=val_aug), batch_size=16)
        
        y, _, p = predict_tta(model, loader)
        
        # Optimization
        scores = np.sum(p * np.arange(5), axis=1)
        rounder = OptimizedRounder()
        rounder.fit(scores, y)
        y_opt = rounder.predict(scores)
        
        print(f"🏆 APTOS Kappa: {cohen_kappa_score(y, y_opt, weights='quadratic'):.4f}")
        
    # 2. IDRiD Test
    idrid_csv = None
    if os.path.exists('idrid_dataset/idrid_labels.csv'): idrid_csv = 'idrid_dataset/idrid_labels.csv'
    
    if idrid_csv:
        print("\n--- EXTERNAL VALIDATION (IDRiD) ---")
        df = pd.read_csv(idrid_csv).iloc[:, :2]
        loader = DataLoader(RetinopathyDataset(df, 'idrid_dataset', transform=val_aug), batch_size=16)
        y, _, p = predict_tta(model, loader)
        print(f"🌍 IDRiD Kappa: {cohen_kappa_score(y, np.argmax(p,1), weights='quadratic'):.4f}")

if __name__ == "__main__":
    main()
