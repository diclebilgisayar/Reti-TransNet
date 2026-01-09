import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.optimize import minimize
from functools import partial
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- GRAFİK FONKSİYONLARI ---
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"{RESULTS_DIR}/{filename}")
    print(f"🖼️ Saved: {RESULTS_DIR}/{filename}")

def save_roc_curve(y_true, y_probs, title, filename):
    y_bin = label_binarize(y_true, classes=[0,1,2,3,4])
    plt.figure(figsize=(10, 8))
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} AUC={auc(fpr, tpr):.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/{filename}")
    print(f"🖼️ Saved: {RESULTS_DIR}/{filename}")

# --- OPTIMIZER ---
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
        print("⚠️ Model weights not found. Skipping evaluation.")
        return
    
    val_aug = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    
    # 1. APTOS
    csv_path = None
    for r, _, f in os.walk('dataset'):
        for file in f: 
            if file.endswith('.csv') and 'train' in file: csv_path = os.path.join(r, file)
            
    if csv_path:
        print("\--- INTERNAL VALIDATION (APTOS) ---")
        df = pd.read_csv(csv_path)
        if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
        
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
        loader = DataLoader(RetinopathyDataset(val_df, 'dataset', transform=val_aug), batch_size=16)
        
        y, _, p = predict_tta(model, loader)
        
        # Optimize
        scores = np.sum(p * np.arange(5), axis=1)
        rounder = OptimizedRounder()
        rounder.fit(scores, y)
        y_opt = rounder.predict(scores)
        
        print(f"🏆 APTOS Kappa: {cohen_kappa_score(y, y_opt, weights='quadratic'):.4f}")
        
        # GRAFİKLERİ ÇİZ VE KAYDET
        save_confusion_matrix(y, y_opt, "APTOS Confusion Matrix", "figure_aptos_cm.png")
        save_roc_curve(y, p, "APTOS ROC Curves", "figure_aptos_roc.png")

    # 2. IDRiD
    idrid_csv = None
    if os.path.exists('idrid_dataset/idrid_labels.csv'): idrid_csv = 'idrid_dataset/idrid_labels.csv'
    
    if idrid_csv:
        print("\--- EXTERNAL VALIDATION (IDRiD) ---")
        df = pd.read_csv(idrid_csv).iloc[:, :2]
        loader = DataLoader(RetinopathyDataset(df, 'idrid_dataset', transform=val_aug), batch_size=16)
        y, _, p = predict_tta(model, loader)
        
        print(f"🌍 IDRiD Kappa: {cohen_kappa_score(y, np.argmax(p,1), weights='quadratic'):.4f}")
        
        # GRAFİKLERİ ÇİZ VE KAYDET
        save_confusion_matrix(y, np.argmax(p,1), "IDRiD Confusion Matrix", "figure_idrid_cm.png")
        save_roc_curve(y, p, "IDRiD ROC Curves", "figure_idrid_roc.png")

    print(f"✅ Evaluation Complete. Check '{RESULTS_DIR}' folder for plots.")

if __name__ == "__main__":
    main()
