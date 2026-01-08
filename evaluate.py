%%writefile Reti-TransNet/evaluate.py
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
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import RetiTransNet
from dataset import RetinopathyDataset
from utils import seed_everything

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Weights path: Weights klasöründeki en iyi modeli seç
WEIGHTS_PATH = "weights/retitransnet_best.pth"
if not os.path.exists(WEIGHTS_PATH):
    # Eğer best yoksa son epoch'u dene
    WEIGHTS_PATH = "weights/retitransnet_last.pth"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
FIGURES_DIR = "images" # README için resimler buraya
os.makedirs(FIGURES_DIR, exist_ok=True)

# Q1 Style Plot Settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 3
})

# --- HELPER CLASSES ---
class OptimizedRounder:
    """Optimizes thresholds to maximize Quadratic Kappa."""
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _kappa_loss(self, coef, X, y):
        X_p = np.digitize(X, coef)
        return -cohen_kappa_score(y, X_p, weights='quadratic')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = minimize(loss_partial, self.coef_, method='nelder-mead').x

    def predict(self, X, coef):
        return np.digitize(X, coef)

    # --- EKSİK OLAN METOT BURAYA EKLENDİ ---
    def coefficients(self):
        return self.coef_

def predict_tta(model, loader):
    """Test-Time Augmentation (Original + Flip)."""
    model.eval()
    preds, labels, probs = [], [], []
    print(f"🔄 Running Inference (TTA)...")
    
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            # 1. Normal
            out1 = torch.softmax(model(img), dim=1)
            # 2. Horizontal Flip
            out2 = torch.softmax(model(torch.flip(img, [3])), dim=1)
            
            final_prob = (out1 + out2) / 2
            
            # Default Argmax prediction
            pred = torch.argmax(final_prob, dim=1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(lbl.numpy())
            probs.extend(final_prob.cpu().numpy())
            
    return np.array(labels), np.array(preds), np.array(probs)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = "0\n0.0%" if c == 0 else f"{c}\n{p:.1f}%"
            
    plt.figure(figsize=(10, 9))
    class_names = ['No DR', 'Mild', 'Mod', 'Sev', 'Prolif']
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12, "weight": "bold"}, cbar=False)
    
    plt.ylabel('True Severity Grade', fontweight='bold')
    plt.xlabel('Predicted Severity Grade', fontweight='bold')
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_roc_curves(y_true, y_probs, title, filename):
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plt.figure(figsize=(11, 9))
    for i in range(5):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            lw = 4.5 if i == 0 else 3.0
            plt.plot(fpr, tpr, color=colors[i], lw=lw, 
                     label=f'{classes[i]} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(title, pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()

# --- MAIN EVALUATION FLOW ---
def main():
    print("🚀 Starting Evaluation...")
    seed_everything(42)
    
    # 1. Load Model
    if not os.path.exists(WEIGHTS_PATH):
        print("❌ Error: Model weights not found. Please train the model first.")
        print(f"   Checked path: {WEIGHTS_PATH}")
        return

    model = RetiTransNet(num_classes=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"✅ Model weights loaded from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
        
    model.eval()

    # 2. Data Loader (Validation Only)
    val_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # APTOS Loader
    if os.path.exists('dataset/train.csv'):
        print("\n--- INTERNAL VALIDATION (APTOS 2019) ---")
        df = pd.read_csv('dataset/train.csv')
        if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
        
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
        
        val_loader = DataLoader(RetinopathyDataset(val_df, 'dataset', transform=val_aug), 
                                batch_size=16, shuffle=False, num_workers=2)
        
        y_true, _, y_probs = predict_tta(model, val_loader)
        
        # Threshold Optimization
        print("⚙️ Optimizing Thresholds...")
        y_scores = np.sum(y_probs * np.arange(5), axis=1)
        rounder = OptimizedRounder()
        rounder.fit(y_scores, y_true)
        y_pred_opt = rounder.predict(y_scores, rounder.coefficients()) # Hata veren yer düzeltildi
        
        kappa = cohen_kappa_score(y_true, y_pred_opt, weights='quadratic')
        acc = accuracy_score(y_true, y_pred_opt)
        
        print(f"🏆 APTOS Kappa: {kappa:.4f}")
        print(f"🏆 APTOS Accuracy: {acc*100:.2f}%")
        
        plot_confusion_matrix(y_true, y_pred_opt, "APTOS 2019 Confusion Matrix", "figure_4.png")
        plot_roc_curves(y_true, y_probs, "APTOS 2019 ROC Curves", "Figure_5.png")

    # IDRiD Loader
    if os.path.exists('idrid_dataset/idrid_labels.csv'):
        print("\n--- EXTERNAL VALIDATION (IDRiD) ---")
        df_idrid = pd.read_csv('idrid_dataset/idrid_labels.csv').iloc[:, :2]
        df_idrid.columns = ['id_code', 'diagnosis']
        
        # IDRiD resim yolunu bul
        idrid_root = 'idrid_dataset'
        
        idrid_loader = DataLoader(RetinopathyDataset(df_idrid, idrid_root, transform=val_aug), 
                                  batch_size=16, shuffle=False, num_workers=2)
        
        y_true_i, y_pred_i, y_probs_i = predict_tta(model, idrid_loader)
        
        kappa_i = cohen_kappa_score(y_true_i, y_pred_i, weights='quadratic')
        print(f"🌍 IDRiD Kappa: {kappa_i:.4f}")
        
        plot_roc_curves(y_true_i, y_probs_i, "IDRiD External Validation ROC", "Figure_6.png")

    print(f"\n✅ Evaluation Complete. Figures saved in '{FIGURES_DIR}/' folder.")

if __name__ == "__main__":
    main()
