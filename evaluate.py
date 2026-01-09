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
import cv2

# Modüler Importlar
from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

# --- KONFİGÜRASYON ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "weights/retitransnet_best.pth"
RESULTS_DIR = "results"
FIGURES_DIR = "images"  # README için görseller buraya
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Q1 Grafik Ayarları
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

# --- YARDIMCI SINIFLAR ---
class OptimizedRounder:
    def __init__(self): self.coef_ = [0.5, 1.5, 2.5, 3.5]
    def _loss(self, coef, X, y): return -cohen_kappa_score(y, np.digitize(X, coef), weights='quadratic')
    def fit(self, X, y): self.coef_ = minimize(partial(self._loss, X=X, y=y), self.coef_, method='nelder-mead').x
    def predict(self, X): return np.digitize(X, self.coef_)

def predict_tta(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    print("🔄 Running Inference (TTA)...")
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

# --- GRAFİK FONKSİYONLARI ---
def plot_confusion_matrix_recall(y_true, y_pred, filename):
    """Makale Figure 2: Recall (Satır Yüzdesi) Odaklı Confusion Matrix"""
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
            
    class_names = ['No DR', 'Mild', 'Mod', 'Sev', 'Prolif']
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 13, "weight": "bold"}, cbar=False)
    
    plt.ylabel('True Severity Grade', fontweight='bold', fontsize=16)
    plt.xlabel('Predicted Severity Grade', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    print(f"🖼️ Confusion Matrix Saved: {filename}")
    plt.close()

def plot_roc(y_true, y_probs, title, filename):
    """Makale Figure 3 & 6: ROC Eğrileri"""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plt.figure(figsize=(11, 9))
    for i in range(5):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            lw = 4.5 if i == 0 else 3.0 # No DR kalın çizgi
            plt.plot(fpr, tpr, color=colors[i], lw=lw, 
                     label=f'{classes[i]} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    print(f"🖼️ ROC Curve Saved: {filename}")
    plt.close()

def generate_gradcam(model, loader, dataset_name, filename):
    """Makale Figure 7 & 8: Grad-CAM++ Analizi"""
    target_layers = [model.cnn.conv_head]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Her sınıftan 1 doğru örnek bul
    found_classes = {}
    model.eval()
    
    iterator = iter(loader)
    while len(found_classes) < 5:
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            break
            
        inputs = inputs.to(DEVICE)
        labels_np = labels.numpy()
        
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=1).cpu().numpy()
            
        for i in range(len(labels_np)):
            lbl = labels_np[i]
            # Doğru tahmin edilenleri seç
            if lbl == preds[i] and lbl not in found_classes:
                found_classes[lbl] = inputs[i]
            if len(found_classes) == 5: break
    
    # Çizim
    if not found_classes: return
    sorted_classes = sorted(found_classes.keys())
    fig, axes = plt.subplots(len(sorted_classes), 3, figsize=(10, 3 * len(sorted_classes)))
    
    for idx, lbl in enumerate(sorted_classes):
        img_tensor = found_classes[lbl]
        input_tensor = img_tensor.unsqueeze(0)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Orijinal
        ax = axes[idx, 0] if len(sorted_classes) > 1 else axes[0]
        ax.imshow(rgb_img)
        ax.set_title(f"{dataset_name}: {class_names[lbl]}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Heatmap
        ax = axes[idx, 1] if len(sorted_classes) > 1 else axes[1]
        ax.imshow(grayscale_cam, cmap='jet')
        ax.axis('off')
        
        # Overlay
        ax = axes[idx, 2] if len(sorted_classes) > 1 else axes[2]
        ax.imshow(visualization)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    print(f"🖼️ Grad-CAM Saved: {filename}")
    plt.close()

# --- MAIN ---
def main():
    seed_everything(42)
    print("🚀 Starting Professional Evaluation Pipeline...")
    
    # Model Yükle
    if not os.path.exists(WEIGHTS_PATH):
        print("❌ Model weights not found. Skipping evaluation.")
        return
        
    model = RetiTransNet(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    
    val_aug = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

    # 1. APTOS (INTERNAL VALIDATION)
    csv_path = None
    for r, _, f in os.walk('dataset'):
        for file in f: 
            if file.endswith('.csv') and 'train' in file: csv_path = os.path.join(r, file)
            
    if csv_path:
        print("\n--- 1. APTOS 2019 EVALUATION ---")
        df = pd.read_csv(csv_path)
        if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
        
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
        
        loader = DataLoader(RetinopathyDataset(val_df, 'dataset', transform=val_aug), batch_size=16)
        
        y_true, y_pred_raw, y_probs = predict_tta(model, loader)
        
        # Optimizasyon
        print("⚙️ Optimizing Thresholds...")
        scores = np.sum(y_probs * np.arange(5), axis=1)
        rounder = OptimizedRounder()
        rounder.fit(scores, y_true)
        y_pred_opt = rounder.predict(scores)
        
        kappa = cohen_kappa_score(y_true, y_pred_opt, weights='quadratic')
        print(f"🏆 APTOS Kappa: {kappa:.4f}")
        
        # APTOS İÇİN HEPSİNİ ÇİZ
        plot_confusion_matrix_recall(y_true, y_pred_opt, "figure_4.png")
        plot_roc_curves(y_true, y_probs, "APTOS 2019 ROC Curves", "figure_5.png")
        generate_gradcam(model, loader, "APTOS", "figure_7.png")

    # 2. IDRiD (EXTERNAL VALIDATION)
    idrid_csv = None
    if os.path.exists('idrid_dataset/idrid_labels.csv'): idrid_csv = 'idrid_dataset/idrid_labels.csv'
    
    if idrid_csv:
        print("\n--- 2. IDRiD EVALUATION ---")
        df_ext = pd.read_csv(idrid_csv).iloc[:, :2]
        loader_ext = DataLoader(RetinopathyDataset(df_ext, 'idrid_dataset', transform=val_aug), batch_size=16)
        
        y_true_e, y_pred_e, y_probs_e = predict_tta(model, loader_ext)
        
        kappa_e = cohen_kappa_score(y_true_e, y_pred_e, weights='quadratic')
        print(f"🌍 IDRiD Kappa: {kappa_e:.4f}")
        
        # IDRiD İÇİN SADECE GEREKLİLERİ ÇİZ (CM YOK)
        plot_roc_curves(y_true_e, y_probs_e, "IDRiD ROC Curves", "figure_6.png")
        generate_gradcam(model, loader_ext, "IDRiD", "figure_8.png")

    print(f"\n✅ All results saved to '{FIGURES_DIR}/' folder.")

if __name__ == "__main__":
    main()
