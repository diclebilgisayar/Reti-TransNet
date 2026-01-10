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

from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Model ağırlıklarını kontrol et
if os.path.exists("weights/retitransnet_best.pth"):
    WEIGHTS_PATH = "weights/retitransnet_best.pth"
else:
    WEIGHTS_PATH = "weights/retitransnet_last.pth" 
    
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

class OptimizedRounder:
    """Optimizes thresholds to maximize Quadratic Kappa."""
    def __init__(self): self.coef_ = [0.5, 1.5, 2.5, 3.5]
    def _loss(self, coef, X, y): return -cohen_kappa_score(y, np.digitize(X, coef), weights='quadratic')
    def fit(self, X, y): self.coef_ = minimize(partial(self._loss, X=X, y=y), self.coef_, method='nelder-mead').x
    def predict(self, X, coef): return np.digitize(X, coef)
    def coefficients(self): return self.coef_

def predict_tta(model, loader):
    """Test-Time Augmentation (Original + Flip)."""
    model.eval()
    preds, labels, probs = [], [], []
    print("🔄 Running Inference (TTA)...")
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            out1 = torch.softmax(model(img), dim=1)
            out2 = torch.softmax(model(torch.flip(img, [3])), dim=1)
            final = (out1 + out2) / 2
            
            # Default Argmax prediction
            pred = torch.argmax(final, dim=1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(lbl.numpy())
            probs.extend(final.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Confusion Matrix Çizer"""
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    # Sıfıra bölme hatasını önle
    cm_sum[cm_sum == 0] = 1 
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
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    print(f"🖼️ Saved: {filename}")
    plt.close()

def plot_roc_curves(y_true, y_probs, title, filename):
    """ROC Eğrilerini Çizer (Hata veren fonksiyon buydu)"""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plt.figure(figsize=(11, 9))
    for i in range(5):
        # Sınıf var mı kontrol et
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            lw = 4.5 if i == 0 else 3.0 # No DR kalın çizgi
            plt.plot(fpr, tpr, color=colors[i], lw=lw, 
                     label=f'{classes[i]} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    print(f"🖼️ Saved: {filename}")
    plt.close()

def generate_gradcam(model, loader, dataset_name, filename):
    target_layers = [model.cnn.conv_head]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    found_classes = {}
    model.eval()
    
    iterator = iter(loader)
    max_batches = 50
    batch_count = 0
    
    while len(found_classes) < 5 and batch_count < max_batches:
        try:
            inputs, labels = next(iterator)
            batch_count += 1
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
    
    if not found_classes:
        print(f"⚠️ Warning: Could not find examples for Grad-CAM in {dataset_name}")
        return

    sorted_classes = sorted(found_classes.keys())
    fig, axes = plt.subplots(len(sorted_classes), 3, figsize=(10, 3 * len(sorted_classes)))
    
    if len(sorted_classes) == 1: axes = np.expand_dims(axes, 0)
    
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
        ax = axes[idx, 0]
        ax.imshow(rgb_img)
        ax.set_title(f"{dataset_name}: {class_names[lbl]}", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Heatmap
        ax = axes[idx, 1]
        ax.imshow(grayscale_cam, cmap='jet')
        ax.set_title("Heatmap", fontsize=12)
        ax.axis('off')
        
        # Overlay
        ax = axes[idx, 2]
        ax.imshow(visualization)
        ax.set_title("Overlay", fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    print(f"🖼️ Saved: {filename}")
    plt.close()

# --- MAIN EVALUATION FLOW ---
def main():
    print("🚀 Starting Evaluation...")
    seed_everything(42)
    
    # 1. Load Model
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Model weights not found at {WEIGHTS_PATH}. Train the model first.")
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
    aptos_csv = None
    aptos_root = None
    for r, d, f in os.walk('dataset'):
        for file in f: 
            if file.endswith('.csv') and 'train' in file: 
                aptos_csv = os.path.join(r, file)
                aptos_root = r # CSV'nin olduğu yer genelde köktür veya resimler altındadır
    
    if aptos_csv:
        print("\n--- INTERNAL VALIDATION (APTOS 2019) ---")
        df = pd.read_csv(aptos_csv)
        if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
        
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
        
        # Resim klasörü için root_dir olarak 'dataset' veriyoruz, dataset.py içindeki smart index bulacak
        loader = DataLoader(RetinopathyDataset(val_df, 'dataset', transform=val_aug), batch_size=16, shuffle=False, num_workers=2)
        
        y_true, _, y_probs = predict_tta(model, loader)
        
        # Optimization
        print("⚙️ Optimizing Thresholds...")
        scores = np.sum(y_probs * np.arange(5), axis=1)
        rounder = OptimizedRounder()
        rounder.fit(scores, y_true)
        y_pred_opt = rounder.predict(scores, rounder.coefficients()) # Düzeltildi
        
        kappa = cohen_kappa_score(y_true, y_pred_opt, weights='quadratic')
        acc = accuracy_score(y_true, y_pred_opt)
        
        print(f"🏆 APTOS Kappa: {kappa:.4f}")
        print(f"🏆 APTOS Accuracy: {acc*100:.2f}%")
        
        plot_confusion_matrix(y_true, y_pred_opt, "APTOS 2019 Confusion Matrix", "figure_4.png")
        plot_roc_curves(y_true, y_probs, "APTOS 2019 ROC Curves", "figure_5.png")
        generate_gradcam(model, loader, "APTOS", "figure_7.png")

    # IDRiD Loader
    idrid_csv = None
    if os.path.exists('idrid_dataset/idrid_labels.csv'): 
        idrid_csv = 'idrid_dataset/idrid_labels.csv'
        idrid_root = 'idrid_dataset'
    
    if idrid_csv:
        print("\n--- EXTERNAL VALIDATION (IDRiD) ---")
        df_idrid = pd.read_csv(idrid_csv).iloc[:, :2]
        df_idrid.columns = ['id_code', 'diagnosis']
        
        idrid_loader = DataLoader(RetinopathyDataset(df_idrid, idrid_root, transform=val_aug), 
                                  batch_size=16, shuffle=False, num_workers=2)
        
        y_true_e, y_pred_e, y_probs_e = predict_tta(model, idrid_loader)
        
        kappa_i = cohen_kappa_score(y_true_e, y_pred_e, weights='quadratic')
        print(f"🌍 IDRiD Kappa: {kappa_i:.4f}")
        
        plot_roc_curves(y_true_e, y_probs_e, "IDRiD External Validation ROC", "figure_6.png")
        generate_gradcam(model, idrid_loader, "IDRiD", "figure_8.png")

    print(f"\n✅ Evaluation Complete. Results saved in '{RESULTS_DIR}/' folder.")

if __name__ == "__main__":
    main()
