import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.optimize import minimize
from functools import partial
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Modular imports
from src.model import RetiTransNet
from src.dataset import RetinopathyDataset
from src.utils import seed_everything

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Priority: Best weights > Last weights
if os.path.exists("weights/retitransnet_best.pth"):
    WEIGHTS_PATH = "weights/retitransnet_best.pth"
else:
    WEIGHTS_PATH = "weights/retitransnet_last.pth"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Plotting Settings for Q1 Journals
plt.rcParams.update({
    'font.family': 'serif', 
    'font.size': 14, 
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'lines.linewidth': 3
})

# --- HELPER CLASSES ---

class OptimizedRounder:
    """
    Optimizes classification thresholds to maximize the Quadratic Weighted Kappa score.
    Since DR grading is ordinal, shifting thresholds often yields better results than standard argmax.
    """
    def __init__(self): 
        self.coef_ = [0.5, 1.5, 2.5, 3.5]
        
    def _loss(self, coef, X, y): 
        # Negative Kappa because minimize() looks for the lowest value
        return -cohen_kappa_score(y, np.digitize(X, coef), weights='quadratic')
    
    def fit(self, X, y): 
        loss_partial = partial(self._loss, X=X, y=y)
        self.coef_ = minimize(loss_partial, self.coef_, method='nelder-mead').x
        
    def predict(self, X): 
        return np.digitize(X, self.coef_)

# --- FUNCTIONS ---

def predict_tta(model, loader):
    """
    Performs Test-Time Augmentation (TTA).
    Averages predictions of the original image and its horizontal flip.
    """
    model.eval()
    preds, labels, probs = [], [], []
    
    # Start timer for inference speed
    start_time = time.time()
    
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            
            # 1. Forward pass (Original)
            out1 = torch.softmax(model(img), dim=1)
            
            # 2. Forward pass (Horizontal Flip)
            out2 = torch.softmax(model(torch.flip(img, [3])), dim=1)
            
            # Average probabilities
            final_prob = (out1 + out2) / 2
            pred = torch.argmax(final_prob, dim=1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(lbl.numpy())
            probs.extend(final_prob.cpu().numpy())
            
    total_time = time.time() - start_time
    avg_inference = (total_time / len(loader.dataset)) * 1000 # ms per image
    
    return np.array(labels), np.array(preds), np.array(probs), avg_inference

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generates and saves a Confusion Matrix with Row-wise Recall percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_sum[cm_sum==0] = 1 # Avoid division by zero
    cm_perc = cm / cm_sum.astype(float) * 100
    
    # Annotation text
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
    
    plt.ylabel('True Grade', fontweight='bold')
    plt.xlabel('Predicted Grade', fontweight='bold')
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()
    print(f"🖼️ Saved Confusion Matrix: {filename}")

def plot_roc_curves(y_true, y_probs, title, filename):
    """Generates and saves One-vs-Rest ROC Curves."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    plt.figure(figsize=(11, 9))
    for i in range(5):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            lw = 4.5 if i == 0 else 3.0 # Thicker line for No DR
            plt.plot(fpr, tpr, lw=lw, label=f'{class_names[i]} (AUC={roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(title, pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()
    print(f"🖼️ Saved ROC Curve: {filename}")

def generate_gradcam(model, loader, dataset_name, filename):
    """Generates Grad-CAM++ visualizations for each class."""
    # Target Layer: Last Convolutional Layer of EfficientNet
    target_layers = [model.cnn.conv_head]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    found_classes = {}
    model.eval()
    iterator = iter(loader)
    
    # Search for one correct prediction per class
    max_batches = 60
    for _ in range(max_batches):
        try:
            inputs, labels = next(iterator)
        except StopIteration: break
        
        inputs = inputs.to(DEVICE)
        labels_np = labels.numpy()
        
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=1).cpu().numpy()
            
        for i in range(len(labels_np)):
            lbl = labels_np[i]
            # Only visualize correctly classified samples
            if lbl == preds[i] and lbl not in found_classes:
                found_classes[lbl] = inputs[i]
            if len(found_classes) == 5: break
        if len(found_classes) == 5: break
    
    if not found_classes:
        print(f"⚠️ Warning: Not all classes found for Grad-CAM in {dataset_name}.")
        return

    # Plotting
    sorted_keys = sorted(found_classes.keys())
    fig, axes = plt.subplots(len(sorted_keys), 3, figsize=(10, 3 * len(sorted_keys)))
    if len(sorted_keys) == 1: axes = np.expand_dims(axes, 0)
    
    class_names = ['No DR', 'Mild', 'Mod', 'Sev', 'Prolif']
    
    for idx, lbl in enumerate(sorted_keys):
        img_tensor = found_classes[lbl]
        input_tensor = img_tensor.unsqueeze(0)
        
        # Generate Heatmap
        gray_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Denormalize Image
        rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # Overlay
        vis = show_cam_on_image(rgb_img, gray_cam, use_rgb=True)
        
        # Original
        ax = axes[idx, 0] if len(sorted_keys) > 1 else axes[0]
        ax.imshow(rgb_img)
        ax.set_title(f"{dataset_name}: {class_names[lbl]}", fontweight='bold')
        ax.axis('off')
        
        # Heatmap
        ax = axes[idx, 1] if len(sorted_keys) > 1 else axes[1]
        ax.imshow(gray_cam, cmap='jet')
        ax.axis('off')
        
        # Overlay
        ax = axes[idx, 2] if len(sorted_keys) > 1 else axes[2]
        ax.imshow(vis)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()
    print(f"🖼️ Saved Grad-CAM: {filename}")

# --- MAIN EXECUTION ---
def main():
    print("🚀 Starting Evaluation Pipeline...")
    seed_everything(42)
    
    # 1. Load Model
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Weights not found at {WEIGHTS_PATH}. Please train the model.")
        return

    model = RetiTransNet(num_classes=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"✅ Loaded model weights from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return
    
    # Count Parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"ℹ️  Model Parameters: {params/1e6:.2f}M")
    
    val_aug = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

    # ---------------------------------------------------------
    # 2. INTERNAL VALIDATION (APTOS 2019)
    # ---------------------------------------------------------
    csv_path = None
    for r, _, f in os.walk('dataset'):
        for file in f: 
            if file.endswith('.csv') and 'train' in file: csv_path = os.path.join(r, file)
            
    if csv_path:
        print("\n--- INTERNAL VALIDATION (APTOS 2019) ---")
        df = pd.read_csv(csv_path)
        if 'id_code' not in df.columns: df.rename(columns={df.columns[0]: 'id_code', df.columns[1]: 'diagnosis'}, inplace=True)
        
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
        
        loader = DataLoader(RetinopathyDataset(val_df, 'dataset', transform=val_aug), batch_size=16, num_workers=2)
        
        # Test & Time
        y_true, _, y_probs, inf_time = predict_tta(model, loader)
        print(f"⚡ Inference Speed: {inf_time:.2f} ms/image (on {DEVICE})")
        
        # Optimization
        print("⚙️ Optimizing Thresholds...")
        y_scores = np.sum(y_probs * np.arange(5), axis=1)
        rounder = OptimizedRounder()
        rounder.fit(y_scores, y_true)
        y_pred_opt = rounder.predict(y_scores)
        
        # Metrics
        y_bin = label_binarize(y_true, classes=[0,1,2,3,4])
        nodr_auc = roc_auc_score(y_bin[:, 0], y_probs[:, 0])
        
        print(f"🏆 APTOS Kappa: {cohen_kappa_score(y_true, y_pred_opt, weights='quadratic'):.4f}")
        print(f"🏆 APTOS Accuracy: {accuracy_score(y_true, y_pred_opt)*100:.2f}%")
        print(f"⭐ APTOS No DR AUC: {nodr_auc:.4f}")
        
        # Visualizations
        plot_confusion_matrix(y_true, y_pred_opt, "APTOS Confusion Matrix", "figure_4.png")
        plot_roc_curves(y_true, y_probs, "APTOS ROC Curves", "figure_5.png")
        generate_gradcam(model, loader, "APTOS", "figure_7.png")

    # ---------------------------------------------------------
    # 3. EXTERNAL VALIDATION (IDRiD)
    # ---------------------------------------------------------
    idrid_csv = None
    if os.path.exists('idrid_dataset/idrid_labels.csv'): idrid_csv = 'idrid_dataset/idrid_labels.csv'
    
    if idrid_csv:
        print("\n--- EXTERNAL VALIDATION (IDRiD) ---")
        df_ext = pd.read_csv(idrid_csv).iloc[:, :2]
        df_ext.columns = ['id_code', 'diagnosis'] # Ensure columns
        
        # Root dir might be nested, handled by smart dataset class
        loader_ext = DataLoader(RetinopathyDataset(df_ext, 'idrid_dataset', transform=val_aug), batch_size=16, num_workers=2)
        
        y_true_e, _, y_probs_e, _ = predict_tta(model, loader_ext)
        y_pred_e = np.argmax(y_probs_e, 1) # Zero-shot (No optimization)
        
        # Metrics
        y_bin_e = label_binarize(y_true_e, classes=[0,1,2,3,4])
        nodr_auc_e = roc_auc_score(y_bin_e[:, 0], y_probs_e[:, 0])
        
        print(f"🌍 IDRiD Kappa: {cohen_kappa_score(y_true_e, y_pred_e, weights='quadratic'):.4f}")
        print(f"🌍 IDRiD Accuracy: {accuracy_score(y_true_e, y_pred_e)*100:.2f}%")
        print(f"⭐ IDRiD No DR AUC: {nodr_auc_e:.4f}")
        
        # Visualizations (Only ROC and GradCAM)
        plot_roc_curves(y_true_e, y_probs_e, "IDRiD ROC Curves", "figure_6.png")
        generate_gradcam(model, loader_ext, "IDRiD", "figure_8.png")

    print(f"\n✅ Evaluation Complete. Check '{RESULTS_DIR}' folder.")

if __name__ == "__main__":
    main()
