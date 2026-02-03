import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, cohen_kappa_score, accuracy_score
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. SETTINGS AND DATA LOADING
# ==========================================

# NOTE: Since we are in Colab local env, we use relative paths.
# Ensure train.py saves results to './results'
LOAD_DIR = './weight'
SAVE_DIR = './result_final'
os.makedirs(SAVE_DIR, exist_ok=True)

# Graph Styling (Paper Quality)
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

class_names = ['No DR', 'Mild', 'Mod', 'Sev', 'Prolif']

def load_data():
    print(f"📂 Loading evaluation data from: {os.path.abspath(LOAD_DIR)}")
    try:
        # APTOS Data
        # Adjust these filenames if your train.py saves them differently
        if os.path.exists(f'{LOAD_DIR}/aptos_final_true.npy'):
            y_true_apt = np.load(f'{LOAD_DIR}/aptos_final_true.npy')
            y_pred_apt = np.load(f'{LOAD_DIR}/aptos_final_opt_preds.npy')
            y_prob_apt = np.load(f'{LOAD_DIR}/aptos_final_probs.npy')
        else:
            # Fallback for generic names
            y_true_apt = np.load(f'{LOAD_DIR}/y_true.npy')
            y_pred_apt = np.load(f'{LOAD_DIR}/y_pred.npy') 
            y_prob_apt = np.load(f'{LOAD_DIR}/y_probs.npy')

        # IDRiD Data (External) - Optional
        if os.path.exists(f'{LOAD_DIR}/idrid_y_true.npy'):
            y_true_idr = np.load(f'{LOAD_DIR}/idrid_y_true.npy')
            y_prob_idr = np.load(f'{LOAD_DIR}/idrid_y_probs.npy')
            
            if os.path.exists(f'{LOAD_DIR}/idrid_y_pred.npy'):
                y_pred_idr = np.load(f'{LOAD_DIR}/idrid_y_pred.npy')
            else:
                y_pred_idr = np.argmax(y_prob_idr, axis=1)
            
            idrid_tuple = (y_true_idr, y_pred_idr, y_prob_idr)
        else:
            idrid_tuple = None

        print("✅ Data loaded successfully.")
        return (y_true_apt, y_pred_apt, y_prob_apt), idrid_tuple

    except Exception as e:
        print(f"❌ ERROR: Could not load .npy files. Please check if training finished successfully.\nError: {e}")
        return None, None

aptos_data, idrid_data = load_data()

# ==========================================
# 2. METRICS
# ==========================================
def print_metrics(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    cm = confusion_matrix(y_true, y_pred)
    tp_nodr = cm[0,0]
    total_nodr = np.sum(cm[0,:])
    recall_nodr = tp_nodr / total_nodr if total_nodr > 0 else 0

    print(f"\n📊 {dataset_name} PERFORMANCE SUMMARY:")
    print(f"   🔹 Accuracy: {acc*100:.2f}%")
    print(f"   🔹 Quadratic Kappa: {kappa:.4f}")
    print(f"   🔹 No DR Recall: {recall_nodr*100:.2f}%")
    print("-" * 30)

if aptos_data: print_metrics(aptos_data[0], aptos_data[1], "APTOS 2019 (Internal)")
if idrid_data: print_metrics(idrid_data[0], idrid_data[1], "IDRiD (External)")

# ==========================================
# 3. PLOTTING
# ==========================================
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

    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 13, "weight": "bold"}, cbar=False)
    plt.ylabel('True Severity Grade', fontweight='bold', fontsize=16)
    plt.xlabel('Predicted Severity Grade', fontweight='bold', fontsize=16)
    plt.title(title, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{filename}', dpi=300)
    print(f"   🖼️ Saved: {filename}")

def plot_roc_curves(y_true, y_probs, title, filename):
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.figure(figsize=(10, 8))
    for i in range(5):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            lw = 4.5 if i == 0 else 3.0
            plt.plot(fpr, tpr, color=colors[i], lw=lw,
                     label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.6)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold', labelpad=15)
    plt.ylabel('True Positive Rate', fontweight='bold', labelpad=15)
    plt.title(title, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{filename}', dpi=300)
    print(f"   🖼️ Saved: {filename}")

print(f"\n📈 Generating Plots in '{SAVE_DIR}'...")
if aptos_data:
    plot_confusion_matrix(aptos_data[0], aptos_data[1], "APTOS 2019 Confusion Matrix (Recall %)", "APTOS_CM.png")
    plot_roc_curves(aptos_data[0], aptos_data[2], "APTOS 2019 ROC Curves", "APTOS_ROC.png")

if idrid_data:
    plot_confusion_matrix(idrid_data[0], idrid_data[1], "IDRiD Confusion Matrix (External)", "IDRiD_CM.png")
    plot_roc_curves(idrid_data[0], idrid_data[2], "IDRiD ROC Curves (External)", "IDRiD_ROC.png")


print(f"✅ All figures saved to local '{SAVE_DIR}' folder.")
