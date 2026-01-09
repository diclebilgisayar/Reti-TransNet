<h1 align="center">
👁️ Reti-TransNet: An Adaptive Gated Hybrid CNN-Transformer Framework for Diabetic Retinopathy Grading
</h1>


<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://www.kaggle.com/c/aptos2019-blindness-detection">
    <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg?logo=kaggle&logoColor=white" alt="Kaggle Dataset">
  </a>
  <a href="https://www.kaggle.com/c/aptos2019-blindness-detection">
    <img src="https://img.shields.io/badge/Task-Medical%20Imaging-blue.svg" alt="Task">
  </a>
  </a>
  <img src="https://img.shields.io/badge/Model-CNN-green.svg" alt="CNN">
  <img src="https://img.shields.io/badge/Model-Vision%20Transformer-purple.svg" alt="Vision Transformer">
  <img src="https://img.shields.io/badge/Metric-F1--Score-success.svg" alt="Metric">
  <a href="#">
</p>

---

## 📌 Abstract

Diabetic Retinopathy (DR) requires precise detection of both minute local lesions and global structural distortions. **Reti-TransNet** is a novel hybrid framework that synergizes the local feature extraction capability of **EfficientNet-B0** with the global context modeling power of **Swin Transformer**. Unlike trivial concatenation strategies, we introduce a learnable **"Adaptive Gated Fusion"** mechanism to dynamically weight the importance of local versus global features based on image complexity.

### 🔑 Key Achievements
*   🏆 **State-of-the-Art Reliability:** Achieved a **Quadratic Weighted Kappa of 0.90** on the internal APTOS 2019 dataset.
*   🌍 **Robust Generalization:** Demonstrated strong zero-shot performance on the external **IDRiD** dataset (**AUC 0.958** for screening healthy patients).
*   🔍 **Explainability:** Integrated **Grad-CAM++** ensures clinical transparency by localizing pathological biomarkers.
*   ⚡ **Efficiency:** Trained in just **25 epochs** on a single NVIDIA Tesla T4 GPU, highlighting computational efficiency.

---

## 🏗️ Architecture

The proposed architecture consists of two parallel branches (CNN & Transformer) fused by a novel **Adaptive Gated Fusion Mechanism**.

<p align="center">
  <img src="images/figure_2.png" width="95%">
  <br><em>Fig 1: Overall workflow of the Reti-TransNet architecture.</em>
</p>

---

## 📊 Experimental Results

Our model demonstrates superior reliability (Kappa) and screening safety compared to recent state-of-the-art methods.

### 1. Internal Validation (APTOS 2019)
Reti-TransNet shows exceptional performance in identifying healthy patients (Screening) and high consistency with expert graders.

| Metric | Result (95% CI) |
| :--- | :--- |
| **Quadratic Kappa ($\kappa$)** | **0.90** (0.88 - 0.92) |
| **Accuracy** | **84.31%** |
| **No DR Recall** | **98%** |
| **AUC (No DR)** | **0.997** |

<p align="center">
  <img src="images/Figure_5.png" width="50%" alt="APTOS ROC"/>
  <br><em>Fig 2: Internal ROC Curves</em>
</p>

### 2. External Validation (IDRiD - Zero-Shot)
Despite domain shift (different camera specifications), the model maintains high screening reliability.

| Metric | Result |
| :--- | :--- |
| **AUC (No DR)** | **0.958** |
| **Kappa ($\kappa$)** | **0.77** |

<p align="center">
  <img src="images/Figure_6.png" width="50%" alt="IDRiD ROC"/>
  <br><em>Fig 3: External ROC Curves (Zero-Shot)</em>
</p>

## 🚀 Quick Start on Google Colab (For Reviewers)

You can reproduce our results directly on Google Colab without installing anything on your local machine.

### 📝 Prerequisites
1.  **👤 Google Account:** To access Colab.
2.  **🔑 Kaggle API Key (`kaggle.json`):** Required to download datasets automatically. [Get it here](https://www.kaggle.com/account).

---

### 🛠 Step-by-Step Instructions

Open a new Colab Notebook:
- Go to [Google Colab](https://colab.research.google.com/).  
- Click **New Notebook**.  
- **⚠️ Important:** Go to `Runtime` → `Change runtime type` → Select **T4 GPU**.

---

You can reproduce the entire study (Setup -> Data -> Train -> Evaluate) by running a single code block.

### Prerequisites
1.  **Download this Repository:** Click **Code -> Download ZIP** on GitHub.
2.  **Get Kaggle API Key:** Have your `kaggle.json` file ready.
3.  **Open Colab:** Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
4.  **Upload ZIP:** Drag and drop the `Reti-TransNet-main.zip` file into the Colab file panel (left side).

### ⚡ Run Everything
Copy and run the following code in a Colab cell. It will handle unzipping, installation, data download, training, and evaluation automatically.

```python
import os
import sys

# --- 1. UNZIP & PROJECT SETUP ---
# Automatically find the uploaded ZIP file
zip_files = [f for f in os.listdir() if f.endswith('.zip')]

if len(zip_files) == 0:
    print("❌ ERROR: No .zip file found! Please drag & drop the repository ZIP file into the file panel.")
else:
    zip_name = zip_files[0]
    target_dir = "Reti-TransNet_Review"

    print(f"📦 Extracting '{zip_name}'...")
    # Unzip quietly (-q) and overwrite (-o) to the target directory
    os.system(f'unzip -q -o "{zip_name}" -d "{target_dir}"')

    # Navigate into the extracted directory
    os.chdir(target_dir)

    # Handle nested folder structure (common in GitHub downloads)
    # If requirements.txt is not in the root, check the immediate subfolder
    if not os.path.exists('requirements.txt'):
        sub_folders = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith('.')]
        if sub_folders:
            os.chdir(sub_folders[0])
            
    print(f"📍 Working Directory: {os.getcwd()}")

    # --- 2. INSTALL DEPENDENCIES ---
    if os.path.exists('requirements.txt'):
        print("⚙️ Installing dependencies...")
        # Install quietly, redirecting output to /dev/null to keep the log clean
        get_ipython().system('pip install -r requirements.txt > /dev/null')
        print("✅ Installation Complete!")
    else:
        print("❌ CRITICAL ERROR: 'requirements.txt' not found. Please check the ZIP file structure.")
        sys.exit()

    # --- 3. KAGGLE API SETUP ---
    if not os.path.exists('kaggle.json'):
        print("\n📂 Please upload your 'kaggle.json' API key now:")
        from google.colab import files
        uploaded = files.upload()
        if 'kaggle.json' in uploaded:
             print("✅ Kaggle key uploaded successfully.")
        else:
             print("⚠️ Warning: 'kaggle.json' was not found in the uploaded files.")

    # --- 4. DATA PREPARATION ---
    print("\n🚀 [1/3] Downloading & Preparing Data...")
    # Importing the script as a module to handle errors gracefully
    try:
        from download_data import download_datasets
        download_datasets()
    except ImportError:
        print("❌ Error: 'download_data.py' not found.")

    # --- 5. TRAINING (REAL-TIME MONITORING) ---
    print("\n🔥 [2/3] Starting Training...")
    print("    (Real-time Loss and Accuracy updates will appear below)\n")
    
    # Using get_ipython().system ensures the progress bar (tqdm) renders correctly in Colab
    get_ipython().system('python train.py')

    # --- 6. EVALUATION ---
    print("\n📊 [3/3] Running Evaluation & Visualization...")
    get_ipython().system('python evaluate.py')
        print("❌ Evaluation Failed.")
