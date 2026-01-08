# ==========================================
# 2. download_data.py
# ==========================================
with open(f"{Reti-TransNet}/download_data.py", "w") as f:
    f.write("""
import os
import zipfile
import sys

def download_datasets():
    print("🚀 Downloading Datasets for Reti-TransNet...")
    
    # Klasörleri Oluştur
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('idrid_dataset', exist_ok=True)

    # Kaggle API Kontrolü
    if not os.path.exists('kaggle.json'):
        print("❌ Error: 'kaggle.json' not found.")
        print("   Please place your 'kaggle.json' file in this directory.")
        return

    # API Ayarları
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    try:
        os.chmod('kaggle.json', 0o600)
    except:
        pass 

    # --- 1. APTOS 2019 (Sovitrath Mirror - Temiz) ---
    # Bu veri seti train.csv'yi ve 224x224 resimleri içerir. Ekstra işleme gerek yok.
    print("\\n📥 Downloading APTOS 2019...")
    if not os.path.exists('dataset/train.csv'):
        os.system('kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-2019-data -p dataset')
        
        zip_path = 'dataset/diabetic-retinopathy-224x224-2019-data.zip'
        if os.path.exists(zip_path):
            print("📦 Extracting APTOS...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall('dataset')
            os.remove(zip_path)
            print("✅ APTOS Ready.")
        else:
            print("❌ Download failed.")
    else:
        print("✅ APTOS already exists.")

    # --- 2. IDRiD (External Validation) ---
    print("\\n📥 Downloading IDRiD...")
    if not os.path.exists('idrid_dataset/idrid_labels.csv'):
        os.system('kaggle datasets download -d mariaherrerot/idrid-dataset -p idrid_dataset')
        
        # Zip bul ve aç
        for file in os.listdir('idrid_dataset'):
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join('idrid_dataset', file), 'r') as z:
                    z.extractall('idrid_dataset')
                os.remove(os.path.join('idrid_dataset', file))
        print("✅ IDRiD Ready.")
    else:
        print("✅ IDRiD already exists.")

    print("\\n🎉 Setup Complete! You can run 'python train.py'.")

if __name__ == "__main__":
    download_datasets()
""")