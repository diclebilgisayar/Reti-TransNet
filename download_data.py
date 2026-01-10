import os
import zipfile
import sys
import shutil

def download_datasets():
    print("🚀 Downloading Datasets for Reti-TransNet...")
    
    # Klasörleri Oluştur
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('idrid_dataset', exist_ok=True)

    if not os.path.exists('kaggle.json'):
        print("🔍 'kaggle.json' not found.")
        if 'google.colab' in sys.modules:
            print("📂 Please upload your 'kaggle.json' file:")
            try:
                from google.colab import files
                uploaded = files.upload()
            except:
                pass
        else:
            print("❌ Please place 'kaggle.json' in this folder.")
            return

    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    os.system('chmod 600 kaggle.json')

    # --- 1. APTOS 2019 ---
    print("\n📥 Downloading APTOS 2019...")
    if not os.path.exists('dataset/train.csv'):
        os.system('kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-2019-data -p dataset')
        
        if os.path.exists('dataset/diabetic-retinopathy-224x224-2019-data.zip'):
            print("📦 Extracting APTOS...")
            with zipfile.ZipFile('dataset/diabetic-retinopathy-224x224-2019-data.zip', 'r') as z:
                z.extractall('dataset')
            os.remove('dataset/diabetic-retinopathy-224x224-2019-data.zip')
            print("✅ APTOS Ready.")
        else:
            print("❌ APTOS Download Failed.")

    # --- 2. IDRiD ---
    print("\n📥 Downloading IDRiD...")
    if not os.path.exists('idrid_dataset/idrid_labels.csv'):
        os.system('kaggle datasets download -d mariaherrerot/idrid-dataset -p idrid_dataset')
        
        for file in os.listdir('idrid_dataset'):
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join('idrid_dataset', file), 'r') as z:
                    z.extractall('idrid_dataset')
                os.remove(os.path.join('idrid_dataset', file))
        print("✅ IDRiD Ready.")

    print("\n🎉 Setup Complete! You can run 'python train.py'.")

if __name__ == "__main__":
    download_datasets()
