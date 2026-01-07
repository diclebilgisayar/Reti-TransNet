import os
import zipfile

def download_datasets():
    print("🚀 Downloading Datasets for Reti-TransNet...")
    
    # 1. Klasörleri Oluştur
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('idrid_dataset', exist_ok=True)

    # 2. Kaggle API Kontrolü
    if not os.path.exists('kaggle.json'):
        print("❌ Error: 'kaggle.json' not found in the root directory.")
        print("Please place your Kaggle API key here to download datasets.")
        return

    # API Key Ayarı
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    os.system('chmod 600 kaggle.json')

    # 3. APTOS 2019 İndir (MaxJen Mirror)
    print("\n📥 Downloading APTOS 2019...")
    if not os.path.exists('dataset/train.csv'):
        os.system('kaggle datasets download -d maxjen/aptos-dataset -p dataset')
        with zipfile.ZipFile('dataset/aptos-dataset.zip', 'r') as z:
            z.extractall('dataset')
        os.remove('dataset/aptos-dataset.zip')
        print("✅ APTOS 2019 Ready.")
    else:
        print("✅ APTOS 2019 already exists.")

    # 4. IDRiD İndir (External Validation)
    print("\n📥 Downloading IDRiD (External Validation)...")
    if not os.path.exists('idrid_dataset/idrid_labels.csv'):
        os.system('kaggle datasets download -d mariaherrerot/idrid-dataset -p idrid_dataset')
        with zipfile.ZipFile('idrid_dataset/idrid-dataset.zip', 'r') as z:
            z.extractall('idrid_dataset')
        os.remove('idrid_dataset/idrid-dataset.zip')
        print("✅ IDRiD Ready.")
    else:
        print("✅ IDRiD already exists.")

if __name__ == "__main__":
    download_datasets()
