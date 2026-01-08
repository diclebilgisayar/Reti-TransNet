import os
import zipfile
import sys

def download_datasets():
    print("🚀 Downloading Datasets for Reti-TransNet...")
    
    # 1. Klasörleri Oluştur
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('idrid_dataset', exist_ok=True)

    # 2. Kaggle API Kontrolü
    if not os.path.exists('kaggle.json'):
        print("❌ Error: 'kaggle.json' not found.")
        return

    # API Key Ayarı
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    os.system('chmod 600 kaggle.json')

    # --- 3. APTOS 2019 (OFFICIAL COMPETITION) ---
    print("\n📥 Downloading APTOS 2019...")
    
    if not os.path.exists('dataset/train.csv'):
        # İndirme komutu
        os.system('kaggle competitions download -c aptos2019-blindness-detection -p dataset')
        
        # Zip Açma
        zip_path = 'dataset/aptos2019-blindness-detection.zip'
        if os.path.exists(zip_path):
            print("📦 Extracting APTOS Zip...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall('dataset')
            os.remove(zip_path)
            
    # --- KONTROL ---
    if os.path.exists('dataset/train.csv'):
        print("✅ APTOS 2019 Ready.")
    else:
        print("\n⚠️ WARNING: 'train.csv' could not be downloaded automatically.")
        print("   Reason: Competition rules not accepted on Kaggle.")
        print("   Action: You will need to upload 'train.csv' manually in the next step.")

    # --- 4. IDRiD İndir ---
    print("\n📥 Downloading IDRiD...")
    if not os.path.exists('idrid_dataset/idrid_labels.csv'):
        os.system('kaggle datasets download -d mariaherrerot/idrid-dataset -p idrid_dataset')
        
        for file in os.listdir('idrid_dataset'):
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join('idrid_dataset', file), 'r') as z:
                    z.extractall('idrid_dataset')
                os.remove(os.path.join('idrid_dataset', file))
        print("✅ IDRiD Ready.")

if __name__ == "__main__":
    download_datasets()
