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
        print("❌ Error: 'kaggle.json' not found in the root directory.")
        print("Please upload your kaggle.json file.")
        return

    # API Key Ayarı
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    os.system('chmod 600 kaggle.json')

    # --- 3. APTOS 2019 (OFFICIAL COMPETITION) ---
    print("\n📥 Downloading APTOS 2019 (Official Competition Data)...")
    
    # Dosya zaten yoksa indirmeyi dene
    if not os.path.exists('dataset/train.csv'):
        # Not: -c competition flag'i kullanılır
        result = os.system('kaggle competitions download -c aptos2019-blindness-detection -p dataset')
        
        # İndirme başarılıysa zip'i aç
        zip_path = 'dataset/aptos2019-blindness-detection.zip'
        if os.path.exists(zip_path):
            print("📦 Extracting APTOS Zip...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall('dataset')
            os.remove(zip_path)
            
    # --- KRİTİK KONTROL: train.csv İndi mi? ---
    if os.path.exists('dataset/train.csv'):
        print("✅ APTOS 2019 Ready.")
    else:
        print("\n⚠️ WARNING: 'train.csv' could not be downloaded!")
        print("Possible Reason: You have not accepted the competition rules.")
        print("👉 Solution: Go to https://www.kaggle.com/c/aptos2019-blindness-detection/rules")
        print("   1. Click 'Join Competition' or 'Late Submission'")
        print("   2. Verify your phone number if asked")
        print("   3. Accept the rules")
        
        # Colab'daysak Manuel Yükleme İste
        try:
            import google.colab
            print("\n📂 Alternative: Please upload 'train.csv' manually from your computer:")
            from google.colab import files
            uploaded = files.upload()
            # Yüklenen dosyayı dataset klasörüne taşı
            for filename in uploaded.keys():
                if 'train' in filename and filename.endswith('.csv'):
                    os.rename(filename, 'dataset/train.csv')
                    print("✅ 'train.csv' manually loaded.")
        except ImportError:
            print("   Then run this script again.")

    # --- 4. IDRiD İndir (External Validation) ---
    print("\n📥 Downloading IDRiD (External Validation)...")
    if not os.path.exists('idrid_dataset/idrid_labels.csv'):
        os.system('kaggle datasets download -d mariaherrerot/idrid-dataset -p idrid_dataset')
        
        # Zip kontrolü ve açma
        # Not: Kaggle bazen zip ismini değiştirebilir, klasördeki zipe bakalım
        for file in os.listdir('idrid_dataset'):
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join('idrid_dataset', file), 'r') as z:
                    z.extractall('idrid_dataset')
                os.remove(os.path.join('idrid_dataset', file))
        print("✅ IDRiD Ready.")
    else:
        print("✅ IDRiD already exists.")

if __name__ == "__main__":
    download_datasets()
