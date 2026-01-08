import os
import zipfile
import shutil
import sys

def download_datasets():
    """
    Automated script to download and setup datasets for Reti-TransNet.
    Compatible with both Google Colab (interactive upload) and Local Machines.
    """
    print("🚀 Starting Data Preparation for Reti-TransNet...")

    # 1. Create directories
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('idrid_dataset', exist_ok=True)

    # 2. Check for Kaggle API Key & Handle Colab Upload
    if not os.path.exists('kaggle.json'):
        # Check if running in Google Colab
        if 'google.colab' in sys.modules:
            print("⚠️ 'kaggle.json' not found. Opening upload window...")
            try:
                from google.colab import files
                uploaded = files.upload()
                if 'kaggle.json' not in uploaded:
                    print("❌ Error: You uploaded the wrong file. Please upload 'kaggle.json'.")
                    return
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                return
        else:
            # Running locally
            print("❌ Error: 'kaggle.json' not found in the root directory.")
            print("   Please place the 'kaggle.json' file here manually.")
            return

    # Set Kaggle Config Directory to current folder
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    
    # Permission fix
    try:
        os.chmod('kaggle.json', 0o600)
    except:
        pass 

    print("✅ Kaggle API Key configured.")

    # ---------------------------------------------------------
    # 3. DOWNLOAD APTOS 2019 (Internal Dataset)
    # Source: sovitrath/diabetic-retinopathy-224x224-2019-data
    # ---------------------------------------------------------
    print("\n📥 Checking APTOS 2019 Dataset...")
    
    if not os.path.exists('dataset/train.csv'):
        print("   Downloading APTOS 2019 (Resized 224x224)...")
        # Download command
        exit_code = os.system('kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-2019-data -p dataset')
        
        if exit_code != 0:
            print("❌ Failed to download APTOS. Check your internet or API key.")
            return

        # Unzip
        zip_path = 'dataset/diabetic-retinopathy-224x224-2019-data.zip'
        if os.path.exists(zip_path):
            print("   Extracting APTOS...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('dataset')
            os.remove(zip_path) # Clean up
            print("✅ APTOS 2019 Ready.")
        else:
            print("❌ Error: Zip file not found.")
    else:
        print("✅ APTOS 2019 already exists.")

    # ---------------------------------------------------------
    # 4. DOWNLOAD IDRiD (External Validation)
    # Source: mariaherrerot/idrid-dataset
    # ---------------------------------------------------------
    print("\n📥 Checking IDRiD Dataset...")

    if not os.path.exists('idrid_dataset/idrid_labels.csv'):
        print("   Downloading IDRiD...")
        exit_code = os.system('kaggle datasets download -d mariaherrerot/idrid-dataset -p idrid_dataset')
        
        if exit_code != 0:
            print("❌ Failed to download IDRiD.")
            return

        # Unzip (Handling potential sub-zips)
        print("   Extracting IDRiD...")
        for file in os.listdir('idrid_dataset'):
            if file.endswith('.zip'):
                zip_path = os.path.join('idrid_dataset', file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('idrid_dataset')
                os.remove(zip_path)
        
        print("✅ IDRiD Ready.")
    else:
        print("✅ IDRiD already exists.")

    print("\n🎉 All datasets are ready! You can proceed to training.")

if __name__ == "__main__":
    download_datasets()
