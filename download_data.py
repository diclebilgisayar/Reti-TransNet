import os
import zipfile
import shutil

def download_datasets():
    """
    Automated script to download and setup datasets for Reti-TransNet.
    Requires 'kaggle.json' in the root directory.
    """
    print("🚀 Starting Data Preparation for Reti-TransNet...")

    # 1. Create directories
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('idrid_dataset', exist_ok=True)

    # 2. Check for Kaggle API Key
    if not os.path.exists('kaggle.json'):
        print("❌ Error: 'kaggle.json' not found in the root directory.")
        print("   Please download your API key from Kaggle -> Settings -> Create New Token")
        print("   and place the 'kaggle.json' file here.")
        return

    # Set Kaggle Config Directory to current folder
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    
    # Permission fix for Linux/Colab
    try:
        os.chmod('kaggle.json', 0o600)
    except:
        pass 

    # ---------------------------------------------------------
    # 3. DOWNLOAD APTOS 2019 (Internal Dataset)
    # Source: sovitrath/diabetic-retinopathy-224x224-2019-data (Public Mirror)
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
    # Source: mariaherrerot/idrid-dataset (Public Mirror)
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
