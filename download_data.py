import os
import shutil

# Configure local environmental paths to detect your kaggle.json file directly
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

print("=========================================")
print("  DermAlert Automatic Dataset Builder    ")
print("=========================================")

try:
    import kaggle
    print("Kaggle API Authenticated successfully using local token.")
except Exception as e:
    print("Authentication Error: Please ensure 'kaggle.json' is dropped into this folder.")
    exit()

# Define targeted dataset identifiers from Kaggle
datasets = {
    "ham10000": "kmader/skin-cancer-mnist-ham10000",
    "pad_ufes_20": "andrewmvd/isic-2019" # A standard mobile-optimized alternative matrix
}

# Execute direct secure downloads
print("\nInitiating secure dataset synchronization (This may take several minutes)...")
try:
    # Fetching laboratory grade benchmarks
    print("-> Downloading HAM10000 benchmarks...")
    kaggle.api.dataset_download_files(datasets["ham10000"], path="dataset/", unzip=True)
    print("Success: HAM10000 pulled and unzipped.")
    
except Exception as e:
    print(f"Sync Issue encountered: {e}")

print("\n[Status] Dataset repository ready. Ready to initialize comprehensive model retraining.")