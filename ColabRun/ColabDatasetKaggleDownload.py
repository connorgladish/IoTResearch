"""
Force download ACI IoT Network Traffic Dataset 2023
"""

# ============================================
# VERIFY KAGGLE SETUP
# ============================================
import os
if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("❌ ERROR: kaggle.json not found!")
    raise Exception("Kaggle credentials missing")

print("✅ Kaggle credentials found!\n")

# ============================================
# INSTALL PACKAGES
# ============================================
print("📦 Installing packages...")
!pip install -q kaggle xgboost imbalanced-learn

import warnings
warnings.filterwarnings('ignore')

print("✅ Packages installed!\n")

# ============================================
# CLEAN UP OLD FILES & FORCE DOWNLOAD
# ============================================
print("🧹 Cleaning up any partial downloads...")
!rm -f aci-iot-network-traffic-dataset-2023.zip*
! rm -rf aci-iot-*

print("\n📥 Starting FRESH download of ACI IoT dataset...")
print("⚠️ This is 88.8GB - will take 30-60 minutes")
print("⚠️ DO NOT close this tab or let computer sleep!")
print("Go do something else but keep this window open!  🍕\n")

# Force fresh download
! kaggle datasets download -d emilynack/aci-iot-network-traffic-dataset-2023 --force

# Check if download completed
import glob

zip_file = 'aci-iot-network-traffic-dataset-2023.zip'
if os.path. exists(zip_file):
    file_size_gb = os.path.getsize(zip_file) / (1024**3)
    print(f"\n✅ Download complete!   File size:   {file_size_gb:.1f} GB")

    if file_size_gb < 80:
        print(f"⚠️ WARNING: File seems incomplete ({file_size_gb:.1f}GB, expected ~88GB)")
        print("The download may have been interrupted. Try again or use a different dataset.")
    else:
        print("\n📦 Unzipping...  (this will take 10-20 minutes)")
        !unzip -q aci-iot-network-traffic-dataset-2023.zip

        print("\n📂 Checking extracted files:")
        ! ls -lh *.csv *.parquet 2>/dev/null || ls -lh

        csv_files = glob.glob('*.csv')
        parquet_files = glob.glob('*.parquet')

        print(f"\n✅ Found {len(csv_files)} CSV files")
        print(f"✅ Found {len(parquet_files)} Parquet files")

        if csv_files or parquet_files:
            all_files = csv_files + parquet_files
            dataset_file = max(all_files, key=lambda x: os.path.getsize(x))
            file_size_gb = os.path.getsize(dataset_file) / (1024**3)
            print(f"\n🎯 Largest file:  {dataset_file} ({file_size_gb:.1f} GB)")
            print("\n✅ DATASET READY!  Proceed to training.")
        else:
            print("\n❌ No CSV or Parquet files found after extraction!")
            print("Checking directory structure...")
            !find . -name "*.csv" -o -name "*.parquet"
else:
    print(f"\n❌ Download failed!  {zip_file} not found.")
    print("Possible issues:")
    print("  1. Network connection interrupted")
    print("  2. Kaggle API rate limit")
    print("  3. Dataset permissions issue")
    print("\nTrying to diagnose...")
    ! kaggle datasets status emilynack/aci-iot-network-traffic-dataset-2023
