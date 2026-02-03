"""
Load ACI IoT dataset from Google Drive
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive mounted!\n")

# ============================================
# LOCATE AND EXTRACT DATASET
# ============================================
import os
import zipfile

# UPDATE THIS PATH to where you uploaded the file. Will vary by person.
zip_path = '/content/drive/MyDrive/dataset/archive.zip'

# Check if file exists
if not os.path.exists(zip_path):
    print(f"❌ File not found:   {zip_path}")
    print("\nLooking for ZIP files in your Drive...")
    !find /content/drive/MyDrive -name "*aci*. zip" -o -name "*iot*.zip"
    print("\nUpdate the zip_path variable above with the correct path")
    raise Exception("Dataset file not found")

file_size_gb = os.path.getsize(zip_path) / (1024**3)
print(f"✅ Found dataset:   {file_size_gb:.1f} GB")

# Extract to Colab local storage (faster than working from Drive)
print("\n📦 Extracting to Colab storage...")
print("⚠️ This will take 10-20 minutes\n")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/aci_iot_data')

print("\n✅ Extraction complete!")

# Check extracted files
print("\n📂 Extracted files:")
!ls -lh /content/aci_iot_data/

import glob
csv_files = glob.glob('/content/aci_iot_data/*.csv')
parquet_files = glob.glob('/content/aci_iot_data/*.parquet')

print(f"\n✅ Found {len(csv_files)} CSV files")
print(f"✅ Found {len(parquet_files)} Parquet files")

if csv_files or parquet_files:
    all_files = csv_files + parquet_files
    dataset_file = max(all_files, key=lambda x: os.path.getsize(x))
    file_size_gb = os.path.getsize(dataset_file) / (1024**3)
    print(f"\n🎯 Using:   {dataset_file} ({file_size_gb:.1f} GB)")
    print("\n✅ READY FOR TRAINING!")
else:
    print("\n⚠️ Checking subdirectories...")
    !find /content/aci_iot_data -name "*.csv" -o -name "*.parquet"
