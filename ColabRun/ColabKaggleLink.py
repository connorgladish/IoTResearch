from google.colab import files
import os
import shutil

print("Please upload your kaggle.json file...")
uploaded = files.upload()

kaggle_dir = '/root/.kaggle'
os.makedirs(kaggle_dir, exist_ok=True)

# Find the uploaded file
uploaded_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded file: {uploaded_filename}")

# Copy to correct location
destination = os.path.join(kaggle_dir, 'kaggle.json')
shutil.copy(uploaded_filename, destination)

# Set permissions
os.chmod(destination, 0o600)

print("✅ Kaggle credentials installed successfully!")

# Verify
print("\n🔍 Verifying credentials:")
!cat /root/.kaggle/kaggle.json

print("\n✅ Ready to download datasets!")
