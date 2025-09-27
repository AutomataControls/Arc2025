import zipfile
import os
from pathlib import Path

model_dir = '/content/arc_models_v4'
output_zip = '/content/arc_models_v4_all.zip'

# List all model files
model_files = list(Path(model_dir).glob('*.pt'))
print(f"Found {len(model_files)} model files:")
for f in model_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"  - {f.name} ({size_mb:.2f} MB)")

# Create zip file
print(f"\n📦 Creating zip file...")
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for model_file in model_files:
        zipf.write(model_file, model_file.name)
        print(f"  ✓ Added {model_file.name}")

# File ready for download
zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
print(f"\n✅ Zip file created: {output_zip} ({zip_size_mb:.2f} MB)")
print("\nNow run this in a Colab cell to download:")
print("from google.colab import files")
print("files.download('/content/arc_models_v4_all.zip')")