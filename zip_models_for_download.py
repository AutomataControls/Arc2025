#!/usr/bin/env python3
"""
Zip all trained models for easy download from Colab
Solves the "Failed to fetch" error when downloading multiple files
"""

import os
import zipfile
from pathlib import Path

def zip_models():
    model_dir = '/content/arc_models_v4'
    output_zip = '/content/arc_models_v4_all.zip'
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory {model_dir} not found!")
        return
    
    # List all model files
    model_files = list(Path(model_dir).glob('*.pt'))
    
    if not model_files:
        print("❌ No model files found!")
        return
    
    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    # Create zip file
    print(f"\n📦 Creating zip file: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for model_file in model_files:
            zipf.write(model_file, model_file.name)
            print(f"  ✓ Added {model_file.name}")
    
    # Check final zip size
    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✅ Zip file created: {output_zip} ({zip_size_mb:.2f} MB)")
    print("\n📥 To download in Colab, run:")
    print("from google.colab import files")
    print("files.download('/content/arc_models_v4_all.zip')")

if __name__ == "__main__":
    zip_models()