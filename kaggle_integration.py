"""
Kaggle Integration for ARC Prize 2025
Handles downloading competition data and submitting predictions
"""

import os
import json
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
from typing import Dict, List
from pathlib import Path

def setup_kaggle():
    """Setup Kaggle API credentials and install package"""
    print("🔧 Setting up Kaggle API...")
    
    # Install kaggle package
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    
    # Check if credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("\n⚠️  Kaggle credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Upload the downloaded kaggle.json file to Colab")
        print("4. Run this cell again")
        
        # Create directory if it doesn't exist
        kaggle_dir.mkdir(exist_ok=True)
        
        # Check if user uploaded to Colab
        if os.path.exists('/content/kaggle.json'):
            import shutil
            shutil.copy('/content/kaggle.json', kaggle_json)
            os.chmod(kaggle_json, 0o600)
            print("✅ Kaggle credentials configured!")
        else:
            return False
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    return True

def download_competition_data(competition_name: str = "arc-prize-2025"):
    """Download official ARC Prize competition data"""
    if not setup_kaggle():
        return None
        
    import kaggle
    
    print(f"\n📥 Downloading {competition_name} competition data...")
    
    # Create data directory
    comp_dir = Path(f'/content/{competition_name}')
    comp_dir.mkdir(exist_ok=True)
    
    try:
        # Download competition files
        kaggle.api.competition_download_files(
            competition_name,
            path=comp_dir,
            unzip=True
        )
        
        print("✅ Competition data downloaded!")
        
        # List downloaded files
        files = list(comp_dir.glob('**/*'))
        print(f"\nDownloaded {len(files)} files:")
        for f in files[:10]:  # Show first 10
            print(f"  - {f.name}")
        
        return comp_dir
        
    except Exception as e:
        print(f"❌ Error downloading competition data: {e}")
        print("\nMake sure you have accepted the competition rules at:")
        print(f"https://www.kaggle.com/competitions/{competition_name}/rules")
        return None

def load_test_data(data_dir: Path):
    """Load ARC Prize test data"""
    test_file = data_dir / 'test.json'
    if not test_file.exists():
        # Try alternative paths
        test_file = data_dir / 'arc-agi_test_challenges.json'
    
    if not test_file.exists():
        print(f"❌ Test file not found in {data_dir}")
        return None
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"✅ Loaded {len(test_data)} test tasks")
    return test_data

def create_submission(predictions: Dict[str, List[List[int]]], 
                     output_path: str = '/content/submission.json'):
    """Create submission file in required format"""
    
    # ARC Prize expects JSON format with task_id -> output mapping
    submission = {}
    
    for task_id, outputs in predictions.items():
        # Each task can have multiple test cases
        submission[task_id] = []
        for output in outputs:
            # Convert numpy array to list if needed
            if isinstance(output, np.ndarray):
                output = output.tolist()
            submission[task_id].append(output)
    
    # Save submission
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"✅ Submission saved to {output_path}")
    return output_path

def submit_to_kaggle(submission_path: str, 
                     competition_name: str = "arc-prize-2025",
                     message: str = "V4 MEGA-SCALE submission"):
    """Submit predictions to Kaggle"""
    if not setup_kaggle():
        return
        
    import kaggle
    
    print(f"\n📤 Submitting to {competition_name}...")
    
    try:
        result = kaggle.api.competition_submit(
            submission_path,
            message=message,
            competition=competition_name
        )
        
        print("✅ Submission successful!")
        print(f"Message: {message}")
        
        # Check submission status
        submissions = kaggle.api.competition_submissions(competition_name)
        if submissions:
            latest = submissions[0]
            print(f"\nSubmission status: {latest.status}")
            if latest.publicScore:
                print(f"Public score: {latest.publicScore}")
                
    except Exception as e:
        print(f"❌ Error submitting: {e}")

def check_leaderboard(competition_name: str = "arc-prize-2025", top_n: int = 10):
    """Check current leaderboard position"""
    if not setup_kaggle():
        return
        
    import kaggle
    
    print(f"\n🏆 Checking {competition_name} leaderboard...")
    
    try:
        leaderboard = kaggle.api.competition_leaderboard(competition_name)
        
        print(f"\nTop {top_n} teams:")
        print("-" * 50)
        
        for i, entry in enumerate(leaderboard[:top_n]):
            print(f"{i+1:3d}. {entry.teamName[:30]:30s} {entry.score:.4f}")
        
        # Find your position
        username = kaggle.api.get_user().username
        for i, entry in enumerate(leaderboard):
            if entry.teamName == username:
                print(f"\nYour position: {i+1} / {len(leaderboard)} (Score: {entry.score:.4f})")
                break
                
    except Exception as e:
        print(f"❌ Error checking leaderboard: {e}")

def inference_with_model(test_data: Dict, model_path: str, device: str = 'cuda'):
    """Run inference on test data using trained model"""
    print(f"\n🔮 Running inference with {model_path}...")
    
    # Load model
    checkpoint = torch.load(model_path)
    
    # You'll need to recreate the model architecture
    # and load the saved weights
    # This is a placeholder - implement based on your model
    
    predictions = {}
    
    # Process each test task
    for task_id, task_data in test_data.items():
        task_predictions = []
        
        for test_input in task_data['test']:
            # Convert to tensor, run inference
            # This is where you'd use your trained model
            
            # Placeholder - replace with actual inference
            grid_size = len(test_input['input'])
            prediction = [[0] * grid_size for _ in range(grid_size)]
            
            task_predictions.append(prediction)
        
        predictions[task_id] = task_predictions
    
    return predictions

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("ARC PRIZE 2025 - KAGGLE INTEGRATION")
    print("="*60)
    
    # Download competition data
    comp_dir = download_competition_data("arc-prize-2025")
    
    if comp_dir:
        # Load test data
        test_data = load_test_data(comp_dir)
        
        if test_data:
            # Run inference (placeholder)
            predictions = inference_with_model(
                test_data,
                "/content/arc_models_v4/minerva_best.pt"
            )
            
            # Create submission
            submission_path = create_submission(predictions)
            
            # Submit to Kaggle
            submit_to_kaggle(
                submission_path,
                message="V4 MEGA-SCALE - A100 80GB optimized"
            )
            
            # Check leaderboard
            check_leaderboard()