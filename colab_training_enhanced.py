# ARC Prize 2025 - ENHANCED Training Script for Google Colab
# This version trains with reconstruction loss to break the 70% barrier

# Install packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "matplotlib", "numpy", "pandas", "tqdm", "onnx", "onnxruntime", "plotly", "scikit-learn", "-q"])
print("✓ Packages installed")

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time
import gc

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# Clone repository
print("\n📥 Cloning ARC Prize 2025 repository...")
if os.path.exists('Arc2025'):
    shutil.rmtree('Arc2025')
os.system("git clone https://github.com/AutomataControls/Arc2025.git")

# Make sure we have the latest version
os.system("cd Arc2025 && git pull")
print("✓ Repository cloned and updated")

# Add to path
sys.path.append('/content/Arc2025')

# Import enhanced models
from Arc2025.models.arc_models_enhanced import create_enhanced_models

# Import monitor
from Arc2025.colab_monitor_integration import setup_colab_monitor

# Enable mixed precision training for A100
from torch.cuda.amp import GradScaler, autocast

# Verify models load correctly
print("\n🔍 Verifying enhanced models...")
try:
    test_models = create_enhanced_models()
    print(f"✓ Successfully loaded {len(test_models)} models: {list(test_models.keys())}")
    del test_models
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Hyperparameters
BATCH_SIZE = 64  # Increased to better utilize A100 GPU
GRADIENT_ACCUMULATION_STEPS = 1  # No need for accumulation with large batch
LEARNING_RATE = 0.0002  # Slightly higher LR for larger batch
NUM_EPOCHS = 100
MAX_GRID_SIZE = 30
NUM_COLORS = 10
DEVICE = device

# Loss weights
RECONSTRUCTION_WEIGHT = 1.0
PATTERN_WEIGHT = 0.1

print("\n⚙️ Configuration:")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Max grid size: {MAX_GRID_SIZE}")
print(f"  Device: {DEVICE}")

# Data is already in the cloned repo
print("\n📊 Using ARC dataset from cloned repository...")
DATA_DIR = '/content/Arc2025/data'
print(f"✓ Dataset location: {DATA_DIR}")

class ARCDatasetEnhanced(Dataset):
    """Enhanced dataset that returns both input and output grids"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = data_dir
        self.split = split
        self.samples = []
        self.pattern_labels = []
        self._load_data()
        
    def _load_data(self):
        """Load training data with input-output pairs"""
        print(f"Loading {self.split} data...")
        
        # Load challenges and solutions
        with open(f'{self.data_dir}/arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        with open(f'{self.data_dir}/arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        # Process each task
        for task_id, task_data in challenges.items():
            train_examples = task_data['train']
            test_examples = task_data['test']
            task_solutions = solutions[task_id]
            
            # Add training examples
            for example in train_examples:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                self.samples.append({
                    'task_id': task_id,
                    'input': input_grid,
                    'output': output_grid,
                    'type': 'train_example'
                })
                
                pattern_type = self._infer_pattern_type(input_grid, output_grid)
                self.pattern_labels.append(pattern_type)
            
            # Add test examples
            for i, test_input in enumerate(test_examples):
                input_grid = np.array(test_input['input'])
                output_grid = np.array(task_solutions[i])
                
                self.samples.append({
                    'task_id': task_id,
                    'input': input_grid,
                    'output': output_grid,
                    'type': 'test_example'
                })
                
                pattern_type = self._infer_pattern_type(input_grid, output_grid)
                self.pattern_labels.append(pattern_type)
        
        # Data augmentation
        self._augment_data()
        print(f"Loaded {len(self.samples)} samples")
    
    def _infer_pattern_type(self, input_grid: np.ndarray, output_grid: np.ndarray) -> int:
        """Infer pattern type from input-output pair"""
        if np.array_equal(input_grid, np.rot90(output_grid)):
            return 1  # Rotation
        elif np.array_equal(input_grid, np.fliplr(output_grid)):
            return 2  # Reflection
        elif input_grid.shape != output_grid.shape:
            return 3  # Size change
        elif not np.array_equal(np.unique(input_grid), np.unique(output_grid)):
            return 4  # Color change
        else:
            return 0  # Other/complex
    
    def _augment_data(self):
        """Add augmented samples"""
        augmented = []
        augmented_labels = []
        
        for idx, sample in enumerate(self.samples):
            input_grid = sample['input']
            output_grid = sample['output']
            
            # Rotation augmentation
            for k in [1, 2, 3]:
                aug_input = np.rot90(input_grid, k)
                aug_output = np.rot90(output_grid, k)
                
                augmented.append({
                    'task_id': sample['task_id'] + f'_rot{k*90}',
                    'input': aug_input,
                    'output': aug_output,
                    'type': 'augmented'
                })
                augmented_labels.append(self.pattern_labels[idx])
            
            # Reflection augmentation
            for axis in [0, 1]:
                aug_input = np.flip(input_grid, axis=axis)
                aug_output = np.flip(output_grid, axis=axis)
                
                augmented.append({
                    'task_id': sample['task_id'] + f'_flip{axis}',
                    'input': aug_input,
                    'output': aug_output,
                    'type': 'augmented'
                })
                augmented_labels.append(self.pattern_labels[idx])
        
        self.samples.extend(augmented)
        self.pattern_labels.extend(augmented_labels)
        print(f"Added {len(augmented)} augmented samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert grids to one-hot encoding
        input_grid = self._to_one_hot(sample['input'])
        output_grid = self._to_one_hot(sample['output'])
        
        # Pad to max size
        input_grid = self._pad_grid(input_grid)
        output_grid = self._pad_grid(output_grid)
        
        return {
            'input': torch.FloatTensor(input_grid),
            'output': torch.FloatTensor(output_grid),
            'pattern_label': torch.LongTensor([self.pattern_labels[idx]]),
            'task_id': sample['task_id']
        }
    
    def _to_one_hot(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to one-hot encoding"""
        h, w = grid.shape
        one_hot = np.zeros((NUM_COLORS, h, w))
        
        for i in range(h):
            for j in range(w):
                color = int(grid[i, j])
                if 0 <= color < NUM_COLORS:
                    one_hot[color, i, j] = 1
        
        return one_hot
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to max size"""
        c, h, w = grid.shape
        if h >= MAX_GRID_SIZE or w >= MAX_GRID_SIZE:
            grid = grid[:, :MAX_GRID_SIZE, :MAX_GRID_SIZE]
            h = min(h, MAX_GRID_SIZE)
            w = min(w, MAX_GRID_SIZE)
        
        padded = np.zeros((c, MAX_GRID_SIZE, MAX_GRID_SIZE))
        padded[:, :h, :w] = grid[:, :h, :w]
        
        return padded


class ReconstructionLoss(nn.Module):
    """Custom loss for grid reconstruction"""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reconstruction loss"""
        B, C, H, W = pred.shape
        
        # Convert one-hot target to class indices
        target_indices = target.argmax(dim=1)  # (B, H, W)
        
        # Reshape for cross entropy
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        target_flat = target_indices.reshape(-1)  # (B*H*W,)
        
        # Compute loss
        loss = self.ce_loss(pred_flat, target_flat)
        loss = loss.reshape(B, H, W)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss


# Main training function
def train_enhanced_models():
    """Train enhanced models with reconstruction loss"""
    print("\n🚀 Starting Enhanced Model Training")
    print("="*60)
    
    # Set up monitor
    monitor = setup_colab_monitor()
    
    # Create dataset
    dataset = ARCDatasetEnhanced(DATA_DIR, split='train')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create models
    models = create_enhanced_models()
    
    # Loss functions
    reconstruction_loss_fn = ReconstructionLoss()
    pattern_loss_fn = nn.CrossEntropyLoss()
    
    # Training results
    training_history = {}
    
    # Train each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🧠 Training {model_name.upper()}")
        print(f"{'='*60}")
        
        model = model.to(DEVICE)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        # Mixed precision scaler
        scaler = GradScaler()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            for batch in pbar:
                input_grids = batch['input'].to(DEVICE)
                output_grids = batch['output'].to(DEVICE)
                pattern_labels = batch['pattern_label'].squeeze(1).to(DEVICE)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast():
                    # Forward pass
                    if model_name == 'chronos':
                        # CHRONOS expects a list of tensors for sequence processing
                        # For training, we'll pass single frames
                        outputs = model([input_grids])
                    else:
                        outputs = model(input_grids, output_grids, mode='train')
                    
                    # Reconstruction loss
                    pred_output = outputs['predicted_output']
                    recon_loss = reconstruction_loss_fn(pred_output, output_grids)
                    
                    # Total loss
                    loss = RECONSTRUCTION_WEIGHT * recon_loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update weights
                if (train_steps + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item()
                train_steps += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_grids = batch['input'].to(DEVICE)
                    output_grids = batch['output'].to(DEVICE)
                    
                    # Mixed precision forward pass
                    with autocast():
                        # Forward pass
                        if model_name == 'chronos':
                            outputs = model([input_grids])
                        else:
                            outputs = model(input_grids)
                        
                        # Compute loss
                        pred_output = outputs['predicted_output']
                        loss = reconstruction_loss_fn(pred_output, output_grids)
                    
                    val_loss += loss.item()
                    
                    # Compute accuracy (exact match)
                    pred_colors = pred_output.argmax(dim=1)
                    target_colors = output_grids.argmax(dim=1)
                    
                    # Check exact grid match
                    matches = (pred_colors == target_colors).all(dim=[1,2])
                    val_correct += matches.sum().item()
                    val_total += input_grids.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / train_steps
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total * 100
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Update monitor
            monitor.update({
                'model': model_name,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'lr': scheduler.get_last_lr()[0]
            })
            
            # Check if we hit 85% target
            if val_accuracy >= 85:
                print(f"\n🎉 {model_name.upper()} reached 85% accuracy!")
                monitor.alert(f"{model_name} reached target accuracy!", "success")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_accuracy
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy
                }, f'/content/arc_models/{model_name}_enhanced_best.pt')
                
                print(f"✅ New best model saved! Val Acc: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Store training history
        training_history[model_name] = history
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': history,
            'final_accuracy': history['val_accuracy'][-1],
            'best_accuracy': best_val_acc
        }, f'/content/arc_models/{model_name}_enhanced_final.pt')
        
        # Export to ONNX
        print(f"\n📦 Exporting {model_name} to ONNX...")
        dummy_input = torch.randn(1, NUM_COLORS, MAX_GRID_SIZE, MAX_GRID_SIZE).to(DEVICE)
        
        if model_name == 'chronos':
            torch.onnx.export(
                model,
                ([dummy_input],),
                f'/content/arc_models/{model_name}_enhanced.onnx',
                input_names=['sequence'],
                output_names=['predicted_output'],
                dynamic_axes={'sequence': {0: 'batch_size'}},
                opset_version=13
            )
        else:
            torch.onnx.export(
                model,
                dummy_input,
                f'/content/arc_models/{model_name}_enhanced.onnx',
                input_names=['input_grid'],
                output_names=['predicted_output'],
                dynamic_axes={'input_grid': {0: 'batch_size'}},
                opset_version=13
            )
        
        print(f"✅ ONNX export complete")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Plot final results
    print("\n📈 Creating visualizations...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    for model_name, h in training_history.items():
        ax1.plot(h['train_loss'], label=f'{model_name} (train)')
        ax1.plot(h['val_loss'], '--', label=f'{model_name} (val)')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    for model_name, h in training_history.items():
        ax2.plot(h['val_accuracy'], label=model_name)
    
    # Add 85% target line
    ax2.axhline(y=85, color='red', linestyle='--', label='85% Target')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy (Grid Reconstruction)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/content/results/enhanced_training_results.png', dpi=300)
    plt.show()
    
    # Print final summary
    print("\n" + "="*60)
    print("🏁 ENHANCED TRAINING COMPLETE!")
    print("="*60)
    
    for model_name, history in training_history.items():
        final_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
        best_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
        print(f"{model_name.upper()}: Final Acc: {final_acc:.2f}%, Best Acc: {best_acc:.2f}%")
    
    # Save training summary
    summary_data = {
        'training_history': training_history,
        'timestamp': datetime.now().isoformat(),
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'total_samples': len(dataset)
    }
    
    with open('/content/results/enhanced_training_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Push to GitHub
    print("\n📤 Pushing results to GitHub...")
    os.system("cd /content/Arc2025 && git config user.name 'AutomataControls'")
    os.system("cd /content/Arc2025 && git config user.email 'noreply@automatanexus.com'")
    os.system("cp /content/arc_models/*.pt /content/Arc2025/models/")
    os.system("cp /content/arc_models/*.onnx /content/Arc2025/models/")
    os.system("cp /content/results/* /content/Arc2025/results/")
    os.system("cd /content/Arc2025 && git add -A")
    os.system(f"cd /content/Arc2025 && git commit -m 'Enhanced model training results - {datetime.now()}'")
    os.system("cd /content/Arc2025 && git push")
    
    print("✅ Results pushed to GitHub!")
    
    monitor.complete()
    print("\n🎉 All done! Check the results folder for training outputs.")


# Run training
if __name__ == "__main__":
    print("="*80)
    print("ARC PRIZE 2025 - ENHANCED MODEL TRAINING")
    print("="*80)
    print("Training models with reconstruction loss to predict actual output grids")
    print("Target: 85% accuracy for $700,000 grand prize")
    print("="*80)
    print()
    
    train_enhanced_models()