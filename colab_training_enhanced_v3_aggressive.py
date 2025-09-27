# ARC Prize 2025 - AGGRESSIVE V3 Training to Force Transformations
# Extreme measures to prevent input copying

# Install packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "matplotlib", "numpy", "pandas", "tqdm", "onnx", "onnxruntime", "plotly", "scikit-learn", "albumentations", "-q"])
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
import matplotlib.patches as mpatches
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
import random
from torchvision.transforms import v2 as transforms_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# Setup paths
sys.path.append('/mnt/d/opt/ARCPrize2025')
from models.arc_models_enhanced import create_enhanced_models
from colab_monitor_integration import setup_colab_monitor

# Enable mixed precision training
from torch.amp import GradScaler, autocast

# AGGRESSIVE HYPERPARAMETERS
BATCH_SIZE = 16  # Smaller for more frequent updates
LEARNING_RATE = 0.01  # 10x higher!
NUM_EPOCHS = 200
MAX_GRID_SIZE = 30
NUM_COLORS = 10
DEVICE = device

# AGGRESSIVE LOSS WEIGHTS - FORCE TRANSFORMATION!
RECONSTRUCTION_WEIGHT = 1.0
TRANSFORMATION_PENALTY = 5.0  # EXTREME penalty for copying
DIVERSITY_REWARD = 2.0  # Reward for generating different outputs
EXACT_MATCH_BONUS = 10.0  # Huge bonus for exact matches

DATA_DIR = '/mnt/d/opt/ARCPrize2025/data'

class AggressiveTransformationLoss(nn.Module):
    """Loss that aggressively penalizes copying and rewards transformation"""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = pred.shape
        
        # Get actual predictions
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        input_indices = input_grid.argmax(dim=1)
        
        # 1. Standard reconstruction loss (with focal loss for hard pixels)
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target_indices.reshape(-1)
        ce_loss = self.ce_loss(pred_flat, target_flat)
        
        # Focal loss to focus on hard pixels
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        reconstruction_loss = focal_loss.mean()
        
        # 2. EXTREME transformation penalty
        # Count pixels that are same as input
        same_as_input = (pred_indices == input_indices).float()
        copy_ratio = same_as_input.mean(dim=[1,2])  # per sample
        
        # Exponential penalty for copying - gets extreme fast!
        transformation_penalty = torch.exp(copy_ratio * 3) - 1  # exp(3) = 20 for full copy
        
        # 3. Diversity reward - reward outputs that are different from input
        different_from_input = 1.0 - same_as_input
        diversity_reward = -different_from_input.mean(dim=[1,2])  # negative because we minimize
        
        # 4. Exact match super bonus
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_bonus = -exact_matches * EXACT_MATCH_BONUS
        
        # 5. Active region focus - must transform active regions
        active_mask = (input_indices != 0) | (target_indices != 0)
        if active_mask.any():
            active_same = (same_as_input * active_mask.float()).sum(dim=[1,2])
            active_total = active_mask.sum(dim=[1,2]).float()
            active_copy_ratio = active_same / (active_total + 1e-6)
            active_penalty = active_copy_ratio * 2.0  # Double penalty in active regions
        else:
            active_penalty = torch.zeros(B).to(pred.device)
        
        # Combine all terms
        total_loss = (
            RECONSTRUCTION_WEIGHT * reconstruction_loss +
            TRANSFORMATION_PENALTY * transformation_penalty.mean() +
            DIVERSITY_REWARD * diversity_reward.mean() +
            active_penalty.mean() +
            exact_bonus.mean()
        )
        
        return {
            'reconstruction': reconstruction_loss,
            'transformation': transformation_penalty.mean(),
            'diversity': -diversity_reward.mean(),
            'active_penalty': active_penalty.mean(),
            'exact_matches': exact_matches.sum(),
            'copy_ratio': copy_ratio.mean(),
            'total': total_loss
        }


class ARCDataset(Dataset):
    """Simple dataset for aggressive training"""
    
    def __init__(self, data_dir: str):
        self.samples = []
        self._load_data(data_dir)
        
    def _load_data(self, data_dir: str):
        print(f"Loading data...")
        
        with open(f'{data_dir}/arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open(f'{data_dir}/arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        for task_id, task_data in challenges.items():
            for i, example in enumerate(task_data['train']):
                sample = {
                    'input': np.array(example['input']),
                    'output': np.array(solutions[task_id][i])
                }
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Add random augmentation sometimes
        if np.random.rand() < 0.5:
            aug_type = np.random.choice(['rotate', 'flip'])
            input_grid = sample['input'].copy()
            output_grid = sample['output'].copy()
            
            if aug_type == 'rotate':
                k = np.random.randint(1, 4)
                input_grid = np.rot90(input_grid, k)
                output_grid = np.rot90(output_grid, k)
            else:
                axis = np.random.randint(0, 2)
                input_grid = np.flip(input_grid, axis=axis)
                output_grid = np.flip(output_grid, axis=axis)
        else:
            input_grid = sample['input']
            output_grid = sample['output']
        
        input_tensor = self._to_one_hot(input_grid)
        output_tensor = self._to_one_hot(output_grid)
        
        return {
            'input': torch.from_numpy(input_tensor).float(),
            'output': torch.from_numpy(output_tensor).float()
        }
    
    def _to_one_hot(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        one_hot = np.zeros((NUM_COLORS, MAX_GRID_SIZE, MAX_GRID_SIZE))
        
        # Pad if needed
        if h > MAX_GRID_SIZE:
            grid = grid[:MAX_GRID_SIZE, :MAX_GRID_SIZE]
            h = MAX_GRID_SIZE
        if w > MAX_GRID_SIZE:
            grid = grid[:, :MAX_GRID_SIZE]
            w = MAX_GRID_SIZE
        
        for i in range(h):
            for j in range(w):
                color = int(grid[i, j])
                if 0 <= color < NUM_COLORS:
                    one_hot[color, i, j] = 1
        
        return one_hot


def train_aggressive():
    """Aggressive training to force transformations"""
    print("\n🔥 AGGRESSIVE V3 TRAINING - FORCE TRANSFORMATIONS!")
    print("="*60)
    
    monitor = setup_colab_monitor()
    
    # Create dataset
    dataset = ARCDataset(DATA_DIR)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=2, pin_memory=True)
    
    print(f"\n📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    os.makedirs('/content/arc_models_aggressive', exist_ok=True)
    
    # Create models
    models = create_enhanced_models()
    loss_fn = AggressiveTransformationLoss()
    
    # Train each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔥 Training {model_name.upper()} - AGGRESSIVE MODE")
        print(f"{'='*60}")
        
        model = model.to(DEVICE)
        
        # Use different optimizer with higher LR
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                              weight_decay=0.01, betas=(0.9, 0.999))
        
        # Aggressive scheduler - reduce LR when stuck
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
        )
        
        scaler = GradScaler('cuda')
        
        best_val_loss = float('inf')
        best_exact = 0
        
        for epoch in range(NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0
            train_metrics = {'copy_ratio': 0, 'exact': 0}
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            
            for batch in pbar:
                input_grids = batch['input'].to(DEVICE)
                output_grids = batch['output'].to(DEVICE)
                
                with autocast('cuda'):
                    outputs = model(input_grids, output_grids, mode='train')
                    pred_output = outputs['predicted_output']
                    losses = loss_fn(pred_output, output_grids, input_grids)
                
                optimizer.zero_grad()
                scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                train_loss += losses['total'].item()
                train_metrics['copy_ratio'] += losses['copy_ratio'].item()
                train_metrics['exact'] += losses['exact_matches'].item()
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.3f}",
                    'copy': f"{losses['copy_ratio'].item():.2%}",
                    'exact': f"{losses['exact_matches'].item():.0f}"
                })
            
            avg_train_loss = train_loss / len(train_loader)
            avg_copy_ratio = train_metrics['copy_ratio'] / len(train_loader)
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_loss = 0
                val_exact = 0
                val_total = 0
                val_copy_ratio = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        input_grids = batch['input'].to(DEVICE)
                        output_grids = batch['output'].to(DEVICE)
                        
                        with autocast('cuda'):
                            outputs = model(input_grids)
                            pred_output = outputs['predicted_output']
                            losses = loss_fn(pred_output, output_grids, input_grids)
                        
                        val_loss += losses['total'].item()
                        val_exact += losses['exact_matches'].item()
                        val_total += input_grids.size(0)
                        val_copy_ratio += losses['copy_ratio'].item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_exact_pct = val_exact / val_total * 100
                avg_val_copy = val_copy_ratio / len(val_loader)
                
                print(f"\nEpoch {epoch+1}:")
                print(f"  Train Loss: {avg_train_loss:.4f}, Copy: {avg_copy_ratio:.2%}")
                print(f"  Val Loss: {avg_val_loss:.4f}, Copy: {avg_val_copy:.2%}, Exact: {val_exact_pct:.2f}%")
                
                # Step scheduler
                scheduler.step(avg_val_loss)
                
                # Save if best
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_exact': val_exact_pct,
                        'val_copy': avg_val_copy
                    }, f'/content/arc_models_aggressive/{model_name}_best.pt')
                    
                    print(f"✅ New best! Exact: {val_exact_pct:.2f}%, Copy: {avg_val_copy:.2%}")
                
                # Early stopping if we hit target
                if val_exact_pct >= 85.0:
                    print(f"🎉 TARGET ACHIEVED! {val_exact_pct:.2f}% exact match!")
                    break
                
                # If still copying too much, make it more aggressive
                if avg_val_copy > 0.5 and epoch > 10:
                    print("⚠️ Still copying too much! Increasing penalties...")
                    global TRANSFORMATION_PENALTY, DIVERSITY_REWARD
                    TRANSFORMATION_PENALTY *= 1.5
                    DIVERSITY_REWARD *= 1.5
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    monitor.complete()
    print("\n🎉 Aggressive training complete!")


if __name__ == "__main__":
    train_aggressive()