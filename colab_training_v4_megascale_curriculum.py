# ARC Prize 2025 - V4 MEGA-SCALE with Curriculum Learning
# Combines massive scale with proven curriculum strategy

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "matplotlib", "numpy", "pandas", "tqdm", "-q"])
print("✓ Packages installed")

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
import time
import gc
from torch.amp import GradScaler, autocast

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'\n🚀 A100 80GB DETECTED! MEGA-SCALE + CURRICULUM MODE!')

# Clone repository
print("\n📥 Setting up ARC Prize 2025...")
if os.path.exists('Arc2025'):
    shutil.rmtree('Arc2025')
os.system("git clone https://github.com/AutomataControls/Arc2025.git")
print("✓ Repository ready")

sys.path.append('/content/Arc2025')
sys.path.append('/content')

try:
    from Arc2025.models.arc_models_enhanced import create_enhanced_models
except:
    sys.path.append('/content/Arc2025/models')
    from arc_models_enhanced import create_enhanced_models

# MEGA-SCALE HYPERPARAMETERS FOR A100 80GB
BATCH_SIZE = 512  # 16x larger!
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size: 2048!
LEARNING_RATE = 0.01  # Scaled with batch size
NUM_EPOCHS = 300
MAX_GRID_SIZE = 30
NUM_COLORS = 10
NUM_WORKERS = 8  # Parallel data loading
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# Enhanced loss weights - FIXED!
RECONSTRUCTION_WEIGHT = 1.0
EDGE_WEIGHT = 0.3
COLOR_BALANCE_WEIGHT = 0.2
STRUCTURE_WEIGHT = 0.3
TRANSFORMATION_PENALTY = 0.5  # POSITIVE to penalize copying!
EXACT_MATCH_BONUS = 5.0  # Big reward for exact matches!

# Curriculum settings
CURRICULUM_STAGES = 3
EPOCHS_PER_STAGE = 100

print(f"\n⚙️ V4 MEGA-SCALE + CURRICULUM Configuration:")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Curriculum stages: {CURRICULUM_STAGES}")
print(f"  Transformation penalty: {TRANSFORMATION_PENALTY} (positive!)")

# Data setup
DATA_DIR = '/content/Arc2025/data' if os.path.exists('/content/Arc2025/data') else '/content/data'
if not os.path.exists(DATA_DIR):
    print("Downloading ARC data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f"wget -q https://github.com/fchollet/ARC-AGI/raw/master/data/training/arc-agi_training_challenges.json -O {DATA_DIR}/arc-agi_training_challenges.json")
    os.system(f"wget -q https://github.com/fchollet/ARC-AGI/raw/master/data/training/arc-agi_training_solutions.json -O {DATA_DIR}/arc-agi_training_solutions.json")

class MegaScaleLoss(nn.Module):
    """Enhanced loss with exact match bonus and proper transformation penalty"""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = pred.shape
        
        # Get predictions and targets
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        input_indices = input_grid.argmax(dim=1)
        
        # 1. Standard reconstruction loss
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target_indices.reshape(-1)
        ce_loss = self.ce_loss(pred_flat, target_flat).reshape(B, H, W)
        
        # 2. Exact match bonus - HUGE reward for getting it perfect
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()  # B
        exact_bonus = -exact_matches * EXACT_MATCH_BONUS  # Negative because we minimize
        
        # 3. Edge-aware loss
        target_edges = self._detect_edges(target_indices)
        edge_weight = 1.0 + target_edges * (EDGE_WEIGHT * 10)
        weighted_loss = ce_loss * edge_weight
        
        reconstruction_loss = weighted_loss.mean(dim=[1,2])  # B
        
        # 4. Transformation PENALTY - only apply if not an identity task
        # Check if this is an identity task (where copying IS correct)
        is_identity_task = (input_indices == target_indices).all(dim=[1,2]).float()  # B
        
        # Calculate similarity to input
        same_as_input = (pred_indices == input_indices).float().mean(dim=[1,2])  # B
        
        # Apply penalty ONLY for non-identity tasks
        transformation_penalty = same_as_input * (1 - is_identity_task)
        
        # 5. Active region focus
        active_mask = (target_indices != 0)
        if active_mask.any():
            active_loss = ce_loss * active_mask.float()
            active_loss = active_loss.sum(dim=[1,2]) / (active_mask.sum(dim=[1,2]).float() + 1e-6)
        else:
            active_loss = torch.zeros(B).to(pred.device)
        
        # Combine with exact match bonus
        total_loss = (
            RECONSTRUCTION_WEIGHT * reconstruction_loss +
            TRANSFORMATION_PENALTY * transformation_penalty +  # Now properly penalizes copying
            0.5 * active_loss +
            exact_bonus  # This can make loss negative for exact matches!
        )
        
        return {
            'reconstruction': reconstruction_loss.mean(),
            'transformation': transformation_penalty.mean(),
            'active': active_loss.mean(),
            'exact_bonus': -exact_bonus.mean(),  # Show as positive in logs
            'exact_count': exact_matches.sum(),
            'total': total_loss.mean()
        }
    
    def _detect_edges(self, grid: torch.Tensor) -> torch.Tensor:
        """Detect edges in grid"""
        dx = torch.abs(grid[:, 1:, :] - grid[:, :-1, :])
        dy = torch.abs(grid[:, :, 1:] - grid[:, :, :-1])
        
        dx = F.pad(dx, (0, 0, 0, 1), value=0)
        dy = F.pad(dy, (0, 1, 0, 0), value=0)
        
        edges = ((dx + dy) > 0).float()
        return edges


class CurriculumMegaScaleDataset(Dataset):
    """High-performance dataset with curriculum learning"""
    
    def __init__(self, data_dir: str, curriculum_stage: int = 0, augment_factor: int = 10):
        self.samples = []
        self.curriculum_stage = curriculum_stage
        self.augment_factor = augment_factor
        self._load_data(data_dir)
        
    def _load_data(self, data_dir: str):
        print(f"Loading curriculum stage {self.curriculum_stage} with {self.augment_factor}x augmentation...")
        
        with open(f'{data_dir}/arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open(f'{data_dir}/arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        for task_id, task_data in challenges.items():
            for i, example in enumerate(task_data['train']):
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                # Assess difficulty
                h, w = input_grid.shape
                oh, ow = output_grid.shape
                n_colors_in = len(np.unique(input_grid))
                n_colors_out = len(np.unique(output_grid))
                grid_size = max(h, w, oh, ow)
                size_diff = abs(h*w - oh*ow)
                
                # Curriculum filtering
                if self.curriculum_stage == 0:  # Easy
                    if grid_size > 15 or n_colors_in > 3 or n_colors_out > 3 or size_diff > 0:
                        continue
                elif self.curriculum_stage == 1:  # Medium
                    if grid_size > 20 or n_colors_in > 5 or n_colors_out > 5:
                        continue
                # Stage 2 accepts all
                
                sample = {'input': input_grid, 'output': output_grid}
                
                # Add original
                self.samples.append(sample)
                
                # Add augmentations
                for _ in range(self.augment_factor - 1):
                    aug_type = np.random.choice(['rotate', 'flip', 'both'])
                    aug_input = input_grid.copy()
                    aug_output = output_grid.copy()
                    
                    if aug_type in ['rotate', 'both']:
                        k = np.random.randint(1, 4)
                        aug_input = np.rot90(aug_input, k)
                        aug_output = np.rot90(aug_output, k)
                    
                    if aug_type in ['flip', 'both']:
                        axis = np.random.randint(0, 2)
                        aug_input = np.flip(aug_input, axis=axis)
                        aug_output = np.flip(aug_output, axis=axis)
                    
                    self.samples.append({
                        'input': aug_input,
                        'output': aug_output
                    })
        
        print(f"Loaded {len(self.samples)} samples for stage {self.curriculum_stage}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_grid = self._to_tensor(sample['input'])
        output_grid = self._to_tensor(sample['output'])
        
        return {
            'input': input_grid,
            'output': output_grid
        }
    
    def _to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        h, w = grid.shape
        # Fast one-hot encoding
        one_hot = torch.zeros(NUM_COLORS, MAX_GRID_SIZE, MAX_GRID_SIZE)
        
        # Pad if needed
        if h > MAX_GRID_SIZE:
            grid = grid[:MAX_GRID_SIZE, :MAX_GRID_SIZE]
            h = MAX_GRID_SIZE
        if w > MAX_GRID_SIZE:
            grid = grid[:, :MAX_GRID_SIZE]
            w = MAX_GRID_SIZE
            
        # Vectorized one-hot
        for color in range(NUM_COLORS):
            mask = (grid == color)
            one_hot[color, :h, :w] = torch.from_numpy(mask.astype(np.float32))
        
        return one_hot


def train_megascale_curriculum():
    """V4 Mega-scale training with curriculum learning"""
    print("\n🚀 Starting V4 MEGA-SCALE + CURRICULUM Training")
    print("="*60)
    
    # Create models and loss
    models = create_enhanced_models()
    loss_fn = MegaScaleLoss()
    
    os.makedirs('/content/arc_models_v4', exist_ok=True)
    
    # Train only CHRONOS and PROMETHEUS (skip already trained models)
    models_to_train = ['chronos', 'prometheus']
    
    for model_name, model in models.items():
        if model_name not in models_to_train:
            print(f"\n⏭️  Skipping {model_name.upper()} - already trained!")
            continue
            
        print(f"\n{'='*60}")
        print(f"🧠 Training {model_name.upper()} - MEGA-SCALE + CURRICULUM")
        print(f"{'='*60}")
        
        model = model.to(device)
        
        # High-momentum optimizer for large batches
        optimizer = optim.SGD(
            model.parameters(), 
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True
        )
        
        # Cosine annealing across all stages
        total_epochs = EPOCHS_PER_STAGE * CURRICULUM_STAGES
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        
        scaler = GradScaler('cuda')
        
        best_exact = 0
        best_val_loss = float('inf')
        global_epoch = 0
        
        # CURRICULUM LOOP
        for stage in range(CURRICULUM_STAGES):
            print(f"\n📚 Starting Curriculum Stage {stage}")
            print("="*40)
            
            # Create dataset for this stage
            dataset = CurriculumMegaScaleDataset(DATA_DIR, curriculum_stage=stage)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # High-performance dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR,
                persistent_workers=True
            )
            
            print(f"Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
            print(f"Batches per epoch: {len(train_loader):,}")
            
            # Train for this stage
            for epoch in range(EPOCHS_PER_STAGE):
                global_epoch += 1
                
                # Training
                model.train()
                train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
                
                pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}/{EPOCHS_PER_STAGE}")
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(pbar):
                    input_grids = batch['input'].to(device, non_blocking=True)
                    output_grids = batch['output'].to(device, non_blocking=True)
                    
                    with autocast('cuda'):
                        # Handle CHRONOS differently - it expects a list of tensors
                        if model_name == 'chronos':
                            outputs = model([input_grids])  # Pass as single-frame sequence
                            pred_output = outputs['predicted_output']
                        else:
                            outputs = model(input_grids, output_grids, mode='train')
                            pred_output = outputs['predicted_output']
                        losses = loss_fn(pred_output, output_grids, input_grids)
                        loss = losses['total'] / GRADIENT_ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    # Update metrics
                    train_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                    train_metrics['exact'] += losses['exact_count'].item()
                    train_metrics['samples'] += input_grids.size(0)
                    
                    pbar.set_postfix({
                        'loss': f"{losses['total'].item():.3f}",
                        'exact': f"{losses['exact_count'].item():.0f}",
                        'trans': f"{losses['transformation'].item():.2f}"
                    })
                
                scheduler.step()
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    model.eval()
                    val_metrics = {'loss': 0, 'exact': 0, 'pixel_acc': 0, 'samples': 0}
                    
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc="Validation"):
                            input_grids = batch['input'].to(device, non_blocking=True)
                            output_grids = batch['output'].to(device, non_blocking=True)
                            
                            with autocast('cuda'):
                                # Handle CHRONOS differently - it expects a list of tensors
                                if model_name == 'chronos':
                                    outputs = model([input_grids])  # Pass as single-frame sequence
                                else:
                                    outputs = model(input_grids)
                                pred_output = outputs['predicted_output']
                                losses = loss_fn(pred_output, output_grids, input_grids)
                            
                            # Metrics
                            pred_indices = pred_output.argmax(dim=1)
                            target_indices = output_grids.argmax(dim=1)
                            
                            exact = (pred_indices == target_indices).all(dim=[1,2]).sum().item()
                            pixel_acc = (pred_indices == target_indices).float().mean().item()
                            
                            val_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                            val_metrics['exact'] += exact
                            val_metrics['pixel_acc'] += pixel_acc * input_grids.size(0)
                            val_metrics['samples'] += input_grids.size(0)
                    
                    # Calculate averages
                    train_loss = train_metrics['loss'] / train_metrics['samples']
                    train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                    
                    val_loss = val_metrics['loss'] / val_metrics['samples']
                    val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                    val_pixel_acc = val_metrics['pixel_acc'] / val_metrics['samples'] * 100
                    
                    print(f"\nGlobal Epoch {global_epoch} (Stage {stage}): "
                          f"Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%")
                    print(f"Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                    
                    # Save best model
                    if val_exact_pct > best_exact:
                        best_exact = val_exact_pct
                        torch.save({
                            'epoch': global_epoch,
                            'stage': stage,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_exact': val_exact_pct,
                            'val_pixel_acc': val_pixel_acc
                        }, f'/content/arc_models_v4/{model_name}_best.pt')
                        
                        print(f"✅ New best model! Exact: {val_exact_pct:.2f}%")
                        
                        # Early stopping if we hit target
                        if val_exact_pct >= 85.0:
                            print(f"🎉 TARGET ACHIEVED! {val_exact_pct:.2f}% exact match!")
                            break
                
                # Check if we should exit early
                if best_exact >= 85.0:
                    break
            
            # Check if we should exit curriculum early
            if best_exact >= 85.0:
                break
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n🎉 V4 MEGA-SCALE + CURRICULUM Training complete!")


if __name__ == "__main__":
    print("="*80)
    print("ARC PRIZE 2025 - V4 MEGA-SCALE + CURRICULUM TRAINING")
    print("="*80)
    print("Combining massive batch sizes with proven curriculum strategy")
    print("Fixed transformation penalty + identity task handling")
    print("Target: 85% exact match accuracy = $700,000")
    print("="*80)
    
    train_megascale_curriculum()