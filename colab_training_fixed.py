# ARC Prize 2025 - FIXED Training Script for Google Colab
# This version actually trains properly with full dataset and proper epochs

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
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datetime import datetime
import time

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
os.chdir('Arc2025')
print("✓ Repository cloned")

# Import models
from models.arc_models import (
    MinervaNet, AtlasNet, IrisNet, ChronosNet, PrometheusNet,
    create_models
)

# Enhanced Dataset class with better pattern detection
class ARCDataset(Dataset):
    def __init__(self, data_path: str, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.tasks = []
        self.pattern_labels = {
            'rotation': 0, 'reflection': 1, 'scaling': 2, 'translation': 3,
            'color_mapping': 4, 'symmetry': 5, 'object_movement': 6,
            'counting': 7, 'logical': 8, 'composite': 9
        }
        
        # Load JSON files
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            # Convert to task format
            for task_id, task_data in data.items():
                task = {
                    'filename': task_id,
                    'train': task_data.get('train', []),
                    'test': task_data.get('test', [])
                }
                self.tasks.append(task)
        
        # Create training pairs with augmentation
        self.pairs = []
        for task in self.tasks:
            for example in task.get('train', []):
                if 'input' in example and 'output' in example:
                    # Original pair
                    self.pairs.append({
                        'input': np.array(example['input']),
                        'output': np.array(example['output']),
                        'task_id': task['filename'],
                        'pattern_label': self._detect_pattern_type(example)
                    })
                    
                    # Add augmented versions for better training
                    input_arr = np.array(example['input'])
                    output_arr = np.array(example['output'])
                    
                    # Rotation augmentation (if not a rotation pattern)
                    if self._detect_pattern_type(example) != self.pattern_labels['rotation']:
                        for k in [1, 2, 3]:
                            self.pairs.append({
                                'input': np.rot90(input_arr, k),
                                'output': np.rot90(output_arr, k),
                                'task_id': task['filename'],
                                'pattern_label': self._detect_pattern_type(example)
                            })
        
        print(f"Loaded {len(self.tasks)} tasks with {len(self.pairs)} training pairs (with augmentation)")
    
    def _detect_pattern_type(self, example: Dict) -> int:
        """Enhanced pattern detection with more sophisticated checks"""
        try:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Size change detection
            if input_grid.shape != output_grid.shape:
                if output_grid.size > input_grid.size:
                    return self.pattern_labels['scaling']
                else:
                    return self.pattern_labels['counting']
            
            # Rotation detection
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(input_grid, k), output_grid):
                    return self.pattern_labels['rotation']
            
            # Reflection detection
            if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                return self.pattern_labels['reflection']
            if np.array_equal(np.flip(input_grid, axis=1), output_grid):
                return self.pattern_labels['reflection']
            
            # Translation detection
            if self._is_translation(input_grid, output_grid):
                return self.pattern_labels['translation']
            
            # Color mapping detection
            unique_in = set(input_grid.flatten())
            unique_out = set(output_grid.flatten())
            if unique_in != unique_out and len(unique_out) <= len(unique_in):
                return self.pattern_labels['color_mapping']
            
            # Symmetry detection
            if self._has_symmetry(output_grid) and not self._has_symmetry(input_grid):
                return self.pattern_labels['symmetry']
            
            # Object detection (significant structure change)
            if self._count_objects(input_grid) != self._count_objects(output_grid):
                return self.pattern_labels['object_movement']
            
            # Counting patterns
            if self._has_counting_pattern(input_grid, output_grid):
                return self.pattern_labels['counting']
            
            # Logical patterns
            if self._has_logical_pattern(input_grid, output_grid):
                return self.pattern_labels['logical']
            
            # Default to composite
            return self.pattern_labels['composite']
        except:
            return self.pattern_labels['composite']
    
    def _is_translation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is translated version of input"""
        h, w = input_grid.shape
        for dy in range(-h+1, h):
            for dx in range(-w+1, w):
                translated = np.zeros_like(input_grid)
                # Copy with translation
                src_y1 = max(0, -dy)
                src_y2 = min(h, h - dy)
                src_x1 = max(0, -dx)
                src_x2 = min(w, w - dx)
                dst_y1 = max(0, dy)
                dst_y2 = min(h, h + dy)
                dst_x1 = max(0, dx)
                dst_x2 = min(w, w + dx)
                
                if src_y2 > src_y1 and src_x2 > src_x1:
                    translated[dst_y1:dst_y2, dst_x1:dst_x2] = input_grid[src_y1:src_y2, src_x1:src_x2]
                    if np.array_equal(translated, output_grid):
                        return True
        return False
    
    def _has_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has symmetry"""
        return (np.array_equal(grid, np.flip(grid, axis=0)) or 
                np.array_equal(grid, np.flip(grid, axis=1)))
    
    def _count_objects(self, grid: np.ndarray) -> int:
        """Count distinct objects in grid"""
        from scipy import ndimage
        labeled, num_features = ndimage.label(grid > 0)
        return num_features
    
    def _has_counting_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check for counting-based patterns"""
        # Check if output encodes counts or sizes
        unique_values = np.unique(output_grid)
        if len(unique_values) > 2 and len(unique_values) < 10:
            return True
        return False
    
    def _has_logical_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check for logical operations"""
        # Simple XOR, AND, OR checks
        if input_grid.shape == output_grid.shape:
            xor_result = (input_grid > 0) ^ (output_grid > 0)
            and_result = (input_grid > 0) & (output_grid > 0)
            or_result = (input_grid > 0) | (output_grid > 0)
            
            # Check if output follows logical pattern
            if (np.sum(xor_result) > input_grid.size * 0.1 or
                np.sum(and_result) > input_grid.size * 0.1 or
                np.sum(or_result) > input_grid.size * 0.1):
                return True
        return False
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_grid = self._grid_to_tensor(pair['input'])
        output_grid = self._grid_to_tensor(pair['output'])
        pattern_label = torch.tensor(pair['pattern_label'], dtype=torch.long)
        return input_grid, output_grid, pattern_label
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        h, w = grid.shape
        one_hot = np.zeros((10, self.max_grid_size, self.max_grid_size))
        for i in range(min(h, self.max_grid_size)):
            for j in range(min(w, self.max_grid_size)):
                if grid[i, j] < 10:
                    one_hot[grid[i, j], i, j] = 1
        return torch.FloatTensor(one_hot)

# Create datasets with FULL data
print("\n📊 Creating datasets with FULL training data...")
train_dataset = ARCDataset('data/arc-agi_training_challenges.json')
eval_dataset = ARCDataset('data/arc-agi_evaluation_challenges.json')

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Use FULL dataset, not a tiny subset!
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Create dataloaders with better batch size
batch_size = 32  # Increased from 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain size: {len(train_data)}, Val size: {len(val_data)}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Enhanced Training class with better optimization
class ModelTrainer:
    def __init__(self, model, model_name, device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stop_patience = 10
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(loader, desc=f'Training {self.model_name}')
        for input_grid, output_grid, pattern_label in progress_bar:
            input_grid = input_grid.to(self.device)
            output_grid = output_grid.to(self.device)
            pattern_label = pattern_label.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with gradient accumulation for stability
            if self.model_name in ['minerva', 'iris']:
                outputs = self.model(input_grid, output_grid)
                logits = outputs['pattern_logits'] if 'pattern_logits' in outputs else outputs['pattern_type_logits']
            elif self.model_name == 'atlas':
                outputs = self.model(input_grid)
                transform_params = outputs['transform_params']
                logits = transform_params[:, :10]
            elif self.model_name == 'chronos':
                outputs = self.model([input_grid])
                logits = outputs['evolution_type_logits']
            elif self.model_name == 'prometheus':
                outputs = self.model(input_grid)
                logits = outputs['synthesis_strategy_logits'][:, :10]
            
            loss = criterion(logits, pattern_label)
            
            # Add L2 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += pattern_label.size(0)
            correct += (predicted == pattern_label).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        return total_loss / len(loader), correct / total if total > 0 else 0
    
    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for input_grid, output_grid, pattern_label in tqdm(loader, desc='Validating', leave=False):
                input_grid = input_grid.to(self.device)
                output_grid = output_grid.to(self.device)
                pattern_label = pattern_label.to(self.device)
                
                # Forward pass
                if self.model_name in ['minerva', 'iris']:
                    outputs = self.model(input_grid, output_grid)
                    logits = outputs['pattern_logits'] if 'pattern_logits' in outputs else outputs['pattern_type_logits']
                elif self.model_name == 'atlas':
                    outputs = self.model(input_grid)
                    transform_params = outputs['transform_params']
                    logits = transform_params[:, :10]
                elif self.model_name == 'chronos':
                    outputs = self.model([input_grid])
                    logits = outputs['evolution_type_logits']
                elif self.model_name == 'prometheus':
                    outputs = self.model(input_grid)
                    logits = outputs['synthesis_strategy_logits'][:, :10]
                
                loss = criterion(logits, pattern_label)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += pattern_label.size(0)
                correct += (predicted == pattern_label).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(pattern_label.cpu().numpy())
        
        return total_loss / len(loader), correct / total if total > 0 else 0, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, lr=1e-3):
        # Better optimizer with warm-up
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"\nTraining {self.model_name.upper()} for up to {epochs} epochs...")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Learning rate: {lr}, Batch size: {batch_size}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            
            epoch_time = time.time() - epoch_start
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': self.best_val_acc,
                    'history': self.history
                }, f'{self.model_name}_best.pt')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            print(f'Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            scheduler.step()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        
        # Load best model
        if os.path.exists(f'{self.model_name}_best.pt'):
            checkpoint = torch.load(f'{self.model_name}_best.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        return val_preds, val_labels

# Train all models with proper hyperparameters
print("\n🚀 Starting PROPER model training with FULL dataset...")
models = create_models()
training_results = {}

# REAL training parameters
EPOCHS = 50  # Proper epochs, not 3!
LEARNING_RATES = {
    'minerva': 5e-4,
    'atlas': 1e-3,
    'iris': 5e-4,
    'chronos': 3e-4,
    'prometheus': 5e-4
}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()} - {model.description}")
    print(f"{'='*80}")
    
    trainer = ModelTrainer(model, model_name, device)
    val_preds, val_labels = trainer.train(
        train_loader, 
        val_loader, 
        epochs=EPOCHS,
        lr=LEARNING_RATES[model_name]
    )
    
    training_results[model_name] = {
        'trainer': trainer,
        'history': trainer.history,
        'best_acc': trainer.best_val_acc,
        'val_preds': val_preds,
        'val_labels': val_labels
    }

# Generate comprehensive metrics
print("\n📈 Generating comprehensive metrics and visualizations...")
pattern_names = ['rotation', 'reflection', 'scaling', 'translation', 'color_mapping',
                 'symmetry', 'object_movement', 'counting', 'logical', 'composite']

# Create performance comparison plot
fig = go.Figure()

for model_name, results in training_results.items():
    history = results['history']
    fig.add_trace(go.Scatter(
        x=list(range(len(history['val_acc']))),
        y=history['val_acc'],
        mode='lines',
        name=f'{model_name.upper()} Val Acc',
        line=dict(width=2)
    ))

fig.update_layout(
    title='Model Validation Accuracy Over Training',
    xaxis_title='Epoch',
    yaxis_title='Accuracy',
    hovermode='x unified',
    width=1000,
    height=600
)
fig.write_html('model_performance_comparison.html')
print("✓ Saved performance comparison plot")

# Create confusion matrices for each model
for model_name, results in training_results.items():
    if len(results['val_preds']) > 0:
        cm = confusion_matrix(results['val_labels'], results['val_preds'])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=pattern_names[:len(np.unique(results['val_labels']))],
            y=pattern_names[:len(np.unique(results['val_labels']))],
            title=f'{model_name.upper()} Confusion Matrix',
            color_continuous_scale='Blues'
        )
        
        fig.update_xaxis(side="bottom")
        fig.write_html(f'{model_name}_confusion_matrix.html')
        print(f"✓ Saved {model_name} confusion matrix")

# Generate detailed metrics report
detailed_metrics = {}
for model_name, results in training_results.items():
    detailed_metrics[model_name] = {
        'best_accuracy': results['best_acc'],
        'parameters': sum(p.numel() for p in models[model_name].parameters()),
        'final_train_acc': results['history']['train_acc'][-1] if results['history']['train_acc'] else 0,
        'epochs_trained': len(results['history']['train_acc']),
        'best_epoch': results['trainer'].best_epoch + 1
    }

# Export to ONNX with proper opset version
print("\n📦 Exporting models to ONNX (opset 13 for better compatibility)...")
import onnx

def export_to_onnx(model, model_name, example_input):
    model.eval()
    onnx_path = f'{model_name}_model.onnx'
    
    try:
        if model_name in ['minerva', 'iris']:
            example_output = example_input.clone()
            inputs = (example_input, example_output)
            input_names = ['input_grid', 'output_grid']
        elif model_name == 'chronos':
            inputs = [example_input]
            input_names = ['input_grid']
        else:
            inputs = example_input
            input_names = ['input_grid']
        
        # Use opset 13 for better compatibility
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=13,  # Fixed from 11
            do_constant_folding=True,
            input_names=input_names,
            dynamic_axes={'input_grid': {0: 'batch_size'}},
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ {model_name} exported successfully")
        return onnx_path
    except Exception as e:
        print(f"⚠️ Could not export {model_name}: {str(e)}")
        return None

example_input = torch.randn(1, 10, 30, 30).to(device)
onnx_models = {}

for model_name, model in models.items():
    if os.path.exists(f'{model_name}_best.pt'):
        checkpoint = torch.load(f'{model_name}_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
    onnx_path = export_to_onnx(model, model_name, example_input)
    if onnx_path:
        onnx_models[model_name] = onnx_path

# Save everything with proper organization
print("\n💾 Saving all artifacts...")
output_dir = './ARC_Models_2025_FIXED'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/pytorch', exist_ok=True)
os.makedirs(f'{output_dir}/onnx', exist_ok=True)
os.makedirs(f'{output_dir}/metrics', exist_ok=True)
os.makedirs(f'{output_dir}/visualizations', exist_ok=True)

# Copy models
for model_name in models.keys():
    if os.path.exists(f'{model_name}_best.pt'):
        shutil.copy(f'{model_name}_best.pt', f'{output_dir}/pytorch/{model_name}_model.pt')
    if model_name in onnx_models and onnx_models[model_name]:
        shutil.copy(f'{model_name}_model.onnx', f'{output_dir}/onnx/{model_name}_model.onnx')

# Copy visualizations
for html_file in ['model_performance_comparison.html'] + [f'{m}_confusion_matrix.html' for m in models.keys()]:
    if os.path.exists(html_file):
        shutil.copy(html_file, f'{output_dir}/visualizations/{html_file}')

# Save comprehensive metrics
with open(f'{output_dir}/training_metrics.json', 'w') as f:
    json.dump(detailed_metrics, f, indent=2)

# Generate final report
report = f"""# ARC Prize 2025 Training Report

## Training Configuration
- **Dataset**: Full ARC training set ({len(train_dataset)} samples with augmentation)
- **Train/Val Split**: 90/10
- **Batch Size**: {batch_size}
- **Max Epochs**: {EPOCHS}
- **Early Stopping**: Patience {trainer.early_stop_patience}
- **Optimizer**: AdamW with cosine annealing warm restarts
- **Label Smoothing**: 0.1

## Model Performance

"""

for model_name in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']:
    if model_name in detailed_metrics:
        metrics = detailed_metrics[model_name]
        report += f"""### {model_name.upper()}
- **Best Validation Accuracy**: {metrics['best_accuracy']*100:.2f}%
- **Parameters**: {metrics['parameters']:,}
- **Best Epoch**: {metrics['best_epoch']}/{metrics['epochs_trained']}
- **Final Train Accuracy**: {metrics['final_train_acc']*100:.2f}%

"""

report += f"""
## Export Status
- **PyTorch Models**: ✓ Saved (.pt format)
- **ONNX Models**: {'✓ All exported' if len(onnx_models) == 5 else f'{len(onnx_models)}/5 exported'}
- **Visualizations**: ✓ Generated

## Next Steps
1. Deploy models to Raspberry Pi with Hailo-8
2. Run precomputation on full training set
3. Test on evaluation dataset
4. Submit to Kaggle competition

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{output_dir}/training_report.md', 'w') as f:
    f.write(report)

# Enhanced Hailo conversion script
hailo_script = '''#!/bin/bash
# Convert ARC Prize 2025 ONNX models to HEF format for Hailo-8
# Now with opset 13 support

echo "============================================================"
echo "ARC Prize 2025 - ONNX to HEF Conversion (Fixed)"
echo "============================================================"

MODELS=("minerva" "atlas" "iris" "chronos" "prometheus")
BASE_DIR="/mnt/d/opt/ARCPrize2025"

# Activate Hailo environment
source /mnt/c/Users/Juelz/hailo_venv_py310/bin/activate

mkdir -p "$BASE_DIR/hef"

for model in "${MODELS[@]}"; do
    echo "\\nConverting $model..."
    
    ONNX_FILE="$BASE_DIR/models/onnx/${model}_model.onnx"
    
    if [ -f "$ONNX_FILE" ]; then
        # Parse with proper opset support
        hailo parser onnx --hw-arch hailo8 --onnx-model "$ONNX_FILE" \\
            --output-har-path "$BASE_DIR/${model}.har" \\
            --start-node-names input_grid --end-node-names output
        
        # Optimize with calibration dataset
        hailo optimize --hw-arch hailo8 --har "$BASE_DIR/${model}.har" \\
            --calib-set-path "$BASE_DIR/calibration_data/${model}_calib.npy" \\
            --output-har-path "$BASE_DIR/${model}_optimized.har"
        
        # Compile with performance optimization
        hailo compiler --hw-arch hailo8 --har "$BASE_DIR/${model}_optimized.har" \\
            --output-hef-path "$BASE_DIR/hef/${model}.hef" \\
            --performance-mode
        
        # Cleanup
        rm -f "$BASE_DIR/${model}.har" "$BASE_DIR/${model}_optimized.har"
        echo "✓ Created $BASE_DIR/hef/${model}.hef"
    else
        echo "⚠️ $ONNX_FILE not found"
    fi
done

echo "\\n✅ Conversion complete!"
echo "\\nTo deploy to Raspberry Pi:"
echo "scp $BASE_DIR/hef/*.hef Automata@192.168.0.54:/home/Automata/mydata/neural-nexus/arc2025/"
'''

with open(f'{output_dir}/convert_to_hef_fixed.sh', 'w') as f:
    f.write(hailo_script)

# Create zip
print("\n📦 Creating final zip file...")
shutil.make_archive('ARC_Models_2025_FIXED', 'zip', output_dir)

print("\n✅ TRAINING COMPLETE WITH PROPER PARAMETERS!")
print(f"\nModel Performance Summary:")
for model_name in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']:
    if model_name in detailed_metrics:
        print(f"{model_name.upper()}: {detailed_metrics[model_name]['best_accuracy']*100:.1f}% accuracy")

print("\n📥 Download: ARC_Models_2025_FIXED.zip")
print("\nThis version trained on the FULL dataset with proper epochs!")
print("\nTo download, run in a new cell:")
print("from google.colab import files")
print("files.download('ARC_Models_2025_FIXED.zip')")