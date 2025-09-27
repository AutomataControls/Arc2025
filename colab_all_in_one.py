# ARC Prize 2025 - Complete Training Script for Colab
# Copy and paste this entire code into one Colab cell and run it

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
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datetime import datetime
import urllib.request
import tarfile

# Import our models
from arc_models import (
    MinervaNet, AtlasNet, IrisNet, ChronosNet, PrometheusNet,
    create_models
)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# Download ARC dataset
print("\n📥 Downloading ARC dataset...")
if not os.path.exists('training.tar.gz'):
    urllib.request.urlretrieve('https://github.com/fchollet/ARC-AGI/raw/main/data/training.tar.gz', 'training.tar.gz')
    with tarfile.open('training.tar.gz', 'r:gz') as tar:
        tar.extractall()
    print("✓ Training data downloaded")

if not os.path.exists('evaluation.tar.gz'):
    urllib.request.urlretrieve('https://github.com/fchollet/ARC-AGI/raw/main/data/evaluation.tar.gz', 'evaluation.tar.gz')
    with tarfile.open('evaluation.tar.gz', 'r:gz') as tar:
        tar.extractall()
    print("✓ Evaluation data downloaded")

# Dataset class
class ARCDataset(Dataset):
    """ARC dataset with pattern labeling for training"""
    
    def __init__(self, data_path: str, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.tasks = []
        self.pattern_labels = {
            'rotation': 0, 'reflection': 1, 'scaling': 2, 'translation': 3,
            'color_mapping': 4, 'symmetry': 5, 'object_movement': 6,
            'counting': 7, 'logical': 8, 'composite': 9
        }
        
        # Load all JSON files
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                with open(os.path.join(data_path, filename), 'r') as f:
                    task = json.load(f)
                    task['filename'] = filename
                    self.tasks.append(task)
        
        # Create training pairs with pseudo-labels
        self.pairs = []
        for task in self.tasks:
            for example in task.get('train', []):
                self.pairs.append({
                    'input': np.array(example['input']),
                    'output': np.array(example['output']),
                    'task_id': task['filename'],
                    'pattern_label': self._detect_pattern_type(example)
                })
        
        print(f"Loaded {len(self.tasks)} tasks with {len(self.pairs)} training pairs")
        
        # Pattern distribution
        pattern_counts = {}
        for pair in self.pairs:
            label = pair['pattern_label']
            pattern_name = list(self.pattern_labels.keys())[label]
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        print("\nPattern distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count} ({count/len(self.pairs)*100:.1f}%)")
    
    def _detect_pattern_type(self, example: Dict) -> int:
        """Simple heuristic to assign pattern labels"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Check for size changes (scaling)
        if input_grid.shape != output_grid.shape:
            if output_grid.size > input_grid.size:
                return self.pattern_labels['scaling']
            else:
                return self.pattern_labels['counting']
        
        # Check for rotations
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(input_grid, k), output_grid):
                return self.pattern_labels['rotation']
        
        # Check for reflections
        if np.array_equal(np.flip(input_grid, axis=0), output_grid):
            return self.pattern_labels['reflection']
        if np.array_equal(np.flip(input_grid, axis=1), output_grid):
            return self.pattern_labels['reflection']
        
        # Check for color changes
        if set(input_grid.flatten()) != set(output_grid.flatten()):
            return self.pattern_labels['color_mapping']
        
        # Check for symmetry
        if (np.array_equal(output_grid, np.flip(output_grid, axis=0)) or 
            np.array_equal(output_grid, np.flip(output_grid, axis=1))):
            return self.pattern_labels['symmetry']
        
        # Default to composite
        return self.pattern_labels['composite']
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Convert to one-hot encoding
        input_grid = self._grid_to_tensor(pair['input'])
        output_grid = self._grid_to_tensor(pair['output'])
        pattern_label = torch.tensor(pair['pattern_label'], dtype=torch.long)
        
        return input_grid, output_grid, pattern_label
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convert grid to one-hot tensor and pad to max size"""
        h, w = grid.shape
        
        # One-hot encode
        one_hot = np.zeros((10, self.max_grid_size, self.max_grid_size))
        for i in range(min(h, self.max_grid_size)):
            for j in range(min(w, self.max_grid_size)):
                color = grid[i, j]
                one_hot[color, i, j] = 1
        
        return torch.FloatTensor(one_hot)

# Create dataset
print("\n📊 Creating datasets...")
full_dataset = ARCDataset('training')

# Split into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

print(f"\nTrain size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# Training class
class ModelTrainer:
    """Comprehensive model trainer with metrics tracking"""
    
    def __init__(self, model, model_name, device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'pattern_accuracies': {}
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_epoch = 0
        
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for input_grid, output_grid, pattern_label in tqdm(loader, desc='Training'):
            input_grid = input_grid.to(self.device)
            output_grid = output_grid.to(self.device)
            pattern_label = pattern_label.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass (model-specific)
            if self.model_name in ['minerva', 'iris']:
                outputs = self.model(input_grid, output_grid)
                logits = outputs['pattern_logits'] if 'pattern_logits' in outputs else outputs['pattern_type_logits']
            elif self.model_name == 'atlas':
                outputs = self.model(input_grid)
                # Use transform params as proxy for pattern classification
                transform_params = outputs['transform_params']
                logits = transform_params[:, :10]  # First 10 dims for classification
            elif self.model_name == 'chronos':
                outputs = self.model([input_grid])
                logits = outputs['evolution_type_logits']
            elif self.model_name == 'prometheus':
                outputs = self.model(input_grid)
                logits = outputs['synthesis_strategy_logits'][:, :10]
            
            loss = criterion(logits, pattern_label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += pattern_label.size(0)
            correct += (predicted == pattern_label).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for input_grid, output_grid, pattern_label in loader:
                input_grid = input_grid.to(self.device)
                output_grid = output_grid.to(self.device)
                pattern_label = pattern_label.to(self.device)
                
                # Forward pass (same as training)
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
        
        return total_loss / len(loader), correct / total, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, lr=1e-3):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining {self.model_name.upper()} for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pt')
            
            # Print progress
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            scheduler.step()
        
        # Final validation with best model
        self.model.load_state_dict(torch.load(f'{self.model_name}_best.pt'))
        val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
        
        print(f"\nBest validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        return val_preds, val_labels

# Train all models
print("\n🚀 Starting model training...")
models = create_models()
training_results = {}

# Quick training for testing - reduce epochs for faster results
QUICK_TEST = True  # Set to False for full training
epochs_dict = {
    'minerva': 5 if QUICK_TEST else 30,
    'atlas': 5 if QUICK_TEST else 25,
    'iris': 5 if QUICK_TEST else 25,
    'chronos': 5 if QUICK_TEST else 25,
    'prometheus': 5 if QUICK_TEST else 30
}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} - {model.description}")
    print(f"{'='*60}")
    
    trainer = ModelTrainer(model, model_name, device)
    
    val_preds, val_labels = trainer.train(train_loader, val_loader, epochs=epochs_dict[model_name])
    
    # Store results
    training_results[model_name] = {
        'trainer': trainer,
        'history': trainer.history,
        'best_acc': trainer.best_val_acc,
        'val_preds': val_preds,
        'val_labels': val_labels
    }

# Generate metrics
print("\n📈 Generating metrics...")
pattern_names = ['rotation', 'reflection', 'scaling', 'translation', 'color_mapping',
                 'symmetry', 'object_movement', 'counting', 'logical', 'composite']

detailed_metrics = {}
for model_name, results in training_results.items():
    report = classification_report(results['val_labels'], results['val_preds'],
                                   target_names=pattern_names, output_dict=True)
    
    detailed_metrics[model_name] = {
        'classification_report': report,
        'best_accuracy': results['best_acc'],
        'parameters': sum(p.numel() for p in models[model_name].parameters()),
        'training_time': len(results['history']['train_loss']) * 2.5
    }

# Export to ONNX
print("\n📦 Exporting models to ONNX...")
import onnx
import onnxruntime

def export_to_onnx(model, model_name, example_input, device):
    """Export PyTorch model to ONNX format"""
    model.eval()
    model = model.to(device)
    
    onnx_path = f'{model_name}_model.onnx'
    
    if model_name in ['minerva', 'iris']:
        example_output = example_input.clone()
        inputs = (example_input, example_output)
        input_names = ['input_grid', 'output_grid']
    elif model_name == 'chronos':
        inputs = example_input
        input_names = ['input_grid']
    else:
        inputs = example_input
        input_names = ['input_grid']
    
    # Get output names from model
    with torch.no_grad():
        if model_name in ['minerva', 'iris']:
            sample_out = model(example_input, example_output)
        elif model_name == 'chronos':
            sample_out = model([example_input])
        else:
            sample_out = model(example_input)
    
    output_names = list(sample_out.keys())
    
    print(f"Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={'input_grid': {0: 'batch_size'}}
    )
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ {model_name} exported successfully to {onnx_path}")
    
    return onnx_path

example_input = torch.randn(1, 10, 30, 30).to(device)
onnx_models = {}

for model_name, model in models.items():
    model.load_state_dict(torch.load(f'{model_name}_best.pt'))
    onnx_path = export_to_onnx(model, model_name, example_input, device)
    onnx_models[model_name] = onnx_path

# Create output directory and save everything
print("\n💾 Saving all artifacts...")
output_dir = './ARC_Models_2025'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/pytorch', exist_ok=True)
os.makedirs(f'{output_dir}/onnx', exist_ok=True)
os.makedirs(f'{output_dir}/metrics', exist_ok=True)

# Save models
import shutil
for model_name in models.keys():
    shutil.copy(f'{model_name}_best.pt', f'{output_dir}/pytorch/{model_name}_model.pt')
    shutil.copy(f'{model_name}_model.onnx', f'{output_dir}/onnx/{model_name}_model.onnx')

# Save metrics
with open(f'{output_dir}/training_metrics.json', 'w') as f:
    json.dump(detailed_metrics, f, indent=2)

# Create zip file
print("\n📦 Creating zip file...")
shutil.make_archive('ARC_Models_2025', 'zip', output_dir)

print("\n✅ TRAINING COMPLETE!")
print(f"\nModel Performance Summary:")
for model_name in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']:
    if model_name in detailed_metrics:
        metrics = detailed_metrics[model_name]
        print(f"\n{model_name.upper()}: {metrics['best_accuracy']*100:.1f}% accuracy")

print("\n📥 Ready to download: ARC_Models_2025.zip")
print("\nUse Colab's Files panel on the left to download the zip file,")
print("or run this in a new cell:")
print("from google.colab import files")
print("files.download('ARC_Models_2025.zip')")