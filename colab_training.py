# ARC Prize 2025 - Complete Training Script for Google Colab
# Copy this entire code into a Colab cell and run it

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

# Dataset class
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
        
        # Create training pairs
        self.pairs = []
        for task in self.tasks:
            for example in task.get('train', []):
                if 'input' in example and 'output' in example:
                    self.pairs.append({
                        'input': np.array(example['input']),
                        'output': np.array(example['output']),
                        'task_id': task['filename'],
                        'pattern_label': self._detect_pattern_type(example)
                    })
        
        print(f"Loaded {len(self.tasks)} tasks with {len(self.pairs)} training pairs")
    
    def _detect_pattern_type(self, example: Dict) -> int:
        try:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if input_grid.shape != output_grid.shape:
                if output_grid.size > input_grid.size:
                    return self.pattern_labels['scaling']
                else:
                    return self.pattern_labels['counting']
            
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(input_grid, k), output_grid):
                    return self.pattern_labels['rotation']
            
            if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                return self.pattern_labels['reflection']
            if np.array_equal(np.flip(input_grid, axis=1), output_grid):
                return self.pattern_labels['reflection']
            
            if set(input_grid.flatten()) != set(output_grid.flatten()):
                return self.pattern_labels['color_mapping']
            
            if (np.array_equal(output_grid, np.flip(output_grid, axis=0)) or 
                np.array_equal(output_grid, np.flip(output_grid, axis=1))):
                return self.pattern_labels['symmetry']
            
            return self.pattern_labels['composite']
        except:
            return self.pattern_labels['composite']
    
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

# Create datasets
print("\n📊 Creating datasets...")
train_dataset = ARCDataset('data/arc-agi_training_challenges.json')
eval_dataset = ARCDataset('data/arc-agi_evaluation_challenges.json')

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Use smaller subset for quick testing
if len(train_dataset) > 200:
    train_subset = torch.utils.data.Subset(train_dataset, list(range(200)))
else:
    train_subset = train_dataset

# Split train into train/val
train_size = int(0.8 * len(train_subset))
val_size = len(train_subset) - train_size
train_data, val_data = random_split(train_subset, [train_size, val_size])

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\nTrain size: {len(train_data)}, Val size: {len(val_data)}")

# Training class
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
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for input_grid, output_grid, pattern_label in tqdm(loader, desc='Training', leave=False):
            input_grid = input_grid.to(self.device)
            output_grid = output_grid.to(self.device)
            pattern_label = pattern_label.to(self.device)
            
            optimizer.zero_grad()
            
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
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += pattern_label.size(0)
            correct += (predicted == pattern_label).sum().item()
        
        return total_loss / len(loader), correct / total if total > 0 else 0
    
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
    
    def train(self, train_loader, val_loader, epochs=10, lr=1e-3):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining {self.model_name.upper()} for {epochs} epochs...")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pt')
            
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            scheduler.step()
        
        # Load best model
        if os.path.exists(f'{self.model_name}_best.pt'):
            self.model.load_state_dict(torch.load(f'{self.model_name}_best.pt'))
        
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        return val_preds, val_labels

# Train all models
print("\n🚀 Starting model training...")
models = create_models()
training_results = {}

# Quick test - 3 epochs per model
EPOCHS = 3  # Change to 30 for full training

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} - {model.description}")
    print(f"{'='*60}")
    
    trainer = ModelTrainer(model, model_name, device)
    val_preds, val_labels = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
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
    detailed_metrics[model_name] = {
        'best_accuracy': results['best_acc'],
        'parameters': sum(p.numel() for p in models[model_name].parameters()),
        'training_time': len(results['history']['train_loss']) * 0.5  # Approximate
    }

# Export to ONNX
print("\n📦 Exporting models to ONNX...")
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
            inputs = example_input
            input_names = ['input_grid']
        else:
            inputs = example_input
            input_names = ['input_grid']
        
        # Get output names
        with torch.no_grad():
            if model_name in ['minerva', 'iris']:
                sample_out = model(example_input, example_output)
            elif model_name == 'chronos':
                sample_out = model([example_input])
            else:
                sample_out = model(example_input)
        
        output_names = list(sample_out.keys())
        
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
        print(f"✓ {model_name} exported")
        return onnx_path
    except Exception as e:
        print(f"⚠️ Could not export {model_name}: {str(e)}")
        return None

example_input = torch.randn(1, 10, 30, 30).to(device)
onnx_models = {}

for model_name, model in models.items():
    if os.path.exists(f'{model_name}_best.pt'):
        model.load_state_dict(torch.load(f'{model_name}_best.pt'))
    onnx_path = export_to_onnx(model, model_name, example_input)
    if onnx_path:
        onnx_models[model_name] = onnx_path

# Save everything
print("\n💾 Saving all artifacts...")
output_dir = './ARC_Models_2025'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/pytorch', exist_ok=True)
os.makedirs(f'{output_dir}/onnx', exist_ok=True)
os.makedirs(f'{output_dir}/metrics', exist_ok=True)

# Copy models
for model_name in models.keys():
    if os.path.exists(f'{model_name}_best.pt'):
        shutil.copy(f'{model_name}_best.pt', f'{output_dir}/pytorch/{model_name}_model.pt')
    if model_name in onnx_models and onnx_models[model_name]:
        shutil.copy(f'{model_name}_model.onnx', f'{output_dir}/onnx/{model_name}_model.onnx')

# Save metrics
with open(f'{output_dir}/training_metrics.json', 'w') as f:
    json.dump(detailed_metrics, f, indent=2)

# Generate Hailo conversion script
hailo_script = '''#!/bin/bash
# Convert ARC Prize 2025 ONNX models to HEF format for Hailo-8

echo "============================================================"
echo "ARC Prize 2025 - ONNX to HEF Conversion"
echo "============================================================"

MODELS=("minerva" "atlas" "iris" "chronos" "prometheus")
BASE_DIR="/mnt/d/opt/ARCPrize2025"

# Activate Hailo environment
source /mnt/c/Users/Juelz/hailo_venv_py310/bin/activate

mkdir -p "$BASE_DIR/hef"

for model in "${MODELS[@]}"; do
    echo "Converting $model..."
    
    ONNX_FILE="$BASE_DIR/models/onnx/${model}_model.onnx"
    
    if [ -f "$ONNX_FILE" ]; then
        hailo parser onnx --hw-arch hailo8 --onnx-model "$ONNX_FILE" \\
            --output-har-path "$BASE_DIR/${model}.har" --start-node-names input_grid
        
        hailo optimize --hw-arch hailo8 --har "$BASE_DIR/${model}.har" \\
            --use-random-calib-set --output-har-path "$BASE_DIR/${model}_optimized.har"
        
        hailo compiler --hw-arch hailo8 --har "$BASE_DIR/${model}_optimized.har" \\
            --output-hef-path "$BASE_DIR/hef/${model}.hef"
        
        rm -f "$BASE_DIR/${model}.har" "$BASE_DIR/${model}_optimized.har"
        echo "✓ Created $BASE_DIR/hef/${model}.hef"
    fi
done

echo "Copy HEF files to Pi:"
echo "scp $BASE_DIR/hef/*.hef Automata@192.168.0.54:/home/Automata/mydata/neural-nexus/arc2025/"
'''

with open(f'{output_dir}/convert_to_hef.sh', 'w') as f:
    f.write(hailo_script)

# Create zip
print("\n📦 Creating zip file...")
shutil.make_archive('ARC_Models_2025', 'zip', output_dir)

print("\n✅ TRAINING COMPLETE!")
print(f"\nModel Performance Summary:")
for model_name in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']:
    if model_name in detailed_metrics:
        print(f"{model_name.upper()}: {detailed_metrics[model_name]['best_accuracy']*100:.1f}% accuracy")

print("\n📥 Download: ARC_Models_2025.zip")
print("\nTo download, run in a new cell:")
print("from google.colab import files")
print("files.download('ARC_Models_2025.zip')")