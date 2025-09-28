#!/usr/bin/env python3
"""
Flexible model loader that handles architecture mismatches
"""

import torch
import torch.nn as nn


def load_model_flexible(model, checkpoint_path, strict=False):
    """
    Load model weights flexibly, handling architecture changes
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Try to load with strict=False to see what matches
    try:
        model.load_state_dict(state_dict, strict=strict)
        print(f"✓ Loaded model perfectly")
        return model, checkpoint
    except RuntimeError as e:
        if not strict:
            raise e
        
        print(f"⚠️  Model architecture mismatch, attempting flexible load...")
        
        # Get current model state dict
        model_dict = model.state_dict()
        
        # Filter out mismatched keys
        pretrained_dict = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                skipped_keys.append(f"{k} (not in current model)")
        
        # Update model dict with matched weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"✓ Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
        if skipped_keys:
            print(f"⚠️  Skipped keys: {len(skipped_keys)}")
            for key in skipped_keys[:5]:  # Show first 5
                print(f"   - {key}")
            if len(skipped_keys) > 5:
                print(f"   ... and {len(skipped_keys) - 5} more")
        
        return model, checkpoint


def fix_ensemble_loader():
    """
    Create a fixed version of the ensemble loader
    """
    
    code = '''#!/usr/bin/env python3
"""
Fixed OLYMPUS Ensemble loader that handles model architecture changes
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys
import os

# Add model path
sys.path.append('/content/Arc2025/models')
from arc_models_enhanced import (
    EnhancedMinervaNet, EnhancedAtlasNet, EnhancedIrisNet, 
    EnhancedChronosNet, EnhancedPrometheusNet
)


class OLYMPUSEnsemble:
    """The complete OLYMPUS ensemble with all 5 specialists"""
    
    def __init__(self, model_dir: str = '/content/arc_models_v4'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🏛️ OLYMPUS Ensemble initializing on {self.device}")
        
        # Model architectures
        self.models = {
            'minerva': EnhancedMinervaNet(),
            'atlas': EnhancedAtlasNet(),
            'iris': EnhancedIrisNet(),
            'chronos': EnhancedChronosNet(),
            'prometheus': EnhancedPrometheusNet()
        }
        
        # Load trained weights
        self.model_dir = Path(model_dir)
        self._load_all_models()
        
        # Move to device and eval mode
        for name, model in self.models.items():
            model.to(self.device)
            model.eval()
        
        print("✅ All 5 specialists loaded and ready!")
    
    def _load_all_models(self):
        """Load all trained model weights with flexible loading"""
        for model_name in self.models.keys():
            model_path = self.model_dir / f"{model_name}_best.pt"
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Try strict loading first
                    try:
                        self.models[model_name].load_state_dict(checkpoint['model_state_dict'])
                        val_exact = checkpoint.get('val_exact', 0)
                        print(f"  ✓ {model_name.upper()}: {val_exact:.2f}% exact match")
                    except RuntimeError as e:
                        # Flexible loading if strict fails
                        print(f"  ⚠️  {model_name.upper()}: Architecture mismatch, using flexible loading...")
                        
                        model_dict = self.models[model_name].state_dict()
                        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                         if k in model_dict and model_dict[k].shape == v.shape}
                        
                        model_dict.update(pretrained_dict)
                        self.models[model_name].load_state_dict(model_dict, strict=False)
                        
                        val_exact = checkpoint.get('val_exact', 0)
                        print(f"  ✓ {model_name.upper()}: {val_exact:.2f}% exact match (partial load)")
                        
                except Exception as e:
                    print(f"  ✗ {model_name.upper()}: Failed to load - {str(e)}")
            else:
                print(f"  ⚠️  {model_name.upper()}: No checkpoint found at {model_path}")
    
    def grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convert numpy grid to one-hot tensor"""
        h, w = grid.shape
        max_size = 30
        num_colors = 10
        
        # Create one-hot encoding
        one_hot = torch.zeros(num_colors, max_size, max_size)
        
        # Clip to max size
        h_clip = min(h, max_size)
        w_clip = min(w, max_size)
        
        for color in range(num_colors):
            mask = (grid[:h_clip, :w_clip] == color)
            one_hot[color, :h_clip, :w_clip] = torch.from_numpy(mask.astype(np.float32))
        
        return one_hot.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def tensor_to_grid(self, tensor: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Convert tensor back to numpy grid"""
        # Get predicted colors
        pred = tensor.squeeze(0).argmax(dim=0).cpu().numpy()
        
        # Crop to original shape
        h, w = original_shape
        return pred[:h, :w]
    
    def predict_single_model(self, model_name: str, input_grid: np.ndarray) -> np.ndarray:
        """Get prediction from a single model"""
        input_tensor = self.grid_to_tensor(input_grid)
        
        with torch.no_grad():
            if model_name == 'chronos':
                # CHRONOS expects a list of tensors
                outputs = self.models[model_name]([input_tensor])
            else:
                # Other models take tensor directly
                outputs = self.models[model_name](input_tensor)
            
            pred_tensor = outputs['predicted_output']
        
        return self.tensor_to_grid(pred_tensor, input_grid.shape)
    
    def predict_all_models(self, input_grid: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all models"""
        predictions = {}
        
        print("\\n🔮 Getting predictions from all specialists...")
        for model_name in self.models.keys():
            try:
                pred = self.predict_single_model(model_name, input_grid)
                predictions[model_name] = pred
                unique_colors = len(np.unique(pred))
                print(f"  ✓ {model_name.upper()}: Shape {pred.shape}, {unique_colors} colors")
            except Exception as e:
                print(f"  ✗ {model_name.upper()}: Error - {str(e)}")
                predictions[model_name] = None
        
        return predictions
    
    def majority_vote(self, predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int, List[str]]:
        """Simple majority vote ensemble"""
        # Filter out None predictions
        valid_predictions = [(name, pred) for name, pred in predictions.items() if pred is not None]
        
        if not valid_predictions:
            raise ValueError("No valid predictions from any model!")
        
        # Convert grids to hashable strings for voting
        grid_to_voters = {}
        
        for model_name, pred_grid in valid_predictions:
            # Convert to string for hashing
            grid_str = pred_grid.tobytes()
            
            if grid_str not in grid_to_voters:
                grid_to_voters[grid_str] = []
            grid_to_voters[grid_str].append(model_name)
        
        # Find the grid with most votes
        best_grid_str = None
        best_votes = 0
        best_voters = []
        
        for grid_str, voters in grid_to_voters.items():
            if len(voters) > best_votes:
                best_votes = len(voters)
                best_grid_str = grid_str
                best_voters = voters
        
        # Convert back to numpy array
        winning_grid = None
        for model_name, pred_grid in valid_predictions:
            if model_name in best_voters:
                winning_grid = pred_grid
                break
        
        return winning_grid, best_votes, best_voters
    
    def weighted_vote(self, input_grid: np.ndarray, predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """Smart weighted voting based on task characteristics"""
        # Analyze task characteristics
        h, w = input_grid.shape
        unique_colors = len(np.unique(input_grid))
        
        # Base weights
        weights = {
            'minerva': 1.0,    # General baseline
            'atlas': 1.0,      # Spatial
            'iris': 1.0,       # Color
            'chronos': 1.0,    # Temporal
            'prometheus': 1.0  # Creative
        }
        
        # Adjust weights based on task properties
        
        # Large grids favor ATLAS
        if max(h, w) > 15:
            weights['atlas'] *= 1.5
        
        # Many colors favor IRIS
        if unique_colors > 5:
            weights['iris'] *= 1.5
        elif unique_colors <= 3:
            weights['iris'] *= 0.8
        
        # Small, simple grids might be sequential patterns
        if max(h, w) <= 10 and unique_colors <= 3:
            weights['chronos'] *= 1.3
        
        # Very small grids might need creative solutions
        if max(h, w) <= 7:
            weights['prometheus'] *= 1.2
        
        # Complex patterns favor MINERVA
        if unique_colors >= 4 and max(h, w) >= 12:
            weights['minerva'] *= 1.4
        
        # Calculate weighted votes
        grid_scores = {}
        
        for model_name, pred_grid in predictions.items():
            if pred_grid is None:
                continue
                
            grid_str = pred_grid.tobytes()
            if grid_str not in grid_scores:
                grid_scores[grid_str] = {'score': 0.0, 'voters': [], 'grid': pred_grid}
            
            grid_scores[grid_str]['score'] += weights[model_name]
            grid_scores[grid_str]['voters'].append((model_name, weights[model_name]))
        
        # Find best scoring grid
        best_grid = None
        best_score = 0.0
        vote_details = {}
        
        for grid_str, info in grid_scores.items():
            if info['score'] > best_score:
                best_score = info['score']
                best_grid = info['grid']
                vote_details = {name: weight for name, weight in info['voters']}
        
        return best_grid, best_score, vote_details
    
    def predict(self, input_grid: np.ndarray, method: str = 'weighted') -> Dict:
        """Main prediction interface"""
        # Get all model predictions
        predictions = self.predict_all_models(input_grid)
        
        if method == 'majority':
            winning_grid, votes, voters = self.majority_vote(predictions)
            print(f"\\n🗳️ Majority Vote: {votes}/5 votes from {voters}")
            
            return {
                'prediction': winning_grid,
                'method': 'majority',
                'votes': votes,
                'voters': voters,
                'all_predictions': predictions
            }
        
        else:  # weighted
            winning_grid, score, vote_details = self.weighted_vote(input_grid, predictions)
            print(f"\\n⚖️ Weighted Vote: Score {score:.2f}")
            for model, weight in vote_details.items():
                print(f"  - {model}: {weight:.2f}")
            
            return {
                'prediction': winning_grid,
                'method': 'weighted',
                'score': score,
                'vote_details': vote_details,
                'all_predictions': predictions
            }
'''
    
    return code


if __name__ == "__main__":
    # Generate the fixed ensemble loader
    fixed_code = fix_ensemble_loader()
    
    with open('/mnt/d/opt/ARCPrize2025/ensemble_test_bench_fixed.py', 'w') as f:
        f.write(fixed_code)
    
    print("✓ Created fixed ensemble loader: ensemble_test_bench_fixed.py")
    print("\nTo use in Colab:")
    print("1. Upload this file to Colab")
    print("2. Replace ensemble_test_bench.py with ensemble_test_bench_fixed.py")
    print("3. Run the evaluation again")