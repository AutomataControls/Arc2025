#!/usr/bin/env python3
"""
Enhanced Inference System for ARC Prize 2025
Incorporates ideas from winning solutions while maintaining our CNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools
from pathlib import Path
import json
import logging

# Import our enhanced models
import sys
sys.path.append('/mnt/d/opt/ARCPrize2025/models')
from arc_models_enhanced import create_enhanced_models

logger = logging.getLogger(__name__)


class EnhancedInferenceEngine:
    """Enhanced inference with multiple predictions and confidence scoring"""
    
    def __init__(self, model_dir: Path, device: str = 'cuda'):
        self.model_dir = model_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all enhanced models"""
        model_dict = create_enhanced_models()
        
        for model_name, model in model_dict.items():
            checkpoint_path = self.model_dir / f'{model_name}_enhanced_best.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model.to(self.device)
                self.models[model_name] = model
                logger.info(f"Loaded {model_name} with accuracy {checkpoint.get('val_accuracy', 0):.2f}%")
            else:
                logger.warning(f"Checkpoint not found for {model_name}")
    
    def generate_multiple_predictions(self, input_grid: torch.Tensor, model: nn.Module, 
                                    n_predictions: int = 5) -> List[Tuple[torch.Tensor, float]]:
        """Generate multiple predictions with confidence scores"""
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            output = model(input_grid)
            pred_logits = output['predicted_output']
            
            # Calculate confidence using entropy
            probs = F.softmax(pred_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            confidence = 1.0 - (entropy.mean().item() / np.log(10))  # Normalize by max entropy
            
            predictions.append((pred_logits, confidence))
        
        # Generate variations using dropout
        if hasattr(model, 'training'):
            model.train()  # Enable dropout
            for _ in range(n_predictions - 1):
                with torch.no_grad():
                    output = model(input_grid)
                    pred_logits = output['predicted_output']
                    
                    probs = F.softmax(pred_logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    confidence = 1.0 - (entropy.mean().item() / np.log(10))
                    
                    predictions.append((pred_logits, confidence))
            model.eval()
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def apply_color_permutations(self, input_grid: torch.Tensor) -> List[torch.Tensor]:
        """Generate color permutation augmentations"""
        augmented = [input_grid]
        
        # Get unique colors in the grid
        colors = torch.unique(input_grid.argmax(dim=1))
        
        if len(colors) <= 5:  # Only permute if few colors
            # Generate a few random permutations
            for _ in range(3):
                perm = torch.randperm(10)
                perm_grid = torch.zeros_like(input_grid)
                
                for i in range(10):
                    mask = (input_grid.argmax(dim=1) == i)
                    if mask.any():
                        perm_grid[perm[i]] = input_grid[i][mask]
                
                augmented.append(perm_grid)
        
        return augmented
    
    def ensemble_predict_with_selection(self, input_grid: np.ndarray) -> np.ndarray:
        """Enhanced ensemble prediction with intelligent selection"""
        input_tensor = self._prepare_input(input_grid)
        
        all_predictions = []
        
        # Generate predictions from each model
        for model_name, model in self.models.items():
            # Apply augmentations
            augmented_inputs = self.apply_color_permutations(input_tensor)
            
            for aug_input in augmented_inputs:
                # Get multiple predictions per input
                preds = self.generate_multiple_predictions(aug_input, model, n_predictions=3)
                
                for pred_logits, confidence in preds:
                    # Convert to grid
                    pred_grid = pred_logits.argmax(dim=1).squeeze().cpu().numpy()
                    
                    # Calculate pattern-specific scores
                    pattern_score = self._evaluate_pattern_consistency(input_grid, pred_grid)
                    
                    # Combined score
                    total_score = confidence * 0.7 + pattern_score * 0.3
                    
                    all_predictions.append({
                        'grid': pred_grid,
                        'confidence': confidence,
                        'pattern_score': pattern_score,
                        'total_score': total_score,
                        'model': model_name
                    })
        
        # Sort by total score
        all_predictions.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Try to find consensus among top predictions
        top_k = min(5, len(all_predictions))
        top_preds = all_predictions[:top_k]
        
        # Check for exact matches in top predictions
        for i in range(top_k):
            for j in range(i+1, top_k):
                if np.array_equal(top_preds[i]['grid'], top_preds[j]['grid']):
                    # Found consensus
                    return self._clean_prediction(top_preds[i]['grid'], input_grid.shape)
        
        # No consensus, return best prediction
        best_pred = all_predictions[0]['grid']
        return self._clean_prediction(best_pred, input_grid.shape)
    
    def _evaluate_pattern_consistency(self, input_grid: np.ndarray, pred_grid: np.ndarray) -> float:
        """Evaluate if prediction follows consistent patterns"""
        score = 1.0
        
        # Check size consistency
        if pred_grid.shape != input_grid.shape:
            score *= 0.5
        
        # Check color usage consistency
        input_colors = set(np.unique(input_grid))
        pred_colors = set(np.unique(pred_grid))
        
        # Reward using similar color palette
        color_overlap = len(input_colors.intersection(pred_colors)) / max(len(input_colors), 1)
        score *= (0.5 + 0.5 * color_overlap)
        
        # Check structural similarity
        if input_grid.shape == pred_grid.shape:
            # Simple structure preservation check
            input_nonzero = (input_grid > 0).astype(float)
            pred_nonzero = (pred_grid > 0).astype(float)
            
            structure_sim = 1.0 - np.abs(input_nonzero - pred_nonzero).mean()
            score *= structure_sim
        
        return score
    
    def _prepare_input(self, grid: np.ndarray) -> torch.Tensor:
        """Convert numpy grid to model input tensor"""
        h, w = grid.shape
        one_hot = np.zeros((1, 10, 30, 30))
        
        for i in range(min(h, 30)):
            for j in range(min(w, 30)):
                if 0 <= grid[i, j] < 10:
                    one_hot[0, int(grid[i, j]), i, j] = 1
        
        return torch.FloatTensor(one_hot).to(self.device)
    
    def _clean_prediction(self, pred_grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Clean and resize prediction to target shape"""
        # Remove padding
        h, w = target_shape
        cleaned = pred_grid[:h, :w]
        
        # Ensure integer values
        cleaned = np.clip(cleaned, 0, 9).astype(int)
        
        return cleaned


class PatternSpecificPostProcessor:
    """Apply pattern-specific post-processing rules"""
    
    @staticmethod
    def detect_pattern_type(input_grid: np.ndarray, pred_grid: np.ndarray) -> str:
        """Detect the likely pattern type"""
        # Check for rotation
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(input_grid, k), pred_grid):
                return 'rotation'
        
        # Check for reflection
        if np.array_equal(np.fliplr(input_grid), pred_grid):
            return 'reflection_h'
        if np.array_equal(np.flipud(input_grid), pred_grid):
            return 'reflection_v'
        
        # Check for color mapping
        if input_grid.shape == pred_grid.shape:
            unique_in = set(np.unique(input_grid))
            unique_out = set(np.unique(pred_grid))
            if len(unique_in) == len(unique_out) and unique_in != unique_out:
                return 'color_map'
        
        return 'complex'
    
    @staticmethod
    def apply_pattern_constraints(input_grid: np.ndarray, pred_grid: np.ndarray, 
                                pattern_type: str) -> np.ndarray:
        """Apply pattern-specific constraints to improve prediction"""
        
        if pattern_type == 'rotation':
            # Ensure exact rotation
            for k in [1, 2, 3]:
                rotated = np.rot90(input_grid, k)
                if pred_grid.shape == rotated.shape:
                    # Find best matching rotation
                    diff = np.abs(pred_grid.astype(float) - rotated.astype(float)).sum()
                    if diff < pred_grid.size * 0.1:  # Less than 10% different
                        return rotated
        
        elif pattern_type in ['reflection_h', 'reflection_v']:
            # Ensure exact reflection
            if pattern_type == 'reflection_h':
                reflected = np.fliplr(input_grid)
            else:
                reflected = np.flipud(input_grid)
            
            if pred_grid.shape == reflected.shape:
                return reflected
        
        elif pattern_type == 'color_map':
            # Build color mapping from examples
            color_map = {}
            flat_in = input_grid.flatten()
            flat_out = pred_grid.flatten()
            
            for i in range(len(flat_in)):
                if flat_in[i] not in color_map:
                    color_map[flat_in[i]] = flat_out[i]
            
            # Apply consistent mapping
            cleaned = np.zeros_like(pred_grid)
            for i in range(pred_grid.shape[0]):
                for j in range(pred_grid.shape[1]):
                    if input_grid[i, j] in color_map:
                        cleaned[i, j] = color_map[input_grid[i, j]]
                    else:
                        cleaned[i, j] = pred_grid[i, j]
            
            return cleaned.astype(int)
        
        return pred_grid


def create_submission(inference_engine: EnhancedInferenceEngine, 
                     challenges_file: str, 
                     output_file: str):
    """Create submission file using enhanced inference"""
    
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    submission = {}
    post_processor = PatternSpecificPostProcessor()
    
    for task_id, task in challenges.items():
        print(f"Processing {task_id}...")
        
        # Get test inputs
        test_inputs = task['test']
        predictions = []
        
        for test_input in test_inputs:
            input_grid = np.array(test_input['input'])
            
            # Get ensemble prediction
            pred_grid = inference_engine.ensemble_predict_with_selection(input_grid)
            
            # Detect pattern and apply constraints
            pattern_type = post_processor.detect_pattern_type(input_grid, pred_grid)
            pred_grid = post_processor.apply_pattern_constraints(input_grid, pred_grid, pattern_type)
            
            predictions.append(pred_grid.tolist())
        
        submission[task_id] = predictions
    
    # Save submission
    with open(output_file, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_file}")


if __name__ == "__main__":
    # Test the enhanced inference system
    print("Testing Enhanced Inference System...")
    
    model_dir = Path('/mnt/d/opt/ARCPrize2025/models')
    engine = EnhancedInferenceEngine(model_dir)
    
    # Create a test submission
    challenges_file = '/mnt/d/opt/ARCPrize2025/data/arc-agi_evaluation_challenges.json'
    output_file = '/mnt/d/opt/ARCPrize2025/enhanced_submission.json'
    
    if Path(challenges_file).exists():
        create_submission(engine, challenges_file, output_file)
        print("Enhanced inference system ready!")
    else:
        print("Evaluation challenges file not found")
