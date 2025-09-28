#!/usr/bin/env python3
"""
Enhanced OLYMPUS Ensemble with Grid Size Prediction
This fixes the shape blindness issue
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append('/content/Arc2025')
from ensemble_test_bench import OLYMPUSEnsemble
from grid_size_predictor_v2 import GridSizePredictorV2


class OLYMPUSEnsembleV2(OLYMPUSEnsemble):
    """Enhanced ensemble that predicts output dimensions"""
    
    def __init__(self, model_dir: str = '/content/arc_models_v4'):
        super().__init__(model_dir)
        self.size_predictor = GridSizePredictorV2()
    
    def predict_all_models_with_shape(self, input_grid: np.ndarray, 
                                     train_examples: List[Dict]) -> Dict[str, np.ndarray]:
        """Get predictions from all models with correct output shape"""
        # First, predict the output shape
        target_shape = self.size_predictor.predict_output_shape(input_grid, train_examples)
        
        predictions = {}
        
        print(f"\n🔮 Getting predictions from all specialists (target shape: {target_shape})...")
        for model_name in self.models.keys():
            try:
                # Get raw 30x30 prediction
                input_tensor = self.grid_to_tensor(input_grid)
                
                with torch.no_grad():
                    if model_name == 'chronos':
                        outputs = self.models[model_name]([input_tensor])
                    else:
                        outputs = self.models[model_name](input_tensor)
                    
                    pred_tensor = outputs['predicted_output']
                
                # Convert to grid with CORRECT shape
                pred = self.tensor_to_grid(pred_tensor, target_shape)
                predictions[model_name] = pred
                
                unique_colors = len(np.unique(pred))
                print(f"  ✓ {model_name.upper()}: Shape {pred.shape}, {unique_colors} colors")
                
            except Exception as e:
                print(f"  ✗ {model_name.upper()}: Error - {str(e)}")
                predictions[model_name] = None
        
        return predictions
    
    def predict_with_shape(self, input_grid: np.ndarray, train_examples: List[Dict], 
                          method: str = 'weighted') -> Dict:
        """
        Main prediction interface with shape prediction
        """
        # Get all model predictions with correct shape
        predictions = self.predict_all_models_with_shape(input_grid, train_examples)
        
        if method == 'majority':
            winning_grid, votes, voters = self.majority_vote(predictions)
            print(f"\n🗳️ Majority Vote: {votes}/5 votes from {voters}")
            
            return {
                'prediction': winning_grid,
                'method': 'majority',
                'votes': votes,
                'voters': voters,
                'all_predictions': predictions,
                'predicted_shape': self.size_predictor.predict_output_shape(input_grid, train_examples)
            }
        
        else:  # weighted
            winning_grid, score, vote_details = self.weighted_vote(input_grid, predictions)
            print(f"\n⚖️ Weighted Vote: Score {score:.2f}")
            for model, weight in vote_details.items():
                print(f"  - {model}: {weight:.2f}")
            
            return {
                'prediction': winning_grid,
                'method': 'weighted',
                'score': score,
                'vote_details': vote_details,
                'all_predictions': predictions,
                'predicted_shape': self.size_predictor.predict_output_shape(input_grid, train_examples)
            }