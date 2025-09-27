"""
ARC Prize 2025 - Ensemble Solver
Combines all 5 models for maximum accuracy on ARC tasks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from collections import Counter
from models.arc_models import create_models
from pattern_detectors import create_all_detectors


class ARCEnsembleSolver:
    """
    Ensemble solver that combines predictions from all 5 models
    Each model votes on pattern type, then specialized models handle their domains
    """
    
    def __init__(self, model_dir: str = "./checkpoints", device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = self._load_models(model_dir)
        self.pattern_detectors = create_all_detectors()
        
        # Model specializations
        self.model_specialties = {
            'minerva': ['composite', 'logical', 'counting'],
            'atlas': ['rotation', 'reflection', 'scaling', 'translation'],
            'iris': ['color_mapping', 'symmetry'],
            'chronos': ['object_movement', 'counting', 'logical'],
            'prometheus': ['composite', 'symmetry', 'object_movement']
        }
        
        # Pattern type mapping
        self.pattern_types = {
            0: 'rotation', 1: 'reflection', 2: 'scaling', 3: 'translation',
            4: 'color_mapping', 5: 'symmetry', 6: 'object_movement',
            7: 'counting', 8: 'logical', 9: 'composite'
        }
        
    def _load_models(self, model_dir: str) -> Dict[str, nn.Module]:
        """Load all trained models"""
        models = create_models()
        loaded_models = {}
        
        for name, model in models.items():
            checkpoint_path = f"{model_dir}/{name}_best.pt"
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                loaded_models[name] = model
                print(f"✓ Loaded {name} (accuracy: {checkpoint.get('best_acc', 0)*100:.1f}%)")
            except:
                print(f"⚠️ Could not load {name}, using untrained model")
                model.to(self.device)
                model.eval()
                loaded_models[name] = model
                
        return loaded_models
    
    def solve_task(self, task: Dict) -> List[List[List[int]]]:
        """
        Solve a single ARC task using ensemble approach
        Returns 2 predictions as required by competition
        """
        train_examples = task['train']
        test_inputs = task['test']
        
        predictions = []
        for test_input in test_inputs:
            # Get 2 different predictions using different strategies
            pred1 = self._ensemble_predict(train_examples, test_input, strategy='weighted_vote')
            pred2 = self._ensemble_predict(train_examples, test_input, strategy='specialist')
            
            predictions.append({
                'attempt_1': pred1.tolist(),
                'attempt_2': pred2.tolist()
            })
            
        return predictions
    
    def _ensemble_predict(self, train_examples: List[Dict], test_input: Dict, 
                         strategy: str = 'weighted_vote') -> np.ndarray:
        """Generate prediction using specified ensemble strategy"""
        
        # Convert to tensors
        test_tensor = self._grid_to_tensor(np.array(test_input['input']))
        
        # Step 1: Identify pattern type using all models
        pattern_votes = self._get_pattern_votes(train_examples, test_tensor)
        detected_pattern = self._determine_pattern_type(pattern_votes)
        
        # Step 2: Use specialized models based on pattern type
        if strategy == 'weighted_vote':
            # All models contribute, weighted by their confidence
            return self._weighted_ensemble_prediction(
                train_examples, test_tensor, detected_pattern
            )
        elif strategy == 'specialist':
            # Only specialist models for this pattern type
            return self._specialist_prediction(
                train_examples, test_tensor, detected_pattern
            )
        else:
            # Fallback to best single model
            return self._best_model_prediction(train_examples, test_tensor)
    
    def _get_pattern_votes(self, train_examples: List[Dict], 
                          test_input: torch.Tensor) -> Dict[str, int]:
        """Get pattern type votes from all models"""
        votes = {}
        
        with torch.no_grad():
            # Get a representative training example for models that need it
            if train_examples:
                sample_output = self._grid_to_tensor(
                    np.array(train_examples[0]['output'])
                )
            
            for name, model in self.models.items():
                test_batch = test_input.unsqueeze(0).to(self.device)
                
                # Get pattern classification from each model
                if name in ['minerva', 'iris']:
                    output_batch = sample_output.unsqueeze(0).to(self.device)
                    outputs = model(test_batch, output_batch)
                    logits = outputs.get('pattern_logits', 
                                       outputs.get('pattern_type_logits'))
                elif name == 'atlas':
                    outputs = model(test_batch)
                    logits = outputs['transform_params'][:, :10]
                elif name == 'chronos':
                    outputs = model([test_batch])
                    logits = outputs['evolution_type_logits']
                elif name == 'prometheus':
                    outputs = model(test_batch)
                    logits = outputs['synthesis_strategy_logits'][:, :10]
                
                # Get predicted pattern type
                pattern_idx = torch.argmax(logits, dim=1).item()
                pattern_type = self.pattern_types[pattern_idx]
                
                if name not in votes:
                    votes[name] = pattern_type
                    
        return votes
    
    def _determine_pattern_type(self, votes: Dict[str, str]) -> str:
        """Determine final pattern type from votes"""
        # Weight votes by model specialty
        weighted_votes = []
        for model, pattern in votes.items():
            weight = 2 if pattern in self.model_specialties[model] else 1
            weighted_votes.extend([pattern] * weight)
        
        # Return most common pattern
        if weighted_votes:
            return Counter(weighted_votes).most_common(1)[0][0]
        return 'composite'  # Default
    
    def _weighted_ensemble_prediction(self, train_examples: List[Dict], 
                                    test_input: torch.Tensor, 
                                    pattern_type: str) -> np.ndarray:
        """Weighted ensemble prediction based on model confidence"""
        predictions = []
        weights = []
        
        with torch.no_grad():
            test_batch = test_input.unsqueeze(0).to(self.device)
            
            for name, model in self.models.items():
                # Higher weight if model specializes in this pattern
                weight = 2.0 if pattern_type in self.model_specialties[name] else 1.0
                
                # Get prediction based on model type
                if name == 'minerva':
                    # Use pattern memory to generate output
                    pred = self._minerva_predict(model, train_examples, test_batch)
                elif name == 'atlas':
                    # Apply spatial transformation
                    pred = self._atlas_predict(model, train_examples, test_batch)
                elif name == 'iris':
                    # Apply color transformation
                    pred = self._iris_predict(model, train_examples, test_batch)
                elif name == 'chronos':
                    # Predict sequence evolution
                    pred = self._chronos_predict(model, train_examples, test_batch)
                elif name == 'prometheus':
                    # Generate creative solution
                    pred = self._prometheus_predict(model, train_examples, test_batch)
                
                if pred is not None:
                    predictions.append(pred)
                    weights.append(weight)
        
        # Combine predictions
        if predictions:
            return self._combine_predictions(predictions, weights)
        else:
            # Fallback: return input unchanged
            return test_input.squeeze(0).argmax(dim=0).cpu().numpy()
    
    def _specialist_prediction(self, train_examples: List[Dict], 
                             test_input: torch.Tensor, 
                             pattern_type: str) -> np.ndarray:
        """Use only specialist models for detected pattern type"""
        specialists = [name for name, specs in self.model_specialties.items() 
                      if pattern_type in specs]
        
        predictions = []
        with torch.no_grad():
            test_batch = test_input.unsqueeze(0).to(self.device)
            
            for name in specialists:
                model = self.models[name]
                
                if name == 'atlas' and pattern_type in ['rotation', 'reflection', 'scaling']:
                    pred = self._atlas_predict(model, train_examples, test_batch)
                elif name == 'iris' and pattern_type in ['color_mapping', 'symmetry']:
                    pred = self._iris_predict(model, train_examples, test_batch)
                elif name == 'chronos' and pattern_type in ['object_movement', 'counting']:
                    pred = self._chronos_predict(model, train_examples, test_batch)
                else:
                    pred = self._general_predict(model, name, train_examples, test_batch)
                
                if pred is not None:
                    predictions.append(pred)
        
        if predictions:
            return self._combine_predictions(predictions, [1.0] * len(predictions))
        else:
            return test_input.squeeze(0).argmax(dim=0).cpu().numpy()
    
    def _minerva_predict(self, model: nn.Module, train_examples: List[Dict], 
                        test_input: torch.Tensor) -> np.ndarray:
        """MINERVA: Use pattern memory for strategic reasoning"""
        # Get pattern memory from training examples
        for example in train_examples:
            input_tensor = self._grid_to_tensor(np.array(example['input'])).unsqueeze(0).to(self.device)
            output_tensor = self._grid_to_tensor(np.array(example['output'])).unsqueeze(0).to(self.device)
            
            outputs = model(input_tensor, output_tensor)
            # Pattern memory is encoded in the model
        
        # Generate prediction
        dummy_output = torch.zeros_like(test_input).unsqueeze(0).to(self.device)
        outputs = model(test_input, dummy_output)
        
        if 'predicted_output' in outputs:
            return outputs['predicted_output'].squeeze(0).argmax(dim=0).cpu().numpy()
        
        # Fallback: use transformation
        return self._apply_learned_transform(train_examples, test_input)
    
    def _atlas_predict(self, model: nn.Module, train_examples: List[Dict], 
                      test_input: torch.Tensor) -> np.ndarray:
        """ATLAS: Apply spatial transformations"""
        outputs = model(test_input)
        transform_params = outputs['transform_params'].squeeze(0)
        
        # Interpret transformation parameters
        rotation = int(transform_params[0].item() * 4) % 4
        flip_h = transform_params[1].item() > 0.5
        flip_v = transform_params[2].item() > 0.5
        
        # Apply transformations
        grid = test_input.squeeze(0).argmax(dim=0).cpu().numpy()
        
        if rotation > 0:
            grid = np.rot90(grid, k=rotation)
        if flip_h:
            grid = np.flip(grid, axis=1)
        if flip_v:
            grid = np.flip(grid, axis=0)
            
        return grid
    
    def _iris_predict(self, model: nn.Module, train_examples: List[Dict], 
                     test_input: torch.Tensor) -> np.ndarray:
        """IRIS: Apply color transformations"""
        # Analyze color mapping from examples
        color_map = self._learn_color_mapping(train_examples)
        
        # Get color transformation from model
        dummy_output = torch.zeros_like(test_input).unsqueeze(0).to(self.device)
        outputs = model(test_input, dummy_output)
        
        # Apply color mapping
        grid = test_input.squeeze(0).argmax(dim=0).cpu().numpy()
        transformed = np.zeros_like(grid)
        
        for old_color, new_color in color_map.items():
            transformed[grid == old_color] = new_color
            
        return transformed
    
    def _chronos_predict(self, model: nn.Module, train_examples: List[Dict], 
                        test_input: torch.Tensor) -> np.ndarray:
        """CHRONOS: Predict temporal evolution"""
        # Create sequence from training examples
        sequence = [test_input]
        
        outputs = model(sequence)
        next_state = outputs['next_state']
        
        if next_state is not None:
            return next_state.squeeze(0).argmax(dim=0).cpu().numpy()
        
        # Fallback: detect movement pattern
        return self._predict_movement(train_examples, test_input)
    
    def _prometheus_predict(self, model: nn.Module, train_examples: List[Dict], 
                           test_input: torch.Tensor) -> np.ndarray:
        """PROMETHEUS: Generate creative solution"""
        outputs = model(test_input)
        
        # Decode generated pattern
        if 'generated_output' in outputs:
            generated = outputs['generated_output']
            return generated.squeeze(0).argmax(dim=0).cpu().numpy()
        
        # Use latent space for generation
        mu = outputs['mu']
        synthesis_logits = outputs['synthesis_strategy_logits']
        
        # Determine synthesis strategy
        strategy_idx = torch.argmax(synthesis_logits, dim=1).item()
        
        # Apply strategy-based generation
        return self._apply_synthesis_strategy(
            train_examples, test_input, strategy_idx
        )
    
    def _general_predict(self, model: nn.Module, model_name: str,
                        train_examples: List[Dict], 
                        test_input: torch.Tensor) -> np.ndarray:
        """General prediction for any model"""
        # Default: learn and apply transformation
        return self._apply_learned_transform(train_examples, test_input)
    
    def _combine_predictions(self, predictions: List[np.ndarray], 
                           weights: List[float]) -> np.ndarray:
        """Combine multiple predictions using weighted voting"""
        if not predictions:
            return predictions[0]
        
        # Ensure all predictions have same shape
        target_shape = predictions[0].shape
        valid_preds = []
        valid_weights = []
        
        for pred, weight in zip(predictions, weights):
            if pred.shape == target_shape:
                valid_preds.append(pred)
                valid_weights.append(weight)
        
        if not valid_preds:
            return predictions[0]
        
        # Weighted voting for each cell
        h, w = target_shape
        result = np.zeros((h, w), dtype=int)
        
        for i in range(h):
            for j in range(w):
                votes = {}
                for pred, weight in zip(valid_preds, valid_weights):
                    color = pred[i, j]
                    votes[color] = votes.get(color, 0) + weight
                
                # Select color with highest weighted votes
                if votes:
                    result[i, j] = max(votes.items(), key=lambda x: x[1])[0]
        
        return result
    
    def _learn_color_mapping(self, train_examples: List[Dict]) -> Dict[int, int]:
        """Learn color mapping from examples"""
        color_map = {}
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if input_grid.shape == output_grid.shape:
                # Find color correspondences
                for color in np.unique(input_grid):
                    mask = input_grid == color
                    output_colors = output_grid[mask]
                    if len(output_colors) > 0:
                        # Most common output color for this input color
                        most_common = Counter(output_colors).most_common(1)[0][0]
                        color_map[color] = most_common
        
        return color_map
    
    def _apply_learned_transform(self, train_examples: List[Dict], 
                               test_input: torch.Tensor) -> np.ndarray:
        """Learn transformation from examples and apply"""
        # Simple approach: find most similar training input
        test_grid = test_input.squeeze(0).argmax(dim=0).cpu().numpy()
        
        best_match = None
        best_score = -1
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            
            # Similarity score (inverse of difference)
            if input_grid.shape == test_grid.shape:
                score = 1.0 / (1.0 + np.sum(input_grid != test_grid))
                if score > best_score:
                    best_score = score
                    best_match = example
        
        if best_match:
            # Apply same transformation
            return np.array(best_match['output'])
        
        return test_grid
    
    def _predict_movement(self, train_examples: List[Dict], 
                         test_input: torch.Tensor) -> np.ndarray:
        """Predict object movement pattern"""
        test_grid = test_input.squeeze(0).argmax(dim=0).cpu().numpy()
        
        # Detect movement pattern from examples
        if len(train_examples) >= 2:
            # Analyze movement between first two examples
            grid1 = np.array(train_examples[0]['input'])
            grid2 = np.array(train_examples[0]['output'])
            
            # Simple: detect translation
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if self._check_translation(grid1, grid2, dy, dx):
                        # Apply same translation
                        return self._apply_translation(test_grid, dy, dx)
        
        return test_grid
    
    def _apply_synthesis_strategy(self, train_examples: List[Dict], 
                                test_input: torch.Tensor, 
                                strategy: int) -> np.ndarray:
        """Apply synthesis strategy for creative generation"""
        test_grid = test_input.squeeze(0).argmax(dim=0).cpu().numpy()
        
        strategies = {
            0: lambda g: g,  # Identity
            1: lambda g: np.flip(g, axis=0),  # Vertical flip
            2: lambda g: np.flip(g, axis=1),  # Horizontal flip
            3: lambda g: np.rot90(g, 1),  # Rotate 90
            4: lambda g: np.maximum(g, np.flip(g, axis=0)),  # Symmetrize vertical
            5: lambda g: np.maximum(g, np.flip(g, axis=1)),  # Symmetrize horizontal
            6: lambda g: self._fill_pattern(g),  # Fill pattern
            7: lambda g: self._extract_objects(g),  # Extract objects
            8: lambda g: self._complete_pattern(g),  # Complete pattern
            9: lambda g: self._combine_examples(train_examples, g)  # Combine
        }
        
        if strategy in strategies:
            return strategies[strategy](test_grid)
        
        return test_grid
    
    def _check_translation(self, grid1: np.ndarray, grid2: np.ndarray, 
                          dy: int, dx: int) -> bool:
        """Check if grid2 is translated version of grid1"""
        if grid1.shape != grid2.shape:
            return False
        
        h, w = grid1.shape
        translated = np.zeros_like(grid1)
        
        for y in range(h):
            for x in range(w):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < h and 0 <= new_x < w:
                    translated[new_y, new_x] = grid1[y, x]
        
        return np.array_equal(translated, grid2)
    
    def _apply_translation(self, grid: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Apply translation to grid"""
        h, w = grid.shape
        translated = np.zeros_like(grid)
        
        for y in range(h):
            for x in range(w):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < h and 0 <= new_x < w:
                    translated[new_y, new_x] = grid[y, x]
        
        return translated
    
    def _fill_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Fill incomplete pattern"""
        # Simple flood fill of largest connected component
        from scipy import ndimage
        
        result = grid.copy()
        if np.any(grid == 0):
            # Find largest non-zero component
            labeled, num_features = ndimage.label(grid > 0)
            if num_features > 0:
                sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                largest = np.argmax(sizes) + 1
                
                # Fill holes in largest component
                mask = labeled == largest
                result[mask] = grid[mask].max()
        
        return result
    
    def _extract_objects(self, grid: np.ndarray) -> np.ndarray:
        """Extract distinct objects"""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(grid > 0)
        
        if num_features > 1:
            # Keep only the largest object
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest = np.argmax(sizes) + 1
            
            result = np.zeros_like(grid)
            result[labeled == largest] = grid[labeled == largest]
            return result
        
        return grid
    
    def _complete_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Complete partial pattern"""
        # Try to detect and complete symmetry
        h, w = grid.shape
        
        # Check for vertical symmetry
        left_half = grid[:, :w//2]
        right_half = grid[:, w//2:]
        
        if np.sum(left_half > 0) > np.sum(right_half > 0):
            # Complete right side
            result = grid.copy()
            result[:, w//2:] = np.flip(left_half, axis=1)
            return result
        
        return grid
    
    def _combine_examples(self, train_examples: List[Dict], 
                         grid: np.ndarray) -> np.ndarray:
        """Combine patterns from training examples"""
        if not train_examples:
            return grid
        
        # Simple: overlay first training output
        output_grid = np.array(train_examples[0]['output'])
        
        if output_grid.shape == grid.shape:
            # Combine non-zero elements
            result = grid.copy()
            mask = output_grid > 0
            result[mask] = output_grid[mask]
            return result
        
        return grid
    
    def _grid_to_tensor(self, grid: np.ndarray, max_size: int = 30) -> torch.Tensor:
        """Convert grid to tensor"""
        h, w = grid.shape
        one_hot = np.zeros((10, max_size, max_size))
        
        for i in range(min(h, max_size)):
            for j in range(min(w, max_size)):
                if grid[i, j] < 10:
                    one_hot[grid[i, j], i, j] = 1
        
        return torch.FloatTensor(one_hot)
    
    def _best_model_prediction(self, train_examples: List[Dict], 
                              test_input: torch.Tensor) -> np.ndarray:
        """Use single best performing model"""
        # Default to MINERVA as it's usually most versatile
        return self._minerva_predict(
            self.models['minerva'], train_examples, test_input
        )


def create_submission(input_path: str, output_path: str, 
                     checkpoint_dir: str = "./checkpoints"):
    """Create submission file for Kaggle"""
    print("Loading ensemble solver...")
    solver = ARCEnsembleSolver(checkpoint_dir)
    
    print("Processing tasks...")
    with open(input_path, 'r') as f:
        tasks = json.load(f)
    
    submission = {}
    
    for task_id, task_data in tasks.items():
        predictions = solver.solve_task(task_data)
        submission[task_id] = predictions
        print(f"✓ Solved {task_id}")
    
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"\nSubmission saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("""
ARC Prize 2025 - Ensemble Solver
═══════════════════════════════════════════════════════════════

This solver combines all 5 models:
- MINERVA: Strategic reasoning
- ATLAS: Spatial transformations  
- IRIS: Color patterns
- CHRONOS: Temporal sequences
- PROMETHEUS: Creative generation

Each task gets 2 predictions:
1. Weighted ensemble (all models vote)
2. Specialist models (pattern-specific)

To use:
    solver = ARCEnsembleSolver('./checkpoints')
    predictions = solver.solve_task(task_data)
    """)