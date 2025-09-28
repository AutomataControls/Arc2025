#!/usr/bin/env python3
"""
Intelligent Task Router for OLYMPUS Ensemble
Analyzes task characteristics to assign optimal weights to each specialist
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class TaskAnalyzer:
    """Analyze ARC task characteristics to determine which models should be trusted"""
    
    def __init__(self):
        self.characteristics = {}
    
    def analyze_task(self, input_grid: np.ndarray, 
                    train_examples: List[Dict]) -> Dict[str, float]:
        """
        Analyze task and return characteristics
        """
        characteristics = {}
        
        # Basic grid properties
        h, w = input_grid.shape
        characteristics['grid_size'] = max(h, w)
        characteristics['aspect_ratio'] = max(h/w, w/h)
        characteristics['total_cells'] = h * w
        
        # Color analysis
        unique_colors = len(np.unique(input_grid))
        characteristics['num_colors'] = unique_colors
        characteristics['color_density'] = unique_colors / 10.0  # Normalize to [0,1]
        
        # Check for background (0 is typically background)
        background_ratio = (input_grid == 0).sum() / (h * w)
        characteristics['background_ratio'] = background_ratio
        characteristics['has_sparse_objects'] = background_ratio > 0.7
        
        # Transformation analysis from training examples
        transformations = self._analyze_transformations(train_examples)
        characteristics.update(transformations)
        
        # Pattern detection
        patterns = self._detect_patterns(input_grid)
        characteristics.update(patterns)
        
        # Object analysis
        objects = self._analyze_objects(input_grid)
        characteristics.update(objects)
        
        return characteristics
    
    def _analyze_transformations(self, train_examples: List[Dict]) -> Dict[str, float]:
        """Analyze what kind of transformations the task involves"""
        trans = {
            'size_change': 0.0,
            'color_mapping': 0.0,
            'spatial_transform': 0.0,
            'object_manipulation': 0.0,
            'pattern_completion': 0.0,
            'is_sequential': 0.0
        }
        
        for i, example in enumerate(train_examples):
            inp = np.array(example['input'])
            out = np.array(example['output'])
            
            # Size changes
            if inp.shape != out.shape:
                trans['size_change'] += 1.0
                
            # Color mapping
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            if out_colors != inp_colors:
                trans['color_mapping'] += 1.0
            
            # Spatial transform (rotation/flip check)
            if inp.shape == out.shape:
                # Check if output is rotated/flipped version
                for k in range(4):
                    if np.array_equal(np.rot90(inp, k), out):
                        trans['spatial_transform'] += 1.0
                        break
                else:
                    if np.array_equal(np.flip(inp, 0), out) or np.array_equal(np.flip(inp, 1), out):
                        trans['spatial_transform'] += 1.0
            
            # Check for sequential patterns across examples
            if i > 0:
                prev_out = np.array(train_examples[i-1]['output'])
                if self._grids_related(prev_out, out):
                    trans['is_sequential'] += 1.0
        
        # Normalize by number of examples
        n = len(train_examples)
        for key in trans:
            trans[key] /= n
        
        return trans
    
    def _detect_patterns(self, grid: np.ndarray) -> Dict[str, float]:
        """Detect various patterns in the grid"""
        patterns = {
            'has_symmetry': 0.0,
            'has_repetition': 0.0,
            'has_regular_structure': 0.0,
            'is_geometric': 0.0
        }
        
        h, w = grid.shape
        
        # Check symmetry
        if np.array_equal(grid, np.flip(grid, 0)):
            patterns['has_symmetry'] = 1.0
        elif np.array_equal(grid, np.flip(grid, 1)):
            patterns['has_symmetry'] = 1.0
        elif h == w and np.array_equal(grid, grid.T):
            patterns['has_symmetry'] = 1.0
        
        # Check for repetition
        for axis in [0, 1]:
            size = grid.shape[axis]
            for period in range(2, size // 2 + 1):
                if self._is_periodic(grid, axis, period):
                    patterns['has_repetition'] = 1.0
                    break
        
        # Check for geometric shapes
        if self._has_geometric_shapes(grid):
            patterns['is_geometric'] = 1.0
        
        return patterns
    
    def _analyze_objects(self, grid: np.ndarray) -> Dict[str, float]:
        """Analyze objects in the grid"""
        from scipy import ndimage
        
        objects = {
            'num_objects': 0,
            'avg_object_size': 0.0,
            'object_complexity': 0.0,
            'has_multiple_objects': 0.0
        }
        
        # Find connected components for each color
        unique_colors = [c for c in np.unique(grid) if c != 0]
        
        total_objects = 0
        sizes = []
        
        for color in unique_colors:
            mask = (grid == color).astype(int)
            labeled, num = ndimage.label(mask)
            total_objects += num
            
            for i in range(1, num + 1):
                size = (labeled == i).sum()
                sizes.append(size)
        
        objects['num_objects'] = total_objects
        if sizes:
            objects['avg_object_size'] = np.mean(sizes) / (grid.shape[0] * grid.shape[1])
            objects['object_complexity'] = np.std(sizes) / (np.mean(sizes) + 1e-6)
        
        objects['has_multiple_objects'] = 1.0 if total_objects > 3 else 0.0
        
        return objects
    
    def _grids_related(self, grid1: np.ndarray, grid2: np.ndarray) -> bool:
        """Check if two grids appear to be related (for sequence detection)"""
        if grid1.shape != grid2.shape:
            return False
        
        # Check if grids share significant structure
        overlap = (grid1 == grid2).sum() / grid1.size
        return overlap > 0.3 and overlap < 0.95
    
    def _is_periodic(self, grid: np.ndarray, axis: int, period: int) -> bool:
        """Check if grid has periodic pattern along axis"""
        size = grid.shape[axis]
        if size % period != 0:
            return False
        
        for i in range(period, size):
            if axis == 0:
                if not np.array_equal(grid[i % period, :], grid[i, :]):
                    return False
            else:
                if not np.array_equal(grid[:, i % period], grid[:, i]):
                    return False
        return True
    
    def _has_geometric_shapes(self, grid: np.ndarray) -> bool:
        """Simple check for geometric patterns"""
        # Look for straight lines
        for row in grid:
            if len(set(row)) == 1 and row[0] != 0:
                return True
        
        for col in grid.T:
            if len(set(col)) == 1 and col[0] != 0:
                return True
        
        return False


class TaskRouter:
    """Route tasks to appropriate specialists based on characteristics"""
    
    def __init__(self):
        self.analyzer = TaskAnalyzer()
        
        # Model specialties based on training results
        self.model_strengths = {
            'minerva': {
                'complex_objects': 1.4,
                'multiple_colors': 1.2,
                'general': 1.0
            },
            'atlas': {
                'spatial_transform': 1.5,
                'geometric': 1.5,
                'large_grids': 1.5,
                'symmetry': 1.3
            },
            'iris': {
                'color_mapping': 1.5,
                'color_density': 1.5,
                'small_grids': 1.2
            },
            'chronos': {
                'sequential': 1.3,
                'pattern_completion': 1.3,
                'repetition': 1.2,
                'small_simple': 1.3
            },
            'prometheus': {
                'creative': 1.2,
                'unusual': 1.2,
                'very_small': 1.2,
                'low_structure': 1.1
            }
        }
    
    def get_model_weights(self, input_grid: np.ndarray, 
                         train_examples: List[Dict]) -> Dict[str, float]:
        """
        Get optimized weights for each model based on task analysis
        """
        # Analyze task
        characteristics = self.analyzer.analyze_task(input_grid, train_examples)
        
        # Initialize base weights
        weights = {
            'minerva': 1.0,
            'atlas': 1.0,
            'iris': 1.0,
            'chronos': 1.0,
            'prometheus': 1.0
        }
        
        # Apply MINERVA bonuses
        if characteristics['num_objects'] > 3:
            weights['minerva'] *= self.model_strengths['minerva']['complex_objects']
        if characteristics['num_colors'] >= 4:
            weights['minerva'] *= self.model_strengths['minerva']['multiple_colors']
        
        # Apply ATLAS bonuses
        if characteristics['spatial_transform'] > 0.5:
            weights['atlas'] *= self.model_strengths['atlas']['spatial_transform']
        if characteristics['is_geometric'] > 0.5:
            weights['atlas'] *= self.model_strengths['atlas']['geometric']
        if characteristics['grid_size'] > 15:
            weights['atlas'] *= self.model_strengths['atlas']['large_grids']
        if characteristics['has_symmetry'] > 0.5:
            weights['atlas'] *= self.model_strengths['atlas']['symmetry']
        
        # Apply IRIS bonuses
        if characteristics['color_mapping'] > 0.5:
            weights['iris'] *= self.model_strengths['iris']['color_mapping']
        if characteristics['num_colors'] > 5:
            weights['iris'] *= self.model_strengths['iris']['color_density']
        elif characteristics['num_colors'] <= 3:
            weights['iris'] *= 0.8  # Penalty for too few colors
            
        # Apply CHRONOS bonuses  
        if characteristics['is_sequential'] > 0.3:
            weights['chronos'] *= self.model_strengths['chronos']['sequential']
        if characteristics['has_repetition'] > 0.5:
            weights['chronos'] *= self.model_strengths['chronos']['repetition']
        if characteristics['grid_size'] <= 10 and characteristics['num_colors'] <= 3:
            weights['chronos'] *= self.model_strengths['chronos']['small_simple']
            
        # Apply PROMETHEUS bonuses
        if characteristics['grid_size'] <= 7:
            weights['prometheus'] *= self.model_strengths['prometheus']['very_small']
        if characteristics.get('object_complexity', 0) < 0.2:
            weights['prometheus'] *= self.model_strengths['prometheus']['low_structure']
        
        # Normalize weights to sum to 5.0 (average 1.0 per model)
        total = sum(weights.values())
        if total > 0:
            factor = 5.0 / total
            for model in weights:
                weights[model] *= factor
        
        return weights
    
    def explain_routing(self, characteristics: Dict[str, float], 
                       weights: Dict[str, float]) -> str:
        """Generate explanation for weight assignment"""
        explanation = []
        
        # Find dominant characteristics
        if characteristics['grid_size'] > 15:
            explanation.append("Large grid → favoring ATLAS")
        if characteristics['num_colors'] > 5:
            explanation.append("Many colors → favoring IRIS")
        if characteristics['is_sequential'] > 0.3:
            explanation.append("Sequential pattern → favoring CHRONOS")
        if characteristics['num_objects'] > 3:
            explanation.append("Multiple objects → favoring MINERVA")
        if characteristics['grid_size'] <= 7:
            explanation.append("Very small grid → favoring PROMETHEUS")
        
        # Report final weights
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        weight_str = ", ".join([f"{m.upper()}: {w:.2f}" for m, w in sorted_weights])
        explanation.append(f"Final weights: {weight_str}")
        
        return " | ".join(explanation)


class SmartEnsemble:
    """Enhanced ensemble with task routing"""
    
    def __init__(self, ensemble, task_router: TaskRouter):
        self.ensemble = ensemble
        self.router = task_router
    
    def predict(self, input_grid: np.ndarray, train_examples: List[Dict], 
                verbose: bool = True) -> Dict:
        """Make prediction using smart routing"""
        # Get optimized weights
        weights = self.router.get_model_weights(input_grid, train_examples)
        
        if verbose:
            characteristics = self.router.analyzer.analyze_task(input_grid, train_examples)
            explanation = self.router.explain_routing(characteristics, weights)
            print(f"\n🧭 Task Routing: {explanation}")
        
        # Get all model predictions with correct shape
        if hasattr(self.ensemble, 'predict_all_models_with_shape'):
            predictions = self.ensemble.predict_all_models_with_shape(input_grid, train_examples)
        else:
            predictions = self.ensemble.predict_all_models(input_grid)
        
        # Apply weighted voting with task-specific weights
        grid_scores = {}
        
        for model_name, pred_grid in predictions.items():
            if pred_grid is None:
                continue
                
            grid_str = pred_grid.tobytes()
            if grid_str not in grid_scores:
                grid_scores[grid_str] = {
                    'score': 0.0, 
                    'voters': [], 
                    'grid': pred_grid
                }
            
            model_weight = weights[model_name]
            grid_scores[grid_str]['score'] += model_weight
            grid_scores[grid_str]['voters'].append((model_name, model_weight))
        
        # Find best scoring grid
        best_grid = None
        best_score = 0.0
        vote_details = {}
        
        for grid_str, info in grid_scores.items():
            if info['score'] > best_score:
                best_score = info['score']
                best_grid = info['grid']
                vote_details = {name: weight for name, weight in info['voters']}
        
        if verbose:
            print(f"\n⚖️ Smart Vote Result: Score {best_score:.2f}")
            for model, weight in sorted(vote_details.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {model}: {weight:.2f}")
        
        return {
            'prediction': best_grid,
            'score': best_score,
            'weights': weights,
            'vote_details': vote_details,
            'all_predictions': predictions
        }


if __name__ == "__main__":
    """Test the task router"""
    print("🧭 Testing Task Router")
    print("="*50)
    
    # Create test scenarios
    test_cases = [
        {
            'name': 'Large Geometric',
            'input': np.random.randint(0, 3, (25, 25)),
            'examples': [{'input': np.zeros((25, 25)), 'output': np.ones((25, 25))}]
        },
        {
            'name': 'Small Colorful',
            'input': np.random.randint(0, 8, (7, 7)),
            'examples': [{'input': np.zeros((7, 7)), 'output': np.ones((7, 7))}]
        }
    ]
    
    router = TaskRouter()
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Grid size: {test['input'].shape}")
        print(f"Colors: {len(np.unique(test['input']))}")
        
        weights = router.get_model_weights(test['input'], test['examples'])
        
        print("Weights:")
        for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {weight:.2f}")