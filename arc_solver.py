#!/usr/bin/env python3
"""
================================================================================
ARC Prize 2025 - Main Solver Implementation
================================================================================
Core solver that runs on Kaggle platform during evaluation

This is OPEN SOURCE software - no commercial license restrictions
Released under MIT License for the ARC Prize 2025 competition

Author: Andrew Jewell Sr.
Company: AutomataNexus, LLC
Date: September 26, 2024
Version: 1.0.0

Description:
    Main ARC solver implementation that loads pre-computed patterns and applies
    them to solve test tasks within Kaggle's 12-hour runtime limit.
    
    This solver implements a multi-strategy approach:
    - Pattern matching using pre-computed library
    - Geometric transformation detection
    - Object-based reasoning
    - Color mapping inference
    - Composite rule application
    
    The solver makes two attempts per test input as required by competition rules.
    Attempt 1: Best matching pattern from library
    Attempt 2: Alternative strategy or variation
================================================================================
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARCSolver:
    """Main solver class that coordinates pattern matching and solution generation"""
    
    def __init__(self, pattern_library_path: str = 'precomputed_patterns.pkl'):
        """Initialize solver with pre-computed pattern library"""
        logger.info("Initializing ARC Solver")
        
        # Load pre-computed patterns
        self.pattern_library = None
        self.task_analyses = None
        self.transformation_cache = None
        
        if Path(pattern_library_path).exists():
            self._load_pattern_library(pattern_library_path)
        else:
            logger.warning(f"Pattern library not found at {pattern_library_path}")
            logger.warning("Solver will use basic strategies only")
        
        # Initialize solution strategies
        self.strategies = {
            'pattern_match': PatternMatchStrategy(self.pattern_library),
            'geometric': GeometricStrategy(),
            'color_map': ColorMapStrategy(),
            'object_based': ObjectBasedStrategy(),
            'symmetry': SymmetryStrategy(),
            'size_based': SizeBasedStrategy(),
            'composite': CompositeStrategy()
        }
        
        logger.info("ARC Solver initialized successfully")
    
    def _load_pattern_library(self, path: str) -> None:
        """Load pre-computed pattern library from pickle file"""
        logger.info(f"Loading pattern library from {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.pattern_library = data.get('patterns', {})
            self.task_analyses = data.get('task_analyses', {})
            self.transformation_cache = data.get('transformation_cache', {})
        
        logger.info(f"Loaded {len(self.pattern_library)} patterns")
        logger.info(f"Loaded analyses for {len(self.task_analyses)} tasks")
    
    def solve(self, task: Dict) -> List[List[List[int]]]:
        """
        Solve an ARC task by making two attempts per test input
        
        Args:
            task: Dictionary containing 'train' and 'test' examples
            
        Returns:
            List of predictions (2 attempts per test input)
        """
        logger.info("Solving new task")
        
        # Extract task features
        task_features = self._extract_task_features(task)
        
        # Find similar patterns in library
        similar_patterns = self._find_similar_patterns(task_features)
        
        # Generate predictions for each test input
        predictions = []
        
        for test_idx, test_input in enumerate(task.get('test', [])):
            logger.info(f"Solving test input {test_idx}")
            
            # Attempt 1: Best matching strategy
            attempt1 = self._make_attempt(task, test_input, similar_patterns, primary=True)
            
            # Attempt 2: Alternative strategy
            attempt2 = self._make_attempt(task, test_input, similar_patterns, primary=False)
            
            predictions.append({
                'attempt_1': attempt1,
                'attempt_2': attempt2
            })
        
        return predictions
    
    def _extract_task_features(self, task: Dict) -> np.ndarray:
        """Extract feature vector from task for similarity matching"""
        features = []
        
        train_examples = task.get('train', [])
        if not train_examples:
            return np.array([])
        
        # Aggregate features across training examples
        for ex in train_examples:
            input_grid = np.array(ex.get('input', []))
            output_grid = np.array(ex.get('output', []))
            
            ex_features = [
                # Size features
                input_grid.shape[0] if input_grid.size > 0 else 0,
                input_grid.shape[1] if input_grid.size > 0 else 0,
                output_grid.shape[0] if output_grid.size > 0 else 0,
                output_grid.shape[1] if output_grid.size > 0 else 0,
                
                # Color features
                len(np.unique(input_grid)) if input_grid.size > 0 else 0,
                len(np.unique(output_grid)) if output_grid.size > 0 else 0,
                
                # Density features
                np.sum(input_grid > 0) / input_grid.size if input_grid.size > 0 else 0,
                np.sum(output_grid > 0) / output_grid.size if output_grid.size > 0 else 0
            ]
            
            features.append(ex_features)
        
        # Average features across examples
        return np.mean(features, axis=0)
    
    def _find_similar_patterns(self, task_features: np.ndarray) -> List[Dict]:
        """Find similar patterns in the pre-computed library"""
        if not self.task_analyses or task_features.size == 0:
            return []
        
        similarities = []
        
        # Compare with each analyzed task
        for task_id, analysis in self.task_analyses.items():
            stored_features = analysis.get('features', np.array([]))
            
            if stored_features.size > 0 and stored_features.shape[-1] == task_features.shape[-1]:
                # Compute cosine similarity
                similarity = self._cosine_similarity(task_features, stored_features.mean(axis=0))
                similarities.append({
                    'task_id': task_id,
                    'similarity': similarity,
                    'patterns': analysis.get('patterns', {}),
                    'transformations': analysis.get('transformations', {})
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:10]  # Return top 10 similar patterns
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _make_attempt(self, task: Dict, test_input: Dict, 
                     similar_patterns: List[Dict], primary: bool) -> List[List[int]]:
        """Make a single attempt to solve the test input"""
        input_grid = np.array(test_input.get('input', []))
        
        if input_grid.size == 0:
            return [[0]]
        
        # Choose strategy based on similar patterns and whether it's primary attempt
        if primary and similar_patterns:
            # Try pattern matching first
            strategy = self.strategies['pattern_match']
            result = strategy.solve(task, input_grid, similar_patterns)
            if result is not None:
                return result.tolist()
        
        # Try different strategies
        strategy_order = ['geometric', 'color_map', 'object_based', 
                         'symmetry', 'size_based', 'composite']
        
        if not primary:
            # Use different order for second attempt
            strategy_order = strategy_order[1:] + strategy_order[:1]
        
        for strategy_name in strategy_order:
            strategy = self.strategies.get(strategy_name)
            if strategy:
                result = strategy.solve(task, input_grid, similar_patterns)
                if result is not None:
                    return result.tolist()
        
        # Default: return input unchanged
        return input_grid.tolist()


class SolverStrategy:
    """Base class for solving strategies"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """
        Attempt to solve using this strategy
        
        Returns:
            Solution grid or None if strategy doesn't apply
        """
        raise NotImplementedError


class PatternMatchStrategy(SolverStrategy):
    """Strategy based on matching pre-computed patterns"""
    
    def __init__(self, pattern_library: Dict):
        self.pattern_library = pattern_library or {}
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Apply best matching pattern from library"""
        if not similar_patterns:
            return None
        
        # Try patterns from most similar tasks
        for similar in similar_patterns[:3]:
            transformations = similar.get('transformations', {})
            
            # Try size change
            size_change = transformations.get('size_change', {})
            if size_change.get('is_consistent') and size_change.get('rule'):
                scale_y, scale_x = size_change['rule']
                new_shape = (int(test_input.shape[0] * scale_y), 
                           int(test_input.shape[1] * scale_x))
                
                if all(s > 0 and s <= 30 for s in new_shape):
                    # Apply scaling
                    result = np.zeros(new_shape, dtype=int)
                    for i in range(new_shape[0]):
                        for j in range(new_shape[1]):
                            orig_i = int(i / scale_y)
                            orig_j = int(j / scale_x)
                            if orig_i < test_input.shape[0] and orig_j < test_input.shape[1]:
                                result[i, j] = test_input[orig_i, orig_j]
                    return result
            
            # Try color mapping
            color_map = transformations.get('color_mapping', {})
            if color_map.get('is_simple_mapping'):
                mapping = color_map.get('consistent_map', {})
                if mapping:
                    result = test_input.copy()
                    for old_color, new_color in mapping.items():
                        result[test_input == old_color] = new_color
                    return result
        
        return None


class GeometricStrategy(SolverStrategy):
    """Strategy for geometric transformations"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Try geometric transformations"""
        train_examples = task.get('train', [])
        
        if not train_examples:
            return None
        
        # Check if all training examples show same geometric transformation
        transformations = []
        
        for ex in train_examples:
            input_grid = np.array(ex.get('input', []))
            output_grid = np.array(ex.get('output', []))
            
            # Check rotations
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(input_grid, k=k), output_grid):
                    transformations.append(('rotate', k))
                    break
            
            # Check flips
            if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                transformations.append(('flip_h', None))
            elif np.array_equal(np.flip(input_grid, axis=1), output_grid):
                transformations.append(('flip_v', None))
        
        # If all examples have same transformation, apply it
        if transformations and all(t == transformations[0] for t in transformations):
            transform_type, param = transformations[0]
            
            if transform_type == 'rotate':
                return np.rot90(test_input, k=param)
            elif transform_type == 'flip_h':
                return np.flip(test_input, axis=0)
            elif transform_type == 'flip_v':
                return np.flip(test_input, axis=1)
        
        return None


class ColorMapStrategy(SolverStrategy):
    """Strategy for color mapping transformations"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Try color mapping transformations"""
        train_examples = task.get('train', [])
        
        if not train_examples:
            return None
        
        # Build color mapping from training examples
        color_mappings = []
        
        for ex in train_examples:
            input_grid = np.array(ex.get('input', []))
            output_grid = np.array(ex.get('output', []))
            
            if input_grid.shape != output_grid.shape:
                continue
            
            # Build mapping for this example
            mapping = {}
            for color in np.unique(input_grid):
                mask = (input_grid == color)
                output_colors = output_grid[mask]
                if len(np.unique(output_colors)) == 1:
                    mapping[int(color)] = int(output_colors[0])
            
            color_mappings.append(mapping)
        
        # Check if mapping is consistent across examples
        if color_mappings and all(m == color_mappings[0] for m in color_mappings):
            # Apply mapping
            result = test_input.copy()
            for old_color, new_color in color_mappings[0].items():
                result[test_input == old_color] = new_color
            return result
        
        return None


class ObjectBasedStrategy(SolverStrategy):
    """Strategy based on object detection and manipulation"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Try object-based transformations"""
        # This is a placeholder for object-based reasoning
        # In a full implementation, this would:
        # 1. Detect objects in input
        # 2. Learn transformation rules from training examples
        # 3. Apply rules to test input
        return None


class SymmetryStrategy(SolverStrategy):
    """Strategy for symmetry-based transformations"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Try symmetry-based transformations"""
        train_examples = task.get('train', [])
        
        if not train_examples:
            return None
        
        # Check if outputs are symmetrical versions of inputs
        for ex in train_examples:
            input_grid = np.array(ex.get('input', []))
            output_grid = np.array(ex.get('output', []))
            
            # Check if output is horizontally symmetric version
            if output_grid.shape[0] == input_grid.shape[0] and output_grid.shape[1] == input_grid.shape[1] * 2:
                left_half = output_grid[:, :input_grid.shape[1]]
                right_half = output_grid[:, input_grid.shape[1]:]
                
                if np.array_equal(left_half, input_grid) and np.array_equal(right_half, np.flip(input_grid, axis=1)):
                    # Create horizontally symmetric output
                    return np.hstack([test_input, np.flip(test_input, axis=1)])
        
        return None


class SizeBasedStrategy(SolverStrategy):
    """Strategy based on size transformations"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Try size-based transformations"""
        train_examples = task.get('train', [])
        
        if not train_examples:
            return None
        
        # Check for consistent size changes
        size_changes = []
        
        for ex in train_examples:
            input_grid = np.array(ex.get('input', []))
            output_grid = np.array(ex.get('output', []))
            
            size_change = (
                output_grid.shape[0] - input_grid.shape[0],
                output_grid.shape[1] - input_grid.shape[1]
            )
            size_changes.append(size_change)
        
        # If consistent size change, apply it
        if size_changes and all(sc == size_changes[0] for sc in size_changes):
            dy, dx = size_changes[0]
            new_shape = (test_input.shape[0] + dy, test_input.shape[1] + dx)
            
            if all(s > 0 and s <= 30 for s in new_shape):
                # Create output with new size
                result = np.zeros(new_shape, dtype=int)
                # Copy input to result (top-left aligned)
                result[:test_input.shape[0], :test_input.shape[1]] = test_input
                return result
        
        return None


class CompositeStrategy(SolverStrategy):
    """Strategy for composite/complex transformations"""
    
    def solve(self, task: Dict, test_input: np.ndarray, 
             similar_patterns: List[Dict]) -> Optional[np.ndarray]:
        """Try composite transformations combining multiple rules"""
        # This is a placeholder for complex multi-step transformations
        # In a full implementation, this would:
        # 1. Detect multi-step patterns in training examples
        # 2. Learn the sequence of transformations
        # 3. Apply the sequence to test input
        return None


def create_submission(challenges: Dict, output_path: str = 'submission.json') -> None:
    """
    Create submission file for ARC Prize
    
    Args:
        challenges: Dictionary of test challenges
        output_path: Path to save submission JSON
    """
    logger.info("Creating submission")
    
    # Initialize solver
    solver = ARCSolver()
    
    # Process each task
    submission = {}
    
    for task_id, task in challenges.items():
        logger.info(f"Processing task {task_id}")
        
        # Solve task
        predictions = solver.solve(task)
        
        # Format predictions for submission
        submission[task_id] = predictions
    
    # Save submission
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    logger.info(f"Submission saved to {output_path}")


def main():
    """Main execution function for testing"""
    # This would be called with actual test data during competition
    logger.info("ARC Solver ready for evaluation")
    
    # Example usage (would be replaced with actual test data)
    # test_challenges = load_test_challenges()
    # create_submission(test_challenges)


if __name__ == "__main__":
    main()