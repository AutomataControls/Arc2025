#!/usr/bin/env python3
"""
================================================================================
ARC Prize 2025 - Pattern Detectors
================================================================================
Individual pattern detection modules for ARC task analysis

This is OPEN SOURCE software - no commercial license restrictions
Released under MIT License for the ARC Prize 2025 competition

Author: Andrew Jewell Sr.
Company: AutomataNexus, LLC
Date: September 26, 2024
Version: 1.0.0

Description:
    This module implements 8 specialized pattern detectors that analyze
    ARC tasks to identify common transformation types. These detectors
    are used during offline pre-computation on the Hailo-8 NPU.
    
    Each detector is optimized to identify specific pattern types found
    in the ARC dataset through our data exploration.
================================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
import itertools


class PatternDetector:
    """Base class for all pattern detectors"""
    
    def __init__(self, name: str):
        self.name = name
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """
        Analyze training examples to detect patterns
        
        Args:
            train_examples: List of training input/output pairs
            
        Returns:
            Dictionary containing detected pattern information
        """
        raise NotImplementedError


class GeometricDetector(PatternDetector):
    """Detects geometric transformations: rotations, reflections, translations"""
    
    def __init__(self):
        super().__init__("geometric")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'transformations': [],
            'confidence': 0.0
        }
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Check rotations
            for k in [1, 2, 3]:
                if input_grid.shape == output_grid.shape:
                    rotated = np.rot90(input_grid, k=k)
                    if np.array_equal(rotated, output_grid):
                        patterns['transformations'].append({
                            'type': 'rotation',
                            'angle': k * 90
                        })
                        patterns['confidence'] = 1.0
                        return patterns
            
            # Check reflections
            if input_grid.shape == output_grid.shape:
                # Horizontal flip
                if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                    patterns['transformations'].append({
                        'type': 'reflection',
                        'axis': 'horizontal'
                    })
                    patterns['confidence'] = 1.0
                    return patterns
                
                # Vertical flip
                if np.array_equal(np.flip(input_grid, axis=1), output_grid):
                    patterns['transformations'].append({
                        'type': 'reflection',
                        'axis': 'vertical'
                    })
                    patterns['confidence'] = 1.0
                    return patterns
            
            # Check translations
            if input_grid.shape == output_grid.shape:
                translation = self._detect_translation(input_grid, output_grid)
                if translation:
                    patterns['transformations'].append({
                        'type': 'translation',
                        'offset': translation
                    })
                    patterns['confidence'] = 1.0
                    return patterns
        
        return patterns
    
    def _detect_translation(self, grid1: np.ndarray, grid2: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect if grid2 is a translation of grid1"""
        h, w = grid1.shape
        
        # Try different offsets
        for dy in range(-h+1, h):
            for dx in range(-w+1, w):
                translated = np.zeros_like(grid1)
                
                # Apply translation
                src_y1 = max(0, -dy)
                src_y2 = min(h, h - dy)
                src_x1 = max(0, -dx)
                src_x2 = min(w, w - dx)
                
                dst_y1 = max(0, dy)
                dst_y2 = min(h, h + dy)
                dst_x1 = max(0, dx)
                dst_x2 = min(w, w + dx)
                
                if src_y2 > src_y1 and src_x2 > src_x1:
                    translated[dst_y1:dst_y2, dst_x1:dst_x2] = grid1[src_y1:src_y2, src_x1:src_x2]
                
                if np.array_equal(translated, grid2):
                    return (dy, dx)
        
        return None


class ColorDetector(PatternDetector):
    """Detects color mapping and transformation patterns"""
    
    def __init__(self):
        super().__init__("color")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'color_map': {},
            'consistency': 0.0,
            'type': 'none'
        }
        
        # Collect all color mappings
        all_mappings = []
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            if input_grid.shape == output_grid.shape:
                mapping = self._extract_color_mapping(input_grid, output_grid)
                if mapping:
                    all_mappings.append(mapping)
        
        if all_mappings:
            # Check if mappings are consistent
            consistent_map = self._find_consistent_mapping(all_mappings)
            if consistent_map:
                patterns['color_map'] = consistent_map
                patterns['consistency'] = 1.0
                patterns['type'] = 'direct_mapping'
            else:
                # Check for conditional mappings
                conditional = self._detect_conditional_mapping(train_examples)
                if conditional:
                    patterns.update(conditional)
        
        return patterns
    
    def _extract_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
        """Extract color mapping between grids of same size"""
        mapping = {}
        
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = int(input_grid[i, j])
                out_color = int(output_grid[i, j])
                
                if in_color in mapping and mapping[in_color] != out_color:
                    return None  # Inconsistent mapping
                
                mapping[in_color] = out_color
        
        return mapping
    
    def _find_consistent_mapping(self, mappings: List[Dict[int, int]]) -> Optional[Dict[int, int]]:
        """Find consistent color mapping across examples"""
        if not mappings:
            return None
        
        # Start with first mapping
        consistent = mappings[0].copy()
        
        # Check consistency with other mappings
        for mapping in mappings[1:]:
            for color, target in mapping.items():
                if color in consistent and consistent[color] != target:
                    return None  # Inconsistent
                consistent[color] = target
        
        return consistent
    
    def _detect_conditional_mapping(self, train_examples: List[Dict]) -> Optional[Dict[str, Any]]:
        """Detect conditional color mappings based on context"""
        # This would implement more complex conditional logic
        # For now, return None
        return None


class CountingDetector(PatternDetector):
    """Detects patterns involving counting, enumeration, or numerical relationships"""
    
    def __init__(self):
        super().__init__("counting")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'type': 'none',
            'rule': {},
            'confidence': 0.0
        }
        
        # Check various counting patterns
        size_pattern = self._detect_size_counting(train_examples)
        if size_pattern['confidence'] > 0.8:
            return size_pattern
        
        object_pattern = self._detect_object_counting(train_examples)
        if object_pattern['confidence'] > 0.8:
            return object_pattern
        
        color_pattern = self._detect_color_counting(train_examples)
        if color_pattern['confidence'] > 0.8:
            return color_pattern
        
        return patterns
    
    def _detect_size_counting(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect if output size relates to counting input features"""
        pattern = {
            'type': 'size_based',
            'rule': {},
            'confidence': 0.0
        }
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Count non-zero elements
            non_zero_count = np.count_nonzero(input_grid)
            
            # Check if output dimensions relate to count
            if output_grid.shape[0] == non_zero_count or output_grid.shape[1] == non_zero_count:
                pattern['rule']['dimension'] = 'non_zero_count'
                pattern['confidence'] = 1.0
                return pattern
        
        return pattern
    
    def _detect_object_counting(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect counting of connected components or objects"""
        pattern = {
            'type': 'object_count',
            'rule': {},
            'confidence': 0.0
        }
        
        # Implementation would use connected component analysis
        # Placeholder for now
        return pattern
    
    def _detect_color_counting(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect patterns based on counting colors"""
        pattern = {
            'type': 'color_count',
            'rule': {},
            'confidence': 0.0
        }
        
        counts = []
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Count unique colors
            input_colors = len(np.unique(input_grid))
            output_colors = len(np.unique(output_grid))
            
            counts.append((input_colors, output_colors))
        
        # Check for consistent relationship
        if len(set(counts)) == 1:
            pattern['rule']['relationship'] = counts[0]
            pattern['confidence'] = 1.0
        
        return pattern


class LogicalDetector(PatternDetector):
    """Detects logical operations: AND, OR, XOR, conditionals"""
    
    def __init__(self):
        super().__init__("logical")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'operation': 'none',
            'conditions': [],
            'confidence': 0.0
        }
        
        # Check for boolean operations on colors
        bool_op = self._detect_boolean_operations(train_examples)
        if bool_op['confidence'] > 0.8:
            return bool_op
        
        # Check for conditional transformations
        conditional = self._detect_conditional_rules(train_examples)
        if conditional['confidence'] > 0.8:
            return conditional
        
        return patterns
    
    def _detect_boolean_operations(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect if output is result of boolean operations on input"""
        pattern = {
            'operation': 'none',
            'confidence': 0.0
        }
        
        # Check if all examples have exactly 2 training examples (common for boolean ops)
        if len(train_examples) >= 2:
            # Get first two inputs as potential operands
            grid1 = np.array(train_examples[0]['input'])
            grid2 = np.array(train_examples[1]['input']) if len(train_examples) > 1 else None
            output = np.array(train_examples[0]['output'])
            
            if grid2 is not None and grid1.shape == grid2.shape == output.shape:
                # Check AND operation
                if np.array_equal((grid1 > 0) & (grid2 > 0), output > 0):
                    pattern['operation'] = 'AND'
                    pattern['confidence'] = 1.0
                # Check OR operation  
                elif np.array_equal((grid1 > 0) | (grid2 > 0), output > 0):
                    pattern['operation'] = 'OR'
                    pattern['confidence'] = 1.0
                # Check XOR operation
                elif np.array_equal((grid1 > 0) ^ (grid2 > 0), output > 0):
                    pattern['operation'] = 'XOR'
                    pattern['confidence'] = 1.0
        
        return pattern
    
    def _detect_conditional_rules(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect conditional transformation rules"""
        pattern = {
            'operation': 'conditional',
            'conditions': [],
            'confidence': 0.0
        }
        
        # Placeholder for more complex conditional detection
        return pattern


class SpatialDetector(PatternDetector):
    """Detects spatial relationships and positional patterns"""
    
    def __init__(self):
        super().__init__("spatial")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'type': 'none',
            'relationships': [],
            'confidence': 0.0
        }
        
        # Check for gravity/falling patterns
        gravity = self._detect_gravity_pattern(train_examples)
        if gravity['confidence'] > 0.8:
            return gravity
        
        # Check for alignment patterns
        alignment = self._detect_alignment_pattern(train_examples)
        if alignment['confidence'] > 0.8:
            return alignment
        
        # Check for boundary patterns
        boundary = self._detect_boundary_pattern(train_examples)
        if boundary['confidence'] > 0.8:
            return boundary
        
        return patterns
    
    def _detect_gravity_pattern(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect if objects fall to bottom"""
        pattern = {
            'type': 'gravity',
            'direction': 'down',
            'confidence': 0.0
        }
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Check if non-zero elements moved down
            if self._check_gravity_movement(input_grid, output_grid):
                pattern['confidence'] = 1.0
        
        return pattern
    
    def _check_gravity_movement(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if objects moved down due to gravity"""
        h, w = input_grid.shape
        
        for j in range(w):
            input_col = input_grid[:, j]
            output_col = output_grid[:, j]
            
            # Get non-zero positions
            input_nonzero = np.where(input_col > 0)[0]
            output_nonzero = np.where(output_col > 0)[0]
            
            if len(input_nonzero) != len(output_nonzero):
                continue
            
            # Check if all moved down or stayed
            for i, pos in enumerate(input_nonzero):
                if output_nonzero[i] < pos:
                    return False  # Moved up
        
        return True
    
    def _detect_alignment_pattern(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect alignment patterns (left, right, center)"""
        pattern = {
            'type': 'alignment',
            'direction': 'none',
            'confidence': 0.0
        }
        
        # Placeholder implementation
        return pattern
    
    def _detect_boundary_pattern(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect patterns involving boundaries or frames"""
        pattern = {
            'type': 'boundary',
            'style': 'none',
            'confidence': 0.0
        }
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Check if output adds a frame
            if self._has_frame(output_grid) and not self._has_frame(input_grid):
                pattern['style'] = 'add_frame'
                pattern['confidence'] = 1.0
                return pattern
        
        return pattern
    
    def _has_frame(self, grid: np.ndarray) -> bool:
        """Check if grid has a frame of non-zero values"""
        if grid.size == 0:
            return False
        
        # Check if border is all non-zero
        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[:, 0]
        right = grid[:, -1]
        
        return (np.all(top > 0) and np.all(bottom > 0) and 
                np.all(left > 0) and np.all(right > 0))


class SymmetryDetector(PatternDetector):
    """Detects symmetry patterns and symmetry-based transformations"""
    
    def __init__(self):
        super().__init__("symmetry")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'type': 'none',
            'axis': [],
            'confidence': 0.0
        }
        
        # Check if outputs are made symmetric
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Check various symmetry types
            symmetries = self._detect_symmetries(output_grid)
            
            if symmetries and not self._detect_symmetries(input_grid):
                # Output is symmetric but input isn't
                patterns['type'] = 'make_symmetric'
                patterns['axis'] = symmetries
                patterns['confidence'] = 1.0
                return patterns
            
            # Check if symmetry is completed
            completion = self._detect_symmetry_completion(input_grid, output_grid)
            if completion['confidence'] > 0.8:
                return completion
        
        return patterns
    
    def _detect_symmetries(self, grid: np.ndarray) -> List[str]:
        """Detect symmetries in a grid"""
        symmetries = []
        
        # Horizontal symmetry
        if np.array_equal(grid, np.flip(grid, axis=0)):
            symmetries.append('horizontal')
        
        # Vertical symmetry
        if np.array_equal(grid, np.flip(grid, axis=1)):
            symmetries.append('vertical')
        
        # Diagonal symmetry
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, grid.T):
                symmetries.append('diagonal')
        
        return symmetries
    
    def _detect_symmetry_completion(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Detect if output completes input's symmetry"""
        pattern = {
            'type': 'complete_symmetry',
            'axis': 'none',
            'confidence': 0.0
        }
        
        # Check if output is input + its reflection
        h_complete = np.vstack([input_grid, np.flip(input_grid, axis=0)])
        if output_grid.shape == h_complete.shape and np.array_equal(output_grid, h_complete):
            pattern['axis'] = 'horizontal'
            pattern['confidence'] = 1.0
            return pattern
        
        v_complete = np.hstack([input_grid, np.flip(input_grid, axis=1)])
        if output_grid.shape == v_complete.shape and np.array_equal(output_grid, v_complete):
            pattern['axis'] = 'vertical'
            pattern['confidence'] = 1.0
            return pattern
        
        return pattern


class ObjectDetector(PatternDetector):
    """Detects object-based patterns: movement, manipulation, extraction"""
    
    def __init__(self):
        super().__init__("object")
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'type': 'none',
            'operation': {},
            'confidence': 0.0
        }
        
        # Detect object extraction
        extraction = self._detect_object_extraction(train_examples)
        if extraction['confidence'] > 0.8:
            return extraction
        
        # Detect object movement
        movement = self._detect_object_movement(train_examples)
        if movement['confidence'] > 0.8:
            return movement
        
        # Detect object combination
        combination = self._detect_object_combination(train_examples)
        if combination['confidence'] > 0.8:
            return combination
        
        return patterns
    
    def _detect_object_extraction(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect if output extracts specific objects from input"""
        pattern = {
            'type': 'extraction',
            'criteria': {},
            'confidence': 0.0
        }
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Check if output is subset of input
            input_objects = self._extract_objects(input_grid)
            output_objects = self._extract_objects(output_grid)
            
            if output_objects and all(self._object_exists_in(obj, input_objects) for obj in output_objects):
                pattern['confidence'] = 1.0
                # Further analysis would determine extraction criteria
        
        return pattern
    
    def _detect_object_movement(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect if objects move according to rules"""
        pattern = {
            'type': 'movement',
            'rule': {},
            'confidence': 0.0
        }
        
        # Placeholder for object tracking and movement detection
        return pattern
    
    def _detect_object_combination(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """Detect if objects are combined or merged"""
        pattern = {
            'type': 'combination',
            'method': 'none',
            'confidence': 0.0
        }
        
        # Placeholder for object combination detection
        return pattern
    
    def _extract_objects(self, grid: np.ndarray) -> List[np.ndarray]:
        """Extract connected components as objects"""
        objects = []
        # Simplified - would use proper connected component analysis
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color != 0:  # Skip background
                mask = (grid == color)
                objects.append(mask)
        return objects
    
    def _object_exists_in(self, obj: np.ndarray, object_list: List[np.ndarray]) -> bool:
        """Check if object exists in list"""
        for other in object_list:
            if obj.shape == other.shape and np.array_equal(obj, other):
                return True
        return False


class CompositeDetector(PatternDetector):
    """Detects complex patterns combining multiple transformations"""
    
    def __init__(self):
        super().__init__("composite")
        
        # Initialize sub-detectors
        self.detectors = {
            'geometric': GeometricDetector(),
            'color': ColorDetector(),
            'counting': CountingDetector(),
            'logical': LogicalDetector(),
            'spatial': SpatialDetector(),
            'symmetry': SymmetryDetector(),
            'object': ObjectDetector()
        }
    
    def detect(self, train_examples: List[Dict]) -> Dict[str, Any]:
        patterns = {
            'type': 'composite',
            'sequence': [],
            'confidence': 0.0
        }
        
        # Try to decompose transformation into steps
        sequence = self._detect_transformation_sequence(train_examples)
        if sequence:
            patterns['sequence'] = sequence
            patterns['confidence'] = 1.0
        
        return patterns
    
    def _detect_transformation_sequence(self, train_examples: List[Dict]) -> List[Dict[str, Any]]:
        """Detect sequence of transformations"""
        sequence = []
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Try to find intermediate steps
            current_grid = input_grid.copy()
            steps = []
            
            # Try each detector to see if it explains part of transformation
            for name, detector in self.detectors.items():
                # Create fake example with current state
                fake_ex = [{'input': current_grid, 'output': output_grid}]
                result = detector.detect(fake_ex)
                
                if result.get('confidence', 0) > 0.8:
                    steps.append({
                        'detector': name,
                        'pattern': result
                    })
                    
                    # Apply transformation to get next state
                    # This is simplified - would need actual transformation application
                    break
            
            if steps:
                sequence.extend(steps)
        
        return sequence if sequence else None


def create_all_detectors() -> Dict[str, PatternDetector]:
    """Create instances of all pattern detectors"""
    return {
        'geometric': GeometricDetector(),
        'color': ColorDetector(),
        'counting': CountingDetector(),
        'logical': LogicalDetector(),
        'spatial': SpatialDetector(),
        'symmetry': SymmetryDetector(),
        'object': ObjectDetector(),
        'composite': CompositeDetector()
    }


def analyze_task_with_all_detectors(train_examples: List[Dict]) -> Dict[str, Any]:
    """Analyze a task using all detectors"""
    detectors = create_all_detectors()
    results = {}
    
    for name, detector in detectors.items():
        results[name] = detector.detect(train_examples)
    
    return results


if __name__ == "__main__":
    # Test with a simple example
    test_examples = [
        {
            'input': [[1, 0], [0, 1]],
            'output': [[0, 1], [1, 0]]
        }
    ]
    
    results = analyze_task_with_all_detectors(test_examples)
    
    print("Pattern Detection Results:")
    for detector_name, result in results.items():
        if result.get('confidence', 0) > 0:
            print(f"\n{detector_name.upper()} Detector:")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Pattern: {result}")