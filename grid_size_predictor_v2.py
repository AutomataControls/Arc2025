#!/usr/bin/env python3
"""
Grid Size Predictor V2 - Advanced Shape Analysis
Implements object-based and content-aware rules for accurate output shape prediction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy import ndimage
from collections import Counter


class GridSizePredictorV2:
    """Advanced predictor that analyzes objects and content to determine output shape"""
    
    def __init__(self):
        self.debug = False
    
    def predict_output_shape(self, input_grid: np.ndarray, 
                           train_examples: List[Dict]) -> Tuple[int, int]:
        """
        Predict output shape using sophisticated rule detection
        """
        # Try multiple prediction strategies in order of sophistication
        strategies = [
            self._try_exact_match_rule,
            self._try_consistent_size_rule,
            self._try_scaling_rule,
            self._try_fractional_scaling_rule,
            self._try_object_based_rule,
            self._try_cropping_rule,
            self._try_color_count_rule,
            self._try_pattern_based_rule,
            self._try_single_dimension_rule,
            self._try_density_based_rule,
            self._try_extreme_reduction_rule,
            self._try_median_fallback
        ]
        
        for strategy in strategies:
            result = strategy(input_grid, train_examples)
            if result is not None:
                if self.debug:
                    print(f"  Shape predicted by: {strategy.__name__} -> {result}")
                return result
        
        # Ultimate fallback: same as input
        return input_grid.shape
    
    def _try_exact_match_rule(self, input_grid: np.ndarray, 
                            train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if all outputs have the exact same shape"""
        output_shapes = []
        for ex in train_examples:
            output = np.array(ex['output'])
            output_shapes.append(output.shape)
        
        # If all outputs are the same shape, use that
        if len(set(output_shapes)) == 1:
            return output_shapes[0]
        
        return None
    
    def _try_consistent_size_rule(self, input_grid: np.ndarray,
                                 train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for consistent input->output size relationships"""
        relationships = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Check if output = input shape
            if inp.shape == out.shape:
                relationships.append('same')
            # Check for square outputs
            elif out.shape[0] == out.shape[1]:
                relationships.append(('square', out.shape[0]))
        
        # If all relationships are 'same'
        if all(r == 'same' for r in relationships):
            return input_grid.shape
        
        # If all are square with same size
        if all(isinstance(r, tuple) and r[0] == 'square' for r in relationships):
            sizes = [r[1] for r in relationships]
            if len(set(sizes)) == 1:
                return (sizes[0], sizes[0])
        
        return None
    
    def _try_scaling_rule(self, input_grid: np.ndarray,
                         train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for consistent scaling factors"""
        h_scales = []
        w_scales = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if inp.shape[0] > 0 and inp.shape[1] > 0:
                h_scales.append(out.shape[0] / inp.shape[0])
                w_scales.append(out.shape[1] / inp.shape[1])
        
        if h_scales and w_scales:
            # Check if scales are consistent
            h_scale_counts = Counter([round(s, 2) for s in h_scales])
            w_scale_counts = Counter([round(s, 2) for s in w_scales])
            
            # If there's a dominant scale factor
            if len(h_scale_counts) > 0 and len(w_scale_counts) > 0:
                h_scale = h_scale_counts.most_common(1)[0][0]
                w_scale = w_scale_counts.most_common(1)[0][0]
                
                # Apply scaling
                new_h = int(round(input_grid.shape[0] * h_scale))
                new_w = int(round(input_grid.shape[1] * w_scale))
                
                if 1 <= new_h <= 30 and 1 <= new_w <= 30:
                    return (new_h, new_w)
        
        return None
    
    def _try_object_based_rule(self, input_grid: np.ndarray,
                              train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for object-based size rules"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Get objects in input
            inp_objects = self._extract_objects(inp)
            
            if inp_objects:
                # Rule 1: Output size = bounding box of all objects
                bbox = self._get_combined_bbox(inp_objects)
                if bbox and out.shape == (bbox[2] - bbox[0], bbox[3] - bbox[1]):
                    rules.append(('bbox_all', None))
                
                # Rule 2: Output size = largest object bbox
                largest_obj = max(inp_objects, key=lambda o: len(o['pixels']))
                largest_bbox = largest_obj['bbox']
                obj_h = largest_bbox[2] - largest_bbox[0]
                obj_w = largest_bbox[3] - largest_bbox[1]
                if out.shape == (obj_h, obj_w):
                    rules.append(('bbox_largest', None))
                
                # Rule 3: Output size based on object count
                num_objects = len(inp_objects)
                if out.shape[0] == num_objects or out.shape[1] == num_objects:
                    rules.append(('object_count', num_objects))
        
        # Check if any rule is consistent
        rule_types = [r[0] for r in rules]
        rule_counts = Counter(rule_types)
        
        if rule_counts:
            most_common_rule = rule_counts.most_common(1)[0][0]
            
            # Apply the most common rule to input
            input_objects = self._extract_objects(input_grid)
            
            if most_common_rule == 'bbox_all' and input_objects:
                bbox = self._get_combined_bbox(input_objects)
                if bbox:
                    return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            
            elif most_common_rule == 'bbox_largest' and input_objects:
                largest = max(input_objects, key=lambda o: len(o['pixels']))
                bbox = largest['bbox']
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            
            elif most_common_rule == 'object_count' and input_objects:
                count = len(input_objects)
                # Try to determine if it's height or width
                for r in rules:
                    if r[0] == 'object_count':
                        # Use the first example's aspect
                        for ex in train_examples:
                            out = np.array(ex['output'])
                            if out.shape[0] == count:
                                return (count, input_grid.shape[1])
                            elif out.shape[1] == count:
                                return (input_grid.shape[0], count)
                            break
        
        return None
    
    def _try_cropping_rule(self, input_grid: np.ndarray,
                          train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is input with empty borders removed"""
        crop_consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Get non-empty bounds
            cropped = self._crop_empty_borders(inp)
            
            if cropped.shape != out.shape:
                crop_consistent = False
                break
        
        if crop_consistent:
            cropped_input = self._crop_empty_borders(input_grid)
            return cropped_input.shape
        
        return None
    
    def _try_color_count_rule(self, input_grid: np.ndarray,
                             train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output size is based on color count"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Count non-background colors
            inp_colors = len([c for c in np.unique(inp) if c != 0])
            
            if out.shape[0] == inp_colors:
                rules.append(('height_equals_colors', None))
            elif out.shape[1] == inp_colors:
                rules.append(('width_equals_colors', None))
            elif out.shape == (inp_colors, inp_colors):
                rules.append(('square_color_size', None))
        
        if rules:
            rule_counts = Counter([r[0] for r in rules])
            most_common = rule_counts.most_common(1)[0][0]
            
            input_colors = len([c for c in np.unique(input_grid) if c != 0])
            
            if most_common == 'height_equals_colors':
                return (input_colors, input_grid.shape[1])
            elif most_common == 'width_equals_colors':
                return (input_grid.shape[0], input_colors)
            elif most_common == 'square_color_size':
                return (input_colors, input_colors)
        
        return None
    
    def _try_fractional_scaling_rule(self, input_grid: np.ndarray,
                                    train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for fractional scaling like 1.5x, 2.5x, 0.5x"""
        h_scales = []
        w_scales = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if inp.shape[0] > 0 and inp.shape[1] > 0:
                h_scale = out.shape[0] / inp.shape[0]
                w_scale = out.shape[1] / inp.shape[1]
                h_scales.append(h_scale)
                w_scales.append(w_scale)
        
        if h_scales and w_scales:
            # Check for common fractional scales
            common_fractions = [0.5, 1.5, 2.5, 0.33, 0.66, 1.33, 1.66]
            
            for fraction in common_fractions:
                h_matches = sum(abs(s - fraction) < 0.1 for s in h_scales)
                w_matches = sum(abs(s - fraction) < 0.1 for s in w_scales)
                
                if h_matches >= len(h_scales) * 0.6 and w_matches >= len(w_scales) * 0.6:
                    new_h = int(round(input_grid.shape[0] * fraction))
                    new_w = int(round(input_grid.shape[1] * fraction))
                    
                    if 1 <= new_h <= 30 and 1 <= new_w <= 30:
                        return (new_h, new_w)
        
        return None
    
    def _try_pattern_based_rule(self, input_grid: np.ndarray,
                               train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for pattern-based transformations"""
        # Check if outputs are related to repeating patterns in input
        pattern_rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Detect repeating patterns
            h_period = self._find_period(inp, axis=0)
            w_period = self._find_period(inp, axis=1)
            
            if h_period and out.shape[0] == h_period:
                pattern_rules.append(('height_equals_h_period', h_period))
            if w_period and out.shape[1] == w_period:
                pattern_rules.append(('width_equals_w_period', w_period))
            if h_period and w_period and out.shape == (h_period, w_period):
                pattern_rules.append(('period_size', (h_period, w_period)))
        
        if pattern_rules:
            rule_types = [r[0] for r in pattern_rules]
            rule_counts = Counter(rule_types)
            
            if rule_counts:
                # Apply to input
                input_h_period = self._find_period(input_grid, axis=0)
                input_w_period = self._find_period(input_grid, axis=1)
                
                most_common = rule_counts.most_common(1)[0][0]
                
                if most_common == 'height_equals_h_period' and input_h_period:
                    return (input_h_period, input_grid.shape[1])
                elif most_common == 'width_equals_w_period' and input_w_period:
                    return (input_grid.shape[0], input_w_period)
                elif most_common == 'period_size' and input_h_period and input_w_period:
                    return (input_h_period, input_w_period)
        
        return None
    
    def _try_single_dimension_rule(self, input_grid: np.ndarray,
                                  train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if one dimension stays same while other changes"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if inp.shape[0] == out.shape[0]:
                rules.append(('height_preserved', out.shape[1]))
            elif inp.shape[1] == out.shape[1]:
                rules.append(('width_preserved', out.shape[0]))
        
        if rules:
            rule_counts = Counter([r[0] for r in rules])
            
            if rule_counts:
                most_common = rule_counts.most_common(1)[0][0]
                
                if most_common == 'height_preserved':
                    # Height stays same, find width pattern
                    widths = [r[1] for r in rules if r[0] == 'height_preserved']
                    if widths:
                        return (input_grid.shape[0], int(np.median(widths)))
                
                elif most_common == 'width_preserved':
                    # Width stays same, find height pattern
                    heights = [r[1] for r in rules if r[0] == 'width_preserved']
                    if heights:
                        return (int(np.median(heights)), input_grid.shape[1])
        
        return None
    
    def _try_density_based_rule(self, input_grid: np.ndarray,
                               train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size based on density of non-background pixels"""
        density_rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Calculate density
            non_zero = np.count_nonzero(inp)
            total = inp.shape[0] * inp.shape[1]
            density = non_zero / total if total > 0 else 0
            
            # Check various relationships
            if density > 0:
                if abs(out.shape[0] - int(density * 30)) <= 2:
                    density_rules.append(('height_from_density', density))
                if abs(out.shape[1] - int(density * 30)) <= 2:
                    density_rules.append(('width_from_density', density))
        
        if density_rules:
            rule_counts = Counter([r[0] for r in density_rules])
            
            if rule_counts:
                # Calculate input density
                input_non_zero = np.count_nonzero(input_grid)
                input_total = input_grid.shape[0] * input_grid.shape[1]
                input_density = input_non_zero / input_total if input_total > 0 else 0
                
                if input_density > 0:
                    predicted_size = int(input_density * 30)
                    if 1 <= predicted_size <= 30:
                        return (predicted_size, predicted_size)
        
        return None
    
    def _try_extreme_reduction_rule(self, input_grid: np.ndarray,
                                   train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for extreme reductions like input -> 1x1"""
        for ex in train_examples:
            out = np.array(ex['output'])
            
            # Check for 1x1 outputs
            if out.shape == (1, 1):
                # This might be a "find the dominant color" type task
                return (1, 1)
        
        return None
    
    def _try_median_fallback(self, input_grid: np.ndarray,
                           train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Use median output size as fallback"""
        heights = []
        widths = []
        
        for ex in train_examples:
            out = np.array(ex['output'])
            heights.append(out.shape[0])
            widths.append(out.shape[1])
        
        if heights and widths:
            # Use median
            median_h = int(np.median(heights))
            median_w = int(np.median(widths))
            return (median_h, median_w)
        
        return None
    
    # Helper methods
    def _extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract all objects (connected components) from grid"""
        objects = []
        
        # Process each non-background color
        for color in np.unique(grid):
            if color == 0:  # Skip background
                continue
            
            # Find connected components
            mask = (grid == color).astype(int)
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                pixels = np.argwhere(labeled == i)
                if len(pixels) > 0:
                    # Get bounding box
                    min_r, min_c = pixels.min(axis=0)
                    max_r, max_c = pixels.max(axis=0)
                    
                    objects.append({
                        'color': color,
                        'pixels': pixels,
                        'bbox': (min_r, min_c, max_r + 1, max_c + 1),
                        'size': len(pixels)
                    })
        
        return objects
    
    def _get_combined_bbox(self, objects: List[Dict]) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box that contains all objects"""
        if not objects:
            return None
        
        min_r = min(obj['bbox'][0] for obj in objects)
        min_c = min(obj['bbox'][1] for obj in objects)
        max_r = max(obj['bbox'][2] for obj in objects)
        max_c = max(obj['bbox'][3] for obj in objects)
        
        return (min_r, min_c, max_r, max_c)
    
    def _crop_empty_borders(self, grid: np.ndarray) -> np.ndarray:
        """Remove empty borders from grid"""
        # Find non-empty rows and columns
        non_empty_rows = np.any(grid != 0, axis=1)
        non_empty_cols = np.any(grid != 0, axis=0)
        
        if not np.any(non_empty_rows) or not np.any(non_empty_cols):
            return grid
        
        # Get bounds
        first_row = np.argmax(non_empty_rows)
        last_row = len(non_empty_rows) - np.argmax(non_empty_rows[::-1])
        first_col = np.argmax(non_empty_cols)
        last_col = len(non_empty_cols) - np.argmax(non_empty_cols[::-1])
        
        return grid[first_row:last_row, first_col:last_col]
    
    def _find_period(self, grid: np.ndarray, axis: int) -> Optional[int]:
        """Find repeating period along an axis"""
        size = grid.shape[axis]
        
        for period in range(2, size // 2 + 1):
            if size % period == 0:
                is_periodic = True
                
                for i in range(period, size):
                    if axis == 0:
                        if not np.array_equal(grid[i % period, :], grid[i, :]):
                            is_periodic = False
                            break
                    else:
                        if not np.array_equal(grid[:, i % period], grid[:, i]):
                            is_periodic = False
                            break
                
                if is_periodic:
                    return period
        
        return None


if __name__ == "__main__":
    # Test the enhanced predictor
    predictor = GridSizePredictorV2()
    predictor.debug = True
    
    print("Testing Enhanced Grid Size Predictor V2")
    print("=" * 50)
    
    # Test 1: Object bounding box
    print("\n1. Object Bounding Box Test")
    examples = [
        {
            'input': np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]),
            'output': np.array([
                [1, 1],
                [1, 1]
            ])
        }
    ]
    test_input = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    
    shape = predictor.predict_output_shape(test_input, examples)
    print(f"Predicted: {shape} (expected: (2, 3))")
    
    # Test 2: Color count
    print("\n2. Color Count Test")
    examples = [
        {
            'input': np.array([[1, 2, 3], [4, 5, 0]]),  # 5 non-zero colors
            'output': np.array([[1, 2, 3, 4, 5]] * 5)   # 5x5 grid
        }
    ]
    test_input = np.array([[1, 2], [3, 0]])  # 3 non-zero colors
    shape = predictor.predict_output_shape(test_input, examples)
    print(f"Predicted: {shape}")
    
    # Test 3: Cropping empty borders
    print("\n3. Border Cropping Test")
    examples = [
        {
            'input': np.array([
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0]
            ]),
            'output': np.array([
                [1, 2],
                [3, 4]
            ])
        }
    ]
    test_input = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 6, 7, 8, 0],
        [0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    shape = predictor.predict_output_shape(test_input, examples)
    print(f"Predicted: {shape} (expected: (3, 3))")