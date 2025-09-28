#!/usr/bin/env python3
"""
Grid Size Predictor V3 - Comprehensive Shape Analysis
Implements exhaustive shape prediction rules based on evaluation analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy import ndimage
from collections import Counter


class GridSizePredictorV2:
    """Comprehensive predictor with exhaustive shape rules based on evaluation patterns"""
    
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
            self._try_transpose_rule,
            self._try_minus_one_rule,
            self._try_specific_size_rule,
            self._try_aspect_ratio_rule,
            self._try_hole_count_rule,
            self._try_line_count_rule,
            self._try_symmetry_based_rule,
            self._try_half_size_rule,
            self._try_double_size_rule,
            self._try_sqrt_area_rule,
            self._try_modulo_based_rule,
            self._try_prime_factor_rule,
            self._try_plus_one_rule,
            self._try_plus_two_rule,
            self._try_dimension_swap_rule,
            self._try_gcd_lcm_rule,
            self._try_fibonacci_rule,
            self._try_power_of_two_rule,
            self._try_third_size_rule,
            self._try_quarter_size_rule,
            self._try_count_objects_rule,
            self._try_max_object_size_rule,
            self._try_union_bbox_rule,
            self._try_non_zero_bbox_rule,
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
    
    def _try_transpose_rule(self, input_grid: np.ndarray,
                          train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is transposed input"""
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if out.shape == (inp.shape[1], inp.shape[0]):
                return (input_grid.shape[1], input_grid.shape[0])
        
        return None
    
    def _try_minus_one_rule(self, input_grid: np.ndarray,
                           train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is input size minus 1 in one or both dimensions"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if out.shape == (inp.shape[0] - 1, inp.shape[1] - 1):
                rules.append('both_minus_one')
            elif out.shape == (inp.shape[0] - 1, inp.shape[1]):
                rules.append('height_minus_one')
            elif out.shape == (inp.shape[0], inp.shape[1] - 1):
                rules.append('width_minus_one')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            if most_common == 'both_minus_one':
                return (max(1, input_grid.shape[0] - 1), max(1, input_grid.shape[1] - 1))
            elif most_common == 'height_minus_one':
                return (max(1, input_grid.shape[0] - 1), input_grid.shape[1])
            elif most_common == 'width_minus_one':
                return (input_grid.shape[0], max(1, input_grid.shape[1] - 1))
        
        return None
    
    def _try_specific_size_rule(self, input_grid: np.ndarray,
                              train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for specific common output sizes"""
        # Extended list based on evaluation patterns
        common_sizes = [(3, 3), (4, 4), (2, 2), (5, 5), (1, 1), 
                       (3, 1), (1, 3), (2, 3), (3, 2), (2, 4), (4, 2),
                       (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
        
        for size in common_sizes:
            all_match = all(np.array(ex['output']).shape == size for ex in train_examples)
            if all_match:
                return size
        
        return None
    
    def _try_aspect_ratio_rule(self, input_grid: np.ndarray,
                             train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Preserve aspect ratio while changing size"""
        ratios = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if inp.shape[0] > 0 and out.shape[0] > 0:
                # Calculate how output relates to input while preserving ratio
                scale = out.shape[0] / inp.shape[0]
                expected_w = int(round(inp.shape[1] * scale))
                if abs(out.shape[1] - expected_w) <= 1:
                    ratios.append(scale)
        
        if ratios and len(ratios) >= len(train_examples) * 0.8:
            avg_scale = np.mean(ratios)
            new_h = int(round(input_grid.shape[0] * avg_scale))
            new_w = int(round(input_grid.shape[1] * avg_scale))
            if 1 <= new_h <= 30 and 1 <= new_w <= 30:
                return (new_h, new_w)
        
        return None
    
    def _try_hole_count_rule(self, input_grid: np.ndarray,
                           train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size based on number of holes/enclosed regions"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Count holes (background regions completely enclosed)
            holes = self._count_holes(inp)
            
            if holes > 0:
                if out.shape == (holes, holes):
                    rules.append('square_holes')
                elif out.shape[0] == holes:
                    rules.append('height_equals_holes')
                elif out.shape[1] == holes:
                    rules.append('width_equals_holes')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            input_holes = self._count_holes(input_grid)
            if input_holes > 0:
                if most_common == 'square_holes':
                    return (input_holes, input_holes)
                elif most_common == 'height_equals_holes':
                    return (input_holes, input_grid.shape[1])
                elif most_common == 'width_equals_holes':
                    return (input_grid.shape[0], input_holes)
        
        return None
    
    def _try_line_count_rule(self, input_grid: np.ndarray,
                           train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size based on number of lines (horizontal/vertical)"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            h_lines = self._count_horizontal_lines(inp)
            v_lines = self._count_vertical_lines(inp)
            
            if out.shape == (h_lines, v_lines) and h_lines > 0 and v_lines > 0:
                rules.append('h_by_v_lines')
            elif out.shape[0] == h_lines and h_lines > 0:
                rules.append('height_equals_h_lines')
            elif out.shape[1] == v_lines and v_lines > 0:
                rules.append('width_equals_v_lines')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            input_h_lines = self._count_horizontal_lines(input_grid)
            input_v_lines = self._count_vertical_lines(input_grid)
            
            if most_common == 'h_by_v_lines' and input_h_lines > 0 and input_v_lines > 0:
                return (input_h_lines, input_v_lines)
            elif most_common == 'height_equals_h_lines' and input_h_lines > 0:
                return (input_h_lines, input_grid.shape[1])
            elif most_common == 'width_equals_v_lines' and input_v_lines > 0:
                return (input_grid.shape[0], input_v_lines)
        
        return None
    
    def _try_symmetry_based_rule(self, input_grid: np.ndarray,
                               train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size based on symmetry properties"""
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Check if output is half of input (folding along axis)
            if out.shape == (inp.shape[0] // 2, inp.shape[1]):
                if input_grid.shape[0] % 2 == 0:
                    return (input_grid.shape[0] // 2, input_grid.shape[1])
            elif out.shape == (inp.shape[0], inp.shape[1] // 2):
                if input_grid.shape[1] % 2 == 0:
                    return (input_grid.shape[0], input_grid.shape[1] // 2)
        
        return None
    
    def _try_half_size_rule(self, input_grid: np.ndarray,
                          train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is exactly half the input size"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if out.shape == (inp.shape[0] // 2, inp.shape[1] // 2):
                rules.append('both_half')
            elif out.shape == (inp.shape[0] // 2, inp.shape[1]):
                rules.append('height_half')
            elif out.shape == (inp.shape[0], inp.shape[1] // 2):
                rules.append('width_half')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            if most_common == 'both_half':
                return (max(1, input_grid.shape[0] // 2), max(1, input_grid.shape[1] // 2))
            elif most_common == 'height_half':
                return (max(1, input_grid.shape[0] // 2), input_grid.shape[1])
            elif most_common == 'width_half':
                return (input_grid.shape[0], max(1, input_grid.shape[1] // 2))
        
        return None
    
    def _try_double_size_rule(self, input_grid: np.ndarray,
                            train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is exactly double the input size"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if out.shape == (inp.shape[0] * 2, inp.shape[1] * 2):
                rules.append('both_double')
            elif out.shape == (inp.shape[0] * 2, inp.shape[1]):
                rules.append('height_double')
            elif out.shape == (inp.shape[0], inp.shape[1] * 2):
                rules.append('width_double')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            h, w = input_grid.shape
            if most_common == 'both_double' and h * 2 <= 30 and w * 2 <= 30:
                return (h * 2, w * 2)
            elif most_common == 'height_double' and h * 2 <= 30:
                return (h * 2, w)
            elif most_common == 'width_double' and w * 2 <= 30:
                return (h, w * 2)
        
        return None
    
    def _try_sqrt_area_rule(self, input_grid: np.ndarray,
                          train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output size is based on square root of input area"""
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Calculate expected size based on sqrt of area
            area = inp.shape[0] * inp.shape[1]
            expected_size = int(np.sqrt(area))
            
            if out.shape != (expected_size, expected_size):
                consistent = False
                break
        
        if consistent:
            input_area = input_grid.shape[0] * input_grid.shape[1]
            size = int(np.sqrt(input_area))
            if 1 <= size <= 30:
                return (size, size)
        
        return None
    
    def _try_modulo_based_rule(self, input_grid: np.ndarray,
                             train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check for modulo-based size relationships"""
        # Common modulo values
        for mod in [3, 4, 5, 6, 7, 8]:
            h_consistent = True
            w_consistent = True
            
            for ex in train_examples:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                
                # Check if output dimensions relate to input via modulo
                if inp.shape[0] % mod != 0 or out.shape[0] != inp.shape[0] // mod:
                    h_consistent = False
                if inp.shape[1] % mod != 0 or out.shape[1] != inp.shape[1] // mod:
                    w_consistent = False
            
            if h_consistent and w_consistent:
                new_h = input_grid.shape[0] // mod
                new_w = input_grid.shape[1] // mod
                if 1 <= new_h <= 30 and 1 <= new_w <= 30:
                    return (new_h, new_w)
        
        return None
    
    def _try_prime_factor_rule(self, input_grid: np.ndarray,
                             train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output size is based on prime factorization"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Get prime factors of input dimensions
            h_factors = self._get_prime_factors(inp.shape[0])
            w_factors = self._get_prime_factors(inp.shape[1])
            
            # Check various relationships
            if h_factors and out.shape[0] == max(h_factors):
                rules.append(('height_largest_prime', max(h_factors)))
            if w_factors and out.shape[1] == max(w_factors):
                rules.append(('width_largest_prime', max(w_factors)))
            
            # Check if output is product of unique prime factors
            if h_factors:
                unique_product = 1
                for p in set(h_factors):
                    unique_product *= p
                if out.shape[0] == unique_product:
                    rules.append(('height_unique_prime_product', unique_product))
        
        if rules:
            rule_types = [r[0] for r in rules]
            rule_counts = Counter(rule_types)
            
            if rule_counts:
                # Apply to input
                input_h_factors = self._get_prime_factors(input_grid.shape[0])
                input_w_factors = self._get_prime_factors(input_grid.shape[1])
                
                most_common = rule_counts.most_common(1)[0][0]
                
                if most_common == 'height_largest_prime' and input_h_factors:
                    return (max(input_h_factors), input_grid.shape[1])
                elif most_common == 'width_largest_prime' and input_w_factors:
                    return (input_grid.shape[0], max(input_w_factors))
        
        return None
    
    def _get_prime_factors(self, n: int) -> List[int]:
        """Get prime factors of a number"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def _try_plus_one_rule(self, input_grid: np.ndarray,
                         train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is input size plus 1 in one or both dimensions"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if out.shape == (inp.shape[0] + 1, inp.shape[1] + 1):
                rules.append('both_plus_one')
            elif out.shape == (inp.shape[0] + 1, inp.shape[1]):
                rules.append('height_plus_one')
            elif out.shape == (inp.shape[0], inp.shape[1] + 1):
                rules.append('width_plus_one')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            h, w = input_grid.shape
            if most_common == 'both_plus_one' and h + 1 <= 30 and w + 1 <= 30:
                return (h + 1, w + 1)
            elif most_common == 'height_plus_one' and h + 1 <= 30:
                return (h + 1, w)
            elif most_common == 'width_plus_one' and w + 1 <= 30:
                return (h, w + 1)
        
        return None
    
    def _try_plus_two_rule(self, input_grid: np.ndarray,
                         train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is input size plus 2 in one or both dimensions"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if out.shape == (inp.shape[0] + 2, inp.shape[1] + 2):
                rules.append('both_plus_two')
            elif out.shape == (inp.shape[0] + 2, inp.shape[1]):
                rules.append('height_plus_two')
            elif out.shape == (inp.shape[0], inp.shape[1] + 2):
                rules.append('width_plus_two')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            h, w = input_grid.shape
            if most_common == 'both_plus_two' and h + 2 <= 30 and w + 2 <= 30:
                return (h + 2, w + 2)
            elif most_common == 'height_plus_two' and h + 2 <= 30:
                return (h + 2, w)
            elif most_common == 'width_plus_two' and w + 2 <= 30:
                return (h, w + 2)
        
        return None
    
    def _try_dimension_swap_rule(self, input_grid: np.ndarray,
                               train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output swaps height and width with modifications"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Check various swap patterns
            if out.shape == (inp.shape[1] - 1, inp.shape[0]):
                rules.append('swap_width_minus_one')
            elif out.shape == (inp.shape[1], inp.shape[0] - 1):
                rules.append('swap_height_minus_one')
            elif out.shape == (inp.shape[1] + 1, inp.shape[0]):
                rules.append('swap_width_plus_one')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            h, w = input_grid.shape
            if most_common == 'swap_width_minus_one' and w - 1 >= 1:
                return (w - 1, h)
            elif most_common == 'swap_height_minus_one' and h - 1 >= 1:
                return (w, h - 1)
            elif most_common == 'swap_width_plus_one' and w + 1 <= 30:
                return (w + 1, h)
        
        return None
    
    def _try_gcd_lcm_rule(self, input_grid: np.ndarray,
                         train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output size is based on GCD or LCM of input dimensions"""
        import math
        
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            gcd = math.gcd(inp.shape[0], inp.shape[1])
            lcm = (inp.shape[0] * inp.shape[1]) // gcd
            
            if out.shape == (gcd, gcd):
                rules.append('gcd_square')
            elif out.shape[0] == gcd or out.shape[1] == gcd:
                rules.append('gcd_dimension')
            elif lcm <= 30 and out.shape == (lcm, lcm):
                rules.append('lcm_square')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            gcd = math.gcd(input_grid.shape[0], input_grid.shape[1])
            lcm = (input_grid.shape[0] * input_grid.shape[1]) // gcd
            
            if most_common == 'gcd_square':
                return (gcd, gcd)
            elif most_common == 'lcm_square' and lcm <= 30:
                return (lcm, lcm)
        
        return None
    
    def _try_fibonacci_rule(self, input_grid: np.ndarray,
                          train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output size follows Fibonacci sequence"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21]
        
        for ex in train_examples:
            out = np.array(ex['output'])
            
            # Check if output dimensions are fibonacci numbers
            if out.shape[0] in fib and out.shape[1] in fib:
                # Find pattern
                h_idx = fib.index(out.shape[0])
                w_idx = fib.index(out.shape[1])
                
                # Apply similar pattern to input
                if input_grid.shape[0] in fib and input_grid.shape[1] in fib:
                    inp_h_idx = fib.index(input_grid.shape[0])
                    inp_w_idx = fib.index(input_grid.shape[1])
                    
                    # Try to find the transformation pattern
                    new_h_idx = inp_h_idx + (h_idx - inp_h_idx)
                    new_w_idx = inp_w_idx + (w_idx - inp_w_idx)
                    
                    if 0 <= new_h_idx < len(fib) and 0 <= new_w_idx < len(fib):
                        return (fib[new_h_idx], fib[new_w_idx])
        
        return None
    
    def _try_power_of_two_rule(self, input_grid: np.ndarray,
                             train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output size is nearest power of 2"""
        def nearest_power_of_2(n):
            powers = [1, 2, 4, 8, 16]
            return min(powers, key=lambda x: abs(x - n))
        
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            expected_h = nearest_power_of_2(inp.shape[0])
            expected_w = nearest_power_of_2(inp.shape[1])
            
            if out.shape != (expected_h, expected_w):
                consistent = False
                break
        
        if consistent:
            new_h = nearest_power_of_2(input_grid.shape[0])
            new_w = nearest_power_of_2(input_grid.shape[1])
            return (new_h, new_w)
        
        return None
    
    def _try_third_size_rule(self, input_grid: np.ndarray,
                           train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is exactly 1/3 the input size"""
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            expected_h = max(1, inp.shape[0] // 3)
            expected_w = max(1, inp.shape[1] // 3)
            
            if out.shape != (expected_h, expected_w):
                consistent = False
                break
        
        if consistent:
            return (max(1, input_grid.shape[0] // 3), max(1, input_grid.shape[1] // 3))
        
        return None
    
    def _try_quarter_size_rule(self, input_grid: np.ndarray,
                             train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if output is exactly 1/4 the input size"""
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            expected_h = max(1, inp.shape[0] // 4)
            expected_w = max(1, inp.shape[1] // 4)
            
            if out.shape != (expected_h, expected_w):
                consistent = False
                break
        
        if consistent:
            return (max(1, input_grid.shape[0] // 4), max(1, input_grid.shape[1] // 4))
        
        return None
    
    def _try_count_objects_rule(self, input_grid: np.ndarray,
                              train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size based on number of distinct objects"""
        rules = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Count distinct objects (connected components)
            objects = self._extract_objects(inp)
            num_objects = len(objects)
            
            if num_objects > 0:
                if out.shape == (num_objects, num_objects):
                    rules.append('square_object_count')
                elif out.shape[0] == num_objects:
                    rules.append('height_object_count')
                elif out.shape[1] == num_objects:
                    rules.append('width_object_count')
        
        if rules:
            rule_counts = Counter(rules)
            most_common = rule_counts.most_common(1)[0][0]
            
            input_objects = self._extract_objects(input_grid)
            count = len(input_objects)
            
            if count > 0 and count <= 30:
                if most_common == 'square_object_count':
                    return (count, count)
                elif most_common == 'height_object_count':
                    # Try to preserve width or use median
                    widths = [np.array(ex['output']).shape[1] for ex in train_examples]
                    return (count, int(np.median(widths)))
                elif most_common == 'width_object_count':
                    heights = [np.array(ex['output']).shape[0] for ex in train_examples]
                    return (int(np.median(heights)), count)
        
        return None
    
    def _try_max_object_size_rule(self, input_grid: np.ndarray,
                                 train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size equals the largest object dimensions"""
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            objects = self._extract_objects(inp)
            if objects:
                # Find largest object
                largest = max(objects, key=lambda o: len(o['pixels']))
                bbox = largest['bbox']
                obj_h = bbox[2] - bbox[0]
                obj_w = bbox[3] - bbox[1]
                
                if out.shape != (obj_h, obj_w):
                    consistent = False
                    break
            else:
                consistent = False
                break
        
        if consistent:
            input_objects = self._extract_objects(input_grid)
            if input_objects:
                largest = max(input_objects, key=lambda o: len(o['pixels']))
                bbox = largest['bbox']
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        return None
    
    def _try_union_bbox_rule(self, input_grid: np.ndarray,
                           train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output size is the bounding box of all non-zero pixels"""
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Get bounding box of all objects
            objects = self._extract_objects(inp)
            if objects:
                bbox = self._get_combined_bbox(objects)
                if bbox:
                    expected_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    if out.shape != expected_shape:
                        consistent = False
                        break
        
        if consistent:
            input_objects = self._extract_objects(input_grid)
            if input_objects:
                bbox = self._get_combined_bbox(input_objects)
                if bbox:
                    return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        return None
    
    def _try_non_zero_bbox_rule(self, input_grid: np.ndarray,
                              train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Output is the tight bounding box around all non-zero pixels"""
        consistent = True
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Find non-zero bounds
            non_zero = np.argwhere(inp != 0)
            if len(non_zero) > 0:
                min_r, min_c = non_zero.min(axis=0)
                max_r, max_c = non_zero.max(axis=0)
                expected_shape = (max_r - min_r + 1, max_c - min_c + 1)
                
                if out.shape != expected_shape:
                    consistent = False
                    break
        
        if consistent:
            non_zero = np.argwhere(input_grid != 0)
            if len(non_zero) > 0:
                min_r, min_c = non_zero.min(axis=0)
                max_r, max_c = non_zero.max(axis=0)
                return (max_r - min_r + 1, max_c - min_c + 1)
        
        return None
    
    def _count_holes(self, grid: np.ndarray) -> int:
        """Count number of enclosed background regions"""
        # Create binary mask (0 for background, 1 for any color)
        binary = (grid != 0).astype(int)
        
        # Invert to find background regions
        inverted = 1 - binary
        
        # Label connected components
        labeled, num_features = ndimage.label(inverted, structure=np.ones((3,3)))
        
        # Count internal holes (not touching border)
        holes = 0
        for i in range(1, num_features + 1):
            component = (labeled == i)
            # Check if component touches border
            if not (component[0,:].any() or component[-1,:].any() or 
                    component[:,0].any() or component[:,-1].any()):
                holes += 1
        
        return holes
    
    def _count_horizontal_lines(self, grid: np.ndarray) -> int:
        """Count continuous horizontal lines of non-background pixels"""
        count = 0
        for row in grid:
            if np.any(row != 0) and np.all(row == row[row != 0][0]):
                count += 1
        return count
    
    def _count_vertical_lines(self, grid: np.ndarray) -> int:
        """Count continuous vertical lines of non-background pixels"""
        count = 0
        for col in grid.T:
            if np.any(col != 0) and np.all(col == col[col != 0][0]):
                count += 1
        return count
    
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
    
    print("Testing Enhanced Grid Size Predictor V2 with V3 features")
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