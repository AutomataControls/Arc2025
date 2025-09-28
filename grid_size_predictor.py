#!/usr/bin/env python3
"""
Grid Size Predictor - Determines correct output dimensions
This is the key to unlocking shape-changing tasks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class GridSizePredictor:
    """Predict output grid dimensions from training examples"""
    
    def predict_output_shape(self, input_grid: np.ndarray, 
                           train_examples: List[Dict]) -> Tuple[int, int]:
        """
        Predict the output shape based on patterns in training examples
        Returns: (height, width)
        """
        input_h, input_w = input_grid.shape
        
        # Collect all the transformation patterns
        patterns = []
        
        for example in train_examples:
            ex_input = np.array(example['input'])
            ex_output = np.array(example['output'])
            
            in_h, in_w = ex_input.shape
            out_h, out_w = ex_output.shape
            
            # Check different transformation types
            if out_h == in_h and out_w == in_w:
                patterns.append(('same', None))
            elif out_h == in_h * 2 and out_w == in_w * 2:
                patterns.append(('scale', 2))
            elif out_h == in_h * 3 and out_w == in_w * 3:
                patterns.append(('scale', 3))
            elif out_h == in_h // 2 and out_w == in_w // 2:
                patterns.append(('scale', 0.5))
            elif out_h == in_h // 3 and out_w == in_w // 3:
                patterns.append(('scale', 1/3))
            elif out_h == out_w and in_h == in_w:
                # Square to square with specific size
                patterns.append(('fixed_square', out_h))
            elif out_h == out_w:
                # Always square output
                patterns.append(('square', out_h))
            else:
                # Check for fixed output size
                patterns.append(('fixed', (out_h, out_w)))
        
        # Find the most consistent pattern
        if not patterns:
            return input_h, input_w
        
        # Check if all examples have the same pattern
        first_pattern = patterns[0]
        
        if all(p[0] == first_pattern[0] for p in patterns):
            pattern_type = first_pattern[0]
            
            if pattern_type == 'same':
                return input_h, input_w
            elif pattern_type == 'scale':
                scale = patterns[0][1]
                return int(input_h * scale), int(input_w * scale)
            elif pattern_type == 'fixed_square':
                # Use the size from examples
                size = patterns[0][1]
                return size, size
            elif pattern_type == 'square':
                # Make it square, preserve area roughly
                area = input_h * input_w
                size = int(np.sqrt(area))
                return size, size
            elif pattern_type == 'fixed':
                # Use the fixed size
                return patterns[0][1]
        
        # If patterns vary, look for common output shapes
        output_shapes = [(np.array(ex['output']).shape) for ex in train_examples]
        
        # If all outputs have the same shape, use that
        if all(shape == output_shapes[0] for shape in output_shapes):
            return output_shapes[0]
        
        # Otherwise, try to find a relationship
        # Check if output dimensions are related to input dimensions
        h_ratios = []
        w_ratios = []
        
        for example in train_examples:
            ex_input = np.array(example['input'])
            ex_output = np.array(example['output'])
            
            if ex_input.shape[0] > 0:
                h_ratios.append(ex_output.shape[0] / ex_input.shape[0])
            if ex_input.shape[1] > 0:
                w_ratios.append(ex_output.shape[1] / ex_input.shape[1])
        
        if h_ratios and w_ratios:
            # Use median ratio
            h_ratio = np.median(h_ratios)
            w_ratio = np.median(w_ratios)
            
            pred_h = int(round(input_h * h_ratio))
            pred_w = int(round(input_w * w_ratio))
            
            # Sanity check - don't predict crazy sizes
            if 1 <= pred_h <= 30 and 1 <= pred_w <= 30:
                return pred_h, pred_w
        
        # Default: same as input
        return input_h, input_w
    
    def extract_from_30x30(self, tensor_30x30: np.ndarray, 
                          predicted_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract the meaningful part from a 30x30 prediction
        
        The models always output 30x30, but the actual content might be
        in a smaller region. This function extracts that region.
        """
        h, w = predicted_shape
        
        # For now, just take the top-left corner
        # TODO: Could be smarter about finding the active region
        return tensor_30x30[:h, :w]


if __name__ == "__main__":
    # Test the predictor
    predictor = GridSizePredictor()
    
    # Test case 1: Same size
    examples = [
        {'input': np.zeros((5, 5)), 'output': np.ones((5, 5))},
        {'input': np.zeros((3, 3)), 'output': np.ones((3, 3))}
    ]
    input_grid = np.zeros((7, 7))
    shape = predictor.predict_output_shape(input_grid, examples)
    print(f"Same size test: {shape} (expected: (7, 7))")
    
    # Test case 2: Double size
    examples = [
        {'input': np.zeros((2, 2)), 'output': np.ones((4, 4))},
        {'input': np.zeros((3, 3)), 'output': np.ones((6, 6))}
    ]
    input_grid = np.zeros((5, 5))
    shape = predictor.predict_output_shape(input_grid, examples)
    print(f"Double size test: {shape} (expected: (10, 10))")
    
    # Test case 3: Fixed output
    examples = [
        {'input': np.zeros((5, 5)), 'output': np.ones((3, 3))},
        {'input': np.zeros((10, 10)), 'output': np.ones((3, 3))}
    ]
    input_grid = np.zeros((7, 7))
    shape = predictor.predict_output_shape(input_grid, examples)
    print(f"Fixed output test: {shape} (expected: (3, 3))")