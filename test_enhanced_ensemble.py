#!/usr/bin/env python3
"""
Test script for enhanced ensemble with GridSizePredictorV2
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.append('/content/Arc2025')
from ensemble_with_size_prediction import OLYMPUSEnsembleV2


def test_shape_prediction():
    """Test the enhanced shape prediction capabilities"""
    
    print("Testing Enhanced OLYMPUS Ensemble with GridSizePredictorV2")
    print("=" * 60)
    
    # Initialize ensemble
    ensemble = OLYMPUSEnsembleV2('/content/arc_models_v4')
    
    # Test Case 1: Object Bounding Box
    print("\n1. Testing Object Bounding Box Shape Prediction")
    print("-" * 50)
    
    train_examples = [
        {
            'input': [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            'output': [
                [1, 1],
                [1, 1]
            ]
        }
    ]
    
    test_input = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    
    # Enable debug mode
    ensemble.size_predictor.debug = True
    
    # Get prediction with shape
    result = ensemble.predict_with_shape(test_input, train_examples, method='weighted')
    
    print(f"\nPredicted shape: {result['predicted_shape']}")
    print(f"Expected shape: (2, 3)")
    print(f"Actual output shape: {result['prediction'].shape}")
    
    # Test Case 2: Color Count Based
    print("\n\n2. Testing Color Count Shape Prediction")
    print("-" * 50)
    
    train_examples = [
        {
            'input': [[1, 2, 3], [4, 5, 0]],  # 5 non-zero colors
            'output': [[1] * 5] * 5           # 5x5 grid
        }
    ]
    
    test_input = np.array([[1, 2], [3, 0]])  # 3 non-zero colors
    
    result = ensemble.predict_with_shape(test_input, train_examples, method='weighted')
    
    print(f"\nPredicted shape: {result['predicted_shape']}")
    print(f"Expected shape based on colors: (3, 3)")
    print(f"Actual output shape: {result['prediction'].shape}")
    
    # Test Case 3: Border Cropping
    print("\n\n3. Testing Border Cropping Shape Prediction")
    print("-" * 50)
    
    train_examples = [
        {
            'input': [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0]
            ],
            'output': [
                [1, 2],
                [3, 4]
            ]
        }
    ]
    
    test_input = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 6, 7, 8, 0],
        [0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    result = ensemble.predict_with_shape(test_input, train_examples, method='weighted')
    
    print(f"\nPredicted shape: {result['predicted_shape']}")
    print(f"Expected shape (cropped): (3, 3)")
    print(f"Actual output shape: {result['prediction'].shape}")
    
    # Test with real ARC task if available
    print("\n\n4. Testing with Real ARC Task (if available)")
    print("-" * 50)
    
    arc_path = Path('/content/arc-agi_evaluation_challenges.json')
    if arc_path.exists():
        with open(arc_path) as f:
            tasks = json.load(f)
        
        # Use first task
        task = tasks[list(tasks.keys())[0]]
        train_examples = task['train']
        test_case = task['test'][0]
        
        test_input = np.array(test_case['input'])
        expected_output = np.array(test_case['output'])
        
        result = ensemble.predict_with_shape(test_input, train_examples, method='weighted')
        
        print(f"\nTask: {list(tasks.keys())[0]}")
        print(f"Predicted shape: {result['predicted_shape']}")
        print(f"Expected shape: {expected_output.shape}")
        print(f"Actual output shape: {result['prediction'].shape}")
        print(f"Shape match: {result['prediction'].shape == expected_output.shape}")
    else:
        print("ARC evaluation file not found, skipping real task test")
    
    print("\n" + "=" * 60)
    print("Enhanced ensemble testing complete!")


if __name__ == "__main__":
    test_shape_prediction()