#!/usr/bin/env python3
"""
Evaluation script for Enhanced OLYMPUS Ensemble with GridSizePredictorV2
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.append('/content/Arc2025')
from ensemble_with_size_prediction import OLYMPUSEnsembleV2
from heuristic_pipeline import HeuristicPipeline


def evaluate_task(task: Dict, ensemble: OLYMPUSEnsembleV2, 
                  pipeline: HeuristicPipeline) -> Dict:
    """Evaluate a single ARC task"""
    
    train_examples = task['train']
    test_cases = task['test']
    
    results = []
    
    for test_idx, test_case in enumerate(test_cases):
        input_grid = np.array(test_case['input'])
        expected_output = np.array(test_case['output'])
        
        # Get ensemble prediction with shape
        ensemble_result = ensemble.predict_with_shape(
            input_grid, train_examples, method='weighted'
        )
        
        raw_prediction = ensemble_result['prediction']
        
        # Apply heuristic pipeline
        pipeline_output = pipeline.apply_pipeline(
            raw_prediction, input_grid, train_examples
        )
        
        # Check results
        exact_match_raw = np.array_equal(raw_prediction, expected_output)
        exact_match_pipeline = np.array_equal(pipeline_output, expected_output)
        
        # Calculate pixel accuracy
        if raw_prediction.shape == expected_output.shape:
            pixel_acc_raw = np.mean(raw_prediction == expected_output)
        else:
            pixel_acc_raw = 0.0
            
        if pipeline_output.shape == expected_output.shape:
            pixel_acc_pipeline = np.mean(pipeline_output == expected_output)
        else:
            pixel_acc_pipeline = 0.0
        
        results.append({
            'test_idx': test_idx,
            'input_shape': input_grid.shape,
            'expected_shape': expected_output.shape,
            'predicted_shape': ensemble_result['predicted_shape'],
            'raw_shape': raw_prediction.shape,
            'pipeline_shape': pipeline_output.shape,
            'shape_correct': raw_prediction.shape == expected_output.shape,
            'exact_match_raw': exact_match_raw,
            'exact_match_pipeline': exact_match_pipeline,
            'pixel_acc_raw': pixel_acc_raw,
            'pixel_acc_pipeline': pixel_acc_pipeline,
            'shape_prediction_method': ensemble.size_predictor.__class__.__name__
        })
    
    return results


def main():
    print("Enhanced OLYMPUS Ensemble Evaluation")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading enhanced ensemble...")
    ensemble = OLYMPUSEnsembleV2('/content/arc_models_v4')
    
    # Initialize pipeline
    print("2. Initializing heuristic pipeline...")
    pipeline = HeuristicPipeline()
    
    # Load evaluation tasks
    print("3. Loading evaluation tasks...")
    eval_path = Path('/content/arc-agi_evaluation_challenges.json')
    
    with open(eval_path) as f:
        tasks = json.load(f)
    
    print(f"   Loaded {len(tasks)} evaluation tasks")
    
    # Evaluate
    print("\n4. Running evaluation...")
    print("-" * 60)
    
    all_results = []
    shape_correct_count = 0
    exact_match_raw_count = 0
    exact_match_pipeline_count = 0
    
    for task_idx, (task_id, task) in enumerate(tasks.items()):
        if task_idx % 10 == 0:
            print(f"   Processing task {task_idx + 1}/{len(tasks)}...")
        
        task_results = evaluate_task(task, ensemble, pipeline)
        
        for result in task_results:
            all_results.append({
                'task_id': task_id,
                **result
            })
            
            if result['shape_correct']:
                shape_correct_count += 1
            if result['exact_match_raw']:
                exact_match_raw_count += 1
            if result['exact_match_pipeline']:
                exact_match_pipeline_count += 1
    
    # Calculate statistics
    total_tests = len(all_results)
    shape_accuracy = shape_correct_count / total_tests * 100
    exact_match_raw = exact_match_raw_count / total_tests * 100
    exact_match_pipeline = exact_match_pipeline_count / total_tests * 100
    
    # Average pixel accuracy (only for correct shapes)
    pixel_accs_raw = [r['pixel_acc_raw'] for r in all_results if r['shape_correct']]
    pixel_accs_pipeline = [r['pixel_acc_pipeline'] for r in all_results if r['shape_correct']]
    
    avg_pixel_acc_raw = np.mean(pixel_accs_raw) * 100 if pixel_accs_raw else 0
    avg_pixel_acc_pipeline = np.mean(pixel_accs_pipeline) * 100 if pixel_accs_pipeline else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal test cases: {total_tests}")
    print(f"\nShape Prediction Accuracy: {shape_accuracy:.2f}%")
    print(f"  - Correct shapes: {shape_correct_count}/{total_tests}")
    
    print(f"\nExact Match Accuracy:")
    print(f"  - Raw ensemble: {exact_match_raw:.2f}% ({exact_match_raw_count}/{total_tests})")
    print(f"  - With pipeline: {exact_match_pipeline:.2f}% ({exact_match_pipeline_count}/{total_tests})")
    
    print(f"\nAverage Pixel Accuracy (for correct shapes):")
    print(f"  - Raw ensemble: {avg_pixel_acc_raw:.2f}%")
    print(f"  - With pipeline: {avg_pixel_acc_pipeline:.2f}%")
    
    # Analyze shape prediction failures
    shape_failures = [r for r in all_results if not r['shape_correct']]
    if shape_failures:
        print(f"\n\nShape Prediction Analysis:")
        print(f"Failed on {len(shape_failures)} cases")
        
        # Sample some failures
        print("\nSample failures:")
        for failure in shape_failures[:5]:
            print(f"  Task {failure['task_id']}:")
            print(f"    Expected: {failure['expected_shape']}")
            print(f"    Predicted: {failure['predicted_shape']}")
            print(f"    Input: {failure['input_shape']}")
    
    # Save detailed results
    output_path = '/content/enhanced_evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'shape_accuracy': shape_accuracy,
                'exact_match_raw': exact_match_raw,
                'exact_match_pipeline': exact_match_pipeline,
                'avg_pixel_acc_raw': avg_pixel_acc_raw,
                'avg_pixel_acc_pipeline': avg_pixel_acc_pipeline
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_path}")
    
    # Compare with previous results
    print("\n" + "=" * 60)
    print("IMPROVEMENT COMPARISON")
    print("=" * 60)
    print("\nPrevious GridSizePredictor: 0.00% exact match")
    print(f"Enhanced GridSizePredictorV2: {exact_match_pipeline:.2f}% exact match")
    print(f"\nImprovement: +{exact_match_pipeline:.2f}%!")


if __name__ == "__main__":
    main()