#!/usr/bin/env python3
"""
OLYMPUS Ensemble Runner - Complete Integration
Combines all components: Models, Task Router, Heuristics
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Add paths
sys.path.append('/content/Arc2025')
sys.path.append('/content/Arc2025/models')

from ensemble_test_bench import OLYMPUSEnsemble
from ensemble_with_size_prediction import OLYMPUSEnsembleV2
from task_router import TaskRouter, SmartEnsemble
from heuristic_solvers import HeuristicPipeline


class OLYMPUSRunner:
    """Complete OLYMPUS system with all enhancements"""
    
    def __init__(self, model_dir: str = '/content/arc_models_v4'):
        print("🏛️ Initializing OLYMPUS Complete System...")
        
        # Load enhanced ensemble with size prediction
        self.ensemble = OLYMPUSEnsembleV2(model_dir)
        
        # Initialize task router
        self.router = TaskRouter()
        self.smart_ensemble = SmartEnsemble(self.ensemble, self.router)
        
        # Initialize heuristics
        self.heuristics = HeuristicPipeline()
        
        print("✅ OLYMPUS ready for predictions!")
    
    def predict(self, input_grid: np.ndarray, train_examples: List[Dict],
                apply_heuristics: bool = True, verbose: bool = True) -> Dict:
        """
        Make a prediction using the complete OLYMPUS pipeline
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🏛️ OLYMPUS PREDICTION PIPELINE")
            print(f"{'='*60}")
            print(f"Input shape: {input_grid.shape}")
            print(f"Training examples: {len(train_examples)}")
        
        # Step 1: Smart ensemble prediction with task routing
        result = self.smart_ensemble.predict(input_grid, train_examples, verbose)
        
        prediction = result['prediction']
        
        # Step 2: Apply heuristics if enabled
        if apply_heuristics and prediction is not None:
            if verbose:
                print(f"\n🔧 Applying post-processing heuristics...")
            
            prediction = self.heuristics.apply(
                input_grid, prediction, train_examples, verbose
            )
        
        # Calculate confidence
        confidence = self._calculate_confidence(result)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n⏱️  Prediction completed in {elapsed:.2f}s")
            print(f"🎯 Confidence: {confidence:.1f}%")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'ensemble_result': result,
            'time_elapsed': elapsed
        }
    
    def _calculate_confidence(self, ensemble_result: Dict) -> float:
        """Calculate confidence score for the prediction"""
        # Base confidence from voting score
        score = ensemble_result.get('score', 0)
        max_possible_score = 5.0  # Sum of all weights
        
        base_confidence = (score / max_possible_score) * 100
        
        # Boost confidence if multiple models agree
        vote_details = ensemble_result.get('vote_details', {})
        if len(vote_details) >= 3:
            base_confidence *= 1.2
        elif len(vote_details) >= 2:
            base_confidence *= 1.1
        
        return min(base_confidence, 100.0)
    
    def _analyze_prediction(self, prediction: np.ndarray, input_grid: np.ndarray, 
                          train_examples: List[Dict]) -> Dict:
        """Analyze a prediction for potential issues"""
        analysis = {
            'shape_mismatch_risk': False,
            'color_pattern_issue': False,
            'confidence': 0.5
        }
        
        # Check shape patterns
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # If training examples show consistent shape transformation
            if inp.shape != out.shape:
                # Check if prediction follows the pattern
                expected_h_ratio = out.shape[0] / inp.shape[0]
                expected_w_ratio = out.shape[1] / inp.shape[1]
                
                actual_h_ratio = prediction.shape[0] / input_grid.shape[0]
                actual_w_ratio = prediction.shape[1] / input_grid.shape[1]
                
                if abs(expected_h_ratio - actual_h_ratio) > 0.3 or abs(expected_w_ratio - actual_w_ratio) > 0.3:
                    analysis['shape_mismatch_risk'] = True
        
        # Check color patterns
        pred_colors = set(np.unique(prediction))
        input_colors = set(np.unique(input_grid))
        
        # Check if prediction has way more or fewer colors than expected
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            inp_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            
            # If outputs typically preserve color count
            if len(inp_colors) == len(out_colors):
                if abs(len(pred_colors) - len(input_colors)) > 2:
                    analysis['color_pattern_issue'] = True
        
        # Calculate confidence based on shape match and color count
        if prediction.shape == input_grid.shape:
            analysis['confidence'] += 0.2
        
        if len(pred_colors) <= len(input_colors) + 1:
            analysis['confidence'] += 0.2
        
        # Check if prediction is mostly background (might be failure)
        if np.count_nonzero(prediction == 0) > prediction.size * 0.9:
            analysis['confidence'] -= 0.3
        
        analysis['confidence'] = max(0, min(1, analysis['confidence']))
        
        return analysis
    
    def predict_task(self, task: Dict) -> List[np.ndarray]:
        """
        Predict all test outputs for a complete task
        Returns top 2 predictions for each test case
        """
        train_examples = task['train']
        test_inputs = [np.array(t['input']) for t in task['test']]
        
        predictions = []
        
        for i, test_input in enumerate(test_inputs):
            print(f"\n📋 Test case {i+1}/{len(test_inputs)}")
            
            # First attempt: Full pipeline with heuristics
            result_with_heur = self.predict(test_input, train_examples, apply_heuristics=True)
            attempt_1 = result_with_heur['prediction']
            
            # Second attempt: Smart strategy based on first attempt analysis
            
            # Get all model predictions
            model_predictions = self.ensemble.predict_all_models_with_shape(
                test_input, train_examples
            )
            
            # Use task router to get weights
            weights = self.router.get_model_weights(test_input, train_examples)
            
            # Analyze first attempt for potential issues
            attempt_1_analysis = self._analyze_prediction(attempt_1, test_input, train_examples)
            
            # Strategy 1: If first attempt has shape mismatch risk, try without heuristics
            if attempt_1_analysis.get('shape_mismatch_risk', False):
                result_no_heur = self.predict(test_input, train_examples, apply_heuristics=False, verbose=False)
                attempt_2 = result_no_heur['prediction']
            
            # Strategy 2: If color patterns seem wrong, try different model
            elif attempt_1_analysis.get('color_pattern_issue', False):
                # Find model with different color handling approach
                color_models = ['IRIS', 'CHRONOS']  # Models better at color patterns
                for model in color_models:
                    if model in model_predictions and not np.array_equal(model_predictions[model], attempt_1):
                        attempt_2 = model_predictions[model]
                        break
                else:
                    # Fall back to highest weighted model
                    best_model = max(weights.items(), key=lambda x: x[1])[0]
                    attempt_2 = model_predictions.get(best_model, attempt_1)
            
            # Strategy 3: If high confidence in first attempt, try a complementary approach
            elif attempt_1_analysis.get('confidence', 0) > 0.8:
                # Try the model that's most different from consensus
                sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                for model, _ in sorted_models:
                    pred = model_predictions.get(model)
                    if pred is not None and not np.array_equal(pred, attempt_1):
                        attempt_2 = pred
                        break
                else:
                    attempt_2 = attempt_1
            
            # Strategy 4: Low confidence - try best individual model
            else:
                best_model = max(weights.items(), key=lambda x: x[1])[0]
                attempt_2 = model_predictions.get(best_model, attempt_1)
            
            # Final check: If both attempts are still the same, try median prediction
            if np.array_equal(attempt_1, attempt_2) and len(model_predictions) > 2:
                # Create a simple median-based prediction
                all_preds = list(model_predictions.values())
                if all_preds:
                    # Use mode for each pixel position
                    from scipy import stats
                    attempt_2 = stats.mode(np.stack(all_preds), axis=0, keepdims=False)[0]
            
            predictions.append([attempt_1, attempt_2])
            print(f"  ✅ Two attempts generated (heuristic + raw best model)")
        
        return predictions
    
    def evaluate_on_training_set(self, data_path: str, n_tasks: int = 50) -> Dict:
        """Evaluate ensemble on training tasks"""
        print("\n📊 EVALUATING OLYMPUS ON TRAINING SET")
        print("="*60)
        
        # Load data
        with open(data_path, 'r') as f:
            all_tasks = json.load(f)
        
        # Sample tasks
        import random
        task_ids = random.sample(list(all_tasks.keys()), min(n_tasks, len(all_tasks)))
        
        # Evaluate
        results = {
            'exact_matches': 0,
            'pixel_accuracy': [],
            'confidence_scores': [],
            'heuristics_helped': 0,
            'task_results': {}
        }
        
        for task_idx, task_id in enumerate(task_ids):
            print(f"\n[{task_idx+1}/{n_tasks}] Task: {task_id}")
            
            task = all_tasks[task_id]
            train_examples = task['train']
            
            # Use first training example as test
            test_input = np.array(train_examples[0]['input'])
            test_output = np.array(train_examples[0]['output'])
            
            # Predict without heuristics first
            result_no_heur = self.predict(
                test_input, train_examples[1:], 
                apply_heuristics=False, verbose=False
            )
            
            # Predict with heuristics
            result_with_heur = self.predict(
                test_input, train_examples[1:], 
                apply_heuristics=True, verbose=False
            )
            
            pred_no_heur = result_no_heur['prediction']
            pred_with_heur = result_with_heur['prediction']
            
            # Handle None predictions
            if pred_no_heur is None or pred_with_heur is None:
                print("  ⚠️  Failed to get prediction")
                continue
            
            # Check if shapes match for comparison
            if pred_with_heur.shape != test_output.shape:
                print(f"  ⚠️  Shape mismatch: predicted {pred_with_heur.shape} vs actual {test_output.shape}")
                pixel_acc = 0.0
                exact_no_heur = False
                exact_with_heur = False
            else:
                # Check results
                exact_no_heur = np.array_equal(pred_no_heur, test_output)
                exact_with_heur = np.array_equal(pred_with_heur, test_output)
                
                # Pixel accuracy
                pixel_acc = (pred_with_heur == test_output).mean() * 100
            
            if exact_with_heur:
                results['exact_matches'] += 1
                print("  ✅ EXACT MATCH!")
            
            if not exact_no_heur and exact_with_heur:
                results['heuristics_helped'] += 1
                print("  ✨ Heuristics fixed it!")
            results['pixel_accuracy'].append(pixel_acc)
            results['confidence_scores'].append(result_with_heur['confidence'])
            
            # Store detailed result
            results['task_results'][task_id] = {
                'exact_match': exact_with_heur,
                'pixel_accuracy': pixel_acc,
                'confidence': result_with_heur['confidence'],
                'heuristics_helped': not exact_no_heur and exact_with_heur
            }
            
            print(f"  Pixel accuracy: {pixel_acc:.1f}%")
            print(f"  Confidence: {result_with_heur['confidence']:.1f}%")
        
        # Summary
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        exact_pct = results['exact_matches'] / n_tasks * 100
        avg_pixel = np.mean(results['pixel_accuracy'])
        avg_conf = np.mean(results['confidence_scores'])
        heur_pct = results['heuristics_helped'] / n_tasks * 100
        
        print(f"Exact matches: {results['exact_matches']}/{n_tasks} ({exact_pct:.1f}%)")
        print(f"Average pixel accuracy: {avg_pixel:.1f}%")
        print(f"Average confidence: {avg_conf:.1f}%")
        print(f"Heuristics helped: {results['heuristics_helped']} tasks ({heur_pct:.1f}%)")
        
        return results
    
    def create_submission(self, test_path: str, output_path: str):
        """Create a submission file for the competition"""
        print("\n🏆 CREATING COMPETITION SUBMISSION")
        print("="*60)
        
        # Load test tasks
        with open(test_path, 'r') as f:
            test_tasks = json.load(f)
        
        submission = {}
        
        for task_idx, (task_id, task) in enumerate(test_tasks.items()):
            print(f"\n[{task_idx+1}/{len(test_tasks)}] Task: {task_id}")
            
            # Get predictions
            predictions = self.predict_task(task)
            
            # Format for submission
            task_preds = []
            for test_idx, (pred1, pred2) in enumerate(predictions):
                task_preds.append({
                    'output': pred1.tolist(),
                    'output_2': pred2.tolist()  # Second guess
                })
            
            submission[task_id] = task_preds
        
        # Save submission
        with open(output_path, 'w') as f:
            json.dump(submission, f)
        
        print(f"\n✅ Submission saved to: {output_path}")
        
        return submission


def run_example():
    """Run an example prediction"""
    print("🏛️ OLYMPUS ENSEMBLE - COMPLETE SYSTEM DEMO")
    print("="*60)
    
    # Create simple test case
    input_grid = np.array([
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ])
    
    train_examples = [
        {
            'input': np.array([[1, 0], [0, 1]]),
            'output': np.array([[2, 0], [0, 2]])  # Color doubling
        },
        {
            'input': np.array([[2, 1], [1, 2]]),
            'output': np.array([[4, 2], [2, 4]])  # Color doubling
        }
    ]
    
    # Initialize OLYMPUS
    olympus = OLYMPUSRunner()
    
    # Make prediction
    result = olympus.predict(input_grid, train_examples)
    
    print("\nPredicted output:")
    print(result['prediction'])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OLYMPUS Ensemble Runner')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--submit', type=str, help='Create submission from test file')
    parser.add_argument('--n_tasks', type=int, default=50, help='Number of tasks to evaluate')
    
    args = parser.parse_args()
    
    if args.demo or (not args.evaluate and not args.submit):
        run_example()
    
    if args.evaluate:
        olympus = OLYMPUSRunner()
        results = olympus.evaluate_on_training_set(
            '/content/Arc2025/data/arc-agi_training_challenges.json',
            n_tasks=args.n_tasks
        )
    
    if args.submit:
        olympus = OLYMPUSRunner()
        olympus.create_submission(
            args.submit,
            'submission.json'
        )