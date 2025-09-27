#!/usr/bin/env python3
"""
================================================================================
ARC Prize 2025 - Testing Framework
================================================================================
Test our solution locally before submitting to Kaggle

This is OPEN SOURCE software - no commercial license restrictions
Released under MIT License for the ARC Prize 2025 competition

Author: Andrew Jewell Sr.
Company: AutomataNexus, LLC
Date: September 26, 2024
Version: 1.0.0

Description:
    This framework allows us to:
    1. Test on evaluation data (if available)
    2. Validate our solver accuracy
    3. Measure performance and timing
    4. Identify weak patterns
    5. Improve before submitting
================================================================================
"""

import json
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import our components
from arc_solver import ARCSolver
from pattern_detectors import analyze_task_with_all_detectors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARCTestFramework:
    """Framework for testing our ARC solution"""
    
    def __init__(self, pattern_library_path: str = None):
        self.solver = ARCSolver(pattern_library_path)
        self.results = {
            'correct': 0,
            'total': 0,
            'accuracy_by_pattern': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'timing': [],
            'failures': []
        }
    
    def load_evaluation_data(self) -> Tuple[Dict, Dict]:
        """Load evaluation challenges and solutions"""
        data_dir = Path('/mnt/d/opt/ARCPrize2025/data')
        
        # Try to load evaluation data
        eval_challenges_path = data_dir / 'arc-agi_evaluation_challenges.json'
        eval_solutions_path = data_dir / 'arc-agi_evaluation_solutions.json'
        
        if not eval_challenges_path.exists():
            logger.warning("Evaluation data not found, using subset of training data")
            # Use last 100 training tasks as mock evaluation
            train_path = data_dir / 'arc-agi_training_challenges.json'
            solutions_path = data_dir / 'arc-agi_training_solutions.json'
            
            with open(train_path, 'r') as f:
                all_challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                all_solutions = json.load(f)
            
            # Take last 100 as evaluation set
            eval_tasks = list(all_challenges.keys())[-100:]
            challenges = {k: all_challenges[k] for k in eval_tasks}
            solutions = {k: all_solutions[k] for k in eval_tasks}
        else:
            with open(eval_challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(eval_solutions_path, 'r') as f:
                solutions = json.load(f)
        
        logger.info(f"Loaded {len(challenges)} evaluation tasks")
        return challenges, solutions
    
    def evaluate_task(self, task_id: str, task: Dict, solution: List) -> Dict[str, Any]:
        """Evaluate a single task"""
        start_time = time.time()
        
        # Get predictions from solver
        predictions = self.solver.solve(task)
        
        solve_time = time.time() - start_time
        
        # Check accuracy
        correct = False
        attempt_correct = [False, False]
        
        for i, (pred, actual) in enumerate(zip(predictions, solution)):
            # Check attempt 1
            if np.array_equal(np.array(pred['attempt_1']), np.array(actual)):
                attempt_correct[0] = True
                correct = True
            # Check attempt 2
            elif np.array_equal(np.array(pred['attempt_2']), np.array(actual)):
                attempt_correct[1] = True
                correct = True
        
        # Analyze what patterns were in this task
        train_examples = task.get('train', [])
        detected_patterns = analyze_task_with_all_detectors(train_examples)
        
        # Find which pattern had highest confidence
        best_pattern = None
        best_confidence = 0
        for pattern_name, result in detected_patterns.items():
            if result.get('confidence', 0) > best_confidence:
                best_confidence = result['confidence']
                best_pattern = pattern_name
        
        return {
            'task_id': task_id,
            'correct': correct,
            'attempt_correct': attempt_correct,
            'solve_time': solve_time,
            'best_pattern': best_pattern,
            'pattern_confidence': best_confidence,
            'predictions': predictions,
            'actual': solution
        }
    
    def run_evaluation(self, max_tasks: int = None):
        """Run full evaluation"""
        logger.info("Starting evaluation...")
        
        # Load evaluation data
        challenges, solutions = self.load_evaluation_data()
        
        # Limit tasks if requested
        if max_tasks:
            task_ids = list(challenges.keys())[:max_tasks]
        else:
            task_ids = list(challenges.keys())
        
        # Evaluate each task
        for i, task_id in enumerate(task_ids):
            logger.info(f"Evaluating task {i+1}/{len(task_ids)}: {task_id}")
            
            task = challenges[task_id]
            solution = solutions[task_id]
            
            result = self.evaluate_task(task_id, task, solution)
            
            # Update statistics
            self.results['total'] += 1
            if result['correct']:
                self.results['correct'] += 1
            else:
                self.results['failures'].append(result)
            
            # Update pattern-specific accuracy
            pattern = result['best_pattern']
            if pattern:
                self.results['accuracy_by_pattern'][pattern]['total'] += 1
                if result['correct']:
                    self.results['accuracy_by_pattern'][pattern]['correct'] += 1
            
            self.results['timing'].append(result['solve_time'])
            
            # Log progress every 10 tasks
            if (i + 1) % 10 == 0:
                current_acc = self.results['correct'] / self.results['total']
                logger.info(f"Progress: {i+1}/{len(task_ids)}, Accuracy: {current_acc:.2%}")
    
    def analyze_results(self):
        """Analyze and display results"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        
        # Overall accuracy
        accuracy = self.results['correct'] / self.results['total']
        logger.info(f"\nOverall Accuracy: {accuracy:.2%} ({self.results['correct']}/{self.results['total']})")
        
        # Need 85% to win
        if accuracy >= 0.85:
            logger.info("✓ MEETS 85% THRESHOLD FOR GRAND PRIZE!")
        else:
            logger.info(f"✗ Need {0.85 - accuracy:.2%} more to reach 85% threshold")
        
        # Timing statistics
        avg_time = np.mean(self.results['timing'])
        max_time = np.max(self.results['timing'])
        total_time = np.sum(self.results['timing'])
        
        logger.info(f"\nTiming Statistics:")
        logger.info(f"  Average time per task: {avg_time:.2f}s")
        logger.info(f"  Max time: {max_time:.2f}s")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        
        # Extrapolate to full test set (240 tasks)
        estimated_full_time = (total_time / self.results['total']) * 240
        logger.info(f"  Estimated time for 240 tasks: {estimated_full_time/3600:.1f} hours")
        
        # Pattern-specific accuracy
        logger.info(f"\nAccuracy by Pattern Type:")
        pattern_stats = []
        for pattern, stats in self.results['accuracy_by_pattern'].items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                pattern_stats.append((pattern, acc, stats['total']))
        
        # Sort by accuracy
        pattern_stats.sort(key=lambda x: x[1], reverse=True)
        for pattern, acc, total in pattern_stats:
            logger.info(f"  {pattern}: {acc:.2%} ({total} tasks)")
        
        # Analyze failures
        if self.results['failures']:
            logger.info(f"\nFailure Analysis:")
            logger.info(f"  Total failures: {len(self.results['failures'])}")
            
            # Group failures by pattern
            failure_patterns = defaultdict(int)
            for failure in self.results['failures']:
                pattern = failure.get('best_pattern', 'unknown')
                failure_patterns[pattern] += 1
            
            logger.info("  Failures by pattern:")
            for pattern, count in sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {pattern}: {count} failures")
    
    def visualize_results(self):
        """Create visualization of results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Pattern accuracy bar chart
        ax = axes[0, 0]
        patterns = []
        accuracies = []
        for pattern, stats in self.results['accuracy_by_pattern'].items():
            if stats['total'] > 0:
                patterns.append(pattern)
                accuracies.append(stats['correct'] / stats['total'])
        
        ax.bar(patterns, accuracies)
        ax.set_title('Accuracy by Pattern Type')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.85, color='r', linestyle='--', label='85% threshold')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Timing distribution
        ax = axes[0, 1]
        ax.hist(self.results['timing'], bins=30)
        ax.set_title('Task Solving Time Distribution')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Count')
        
        # 3. Cumulative accuracy over time
        ax = axes[1, 0]
        cumulative_acc = []
        for i in range(1, self.results['total'] + 1):
            acc = sum(1 for f in self.results['failures'][:i] if not f['correct']) / i
            cumulative_acc.append(1 - acc)
        ax.plot(cumulative_acc)
        ax.set_title('Cumulative Accuracy')
        ax.set_xlabel('Tasks Evaluated')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.85, color='r', linestyle='--')
        
        # 4. Summary text
        ax = axes[1, 1]
        ax.text(0.1, 0.9, f"Overall Accuracy: {self.results['correct']/self.results['total']:.2%}", 
                transform=ax.transAxes, fontsize=14, weight='bold')
        ax.text(0.1, 0.7, f"Total Tasks: {self.results['total']}", transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.5, f"Correct: {self.results['correct']}", transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.3, f"Failed: {len(self.results['failures'])}", transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('/mnt/d/opt/ARCPrize2025/test_results.png')
        logger.info("Saved visualization to test_results.png")
    
    def export_failure_analysis(self):
        """Export detailed failure analysis for improvement"""
        output = {
            'summary': {
                'total': self.results['total'],
                'correct': self.results['correct'],
                'accuracy': self.results['correct'] / self.results['total'],
                'failures': len(self.results['failures'])
            },
            'failures': []
        }
        
        # Add detailed failure info
        for failure in self.results['failures'][:10]:  # First 10 failures
            output['failures'].append({
                'task_id': failure['task_id'],
                'pattern': failure['best_pattern'],
                'confidence': failure['pattern_confidence'],
                'attempts': failure['attempt_correct']
            })
        
        # Save to file
        with open('/mnt/d/opt/ARCPrize2025/failure_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info("Exported failure analysis to failure_analysis.json")


def main():
    """Run testing framework"""
    # Initialize framework
    tester = ARCTestFramework()
    
    # Run evaluation on subset first
    logger.info("Running evaluation on 20 tasks for quick test...")
    tester.run_evaluation(max_tasks=20)
    
    # Analyze results
    tester.analyze_results()
    
    # Create visualizations
    tester.visualize_results()
    
    # Export failure analysis
    tester.export_failure_analysis()
    
    logger.info("\nTesting complete! Check results and visualizations.")


if __name__ == "__main__":
    main()