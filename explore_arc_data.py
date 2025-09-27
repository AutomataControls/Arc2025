#!/usr/bin/env python3
"""
================================================================================
ARC Prize 2025 - Data Exploration Script
================================================================================
Explores and analyzes ARC training data to understand patterns

This is OPEN SOURCE software - no commercial license restrictions
Released under MIT License for the ARC Prize 2025 competition

Author: Andrew Jewell Sr.
Company: AutomataNexus, LLC
Date: September 26, 2024
Version: 1.0.0

Description:
    This script explores the ARC training data to understand:
    - Task complexity and variety
    - Common transformation patterns
    - Grid size distributions
    - Color usage patterns
    - Example structures
    
    This analysis helps design appropriate pattern detectors for the competition.
================================================================================
"""

import json
import numpy as np
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple

# Color mapping for visualization
COLOR_MAP = {
    0: '#000000',  # Black
    1: '#0074D9',  # Blue
    2: '#FF4136',  # Red
    3: '#2ECC40',  # Green
    4: '#FFDC00',  # Yellow
    5: '#AAAAAA',  # Gray
    6: '#F012BE',  # Fuchsia
    7: '#FF851B',  # Orange
    8: '#7FDBFF',  # Teal
    9: '#870C25'   # Brown
}


def extract_and_load_data(zip_path: str) -> Tuple[Dict, Dict]:
    """Extract and load ARC data from zip file or existing directory"""
    print(f"Loading data from existing directory")
    
    # Data already extracted to /mnt/d/opt/ARCPrize2025/data
    extract_dir = Path("/mnt/d/opt/ARCPrize2025/data")
    
    # Load training challenges and solutions
    train_challenges_path = extract_dir / "arc-agi_training_challenges.json"
    train_solutions_path = extract_dir / "arc-agi_training_solutions.json"
    
    with open(train_challenges_path, 'r') as f:
        challenges = json.load(f)
    
    with open(train_solutions_path, 'r') as f:
        solutions = json.load(f)
    
    print(f"Loaded {len(challenges)} training challenges")
    return challenges, solutions


def analyze_task_statistics(challenges: Dict, solutions: Dict) -> Dict:
    """Analyze statistics of the tasks"""
    stats = {
        'total_tasks': len(challenges),
        'grid_sizes': defaultdict(int),
        'train_examples_count': defaultdict(int),
        'test_examples_count': defaultdict(int),
        'colors_used': defaultdict(int),
        'size_changes': defaultdict(int),
        'complexity_scores': []
    }
    
    for task_id, task in challenges.items():
        # Count train examples
        train_count = len(task.get('train', []))
        stats['train_examples_count'][train_count] += 1
        
        # Count test examples
        test_count = len(task.get('test', []))
        stats['test_examples_count'][test_count] += 1
        
        # Analyze grid sizes and colors
        all_grids = []
        for ex in task.get('train', []):
            all_grids.append(('input', np.array(ex['input'])))
            all_grids.append(('output', np.array(ex['output'])))
        
        for test_ex in task.get('test', []):
            all_grids.append(('test_input', np.array(test_ex['input'])))
        
        # Add solution if available
        if task_id in solutions:
            solution_data = solutions[task_id]
            # Solutions is a list of outputs
            for i, sol in enumerate(solution_data):
                all_grids.append(('solution', np.array(sol)))
        
        # Analyze grids
        for grid_type, grid in all_grids:
            size_key = f"{grid.shape[0]}x{grid.shape[1]}"
            stats['grid_sizes'][size_key] += 1
            
            # Count colors
            unique_colors = np.unique(grid)
            stats['colors_used'][len(unique_colors)] += 1
        
        # Analyze size changes
        if task.get('train'):
            for ex in task['train']:
                input_shape = np.array(ex['input']).shape
                output_shape = np.array(ex['output']).shape
                
                if input_shape == output_shape:
                    stats['size_changes']['same'] += 1
                elif output_shape[0] > input_shape[0] or output_shape[1] > input_shape[1]:
                    stats['size_changes']['larger'] += 1
                else:
                    stats['size_changes']['smaller'] += 1
        
        # Calculate complexity score
        complexity = calculate_task_complexity(task)
        stats['complexity_scores'].append(complexity)
    
    return stats


def calculate_task_complexity(task: Dict) -> float:
    """Calculate a complexity score for a task"""
    complexity = 0.0
    
    # Factor 1: Number of training examples
    train_count = len(task.get('train', []))
    complexity += (5 - train_count) * 0.1  # Fewer examples = harder
    
    # Factor 2: Grid size variance
    sizes = []
    for ex in task.get('train', []):
        sizes.append(np.array(ex['input']).size)
        sizes.append(np.array(ex['output']).size)
    
    if sizes:
        size_variance = np.var(sizes)
        complexity += min(size_variance / 100, 1.0)
    
    # Factor 3: Color complexity
    all_colors = set()
    for ex in task.get('train', []):
        all_colors.update(np.unique(ex['input']))
        all_colors.update(np.unique(ex['output']))
    
    complexity += len(all_colors) * 0.1
    
    return complexity


def visualize_task(task: Dict, task_id: str, solution: Dict = None, save_path: str = None):
    """Visualize a single ARC task"""
    train_examples = task.get('train', [])
    test_examples = task.get('test', [])
    
    # Calculate layout
    n_train = len(train_examples)
    n_test = len(test_examples)
    
    # Create figure
    fig_height = 2 * n_train + 2 * n_test
    fig = plt.figure(figsize=(10, fig_height))
    
    # Plot training examples
    for i, ex in enumerate(train_examples):
        # Input
        ax = plt.subplot(n_train + n_test, 2, 2*i + 1)
        plot_grid(ax, ex['input'], f'Train {i+1} Input')
        
        # Output
        ax = plt.subplot(n_train + n_test, 2, 2*i + 2)
        plot_grid(ax, ex['output'], f'Train {i+1} Output')
    
    # Plot test examples
    for i, test_ex in enumerate(test_examples):
        row = n_train + i
        
        # Test input
        ax = plt.subplot(n_train + n_test, 2, 2*row + 1)
        plot_grid(ax, test_ex['input'], f'Test {i+1} Input')
        
        # Test output (solution if available)
        ax = plt.subplot(n_train + n_test, 2, 2*row + 2)
        if solution and i < len(solution):
            plot_grid(ax, solution[i], f'Test {i+1} Solution')
        else:
            ax.text(0.5, 0.5, '?', ha='center', va='center', fontsize=40)
            ax.set_title(f'Test {i+1} Output (Unknown)')
            ax.axis('off')
    
    plt.suptitle(f'Task: {task_id}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_grid(ax, grid, title):
    """Plot a single grid"""
    grid = np.array(grid)
    height, width = grid.shape
    
    # Create image
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=9)
    
    # Add grid lines
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    # Add border
    ax.add_patch(patches.Rectangle((-0.5, -0.5), width, height, 
                                  linewidth=2, edgecolor='black', facecolor='none'))


def find_pattern_examples(challenges: Dict) -> Dict[str, List[str]]:
    """Find examples of different pattern types"""
    patterns = {
        'rotation': [],
        'reflection': [],
        'scaling': [],
        'color_mapping': [],
        'symmetry': [],
        'object_movement': [],
        'counting': [],
        'conditional': []
    }
    
    for task_id, task in challenges.items():
        train = task.get('train', [])
        
        if not train:
            continue
        
        # Check for specific patterns
        if is_rotation_pattern(train):
            patterns['rotation'].append(task_id)
        
        if is_reflection_pattern(train):
            patterns['reflection'].append(task_id)
        
        if is_scaling_pattern(train):
            patterns['scaling'].append(task_id)
        
        if is_color_mapping_pattern(train):
            patterns['color_mapping'].append(task_id)
        
        if is_symmetry_pattern(train):
            patterns['symmetry'].append(task_id)
        
        # Limit to first 5 examples of each type
        for pattern_type in patterns:
            patterns[pattern_type] = patterns[pattern_type][:5]
    
    return patterns


def is_rotation_pattern(train_examples: List[Dict]) -> bool:
    """Check if examples show rotation pattern"""
    for ex in train_examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])
        
        # Check for 90, 180, 270 degree rotations
        for k in [1, 2, 3]:
            if input_grid.shape == output_grid.shape and np.array_equal(np.rot90(input_grid, k=k), output_grid):
                return True
    return False


def is_reflection_pattern(train_examples: List[Dict]) -> bool:
    """Check if examples show reflection pattern"""
    for ex in train_examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])
        
        if input_grid.shape == output_grid.shape:
            # Check horizontal and vertical flips
            if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                return True
            if np.array_equal(np.flip(input_grid, axis=1), output_grid):
                return True
    return False


def is_scaling_pattern(train_examples: List[Dict]) -> bool:
    """Check if examples show scaling pattern"""
    for ex in train_examples:
        input_shape = np.array(ex['input']).shape
        output_shape = np.array(ex['output']).shape
        
        # Check if output is scaled version
        if (output_shape[0] % input_shape[0] == 0 and 
            output_shape[1] % input_shape[1] == 0 and
            (output_shape[0] > input_shape[0] or output_shape[1] > input_shape[1])):
            return True
    return False


def is_color_mapping_pattern(train_examples: List[Dict]) -> bool:
    """Check if examples show consistent color mapping"""
    if not train_examples:
        return False
    
    # Check if all examples have same size transformation
    first_in_shape = np.array(train_examples[0]['input']).shape
    first_out_shape = np.array(train_examples[0]['output']).shape
    
    for ex in train_examples[1:]:
        if (np.array(ex['input']).shape != first_in_shape or 
            np.array(ex['output']).shape != first_out_shape):
            return False
    
    # Check for consistent color changes
    return first_in_shape == first_out_shape


def is_symmetry_pattern(train_examples: List[Dict]) -> bool:
    """Check if examples involve symmetry"""
    for ex in train_examples:
        output_grid = np.array(ex['output'])
        
        # Check if output is symmetric
        if np.array_equal(output_grid, np.flip(output_grid, axis=0)):
            return True
        if np.array_equal(output_grid, np.flip(output_grid, axis=1)):
            return True
    return False


def print_statistics(stats: Dict):
    """Print analysis statistics"""
    print("\n" + "="*60)
    print("ARC TRAINING DATA STATISTICS")
    print("="*60)
    
    print(f"\nTotal Tasks: {stats['total_tasks']}")
    
    print("\nTraining Examples per Task:")
    for count, freq in sorted(stats['train_examples_count'].items()):
        print(f"  {count} examples: {freq} tasks")
    
    print("\nTest Examples per Task:")
    for count, freq in sorted(stats['test_examples_count'].items()):
        print(f"  {count} examples: {freq} tasks")
    
    print("\nMost Common Grid Sizes:")
    size_counter = Counter(stats['grid_sizes'])
    for size, count in size_counter.most_common(10):
        print(f"  {size}: {count} occurrences")
    
    print("\nColors Used Distribution:")
    for n_colors, count in sorted(stats['colors_used'].items()):
        print(f"  {n_colors} colors: {count} grids")
    
    print("\nSize Change Patterns:")
    for change_type, count in stats['size_changes'].items():
        print(f"  {change_type}: {count} examples")
    
    print("\nComplexity Statistics:")
    complexities = stats['complexity_scores']
    print(f"  Mean: {np.mean(complexities):.2f}")
    print(f"  Std: {np.std(complexities):.2f}")
    print(f"  Min: {np.min(complexities):.2f}")
    print(f"  Max: {np.max(complexities):.2f}")


def main():
    """Main exploration function"""
    # Extract and load data
    zip_path = "/mnt/d/Downloads/arc-prize-2025.zip"
    challenges, solutions = extract_and_load_data(zip_path)
    
    # Analyze statistics
    stats = analyze_task_statistics(challenges, solutions)
    print_statistics(stats)
    
    # Find pattern examples
    print("\n" + "="*60)
    print("PATTERN EXAMPLES FOUND")
    print("="*60)
    
    patterns = find_pattern_examples(challenges)
    for pattern_type, task_ids in patterns.items():
        if task_ids:
            print(f"\n{pattern_type.upper()} Pattern Examples:")
            for task_id in task_ids[:3]:  # Show first 3
                print(f"  - {task_id}")
    
    # Visualize a few interesting tasks
    print("\n" + "="*60)
    print("VISUALIZING SAMPLE TASKS")
    print("="*60)
    
    # Create visualization directory
    viz_dir = Path("/mnt/d/opt/ARCPrize2025/visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # Visualize first few tasks
    task_ids = list(challenges.keys())[:5]
    for task_id in task_ids:
        print(f"\nVisualizing task: {task_id}")
        solution = solutions.get(task_id, None)
        save_path = viz_dir / f"task_{task_id}.png"
        visualize_task(challenges[task_id], task_id, solution, str(save_path))
    
    print(f"\nVisualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()