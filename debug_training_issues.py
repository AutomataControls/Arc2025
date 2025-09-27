#!/usr/bin/env python3
"""
Debug script to diagnose why accuracy is so low
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def analyze_predictions(pred_output, target_output):
    """
    Analyze prediction quality beyond exact match
    """
    # Get predicted and target colors
    pred_colors = pred_output.argmax(dim=1)  # (B, H, W)
    target_colors = target_output.argmax(dim=1)
    
    # Calculate different accuracy metrics
    B, H, W = pred_colors.shape
    
    # 1. Exact match accuracy (current metric)
    exact_matches = (pred_colors == target_colors).all(dim=[1,2])
    exact_accuracy = exact_matches.float().mean().item() * 100
    
    # 2. Pixel-wise accuracy
    pixel_correct = (pred_colors == target_colors).float()
    pixel_accuracy = pixel_correct.mean().item() * 100
    
    # 3. Non-zero pixel accuracy (ignoring background)
    non_zero_mask = target_colors != 0
    if non_zero_mask.any():
        non_zero_correct = pixel_correct[non_zero_mask]
        non_zero_accuracy = non_zero_correct.mean().item() * 100
    else:
        non_zero_accuracy = 0
    
    # 4. Color distribution similarity
    color_similarity_scores = []
    for b in range(B):
        pred_hist = torch.histc(pred_colors[b].float(), bins=10, min=0, max=9)
        target_hist = torch.histc(target_colors[b].float(), bins=10, min=0, max=9)
        
        # Normalize histograms
        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()
        
        # Calculate similarity (1 - JS divergence)
        m = 0.5 * (pred_hist + target_hist)
        js_div = 0.5 * F.kl_div(pred_hist.log(), m, reduction='sum') + \
                 0.5 * F.kl_div(target_hist.log(), m, reduction='sum')
        similarity = 1.0 - js_div.item()
        color_similarity_scores.append(similarity)
    
    color_similarity = np.mean(color_similarity_scores) * 100
    
    # 5. Structure similarity (edge detection)
    def get_edges(grid):
        # Simple edge detection
        dx = torch.abs(grid[:, :, 1:] - grid[:, :, :-1])
        dy = torch.abs(grid[:, 1:, :] - grid[:, :-1, :])
        edges_x = F.pad(dx, (0, 1, 0, 0))
        edges_y = F.pad(dy, (0, 0, 0, 1))
        return (edges_x + edges_y) > 0
    
    pred_edges = get_edges(pred_colors)
    target_edges = get_edges(target_colors)
    edge_accuracy = (pred_edges == target_edges).float().mean().item() * 100
    
    # 6. IoU for each color
    color_ious = []
    for color in range(10):
        pred_mask = (pred_colors == color)
        target_mask = (target_colors == color)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        if union > 0:
            iou = intersection / union
            color_ious.append(iou.item())
    
    avg_iou = np.mean(color_ious) * 100 if color_ious else 0
    
    return {
        'exact_match': exact_accuracy,
        'pixel_wise': pixel_accuracy,
        'non_zero_pixel': non_zero_accuracy,
        'color_distribution': color_similarity,
        'edge_accuracy': edge_accuracy,
        'avg_color_iou': avg_iou
    }

def visualize_predictions(input_grid, target_grid, pred_grid, save_path=None):
    """
    Visualize input, target, and prediction side by side
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Convert to numpy for visualization
    if torch.is_tensor(input_grid):
        input_np = input_grid.argmax(dim=0).cpu().numpy()
    else:
        input_np = input_grid
    
    if torch.is_tensor(target_grid):
        target_np = target_grid.argmax(dim=0).cpu().numpy()
    else:
        target_np = target_grid
        
    if torch.is_tensor(pred_grid):
        pred_np = pred_grid.argmax(dim=0).cpu().numpy()
    else:
        pred_np = pred_grid
    
    # Plot input
    im1 = axes[0].imshow(input_np, cmap='tab10', vmin=0, vmax=9)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # Plot target
    im2 = axes[1].imshow(target_np, cmap='tab10', vmin=0, vmax=9)
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    # Plot prediction
    im3 = axes[2].imshow(pred_np, cmap='tab10', vmin=0, vmax=9)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def check_grid_sizes(dataset):
    """
    Check the distribution of grid sizes in the dataset
    """
    sizes = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        input_grid = sample['input']
        output_grid = sample['output']
        
        # Get actual size before padding
        input_size = (input_grid.sum(dim=0) > 0).any(dim=0).sum().item()
        output_size = (output_grid.sum(dim=0) > 0).any(dim=0).sum().item()
        
        sizes.append({
            'input_h': (input_grid.sum(dim=0) > 0).any(dim=1).sum().item(),
            'input_w': input_size,
            'output_h': (output_grid.sum(dim=0) > 0).any(dim=1).sum().item(),
            'output_w': output_size
        })
    
    # Calculate statistics
    avg_input_size = np.mean([(s['input_h'] + s['input_w']) / 2 for s in sizes])
    avg_output_size = np.mean([(s['output_h'] + s['output_w']) / 2 for s in sizes])
    
    print(f"Average input grid size: {avg_input_size:.1f}")
    print(f"Average output grid size: {avg_output_size:.1f}")
    
    # Check how many have size changes
    size_changes = sum(1 for s in sizes if s['input_h'] != s['output_h'] or s['input_w'] != s['output_w'])
    print(f"Grids with size changes: {size_changes}/{len(sizes)} ({size_changes/len(sizes)*100:.1f}%)")

def diagnose_training_issue():
    """
    Main diagnostic function
    """
    print("=" * 80)
    print("DIAGNOSING LOW ACCURACY ISSUE")
    print("=" * 80)
    
    # Test with a simple pattern
    print("\n1. Testing with simple rotation pattern...")
    
    # Create a simple test case
    test_input = torch.zeros(1, 10, 30, 30)
    test_input[0, 1, 5:10, 5:10] = 1  # Red square
    test_input[0, 2, 15:20, 15:20] = 1  # Green square
    
    # Target is 90 degree rotation
    test_target = torch.zeros(1, 10, 30, 30)
    test_target[0, 1, 5:10, 20:25] = 1  # Red square rotated
    test_target[0, 2, 15:20, 10:15] = 1  # Green square rotated
    
    # Simulate model prediction (perfect for testing)
    test_pred = test_target.clone()
    # Add small noise
    test_pred[0, 3, 0, 0] = 0.1  # Tiny noise
    
    metrics = analyze_predictions(test_pred, test_target)
    print("\nMetrics for perfect prediction with tiny noise:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}%")
    
    # Test with slightly imperfect prediction
    print("\n2. Testing with slightly imperfect prediction...")
    test_pred_imperfect = test_target.clone()
    test_pred_imperfect[0, 1, 5, 5] = 0  # Remove one pixel
    test_pred_imperfect[0, 3, 5, 5] = 1  # Add wrong color pixel
    
    metrics = analyze_predictions(test_pred_imperfect, test_target)
    print("\nMetrics for slightly imperfect prediction:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}%")
    
    print("\n" + "=" * 80)
    print("INSIGHTS:")
    print("1. Exact match accuracy is extremely strict - even one pixel off = 0%")
    print("2. We should track pixel-wise accuracy during training")
    print("3. The model might be learning patterns but not achieving exact reconstruction")
    print("4. Consider using a softer accuracy metric for early stopping")
    print("=" * 80)

def suggest_improvements():
    """
    Suggest improvements to the training process
    """
    print("\nSUGGESTED IMPROVEMENTS:")
    print("1. Add pixel-wise accuracy tracking to monitor")
    print("2. Use IoU-based metrics for validation")
    print("3. Add intermediate supervision (auxiliary losses)")
    print("4. Implement curriculum learning (start with simpler patterns)")
    print("5. Use teacher forcing for initial epochs")
    print("6. Add attention visualization to debug what model is learning")
    
if __name__ == "__main__":
    diagnose_training_issue()
    suggest_improvements()
