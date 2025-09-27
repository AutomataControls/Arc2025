#!/usr/bin/env python3
"""
Improved accuracy metrics for ARC reconstruction tasks
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class ARCAccuracyMetrics:
    """Calculate various accuracy metrics for ARC predictions"""
    
    @staticmethod
    def calculate_metrics(pred_output: torch.Tensor, target_output: torch.Tensor) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics
        
        Args:
            pred_output: Predicted logits (B, C, H, W)
            target_output: Target one-hot (B, C, H, W)
        
        Returns:
            Dictionary of metrics
        """
        # Get predicted and target colors
        pred_colors = pred_output.argmax(dim=1)  # (B, H, W)
        target_colors = target_output.argmax(dim=1)
        
        B, H, W = pred_colors.shape
        
        # 1. Exact match accuracy (strict)
        exact_matches = (pred_colors == target_colors).all(dim=[1,2])
        exact_accuracy = exact_matches.float().mean().item() * 100
        
        # 2. Pixel-wise accuracy (more forgiving)
        pixel_correct = (pred_colors == target_colors).float()
        pixel_accuracy = pixel_correct.mean().item() * 100
        
        # 3. Active region accuracy (ignore padding)
        # Find active regions (non-zero in either pred or target)
        active_mask = (target_colors != 0) | (pred_colors != 0)
        if active_mask.any():
            active_correct = pixel_correct[active_mask]
            active_accuracy = active_correct.mean().item() * 100
        else:
            active_accuracy = 100.0  # All padding
        
        # 4. Per-sample pixel accuracy (for better tracking)
        per_sample_pixel_acc = pixel_correct.view(B, -1).mean(dim=1)
        avg_sample_accuracy = per_sample_pixel_acc.mean().item() * 100
        
        # 5. IoU-based accuracy
        total_iou = 0.0
        for b in range(B):
            # Calculate IoU for the active region
            pred_active = pred_colors[b] != 0
            target_active = target_colors[b] != 0
            
            # Color-wise IoU
            color_ious = []
            for color in range(1, 10):  # Skip background (0)
                pred_mask = pred_colors[b] == color
                target_mask = target_colors[b] == color
                
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                
                if union > 0:
                    iou = intersection / union
                    color_ious.append(iou.item())
            
            if color_ious:
                sample_iou = np.mean(color_ious)
            else:
                sample_iou = 1.0 if not target_active.any() else 0.0
            
            total_iou += sample_iou
        
        avg_iou = (total_iou / B) * 100
        
        # 6. Pattern-aware accuracy (check if key structures are preserved)
        structure_score = ARCAccuracyMetrics._calculate_structure_score(pred_colors, target_colors)
        
        return {
            'exact_match': exact_accuracy,
            'pixel_wise': pixel_accuracy,
            'active_region': active_accuracy,
            'avg_sample': avg_sample_accuracy,
            'avg_iou': avg_iou,
            'structure': structure_score
        }
    
    @staticmethod
    def _calculate_structure_score(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate how well structural patterns are preserved"""
        B = pred.shape[0]
        scores = []
        
        for b in range(B):
            pred_b = pred[b]
            target_b = target[b]
            
            # Check connected components preservation
            # Simple proxy: check if number of unique colors is similar
            pred_colors = torch.unique(pred_b)
            target_colors = torch.unique(target_b)
            
            color_diff = abs(len(pred_colors) - len(target_colors))
            color_score = 1.0 / (1.0 + color_diff)
            
            # Check spatial distribution
            # Calculate center of mass for each color
            spatial_score = 0.0
            matched_colors = 0
            
            for color in target_colors:
                if color == 0:  # Skip background
                    continue
                    
                target_mask = (target_b == color).float()
                if color in pred_colors:
                    pred_mask = (pred_b == color).float()
                    
                    # Calculate normalized distance between centers
                    if target_mask.sum() > 0 and pred_mask.sum() > 0:
                        target_com = ARCAccuracyMetrics._center_of_mass(target_mask)
                        pred_com = ARCAccuracyMetrics._center_of_mass(pred_mask)
                        
                        dist = torch.norm(target_com - pred_com)
                        max_dist = torch.tensor(target_b.shape).float().norm()
                        norm_dist = dist / max_dist
                        
                        spatial_score += 1.0 - norm_dist.item()
                        matched_colors += 1
            
            if matched_colors > 0:
                spatial_score /= matched_colors
            else:
                spatial_score = 0.0
            
            # Combine scores
            structure_score = (color_score + spatial_score) / 2.0
            scores.append(structure_score)
        
        return np.mean(scores) * 100
    
    @staticmethod
    def _center_of_mass(mask: torch.Tensor) -> torch.Tensor:
        """Calculate center of mass for a binary mask"""
        H, W = mask.shape
        y_coords = torch.arange(H, device=mask.device).float().unsqueeze(1)
        x_coords = torch.arange(W, device=mask.device).float().unsqueeze(0)
        
        total_mass = mask.sum()
        if total_mass == 0:
            return torch.tensor([H/2, W/2], device=mask.device)
        
        y_com = (mask * y_coords).sum() / total_mass
        x_com = (mask * x_coords).sum() / total_mass
        
        return torch.tensor([y_com, x_com], device=mask.device)
    
    @staticmethod
    def get_training_accuracy(metrics: Dict[str, float]) -> float:
        """
        Get a composite accuracy score for training monitoring
        
        Uses a weighted combination of metrics that's more forgiving
        than exact match but still meaningful
        """
        # Weighted combination
        weights = {
            'pixel_wise': 0.3,
            'active_region': 0.3,
            'avg_iou': 0.2,
            'structure': 0.2
        }
        
        composite = sum(metrics[k] * w for k, w in weights.items())
        return composite


def test_metrics():
    """Test the metrics with example predictions"""
    print("Testing improved accuracy metrics...")
    
    # Create test data
    B, C, H, W = 2, 10, 30, 30
    
    # Perfect prediction
    target = torch.zeros(B, C, H, W)
    target[0, 1, 5:10, 5:10] = 1  # Red square
    target[0, 2, 15:20, 15:20] = 1  # Green square
    target[1, 3, 10:15, 10:15] = 1  # Blue square
    
    pred_perfect = target.clone()
    
    # Slightly imperfect prediction
    pred_imperfect = target.clone()
    pred_imperfect[0, 1, 5, 5] = 0  # Remove one pixel
    pred_imperfect[0, 1, 10, 10] = 1  # Add one pixel in wrong place
    
    # Calculate metrics
    metrics_perfect = ARCAccuracyMetrics.calculate_metrics(pred_perfect, target)
    metrics_imperfect = ARCAccuracyMetrics.calculate_metrics(pred_imperfect, target)
    
    print("\nPerfect prediction metrics:")
    for k, v in metrics_perfect.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  Composite: {ARCAccuracyMetrics.get_training_accuracy(metrics_perfect):.2f}%")
    
    print("\nSlightly imperfect prediction metrics:")
    for k, v in metrics_imperfect.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  Composite: {ARCAccuracyMetrics.get_training_accuracy(metrics_imperfect):.2f}%")

if __name__ == "__main__":
    test_metrics()
