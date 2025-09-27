#!/usr/bin/env python3
"""
================================================================================
ARC Prize 2025 - Custom Neural Network Models
================================================================================
Neural network architectures for ARC pattern recognition

This is OPEN SOURCE software - no commercial license restrictions
Released under MIT License for the ARC Prize 2025 competition

Author: Andrew Jewell Sr.
Company: AutomataNexus, LLC
Date: September 26, 2024
Version: 1.0.0

Description:
    Custom neural network models designed for ARC pattern recognition:
    1. GridTransformerNet - For geometric transformations
    2. ColorMappingNet - For color pattern detection
    3. SymmetryNet - For symmetry detection
    4. ObjectRelationNet - For object relationships
    5. PatternEnsemble - Ensemble of all models
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any


class GridEncoder(nn.Module):
    """Shared encoder for grid inputs"""
    
    def __init__(self, input_channels: int = 10, hidden_dim: int = 128):
        super(GridEncoder, self).__init__()
        
        # Convolutional layers with residual connections
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode grid
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        features = F.relu(self.bn3(self.conv3(x)))
        
        # Global context
        context = self.global_pool(features).squeeze(-1).squeeze(-1)
        context = F.relu(self.context_fc(context))
        
        return features, context


class GridTransformerNet(nn.Module):
    """Network for learning geometric transformations"""
    
    def __init__(self, max_grid_size: int = 30):
        super(GridTransformerNet, self).__init__()
        
        self.encoder = GridEncoder()
        
        # Transformation prediction heads
        self.rotation_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 0°, 90°, 180°, 270°
        )
        
        self.flip_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # none, horizontal, vertical
        )
        
        self.scale_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)  # scale factors 1x to 3x in each dimension
        )
        
        # Spatial transformer for arbitrary transformations
        self.localization = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 2x3 affine matrix
        )
        
        # Initialize the weights/bias with identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode both grids
        _, input_context = self.encoder(input_grid)
        _, output_context = self.encoder(output_grid)
        
        # Combine contexts
        combined = input_context + output_context
        
        # Predict transformations
        rotation_logits = self.rotation_head(combined)
        flip_logits = self.flip_head(combined)
        scale_logits = self.scale_head(combined)
        
        # Spatial transformer parameters
        theta = self.localization(combined)
        theta = theta.view(-1, 2, 3)
        
        return {
            'rotation': rotation_logits,
            'flip': flip_logits,
            'scale': scale_logits,
            'theta': theta
        }


class ColorMappingNet(nn.Module):
    """Network for learning color transformations"""
    
    def __init__(self, num_colors: int = 10):
        super(ColorMappingNet, self).__init__()
        
        self.num_colors = num_colors
        self.encoder = GridEncoder()
        
        # Color attention mechanism
        self.color_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        
        # Color mapping prediction
        self.color_transform = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_colors * num_colors)  # Full color mapping matrix
        )
        
        # Conditional color rules
        self.rule_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Types: direct, conditional, context-based, etc.
        )
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode grids
        input_features, input_context = self.encoder(input_grid)
        output_features, output_context = self.encoder(output_grid)
        
        # Combine contexts with attention
        contexts = torch.stack([input_context, output_context], dim=1)
        attended, _ = self.color_attention(contexts, contexts, contexts)
        combined_context = attended.mean(dim=1)
        
        # Predict color mapping
        color_map = self.color_transform(torch.cat([input_context, output_context], dim=1))
        color_map = color_map.view(-1, self.num_colors, self.num_colors)
        
        # Predict rule type
        rule_type = self.rule_detector(combined_context)
        
        return {
            'color_map': color_map,
            'rule_type': rule_type,
            'attention': attended
        }


class SymmetryNet(nn.Module):
    """Network for detecting and applying symmetry patterns"""
    
    def __init__(self):
        super(SymmetryNet, self).__init__()
        
        self.encoder = GridEncoder()
        
        # Symmetry detection
        self.symmetry_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # horizontal, vertical, diagonal, rotational
        )
        
        # Symmetry axis localization
        self.axis_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # x_center, y_center, angle, type
        )
        
        # Symmetry completion network
        self.completion_net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 10, 3, padding=1)
        )
        
    def forward(self, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode input
        features, context = self.encoder(input_grid)
        
        # Detect symmetries
        symmetry_types = self.symmetry_detector(context)
        axis_params = self.axis_predictor(context)
        
        # Generate symmetry completion
        completed = self.completion_net(features)
        
        return {
            'symmetry_types': symmetry_types,
            'axis_params': axis_params,
            'completed_grid': completed
        }


class ObjectRelationNet(nn.Module):
    """Network for understanding object relationships and movements"""
    
    def __init__(self, max_objects: int = 10):
        super(ObjectRelationNet, self).__init__()
        
        self.max_objects = max_objects
        self.encoder = GridEncoder()
        
        # Object detection and segmentation
        self.object_detector = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, max_objects + 1, 1)  # +1 for background
        )
        
        # Object feature extraction
        self.object_features = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Relational reasoning
        self.relation_net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True),
            num_layers=2
        )
        
        # Movement prediction
        self.movement_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)  # dx, dy, rotate, scale, appear, disappear
        )
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode grids
        input_features, input_context = self.encoder(input_grid)
        output_features, output_context = self.encoder(output_grid)
        
        # Detect objects
        input_objects = self.object_detector(input_features)
        output_objects = self.object_detector(output_features)
        
        # Extract object features
        b, c, h, w = input_features.shape
        object_masks = F.softmax(input_objects, dim=1)
        
        # Pool features for each object
        object_feats = []
        for i in range(self.max_objects):
            mask = object_masks[:, i:i+1, :, :]
            pooled = (input_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-6)
            obj_feat = self.object_features(pooled)
            object_feats.append(obj_feat)
        
        object_feats = torch.stack(object_feats, dim=1)
        
        # Relational reasoning
        relations = self.relation_net(object_feats)
        
        # Predict movements
        movements = self.movement_head(relations)
        
        return {
            'input_objects': input_objects,
            'output_objects': output_objects,
            'object_features': object_feats,
            'relations': relations,
            'movements': movements
        }


class PatternEnsemble(nn.Module):
    """Ensemble of all pattern detection models"""
    
    def __init__(self):
        super(PatternEnsemble, self).__init__()
        
        # Individual models
        self.transformer_net = GridTransformerNet()
        self.color_net = ColorMappingNet()
        self.symmetry_net = SymmetryNet()
        self.object_net = ObjectRelationNet()
        
        # Ensemble combination network
        self.ensemble_combine = nn.Sequential(
            nn.Linear(512, 256),  # Combined features from all models
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Confidence scores for each pattern type
        )
        
        # Final prediction network
        self.predictor = nn.Sequential(
            nn.Linear(128 + 10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Overall confidence
        )
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor = None) -> Dict[str, Any]:
        results = {}
        
        # Get predictions from each model
        if output_grid is not None:
            transform_out = self.transformer_net(input_grid, output_grid)
            color_out = self.color_net(input_grid, output_grid)
            object_out = self.object_net(input_grid, output_grid)
            results.update({
                'transform': transform_out,
                'color': color_out,
                'object': object_out
            })
        
        symmetry_out = self.symmetry_net(input_grid)
        results['symmetry'] = symmetry_out
        
        # Combine features for ensemble prediction
        # This is a simplified version - you'd extract key features from each model
        dummy_features = torch.randn(input_grid.shape[0], 512)  # Placeholder
        pattern_scores = self.ensemble_combine(dummy_features)
        
        # Get encoder features for final prediction
        _, context = self.transformer_net.encoder(input_grid)
        combined = torch.cat([context, pattern_scores], dim=1)
        confidence = self.predictor(combined)
        
        results['pattern_scores'] = pattern_scores
        results['confidence'] = confidence
        
        return results


def create_models() -> Dict[str, nn.Module]:
    """Create all ARC pattern models"""
    return {
        'transformer': GridTransformerNet(),
        'color': ColorMappingNet(),
        'symmetry': SymmetryNet(),
        'object': ObjectRelationNet(),
        'ensemble': PatternEnsemble()
    }


if __name__ == "__main__":
    # Test models
    models = create_models()
    
    # Test input
    batch_size = 4
    input_grid = torch.randn(batch_size, 10, 30, 30)  # 10 colors, 30x30 grid
    output_grid = torch.randn(batch_size, 10, 30, 30)
    
    print("Testing models...")
    for name, model in models.items():
        print(f"\n{name.upper()} Model:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if name in ['transformer', 'color', 'object']:
            out = model(input_grid, output_grid)
        else:
            out = model(input_grid)
        
        print(f"  Output keys: {list(out.keys())}")
    
    print("\nModels created successfully!")