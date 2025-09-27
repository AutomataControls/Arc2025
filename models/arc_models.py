#!/usr/bin/env python3
"""
================================================================================
ARC Prize 2025 - Neural Network Models
================================================================================
Neural network architectures for ARC pattern recognition

This is OPEN SOURCE software - no commercial license restrictions
Released under MIT License for the ARC Prize 2025 competition

Author: Andrew Jewell Sr.
Company: AutomataNexus, LLC
Date: September 26, 2024
Version: 1.0.0

Description:
    AI models for ARC pattern recognition:
    1. MINERVA - Wisdom and Strategic Pattern Analysis
    2. ATLAS - Spatial Transformation and Structure Support  
    3. IRIS - Rainbow Color Pattern Recognition
    4. CHRONOS - Temporal Sequence and Evolution Detection
    5. PROMETHEUS - Creative Pattern Generation and Foresight
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any


class AttentionBlock(nn.Module):
    """Multi-head attention block for pattern recognition"""
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class MinervaNet(nn.Module):
    """MINERVA - Goddess of Wisdom: Strategic Pattern Analysis and Decision Making"""
    
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256):
        super(MinervaNet, self).__init__()
        self.name = "MINERVA"
        self.description = "Wisdom and Strategic Pattern Analysis"
        
        # Visual cortex - enhanced pattern recognition
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Strategic reasoning module
        self.reasoning_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Pattern memory bank
        self.pattern_memory = nn.Parameter(torch.randn(100, hidden_dim))
        
        # Decision heads
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 20)  # 20 pattern types
        )
        
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # 8 solving strategies
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        b, c, h, w = input_grid.shape
        
        # Encode visual features
        features = self.visual_encoder(input_grid)
        
        # Flatten and create sequence
        features_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add pattern memory as additional context
        memory_expanded = self.pattern_memory.unsqueeze(0).expand(b, -1, -1)
        sequence = torch.cat([features_flat, memory_expanded], dim=1)
        
        # Strategic reasoning
        reasoned = self.reasoning_transformer(sequence)
        
        # Global representation
        global_repr = reasoned.mean(dim=1)
        
        # Make decisions
        pattern_logits = self.pattern_classifier(global_repr)
        strategy_logits = self.strategy_selector(global_repr)
        confidence = self.confidence_estimator(global_repr)
        
        outputs = {
            'pattern_logits': pattern_logits,
            'strategy_logits': strategy_logits,
            'confidence': confidence,
            'features': features,
            'reasoning': reasoned
        }
        
        # If output grid provided, analyze transformation
        if output_grid is not None:
            output_features = self.visual_encoder(output_grid)
            transformation_features = torch.cat([features, output_features], dim=1).mean(dim=[2, 3])
            outputs['transformation_embedding'] = transformation_features
        
        return outputs


class AtlasNet(nn.Module):
    """ATLAS - Titan of Endurance: Spatial Support and Structural Transformations"""
    
    def __init__(self, max_grid_size: int = 30):
        super(AtlasNet, self).__init__()
        self.name = "ATLAS"
        self.description = "Spatial Transformation and Structure Support"
        
        # Structural analysis layers
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Spatial transformer network
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 2x3 affine matrix
        )
        
        # Initialize with identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Structural integrity analyzer
        self.integrity_checker = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)  # stability, symmetry, completeness
        )
        
        # Transformation predictor
        self.transform_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24)  # rotation(4) + flip(3) + scale(9) + shift(8)
        )
    
    def spatial_transform(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformation"""
        grid = F.affine_grid(theta.view(-1, 2, 3), x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)
    
    def forward(self, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Analyze structure
        structure = self.structure_encoder(input_grid)
        
        # Predict transformation parameters
        theta = self.localization(structure)
        theta = theta.view(-1, 2, 3)
        
        # Apply transformation
        transformed = self.spatial_transform(input_grid, theta)
        
        # Check structural integrity
        integrity = self.integrity_checker(structure)
        integrity_scores = F.sigmoid(integrity.mean(dim=[2, 3]))
        
        # Predict transformation components
        transform_params = self.transform_predictor(structure)
        
        return {
            'transformed_grid': transformed,
            'transformation_matrix': theta,
            'transform_params': transform_params,
            'integrity_scores': integrity_scores,
            'structure_features': structure
        }


class IrisNet(nn.Module):
    """IRIS - Goddess of the Rainbow: Color Pattern Recognition and Harmony"""
    
    def __init__(self, num_colors: int = 10):
        super(IrisNet, self).__init__()
        self.name = "IRIS"
        self.description = "Rainbow Color Pattern Recognition"
        self.num_colors = num_colors
        
        # Color feature extraction
        self.color_encoder = nn.Sequential(
            nn.Conv2d(num_colors, 32, 1),  # Per-color features
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Color relationship attention
        self.color_attention = AttentionBlock(128, heads=num_colors, dim_head=32)
        
        # Harmony analyzer
        self.harmony_net = nn.Sequential(
            nn.Linear(num_colors * num_colors, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # harmony metrics
        )
        
        # Color transformation matrix
        self.color_transformer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_colors * num_colors),
            nn.Unflatten(1, (num_colors, num_colors))
        )
        
        # Pattern type classifier
        self.pattern_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 color pattern types
        )
    
    def analyze_color_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Analyze color distribution in grid"""
        b, c, h, w = x.shape
        color_counts = x.sum(dim=[2, 3]) / (h * w)  # [B, C]
        
        # Create color co-occurrence matrix
        x_flat = x.view(b, c, -1).transpose(1, 2)  # [B, H*W, C]
        cooccurrence = torch.bmm(x_flat.transpose(1, 2), x_flat) / (h * w)
        
        return cooccurrence.view(b, -1)
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        b = input_grid.shape[0]
        
        # Extract color features
        color_features = self.color_encoder(input_grid)
        
        # Analyze color relationships with attention
        h, w = color_features.shape[2:]
        features_seq = color_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attended_features = self.color_attention(features_seq)
        attended_features = attended_features.transpose(1, 2).view(b, -1, h, w)
        
        # Global color representation
        global_color = attended_features.mean(dim=[2, 3])
        
        # Predict color transformation matrix
        color_transform_matrix = self.color_transformer(global_color)
        
        # Analyze harmony
        color_distribution = self.analyze_color_distribution(input_grid)
        harmony_scores = self.harmony_net(color_distribution)
        
        # Classify pattern type
        pattern_type = self.pattern_classifier(attended_features)
        
        outputs = {
            'color_transform_matrix': color_transform_matrix,
            'harmony_scores': F.sigmoid(harmony_scores),
            'pattern_type_logits': pattern_type,
            'color_features': attended_features,
            'color_distribution': color_distribution.view(b, self.num_colors, self.num_colors)
        }
        
        # If output provided, analyze color mapping
        if output_grid is not None:
            output_dist = self.analyze_color_distribution(output_grid)
            outputs['output_distribution'] = output_dist.view(b, self.num_colors, self.num_colors)
            outputs['distribution_change'] = outputs['output_distribution'] - outputs['color_distribution']
        
        return outputs


class ChronosNet(nn.Module):
    """CHRONOS - Titan of Time: Temporal Sequence and Pattern Evolution"""
    
    def __init__(self, hidden_dim: int = 256, max_sequence: int = 10):
        super(ChronosNet, self).__init__()
        self.name = "CHRONOS"
        self.description = "Temporal Sequence and Evolution Detection"
        self.hidden_dim = hidden_dim
        
        # Temporal encoder
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )
        
        # Temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=128 * 8 * 8,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Evolution predictor
        self.evolution_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # evolution types
        )
        
        # Next state predictor
        self.next_state_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 10, 3, padding=1),
        )
        
        # Pattern cycle detector
        self.cycle_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_sequence)  # cycle length prediction
        )
    
    def forward(self, grid_sequence: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a sequence of grids to understand temporal patterns"""
        if not isinstance(grid_sequence, list):
            grid_sequence = [grid_sequence]
        
        batch_size = grid_sequence[0].shape[0]
        
        # Encode each grid
        encoded_sequence = []
        for grid in grid_sequence:
            encoded = self.grid_encoder(grid)
            encoded = encoded.flatten(1)
            encoded_sequence.append(encoded)
        
        # Stack into sequence
        sequence_tensor = torch.stack(encoded_sequence, dim=1)  # [B, T, F]
        
        # Process temporal sequence
        lstm_out, (h_n, c_n) = self.lstm(sequence_tensor)
        
        # Get final representation
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # Bidirectional
        
        # Predict evolution type
        evolution_type = self.evolution_net(final_hidden)
        
        # Predict next state
        next_state = self.next_state_decoder(final_hidden)
        next_state = F.interpolate(next_state, size=(30, 30), mode='bilinear', align_corners=False)
        
        # Detect cycles
        cycle_logits = self.cycle_detector(final_hidden)
        
        return {
            'evolution_type_logits': evolution_type,
            'next_state_prediction': next_state,
            'cycle_length_logits': cycle_logits,
            'temporal_features': lstm_out,
            'final_hidden_state': final_hidden
        }


class PrometheusNet(nn.Module):
    """PROMETHEUS - Titan of Foresight: Creative Pattern Generation and Innovation"""
    
    def __init__(self, latent_dim: int = 512):
        super(PrometheusNet, self).__init__()
        self.name = "PROMETHEUS"
        self.description = "Creative Pattern Generation and Foresight"
        self.latent_dim = latent_dim
        
        # Visionary encoder - sees beyond the obvious
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Latent space projector
        self.to_latent = nn.Sequential(
            nn.Linear(512, latent_dim * 2),
        )
        
        # Creative generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 10, 3, padding=1),
        )
        
        # Innovation scorer
        self.innovation_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # novelty, creativity, validity, complexity, elegance
        )
        
        # Pattern synthesizer
        self.pattern_synthesizer = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 15)  # synthesis strategies
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input_grid: torch.Tensor, generate_variations: bool = False) -> Dict[str, torch.Tensor]:
        b = input_grid.shape[0]
        
        # Encode to latent space
        encoded = self.vision_encoder(input_grid)
        latent_params = self.to_latent(encoded)
        mu, logvar = latent_params.chunk(2, dim=1)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Generate reconstruction
        reconstruction = self.generator(z)
        reconstruction = F.interpolate(reconstruction, size=(30, 30), mode='bilinear', align_corners=False)
        
        # Score innovation potential
        innovation_scores = self.innovation_net(z)
        
        # Predict synthesis strategy
        combined_latent = torch.cat([mu, z], dim=1)
        synthesis_strategy = self.pattern_synthesizer(combined_latent)
        
        outputs = {
            'reconstruction': reconstruction,
            'latent_mean': mu,
            'latent_logvar': logvar,
            'latent_sample': z,
            'innovation_scores': F.sigmoid(innovation_scores),
            'synthesis_strategy_logits': synthesis_strategy
        }
        
        # Generate creative variations
        if generate_variations:
            variations = []
            for _ in range(3):
                z_var = torch.randn_like(z) * 0.5 + z
                var_grid = self.generator(z_var)
                var_grid = F.interpolate(var_grid, size=(30, 30), mode='bilinear', align_corners=False)
                variations.append(var_grid)
            outputs['variations'] = torch.stack(variations, dim=1)
        
        return outputs


def create_models() -> Dict[str, nn.Module]:
    """Create all ARC pattern models"""
    return {
        'minerva': MinervaNet(),
        'atlas': AtlasNet(),
        'iris': IrisNet(),
        'chronos': ChronosNet(),
        'prometheus': PrometheusNet()
    }


if __name__ == "__main__":
    # Test models
    models = create_models()
    
    # Test input
    batch_size = 4
    input_grid = torch.randn(batch_size, 10, 30, 30)  # 10 colors, 30x30 grid
    output_grid = torch.randn(batch_size, 10, 30, 30)
    
    print("Testing ARC Models...")
    print("="*50)
    
    for name, model in models.items():
        print(f"\n{name.upper()} - {model.description}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if name == 'minerva':
            out = model(input_grid, output_grid)
        elif name == 'chronos':
            # Test with sequence
            sequence = [input_grid, output_grid]
            out = model(sequence)
        else:
            out = model(input_grid)
        
        print(f"  Output keys: {list(out.keys())}")
        
        # Show first output shape
        first_key = list(out.keys())[0]
        print(f"  {first_key} shape: {out[first_key].shape}")
    
    print(f"\n{'='*50}")
    print("All models created successfully!")