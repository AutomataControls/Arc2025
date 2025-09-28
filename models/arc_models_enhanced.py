"""
Enhanced ARC Models - Breaking the 70% Barrier
These models include critical improvements for ARC reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class GridAttention(nn.Module):
    """Grid-aware attention mechanism for ARC tasks"""
    def __init__(self, channels: int, grid_size: int = 30):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        
        # Learnable position embeddings for grid
        self.row_embed = nn.Parameter(torch.randn(grid_size, channels // 2))
        self.col_embed = nn.Parameter(torch.randn(grid_size, channels // 2))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        
        # Projection layers for dimension matching
        self.input_proj = None
        self.output_proj = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Initialize projection layers if needed
        if self.input_proj is None and C != self.channels:
            self.input_proj = nn.Linear(C, self.channels).to(x.device)
            self.output_proj = nn.Linear(self.channels, C).to(x.device)
        
        # Apply attention with position awareness
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # B, H*W, C
        
        # Project to attention dimension if needed
        if C != self.channels:
            x_flat = self.input_proj(x_flat)
        
        # Add positional embeddings
        row_emb = self.row_embed[:H, :].unsqueeze(1).expand(-1, W, -1)  # H, W, channels//2
        col_emb = self.col_embed[:W, :].unsqueeze(0).expand(H, -1, -1)  # H, W, channels//2
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # H, W, channels
        pos_emb = pos_emb.reshape(H*W, self.channels).unsqueeze(0).expand(B, -1, -1)  # B, H*W, channels
        
        x_with_pos = x_flat + pos_emb
        
        # Apply attention
        attended, _ = self.attention(x_with_pos, x_with_pos, x_with_pos)
        
        # Project back if needed
        if C != self.channels:
            attended = self.output_proj(attended)
        
        return attended.permute(0, 2, 1).view(B, C, H, W)


class ObjectEncoder(nn.Module):
    """Extract and encode objects from grids"""
    def __init__(self, in_channels: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, 1)
        
        # Object detection head
        self.object_conv = nn.Conv2d(hidden_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features with residual connections
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        features = self.conv3(h2)
        
        # Detect objects with stronger activation
        object_masks = torch.sigmoid(self.object_conv(features) * 2.0)  # Scale up for sharper masks
        
        # Masked features with residual
        object_features = features * object_masks + features * 0.2  # Keep some global features
        
        return object_features, object_masks


class RelationalReasoning(nn.Module):
    """Reason about relationships between grid elements"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Relation network
        self.relation_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        
        # Use a more efficient approach - compute relations on downsampled features
        # Downsample to reduce memory usage
        if H > 8 or W > 8:
            features_small = F.adaptive_avg_pool2d(features, (8, 8))
            h_small, w_small = 8, 8
        else:
            features_small = features
            h_small, w_small = H, W
        
        # Flatten spatial dimensions
        features_flat = features_small.view(B, C, -1).permute(0, 2, 1)  # B, h*w, C
        N = features_flat.shape[1]
        
        # Instead of all pairs, use local neighborhood relations
        # Create a local attention pattern (e.g., 3x3 neighborhood)
        kernel_size = 3
        padding = kernel_size // 2
        
        # Unfold to get local neighborhoods
        features_unfold = F.unfold(features_small, kernel_size=kernel_size, padding=padding)  # B, C*k*k, h*w
        features_unfold = features_unfold.view(B, C, kernel_size*kernel_size, -1).permute(0, 3, 2, 1)  # B, h*w, k*k, C
        
        # Center features
        center_features = features_flat.unsqueeze(2)  # B, h*w, 1, C
        
        # Compute local relations
        local_relations = []
        for i in range(kernel_size * kernel_size):
            neighbor = features_unfold[:, :, i:i+1, :]  # B, h*w, 1, C
            pair = torch.cat([center_features, neighbor], dim=-1)  # B, h*w, 1, 2*C
            relation = self.relation_net(pair.view(-1, 2*C))  # (B*h*w, C)
            local_relations.append(relation.view(B, N, C))
        
        # Aggregate local relations
        aggregated = torch.stack(local_relations, dim=2).mean(dim=2)  # B, N, C
        aggregated = aggregated.permute(0, 2, 1).view(B, C, h_small, w_small)
        
        # Upsample back to original size if needed
        if H != h_small or W != w_small:
            aggregated = F.interpolate(aggregated, size=(H, W), mode='bilinear', align_corners=False)
        
        return aggregated


class TransformationPredictor(nn.Module):
    """Learn to predict transformations from examples"""
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Encode input-output pairs
        self.pair_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Predict transformation parameters
        self.transform_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, input_features: torch.Tensor, output_features: torch.Tensor) -> torch.Tensor:
        # Global pooling
        input_global = F.adaptive_avg_pool2d(input_features, 1).squeeze(-1).squeeze(-1)
        output_global = F.adaptive_avg_pool2d(output_features, 1).squeeze(-1).squeeze(-1)
        
        # Encode transformation
        pair_features = torch.cat([input_global, output_global], dim=1)
        encoded = self.pair_encoder(pair_features)
        
        # Predict transformation
        transform_params = self.transform_head(encoded)
        
        return transform_params


class EnhancedMinervaNet(nn.Module):
    """Enhanced MINERVA with grid reasoning capabilities"""
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        
        # Grid encoder with object awareness
        self.object_encoder = ObjectEncoder(10, hidden_dim)
        
        # Grid attention
        self.grid_attention = GridAttention(hidden_dim, max_grid_size)
        
        # Relational reasoning
        self.relational = RelationalReasoning(hidden_dim)
        
        # Transformation learning
        self.transform_predictor = TransformationPredictor(hidden_dim)
        
        # Memory bank for pattern storage - initialize with more diverse patterns
        self.pattern_memory = nn.Parameter(torch.randn(200, hidden_dim) * 0.5)
        
        # Pattern attention for better matching
        self.pattern_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Transform projection layer
        self.transform_proj = nn.Linear(128, hidden_dim)
        
        # Simple mixing parameter - start at 0.05 to heavily favor transformations
        self.mix_param = nn.Parameter(torch.tensor(0.05))
        
        # Output decoder - predicts actual output grid
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 10, 1),  # 10 colors
            # NO SOFTMAX - CrossEntropyLoss expects raw logits
        )
        
        # Initialize final layer with stronger values for more decisive predictions
        # Use xavier with higher gain for bolder outputs
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=2.0)
        # Initialize bias to encourage all colors equally
        # Give all colors including background equal chance
        bias_values = torch.ones(10) * 0.1  # Larger bias for stronger predictions
        self.decoder[-1].bias.data = bias_values
        
        self.description = "Enhanced Strategic Pattern Analysis with Grid Reasoning"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        
        # Extract object features
        input_features, input_objects = self.object_encoder(input_grid)
        
        # Apply grid attention
        attended_features = self.grid_attention(input_features)
        
        # Relational reasoning
        relational_features = self.relational(attended_features)
        
        # Combine features
        combined_features = attended_features + relational_features
        
        if mode == 'train' and output_grid is not None:
            # Learn transformation
            output_features, _ = self.object_encoder(output_grid)
            transform_params = self.transform_predictor(combined_features, output_features)
            
            # Apply transformation to predict output
            transformed = self._apply_transform(combined_features, transform_params)
            predicted_output = self.decoder(torch.cat([combined_features, transformed], dim=1))
            
            # Mix prediction with input using learnable parameter
            # Sigmoid to keep in [0, 1] range - but favor the prediction!
            mix = torch.sigmoid(self.mix_param)
            
            # During training, use even less input mixing to learn transformations
            if self.training:
                mix = mix * 0.2  # Reduce input contribution by 80% during training
                
            predicted_output = predicted_output * (1 - mix) + input_grid * mix  # Inverted to favor prediction
            
            return {
                'predicted_output': predicted_output,
                'transform_params': transform_params,
                'object_masks': input_objects,
                'features': combined_features
            }
        else:
            # Inference mode - find best matching pattern
            best_transform = self._find_best_pattern(combined_features)
            transformed = self._apply_transform(combined_features, best_transform)
            predicted_output = self.decoder(torch.cat([combined_features, transformed], dim=1))
            
            # Mix prediction with input
            mix = torch.sigmoid(self.mix_param)
            
            # During inference, still favor the transformation heavily
            if self.training:
                mix = mix * 0.2
            else:
                mix = mix * 0.3
                
            predicted_output = predicted_output * (1 - mix) + input_grid * mix  # Inverted to favor prediction
            
            return {
                'predicted_output': predicted_output,
                'transform_params': best_transform,
                'object_masks': input_objects
            }
    
    def _apply_transform(self, features: torch.Tensor, transform_params: torch.Tensor) -> torch.Tensor:
        """Apply learned transformation to features"""
        B, C, H, W = features.shape
        
        # Project transform params to match feature channels
        transform_params = self.transform_proj(transform_params)
        
        # Use transform params to modulate features
        transform_matrix = transform_params.view(B, C, 1, 1)
        
        # Apply EVEN STRONGER transformation - use tanh for wider range [-1, 1]
        # Then scale up to allow big changes
        transformed = features * (1.0 + 3.0 * torch.tanh(transform_matrix))
        
        return transformed
    
    def _find_best_pattern(self, features: torch.Tensor) -> torch.Tensor:
        """Find best matching pattern from memory using attention"""
        B = features.shape[0]
        
        # Pool features for comparison
        features_pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # B, hidden_dim
        
        # Use attention to find best patterns
        query = features_pooled.unsqueeze(1)  # B, 1, hidden_dim
        keys = self.pattern_memory.unsqueeze(0).expand(B, -1, -1)  # B, 200, hidden_dim
        
        # Apply pattern attention
        attended_patterns, attention_weights = self.pattern_attention(query, keys, keys)
        attended_patterns = attended_patterns.squeeze(1)  # B, hidden_dim
        
        # Also get top-k patterns for diversity
        similarity = F.cosine_similarity(features_pooled.unsqueeze(1), self.pattern_memory.unsqueeze(0), dim=2)
        top_k_values, top_k_idx = similarity.topk(3, dim=1)
        
        # Weighted combination of top patterns
        weighted_patterns = torch.zeros_like(attended_patterns)
        for i in range(3):
            weight = F.softmax(top_k_values, dim=1)[:, i:i+1]
            pattern_idx = top_k_idx[:, i]
            weighted_patterns += weight * self.pattern_memory[pattern_idx]
        
        # Combine attended and weighted patterns
        combined_patterns = 0.7 * attended_patterns + 0.3 * weighted_patterns
        
        # Return as transform parameters
        transform_params = torch.zeros(B, 128).to(features.device)
        transform_params[:, :min(128, combined_patterns.shape[1])] = combined_patterns[:, :min(128, combined_patterns.shape[1])]
        
        return transform_params


class EnhancedAtlasNet(nn.Module):
    """Enhanced ATLAS with better spatial transformation learning"""
    def __init__(self, max_grid_size: int = 30):
        super().__init__()
        self.max_grid_size = max_grid_size
        
        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Spatial transformer network
        self.localization = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Affine transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 2x3 affine matrix
        )
        
        # Rotation predictor
        self.rotation_head = nn.Linear(128, 4)  # 0°, 90°, 180°, 270°
        
        # Reflection predictor  
        self.reflection_head = nn.Linear(128, 3)  # None, Horizontal, Vertical
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer with strong values
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.2)
        # Equal bias for all colors in ATLAS
        self.decoder[-1].bias.data = torch.ones(10) * 0.05
        
        # Initialize affine matrix to identity
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Mix parameter - start at 0.05 to heavily favor transformations
        self.mix_param = nn.Parameter(torch.tensor(0.05))
        
        self.description = "Enhanced Spatial Transformer with Rotation/Reflection"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B = input_grid.shape[0]
        
        # Encode features
        features = self.encoder(input_grid)
        
        # Predict spatial transformation
        loc_features = self.localization(features)
        theta = self.fc_loc(loc_features.view(B, -1))
        theta = theta.view(-1, 2, 3)
        
        # Create sampling grid
        grid = F.affine_grid(theta, input_grid.size(), align_corners=False)
        
        # Apply spatial transformation
        transformed_input = F.grid_sample(input_grid, grid, align_corners=False)
        transformed_features = F.grid_sample(features, grid, align_corners=False)
        
        # Predict rotation and reflection
        pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        rotation_logits = self.rotation_head(pooled)
        reflection_logits = self.reflection_head(pooled)
        
        # Apply discrete transformations
        transformed_features = self._apply_discrete_transforms(
            transformed_features, rotation_logits, reflection_logits
        )
        
        # Decode to output
        predicted_output = self.decoder(transformed_features)
        
        # Add minimal residual for ATLAS to prevent collapse
        mix = torch.sigmoid(self.mix_param)
        if self.training:
            mix = mix * 0.2  # Use much less residual for ATLAS
        else:
            mix = mix * 0.3
        predicted_output = predicted_output * (1 - mix) + input_grid * mix
        
        return {
            'predicted_output': predicted_output,
            'theta': theta,
            'rotation_logits': rotation_logits,
            'reflection_logits': reflection_logits,
            'transformed_input': transformed_input
        }
    
    def _apply_discrete_transforms(self, features: torch.Tensor, 
                                  rotation_logits: torch.Tensor,
                                  reflection_logits: torch.Tensor) -> torch.Tensor:
        """Apply predicted rotations and reflections"""
        B = features.shape[0]
        
        # Get predictions
        rotation_idx = rotation_logits.argmax(dim=1)
        reflection_idx = reflection_logits.argmax(dim=1)
        
        output = features.clone()
        
        for i in range(B):
            # Apply rotation
            rot = rotation_idx[i].item()
            if rot > 0:
                output[i] = torch.rot90(output[i], k=rot, dims=[1, 2])
            
            # Apply reflection
            ref = reflection_idx[i].item()
            if ref == 1:  # Horizontal
                output[i] = torch.flip(output[i], dims=[1])
            elif ref == 2:  # Vertical
                output[i] = torch.flip(output[i], dims=[2])
        
        return output


class EnhancedIrisNet(nn.Module):
    """Enhanced IRIS with color relationship learning"""
    def __init__(self, max_grid_size: int = 30):
        super().__init__()
        
        # Color embedding
        self.color_embed = nn.Embedding(10, 64)
        
        # Color attention mechanism
        self.color_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        
        # Spatial color encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # Color mapping predictor
        self.color_mapper = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10 * 10)  # 10x10 color mapping matrix
        )
        
        # Pattern-based color rules
        self.rule_encoder = nn.LSTM(64, 128, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer for strong color changes
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.5)
        # Slightly varied bias for IRIS color specialization
        self.decoder[-1].bias.data = torch.tensor([0.0, 0.08, 0.06, 0.07, 0.05, 0.06, 0.08, 0.09, 0.07, 0.06])
        
        self.description = "Enhanced Color Pattern Recognition with Attention"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B, C, H, W = input_grid.shape
        
        # Get color distribution
        color_indices = input_grid.argmax(dim=1)  # B, H, W
        
        # Embed colors
        color_embeddings = self.color_embed(color_indices)  # B, H, W, 64
        color_flat = color_embeddings.view(B, -1, 64)
        
        # Color attention
        attended_colors, color_weights = self.color_attention(color_flat, color_flat, color_flat)
        attended_colors = attended_colors.view(B, H, W, 64).permute(0, 3, 1, 2)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(input_grid)
        
        # Combine color and spatial
        combined = spatial_features + attended_colors
        
        # Predict color mapping
        global_features = F.adaptive_avg_pool2d(combined, 1).squeeze(-1).squeeze(-1)
        color_map_logits = self.color_mapper(global_features).view(B, 10, 10)
        
        # Apply color mapping
        mapped_output = self._apply_color_mapping(input_grid, color_map_logits)
        
        # Final decode
        predicted_output = self.decoder(combined)
        
        # For IRIS, we don't add residual - it does color mapping
        # The mapped_output already contains the transformed colors
        
        return {
            'predicted_output': predicted_output,
            'color_map': F.softmax(color_map_logits, dim=-1),
            'color_attention': color_weights,
            'mapped_output': mapped_output
        }
    
    def _apply_color_mapping(self, input_grid: torch.Tensor, color_map: torch.Tensor) -> torch.Tensor:
        """Apply learned color mapping"""
        B, C, H, W = input_grid.shape
        
        # Get input colors
        color_indices = input_grid.argmax(dim=1, keepdim=True).float()  # B, 1, H, W
        
        # Apply mapping
        output = torch.zeros_like(input_grid)
        
        for b in range(B):
            for c_in in range(10):
                mask = (color_indices[b, 0] == c_in)
                if mask.any():
                    # Get mapped color probabilities
                    mapped_probs = color_map[b, c_in]
                    # Take argmax as new color
                    new_color = mapped_probs.argmax().item()
                    output[b, new_color, mask] = 1.0
        
        return output


class EnhancedChronosNet(nn.Module):
    """Enhanced CHRONOS with sequence pattern learning"""
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256):
        super().__init__()
        
        # Grid encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Object tracking
        self.object_encoder = ObjectEncoder(10, 128)
        
        # Temporal reasoning with attention  
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.hidden_dim = hidden_dim
        
        # Projection layer for feature dimension matching
        self.feature_proj = nn.Linear(128, hidden_dim) if hidden_dim != 128 else nn.Identity()
        
        # Sequence predictor
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Movement predictor
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 10, 1)
        )
        
        # Initialize final layer with strong values
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.3)
        # Equal small bias for CHRONOS
        self.decoder[-1].bias.data = torch.ones(10) * 0.05
        
        # Mix parameter - start at 0.05 to heavily favor transformations
        self.mix_param = nn.Parameter(torch.tensor(0.05))
        
        self.description = "Enhanced Temporal Sequence Analysis with Attention"
        
    def forward(self, sequence: List[torch.Tensor], target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if len(sequence) == 1:
            # Single frame - treat as static
            return self._forward_single(sequence[0])
        
        # Encode all frames
        encoded_sequence = []
        object_sequences = []
        
        for frame in sequence:
            features = self.encoder(frame)
            obj_features, obj_masks = self.object_encoder(frame)
            
            encoded_sequence.append(features)
            object_sequences.append(obj_features)
        
        # Stack sequences
        # Pool features to get fixed dimension
        seq_features = []
        for features in encoded_sequence:
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # B, 128
            # Project to hidden_dim if needed
            pooled = self.feature_proj(pooled)
            seq_features.append(pooled)
        
        seq_tensor = torch.stack(seq_features, dim=1)  # B, seq_len, hidden_dim
        
        # Temporal attention
        attended_seq, attention_weights = self.temporal_attention(seq_tensor, seq_tensor, seq_tensor)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(attended_seq)
        
        # Predict next movement
        movement_params = self.movement_head(lstm_out[:, -1])
        
        # Apply movement to last frame
        last_frame_features = encoded_sequence[-1]
        last_objects = object_sequences[-1]
        
        moved_features = self._apply_movement(last_objects, movement_params)
        
        # Decode
        combined = torch.cat([last_frame_features, moved_features], dim=1)
        predicted_output = self.decoder(combined)
        
        # Minimal residual from last frame
        mix = torch.sigmoid(self.mix_param)
        predicted_output = predicted_output * (1 - mix) + sequence[-1] * mix
        
        return {
            'predicted_output': predicted_output,
            'movement_params': movement_params,
            'attention_weights': attention_weights,
            'temporal_features': lstm_out
        }
    
    def _forward_single(self, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Handle single frame input"""
        features = self.encoder(input_grid)
        obj_features, obj_masks = self.object_encoder(input_grid)
        
        # Simple forward without temporal processing
        combined = torch.cat([features, obj_features], dim=1)
        predicted_output = self.decoder(combined)
        
        # Minimal residual
        mix = torch.sigmoid(self.mix_param)
        predicted_output = predicted_output * (1 - mix) + input_grid * mix
        
        return {
            'predicted_output': predicted_output,
            'movement_params': torch.zeros(input_grid.shape[0], 128).to(input_grid.device),
            'temporal_features': features
        }
    
    def _apply_movement(self, features: torch.Tensor, movement_params: torch.Tensor) -> torch.Tensor:
        """Apply predicted movement to features"""
        B, C, H, W = features.shape
        
        # Interpret movement parameters as displacement field
        # Simplified - enhance this with actual movement logic
        moved = features.clone()
        
        # Add movement-based modulation
        movement_field = movement_params[:, :C].view(B, C, 1, 1)
        moved = moved * torch.sigmoid(movement_field)
        
        return moved


class EnhancedPrometheusNet(nn.Module):
    """Enhanced PROMETHEUS with better pattern generation"""
    def __init__(self, max_grid_size: int = 30, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Enhanced encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Object and pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # VAE components
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Pattern synthesis network
        self.synthesis_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4)
        )
        
        # Rule generator
        self.rule_generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer for creative generation
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=2.0)
        # Small varied bias for PROMETHEUS creativity
        self.decoder[-1].bias.data = torch.tensor([0.0, 0.1, 0.08, 0.06, 0.09, 0.07, 0.11, 0.12, 0.1, 0.08])
        
        self.description = "Enhanced Creative Pattern Generation with VAE"
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution"""
        features = self.encoder(x)
        pattern_features = self.pattern_encoder(features)
        pattern_flat = pattern_features.view(pattern_features.shape[0], -1)
        
        mu = self.fc_mu(pattern_flat)
        log_var = self.fc_var(pattern_flat)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, input_shape: torch.Size) -> torch.Tensor:
        """Decode from latent space"""
        # Generate features
        features = self.synthesis_net(z)
        features = features.view(-1, 256, 4, 4)
        
        # Upsample to match input size
        B, _, H, W = input_shape
        
        # Decode
        output = self.decoder(features)
        
        # Ensure output matches input spatial size
        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output
    
    def forward(self, input_grid: torch.Tensor, target_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        # Encode
        mu, log_var = self.encode(input_grid)
        z = self.reparameterize(mu, log_var)
        
        # Generate transformation rules
        rules = self.rule_generator(z)
        
        # Decode
        predicted_output = self.decode(z, input_grid.shape)
        
        # Apply rules to refine output
        refined_output = self._apply_rules(predicted_output, rules, input_grid)
        
        # No additional residual here - already handled in _apply_rules
        
        outputs = {
            'predicted_output': refined_output,
            'raw_output': predicted_output,
            'mu': mu,
            'log_var': log_var,
            'latent': z,
            'rules': rules
        }
        
        return outputs
    
    def _apply_rules(self, generated: torch.Tensor, rules: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Apply learned rules to refine generation"""
        # Simple rule application - enhance this
        B = generated.shape[0]
        
        # Use rules to modulate generation
        rule_weights = torch.sigmoid(rules[:, :10]).view(B, 10, 1, 1)
        
        # Blend with input based on rules
        refined = generated * rule_weights + input_grid * (1 - rule_weights)
        
        return refined


def create_enhanced_models() -> Dict[str, nn.Module]:
    """Create all enhanced models"""
    return {
        'minerva': EnhancedMinervaNet(),
        'atlas': EnhancedAtlasNet(),
        'iris': EnhancedIrisNet(),
        'chronos': EnhancedChronosNet(),
        'prometheus': EnhancedPrometheusNet()
    }