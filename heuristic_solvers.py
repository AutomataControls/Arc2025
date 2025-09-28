#!/usr/bin/env python3
"""
Post-Processing Heuristic Solvers for OLYMPUS Ensemble
These "finishers" fix common prediction errors to boost exact match accuracy
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage


class HeuristicSolver:
    """Base class for heuristic solvers"""
    
    def __init__(self, name: str):
        self.name = name
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray, 
                  train_examples: List[Dict]) -> bool:
        """Check if this heuristic should be applied"""
        raise NotImplementedError
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Apply the heuristic to fix the prediction"""
        raise NotImplementedError


class SymmetrySolver(HeuristicSolver):
    """Fix almost-symmetric grids to be perfectly symmetric"""
    
    def __init__(self, threshold: float = 0.85):
        super().__init__("SymmetrySolver")
        self.threshold = threshold
    
    def _check_symmetry(self, grid: np.ndarray, axis: str) -> float:
        """Check how symmetric a grid is along an axis"""
        if axis == 'horizontal':
            flipped = np.flip(grid, axis=0)
        elif axis == 'vertical':
            flipped = np.flip(grid, axis=1)
        elif axis == 'diagonal':
            flipped = grid.T
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        # Handle size mismatch for diagonal
        if flipped.shape != grid.shape:
            return 0.0
        
        matches = (grid == flipped).sum()
        total = grid.size
        return matches / total
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs should be symmetric"""
        # Check if training outputs are all symmetric
        symmetric_count = 0
        
        for example in train_examples:
            output = np.array(example['output'])
            
            # Check all symmetry types
            h_sym = self._check_symmetry(output, 'horizontal')
            v_sym = self._check_symmetry(output, 'vertical')
            d_sym = self._check_symmetry(output, 'diagonal') if output.shape[0] == output.shape[1] else 0
            
            if max(h_sym, v_sym, d_sym) > 0.95:
                symmetric_count += 1
        
        # If most training outputs are symmetric
        if symmetric_count / len(train_examples) > 0.7:
            # Check if prediction is almost symmetric
            h_sym = self._check_symmetry(pred_grid, 'horizontal')
            v_sym = self._check_symmetry(pred_grid, 'vertical')
            d_sym = self._check_symmetry(pred_grid, 'diagonal') if pred_grid.shape[0] == pred_grid.shape[1] else 0
            
            return max(h_sym, v_sym, d_sym) > self.threshold
        
        return False
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Make the grid perfectly symmetric"""
        h_sym = self._check_symmetry(pred_grid, 'horizontal')
        v_sym = self._check_symmetry(pred_grid, 'vertical')
        d_sym = self._check_symmetry(pred_grid, 'diagonal') if pred_grid.shape[0] == pred_grid.shape[1] else 0
        
        # Apply the strongest symmetry
        if h_sym >= max(v_sym, d_sym):
            # Horizontal symmetry
            mid = pred_grid.shape[0] // 2
            if pred_grid.shape[0] % 2 == 0:
                top_half = pred_grid[:mid, :]
                pred_grid[mid:, :] = np.flip(top_half, axis=0)
            else:
                top_half = pred_grid[:mid, :]
                pred_grid[mid+1:, :] = np.flip(top_half, axis=0)
        
        elif v_sym >= d_sym:
            # Vertical symmetry
            mid = pred_grid.shape[1] // 2
            if pred_grid.shape[1] % 2 == 0:
                left_half = pred_grid[:, :mid]
                pred_grid[:, mid:] = np.flip(left_half, axis=1)
            else:
                left_half = pred_grid[:, :mid]
                pred_grid[:, mid+1:] = np.flip(left_half, axis=1)
        
        else:
            # Diagonal symmetry
            for i in range(pred_grid.shape[0]):
                for j in range(i+1, pred_grid.shape[1]):
                    pred_grid[j, i] = pred_grid[i, j]
        
        return pred_grid


class GridSizeSolver(HeuristicSolver):
    """Enforce consistent output grid size rules"""
    
    def __init__(self):
        super().__init__("GridSizeSolver")
    
    def _infer_size_rule(self, train_examples: List[Dict]) -> Optional[Dict]:
        """Infer the output size rule from training examples"""
        rules = []
        
        for example in train_examples:
            input_shape = np.array(example['input']).shape
            output_shape = np.array(example['output']).shape
            
            # Check various rules
            if output_shape == input_shape:
                rules.append({'type': 'same', 'shape': output_shape})
            elif output_shape[0] == output_shape[1]:
                # Square output
                rules.append({'type': 'square', 'size': output_shape[0]})
            elif output_shape[0] == input_shape[0] * 2 and output_shape[1] == input_shape[1] * 2:
                rules.append({'type': 'double', 'factor': 2})
            elif output_shape[0] == input_shape[0] // 2 and output_shape[1] == input_shape[1] // 2:
                rules.append({'type': 'half', 'factor': 0.5})
        
        # Check if all examples follow same rule
        if len(rules) == len(train_examples):
            first_rule = rules[0]
            if all(r['type'] == first_rule['type'] for r in rules):
                return first_rule
        
        return None
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if there's a consistent size rule being violated"""
        rule = self._infer_size_rule(train_examples)
        if rule is None:
            return False
        
        # Check if prediction violates the rule
        if rule['type'] == 'same':
            return pred_grid.shape != input_grid.shape
        elif rule['type'] == 'square':
            return pred_grid.shape[0] != pred_grid.shape[1]
        
        return False
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Resize grid according to inferred rule"""
        rule = self._infer_size_rule(train_examples)
        
        if rule['type'] == 'same':
            # Resize to match input
            target_shape = input_grid.shape
        elif rule['type'] == 'square':
            # Make it square (use first training example size)
            size = np.array(train_examples[0]['output']).shape[0]
            target_shape = (size, size)
        else:
            return pred_grid
        
        # Resize if needed
        if pred_grid.shape != target_shape:
            # Use nearest neighbor to preserve colors
            h_scale = target_shape[0] / pred_grid.shape[0]
            w_scale = target_shape[1] / pred_grid.shape[1]
            
            resized = ndimage.zoom(pred_grid, (h_scale, w_scale), order=0)
            return resized.astype(np.int32)
        
        return pred_grid


class ColorPaletteSolver(HeuristicSolver):
    """Ensure output only uses colors that appear in input"""
    
    def __init__(self):
        super().__init__("ColorPaletteSolver")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if output has invalid colors"""
        # First check if this rule applies to training examples
        rule_applies = True
        
        for example in train_examples:
            input_colors = set(np.array(example['input']).flatten())
            output_colors = set(np.array(example['output']).flatten())
            
            if not output_colors.issubset(input_colors):
                rule_applies = False
                break
        
        if not rule_applies:
            return False
        
        # Check if prediction violates this
        input_colors = set(input_grid.flatten())
        pred_colors = set(pred_grid.flatten())
        
        return not pred_colors.issubset(input_colors)
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Replace invalid colors with nearest valid color"""
        input_colors = sorted(set(input_grid.flatten()))
        pred_colors = set(pred_grid.flatten())
        
        fixed_grid = pred_grid.copy()
        
        for color in pred_colors:
            if color not in input_colors:
                # Find nearest valid color
                nearest = min(input_colors, key=lambda c: abs(c - color))
                fixed_grid[fixed_grid == color] = nearest
        
        return fixed_grid


class ObjectIntegritySolver(HeuristicSolver):
    """Fix single-pixel holes and remove isolated noise"""
    
    def __init__(self, min_object_size: int = 3):
        super().__init__("ObjectIntegritySolver")
        self.min_object_size = min_object_size
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Always try to clean up objects"""
        return True
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Fill holes and remove small noise"""
        fixed_grid = pred_grid.copy()
        
        # Process each non-background color
        unique_colors = sorted(set(fixed_grid.flatten()))
        if 0 in unique_colors:
            unique_colors.remove(0)  # Skip background
        
        for color in unique_colors:
            # Create binary mask for this color
            mask = (fixed_grid == color).astype(np.uint8)
            
            # Fill holes
            filled = ndimage.binary_fill_holes(mask)
            
            # Remove small objects
            labeled, num_features = ndimage.label(filled)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if component.sum() < self.min_object_size:
                    filled[component] = 0
            
            # Apply back to grid
            fixed_grid[filled.astype(bool)] = color
            fixed_grid[(mask.astype(bool)) & (~filled.astype(bool))] = 0
        
        return fixed_grid


class PatternCompletionSolver(HeuristicSolver):
    """Complete partial patterns based on training examples"""
    
    def __init__(self):
        super().__init__("PatternCompletionSolver")
    
    def _find_pattern_period(self, grid: np.ndarray, axis: int) -> Optional[int]:
        """Find if there's a repeating pattern along an axis"""
        size = grid.shape[axis]
        
        for period in range(2, size // 2 + 1):
            if size % period == 0:
                # Check if pattern repeats
                is_periodic = True
                
                for i in range(period, size):
                    if axis == 0:
                        if not np.array_equal(grid[i % period, :], grid[i, :]):
                            is_periodic = False
                            break
                    else:
                        if not np.array_equal(grid[:, i % period], grid[:, i]):
                            is_periodic = False
                            break
                
                if is_periodic:
                    return period
        
        return None
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs should have repeating patterns"""
        pattern_count = 0
        
        for example in train_examples:
            output = np.array(example['output'])
            
            # Check for patterns in both dimensions
            h_period = self._find_pattern_period(output, 0)
            v_period = self._find_pattern_period(output, 1)
            
            if h_period is not None or v_period is not None:
                pattern_count += 1
        
        return pattern_count / len(train_examples) > 0.5
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Complete partial patterns"""
        # Try to detect partial pattern in prediction
        h_period = self._find_pattern_period(pred_grid[:pred_grid.shape[0]//2, :], 0)
        v_period = self._find_pattern_period(pred_grid[:, :pred_grid.shape[1]//2], 1)
        
        fixed_grid = pred_grid.copy()
        
        if h_period is not None:
            # Complete horizontal pattern
            for i in range(h_period, fixed_grid.shape[0]):
                fixed_grid[i, :] = fixed_grid[i % h_period, :]
        
        elif v_period is not None:
            # Complete vertical pattern
            for j in range(v_period, fixed_grid.shape[1]):
                fixed_grid[:, j] = fixed_grid[:, j % v_period]
        
        return fixed_grid


class SinglePixelFixer(HeuristicSolver):
    """Fix isolated missing or extra pixels"""
    
    def __init__(self):
        super().__init__("SinglePixelFixer")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Always try this for high-accuracy predictions"""
        return True
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Fix isolated pixels that don't match neighbors"""
        fixed_grid = pred_grid.copy()
        h, w = fixed_grid.shape
        
        for i in range(h):
            for j in range(w):
                # Count neighbor colors
                neighbors = []
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbors.append(fixed_grid[ni, nj])
                
                if neighbors:
                    # If all neighbors agree and differ from current
                    if len(set(neighbors)) == 1 and neighbors[0] != fixed_grid[i, j]:
                        # Consider fixing if this creates more consistency
                        if neighbors.count(neighbors[0]) >= 3:
                            fixed_grid[i, j] = neighbors[0]
        
        return fixed_grid


class EdgeCompleter(HeuristicSolver):
    """Complete partial edges and borders"""
    
    def __init__(self):
        super().__init__("EdgeCompleter")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs have consistent edges"""
        edge_count = 0
        for example in train_examples:
            output = np.array(example['output'])
            # Check if output has consistent edge colors
            edges = [output[0, :], output[-1, :], output[:, 0], output[:, -1]]
            for edge in edges:
                if len(np.unique(edge)) == 1 and edge[0] != 0:
                    edge_count += 1
        return edge_count > len(train_examples)
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Complete partial edges"""
        fixed_grid = pred_grid.copy()
        h, w = fixed_grid.shape
        
        # Check each edge for near-completion
        edges = [
            (fixed_grid[0, :], 0, 'horizontal'),
            (fixed_grid[-1, :], h-1, 'horizontal'),
            (fixed_grid[:, 0], 0, 'vertical'),
            (fixed_grid[:, -1], w-1, 'vertical')
        ]
        
        for edge, pos, direction in edges:
            colors = edge[edge != 0]  # Non-background colors
            if len(colors) > 0:
                # If edge is mostly one color (80%+)
                unique_colors, counts = np.unique(colors, return_counts=True)
                if len(unique_colors) > 0:
                    dominant_color = unique_colors[np.argmax(counts)]
                    if np.max(counts) / len(colors) > 0.8:
                        # Complete the edge
                        if direction == 'horizontal':
                            fixed_grid[pos, :] = dominant_color
                        else:
                            fixed_grid[:, pos] = dominant_color
        
        return fixed_grid


class HolesFiller(HeuristicSolver):
    """Fill small holes in objects"""
    
    def __init__(self):
        super().__init__("HolesFiller")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs have solid objects"""
        return True
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Fill holes in objects"""
        fixed_grid = pred_grid.copy()
        
        # For each color, check for holes
        for color in np.unique(pred_grid):
            if color == 0:  # Skip background
                continue
            
            # Create mask for this color
            mask = (pred_grid == color).astype(int)
            
            # Fill holes using morphological closing
            filled = ndimage.binary_fill_holes(mask).astype(int)
            
            # Only fill small holes (1-2 pixels)
            diff = filled - mask
            hole_size = np.sum(diff)
            
            if 0 < hole_size <= 2:
                # Fill the holes with the object color
                fixed_grid[diff == 1] = color
        
        return fixed_grid


class CornerFixer(HeuristicSolver):
    """Fix missing or extra pixels in corners"""
    
    def __init__(self):
        super().__init__("CornerFixer")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs have regular rectangular shapes"""
        # Analyze training examples for rectangular patterns
        rect_count = 0
        for example in train_examples:
            output = np.array(example['output'])
            # Check for rectangular objects
            for color in np.unique(output):
                if color == 0:
                    continue
                mask = (output == color)
                labeled, num = ndimage.label(mask)
                for i in range(1, num + 1):
                    component = (labeled == i)
                    # Check if component is roughly rectangular
                    rows, cols = np.where(component)
                    if len(rows) > 0:
                        min_r, max_r = rows.min(), rows.max()
                        min_c, max_c = cols.min(), cols.max()
                        expected_pixels = (max_r - min_r + 1) * (max_c - min_c + 1)
                        actual_pixels = component.sum()
                        if actual_pixels >= expected_pixels * 0.8:
                            rect_count += 1
        
        return rect_count > len(train_examples)
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Fix corners to complete rectangles"""
        fixed_grid = pred_grid.copy()
        
        for color in np.unique(pred_grid):
            if color == 0:
                continue
            
            mask = (fixed_grid == color)
            labeled, num = ndimage.label(mask)
            
            for i in range(1, num + 1):
                component = (labeled == i)
                rows, cols = np.where(component)
                
                if len(rows) > 0:
                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()
                    
                    # Check corners
                    corners = [
                        (min_r, min_c), (min_r, max_c),
                        (max_r, min_c), (max_r, max_c)
                    ]
                    
                    # If 3 corners are filled, fill the 4th
                    filled_corners = sum(1 for r, c in corners if fixed_grid[r, c] == color)
                    if filled_corners == 3:
                        for r, c in corners:
                            if fixed_grid[r, c] != color:
                                fixed_grid[r, c] = color
        
        return fixed_grid


class DiagonalLineFixer(HeuristicSolver):
    """Fix broken diagonal lines"""
    
    def __init__(self):
        super().__init__("DiagonalLineFixer")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs contain diagonal patterns"""
        diag_count = 0
        for example in train_examples:
            output = np.array(example['output'])
            # Check for diagonal patterns
            h, w = output.shape
            for i in range(h-1):
                for j in range(w-1):
                    if output[i,j] != 0 and output[i,j] == output[i+1,j+1]:
                        diag_count += 1
            
            # Also check anti-diagonals
            for i in range(h-1):
                for j in range(1, w):
                    if output[i,j] != 0 and output[i,j] == output[i+1,j-1]:
                        diag_count += 1
        
        return diag_count > 5 * len(train_examples)
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Fix gaps in diagonal lines"""
        fixed_grid = pred_grid.copy()
        h, w = fixed_grid.shape
        
        # Fix main diagonals
        for i in range(1, h-1):
            for j in range(1, w-1):
                if fixed_grid[i,j] == 0:  # Gap
                    # Check if it's part of a diagonal line
                    if (fixed_grid[i-1,j-1] != 0 and fixed_grid[i+1,j+1] != 0 and
                        fixed_grid[i-1,j-1] == fixed_grid[i+1,j+1]):
                        fixed_grid[i,j] = fixed_grid[i-1,j-1]
                    
                    # Check anti-diagonal
                    elif (fixed_grid[i-1,j+1] != 0 and fixed_grid[i+1,j-1] != 0 and
                          fixed_grid[i-1,j+1] == fixed_grid[i+1,j-1]):
                        fixed_grid[i,j] = fixed_grid[i-1,j+1]
        
        return fixed_grid


class ColorBalancer(HeuristicSolver):
    """Balance color distribution based on training patterns"""
    
    def __init__(self):
        super().__init__("ColorBalancer")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if there's a consistent color distribution pattern"""
        # Analyze color ratios in training examples
        ratios = []
        for example in train_examples:
            input_ex = np.array(example['input'])
            output_ex = np.array(example['output'])
            
            input_colors = dict(zip(*np.unique(input_ex, return_counts=True)))
            output_colors = dict(zip(*np.unique(output_ex, return_counts=True)))
            
            # Check if colors maintain similar ratios
            common_colors = set(input_colors.keys()) & set(output_colors.keys())
            if len(common_colors) >= 2:
                ratios.append(len(common_colors))
        
        return len(ratios) > 0 and np.mean(ratios) >= 2
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Adjust color distribution to match expected patterns"""
        # This is a subtle adjustment - only fix obvious outliers
        fixed_grid = pred_grid.copy()
        
        # Get expected color distribution from training
        expected_ratios = {}
        for example in train_examples:
            output = np.array(example['output'])
            colors, counts = np.unique(output, return_counts=True)
            total = output.size
            for color, count in zip(colors, counts):
                if color not in expected_ratios:
                    expected_ratios[color] = []
                expected_ratios[color].append(count / total)
        
        # Average the ratios
        for color in expected_ratios:
            expected_ratios[color] = np.mean(expected_ratios[color])
        
        # Check current distribution
        current_colors, current_counts = np.unique(pred_grid, return_counts=True)
        current_total = pred_grid.size
        
        # Only adjust if significantly off (>50% difference)
        for color, count in zip(current_colors, current_counts):
            if color in expected_ratios:
                current_ratio = count / current_total
                expected_ratio = expected_ratios[color]
                
                if abs(current_ratio - expected_ratio) > 0.5 * expected_ratio:
                    # Don't apply aggressive changes, just mark for review
                    pass
        
        return fixed_grid


class BoundarySmootherSolver(HeuristicSolver):
    """Smooth jagged boundaries between regions"""
    
    def __init__(self):
        super().__init__("BoundarySmootherSolver")
    
    def can_apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
                  train_examples: List[Dict]) -> bool:
        """Check if outputs have smooth boundaries"""
        smooth_count = 0
        for example in train_examples:
            output = np.array(example['output'])
            # Check for smooth transitions (no isolated pixels at boundaries)
            h, w = output.shape
            jagged_pixels = 0
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # Check if pixel differs from most neighbors
                    neighbors = [
                        output[i-1, j], output[i+1, j],
                        output[i, j-1], output[i, j+1]
                    ]
                    if output[i, j] != 0:
                        different = sum(1 for n in neighbors if n != output[i, j])
                        if different >= 3:
                            jagged_pixels += 1
            
            if jagged_pixels < 2:
                smooth_count += 1
        
        return smooth_count / len(train_examples) > 0.7
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict]) -> np.ndarray:
        """Smooth boundaries by removing isolated boundary pixels"""
        fixed_grid = pred_grid.copy()
        h, w = fixed_grid.shape
        
        changes = []
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if fixed_grid[i, j] != 0:
                    # Get all neighbors
                    neighbors = [
                        fixed_grid[i-1, j], fixed_grid[i+1, j],
                        fixed_grid[i, j-1], fixed_grid[i, j+1]
                    ]
                    
                    # Count how many neighbors differ
                    different = sum(1 for n in neighbors if n != fixed_grid[i, j])
                    
                    # If isolated (3+ neighbors differ), change to most common neighbor
                    if different >= 3:
                        neighbor_counts = {}
                        for n in neighbors:
                            if n != fixed_grid[i, j]:
                                neighbor_counts[n] = neighbor_counts.get(n, 0) + 1
                        
                        if neighbor_counts:
                            new_color = max(neighbor_counts, key=neighbor_counts.get)
                            changes.append((i, j, new_color))
        
        # Apply changes
        for i, j, color in changes:
            fixed_grid[i, j] = color
        
        return fixed_grid


class HeuristicPipeline:
    """Apply multiple heuristics in sequence"""
    
    def __init__(self):
        self.solvers = [
            ColorPaletteSolver(),      # First ensure valid colors
            GridSizeSolver(),          # Then fix size
            ObjectIntegritySolver(),   # Clean up objects
            SinglePixelFixer(),        # Fix isolated pixels
            HolesFiller(),             # Fill small holes
            CornerFixer(),             # Fix missing corners
            DiagonalLineFixer(),       # Fix diagonal lines
            EdgeCompleter(),           # Complete edges
            BoundarySmootherSolver(),  # Smooth jagged boundaries
            SymmetrySolver(),          # Apply symmetry if needed
            PatternCompletionSolver(), # Complete patterns
            ColorBalancer()            # Final color balance check
        ]
    
    def apply(self, input_grid: np.ndarray, pred_grid: np.ndarray,
             train_examples: List[Dict], verbose: bool = True) -> np.ndarray:
        """Apply all applicable heuristics"""
        current_grid = pred_grid.copy()
        applied = []
        
        for solver in self.solvers:
            if solver.can_apply(input_grid, current_grid, train_examples):
                if verbose:
                    print(f"  📐 Applying {solver.name}")
                current_grid = solver.apply(input_grid, current_grid, train_examples)
                applied.append(solver.name)
        
        if verbose and applied:
            print(f"  ✨ Applied heuristics: {', '.join(applied)}")
        
        return current_grid


if __name__ == "__main__":
    """Test heuristics on a simple example"""
    print("🔧 Testing Heuristic Solvers")
    print("="*50)
    
    # Test symmetry solver
    test_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 4]  # Almost symmetric to first row
    ])
    
    train_examples = [
        {
            'input': np.array([[1, 2], [3, 4]]),
            'output': np.array([[1, 2], [1, 2]])  # Symmetric
        }
    ]
    
    solver = SymmetrySolver()
    if solver.can_apply(test_grid, test_grid, train_examples):
        fixed = solver.apply(test_grid, test_grid, train_examples)
        print("Original grid:")
        print(test_grid)
        print("\nFixed grid (symmetric):")
        print(fixed)