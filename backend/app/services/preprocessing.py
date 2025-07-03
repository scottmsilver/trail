"""
Preprocessing module for optimizing pathfinding data.

This module implements various preprocessing techniques to improve pathfinding performance:
1. Neighbor precomputation
2. Passability masks
3. Terrain analysis
4. Distance field computation
"""

import numpy as np
from typing import Dict, Tuple, List, Set
from math import sqrt
import heapq


class PathfindingPreprocessor:
    """Handles preprocessing of terrain data for faster pathfinding."""
    
    def __init__(self):
        self.neighbor_cache = {}
        self.passability_cache = {}
        self.distance_field_cache = {}
        self.reach_cache = {}
        
    def preprocess_tile(self, cost_surface: np.ndarray, indices: np.ndarray, 
                       out_trans, max_cost: float = 9999) -> Dict:
        """
        Preprocess a tile for optimized pathfinding.
        
        Returns a dictionary with preprocessed data:
        - passability_mask: Boolean array of passable cells
        - neighbor_map: Precomputed valid neighbors for each cell
        - distance_field: Distance to nearest obstacle
        - reach_map: Reach values for pruning
        """
        height, width = cost_surface.shape
        
        # 1. Create passability mask
        passability_mask = cost_surface < max_cost
        
        # 2. Precompute neighbors for all positions
        neighbor_map = self._precompute_neighbors(height, width, passability_mask)
        
        # 3. Compute distance field (distance to nearest obstacle)
        distance_field = self._compute_distance_field(passability_mask)
        
        # 4. Compute reach values (optional, for advanced pruning)
        # reach_map = self._compute_reach_values(cost_surface, neighbor_map)
        
        return {
            'passability_mask': passability_mask,
            'neighbor_map': neighbor_map,
            'distance_field': distance_field,
            'cost_statistics': {
                'min_cost': np.min(cost_surface[passability_mask]) if np.any(passability_mask) else 0,
                'max_cost': np.max(cost_surface[passability_mask]) if np.any(passability_mask) else 0,
                'avg_cost': np.mean(cost_surface[passability_mask]) if np.any(passability_mask) else 0,
                'passable_ratio': np.sum(passability_mask) / (height * width)
            }
        }
    
    def _precompute_neighbors(self, height: int, width: int, 
                            passability_mask: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
        """Precompute valid neighbors for each passable position."""
        neighbor_map = {}
        
        # 8-directional movement offsets
        offsets = [
            (-1, -1, sqrt(2)), (-1, 0, 1.0), (-1, 1, sqrt(2)),
            (0, -1, 1.0),                     (0, 1, 1.0),
            (1, -1, sqrt(2)),  (1, 0, 1.0),  (1, 1, sqrt(2))
        ]
        
        indices = np.arange(height * width).reshape(height, width)
        
        for row in range(height):
            for col in range(width):
                if not passability_mask[row, col]:
                    continue
                    
                idx = indices[row, col]
                neighbors = []
                
                for dy, dx, dist in offsets:
                    new_row = row + dy
                    new_col = col + dx
                    
                    if (0 <= new_row < height and 
                        0 <= new_col < width and 
                        passability_mask[new_row, new_col]):
                        neighbor_idx = indices[new_row, new_col]
                        neighbors.append((neighbor_idx, dist))
                
                if neighbors:
                    neighbor_map[idx] = neighbors
        
        return neighbor_map
    
    def _compute_distance_field(self, passability_mask: np.ndarray) -> np.ndarray:
        """
        Compute distance to nearest obstacle for each cell.
        Useful for path smoothing and corridor-based pathfinding.
        """
        from scipy.ndimage import distance_transform_edt
        
        # Distance transform gives distance to nearest False (obstacle)
        distance_field = distance_transform_edt(passability_mask)
        
        return distance_field
    
    def _compute_reach_values(self, cost_surface: np.ndarray, 
                            neighbor_map: Dict) -> Dict[int, float]:
        """
        Compute reach values for each node.
        Reach(v) = max over all shortest paths P through v of:
            min(dist(start_of_P, v), dist(v, end_of_P))
        
        This is used for reach-based pruning in pathfinding.
        """
        # This is a complex preprocessing step that requires running
        # shortest paths from many nodes. For now, we'll skip it
        # but it can provide significant speedups for large graphs.
        return {}
    
    def create_hierarchical_graph(self, cost_surface: np.ndarray, 
                                level_factor: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        Create a hierarchical representation of the terrain.
        This creates a coarser grid for long-distance planning.
        """
        height, width = cost_surface.shape
        
        # Create coarse grid
        coarse_height = height // level_factor
        coarse_width = width // level_factor
        
        # Aggregate costs (use minimum cost in each block)
        coarse_cost = np.zeros((coarse_height, coarse_width))
        
        for i in range(coarse_height):
            for j in range(coarse_width):
                # Get the block of fine cells
                block = cost_surface[
                    i*level_factor:(i+1)*level_factor,
                    j*level_factor:(j+1)*level_factor
                ]
                # Use minimum cost (most optimistic)
                coarse_cost[i, j] = np.min(block)
        
        # Create mapping between coarse and fine grids
        mapping = {
            'factor': level_factor,
            'fine_shape': (height, width),
            'coarse_shape': (coarse_height, coarse_width)
        }
        
        return coarse_cost, mapping
    
    def compute_landmarks(self, cost_surface: np.ndarray, num_landmarks: int = 16) -> Dict:
        """
        Select landmark nodes and precompute distances from all nodes to landmarks.
        Used for landmark-based heuristics (ALT algorithm).
        """
        height, width = cost_surface.shape
        passability_mask = cost_surface < 9999
        
        # Select landmarks using farthest-first traversal
        landmarks = []
        landmark_indices = []
        
        # Start with a random passable node
        passable_indices = np.where(passability_mask.flatten())[0]
        if len(passable_indices) == 0:
            return {'landmarks': [], 'distances': {}}
        
        # First landmark: random passable node
        first_idx = np.random.choice(passable_indices)
        landmark_indices.append(first_idx)
        
        # Select remaining landmarks
        while len(landmark_indices) < num_landmarks and len(landmark_indices) < len(passable_indices):
            # Find node farthest from all current landmarks
            max_min_dist = -1
            best_candidate = None
            
            # This is a simplified version - in practice, you'd run
            # actual shortest path algorithms
            for idx in passable_indices:
                if idx in landmark_indices:
                    continue
                
                # Find minimum distance to any landmark
                row, col = np.unravel_index(idx, (height, width))
                min_dist = float('inf')
                
                for landmark_idx in landmark_indices:
                    lrow, lcol = np.unravel_index(landmark_idx, (height, width))
                    dist = sqrt((row - lrow)**2 + (col - lcol)**2)
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = idx
            
            if best_candidate is not None:
                landmark_indices.append(best_candidate)
        
        # Convert to row/col coordinates
        for idx in landmark_indices:
            row, col = np.unravel_index(idx, (height, width))
            landmarks.append((row, col))
        
        return {
            'landmarks': landmarks,
            'landmark_indices': landmark_indices,
            'num_landmarks': len(landmarks)
        }