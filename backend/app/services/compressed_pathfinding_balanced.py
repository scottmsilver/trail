"""
Balanced compressed pathfinding - less strict but obstacle-aware
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import heapq
from collections import defaultdict

@dataclass
class Region:
    """A region of similar terrain"""
    id: int
    cells: Set[Tuple[int, int]]
    avg_slope: float
    avg_cost: float
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    neighbors: Set[int] = None
    has_obstacles: bool = False  # Track if region contains obstacles
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()
    
    def distance_to(self, other: 'Region') -> float:
        """Approximate distance between region centers"""
        center1_row = (self.min_row + self.max_row) / 2
        center1_col = (self.min_col + self.max_col) / 2
        center2_row = (other.min_row + other.max_row) / 2
        center2_col = (other.min_col + other.max_col) / 2
        
        return np.sqrt((center2_row - center1_row)**2 + (center2_col - center1_col)**2)


class BalancedCompressedPathfinder:
    """Balanced compression that groups terrain while respecting major barriers"""
    
    def __init__(self, slope_threshold=5.0, cost_threshold=3.0, min_region_size=9):
        """Less strict parameters for better coverage"""
        self.slope_threshold = slope_threshold
        self.cost_threshold = cost_threshold
        self.min_region_size = min_region_size
        
    def create_regions(self, slope_degrees: np.ndarray, cost_surface: np.ndarray, 
                      obstacle_mask: np.ndarray = None) -> Dict[int, Region]:
        """Create regions with simple obstacle awareness"""
        height, width = slope_degrees.shape
        visited = np.zeros((height, width), dtype=bool)
        regions = {}
        region_id = 0
        
        # Don't exclude obstacles from regions, but track them
        for i in range(height):
            for j in range(width):
                if not visited[i, j]:
                    # Start new region
                    region_cells = self._grow_region(i, j, slope_degrees, cost_surface, 
                                                   visited, obstacle_mask)
                    
                    if len(region_cells) >= self.min_region_size:
                        # Create region
                        rows = [c[0] for c in region_cells]
                        cols = [c[1] for c in region_cells]
                        slopes = [slope_degrees[r, c] for r, c in region_cells]
                        costs = [cost_surface[r, c] for r, c in region_cells]
                        
                        # Check if region contains obstacles
                        has_obstacles = False
                        if obstacle_mask is not None:
                            has_obstacles = any(obstacle_mask[r, c] for r, c in region_cells)
                        
                        region = Region(
                            id=region_id,
                            cells=set(region_cells),
                            avg_slope=np.mean(slopes),
                            avg_cost=np.mean(costs),
                            min_row=min(rows),
                            max_row=max(rows),
                            min_col=min(cols),
                            max_col=max(cols),
                            has_obstacles=has_obstacles
                        )
                        regions[region_id] = region
                        region_id += 1
        
        # Find neighbors
        self._find_neighbors(regions, height, width)
        
        return regions
    
    def _grow_region(self, start_row: int, start_col: int, 
                     slope_degrees: np.ndarray, cost_surface: np.ndarray,
                     visited: np.ndarray, obstacle_mask: np.ndarray = None) -> List[Tuple[int, int]]:
        """Grow region with relaxed constraints"""
        height, width = slope_degrees.shape
        stack = [(start_row, start_col)]
        region_cells = []
        
        base_slope = slope_degrees[start_row, start_col]
        base_cost = cost_surface[start_row, start_col]
        
        # Don't start regions in very high cost areas
        if base_cost > 5000:
            visited[start_row, start_col] = True
            return []
        
        while stack:
            row, col = stack.pop()
            
            if visited[row, col]:
                continue
                
            # Check if this cell is similar enough
            slope_diff = abs(slope_degrees[row, col] - base_slope)
            cost_ratio = max(cost_surface[row, col] / base_cost, 
                           base_cost / cost_surface[row, col]) if base_cost > 0 else float('inf')
            
            # Don't include extremely high cost cells
            if cost_surface[row, col] > 5000:
                visited[row, col] = True
                continue
            
            if slope_diff <= self.slope_threshold and cost_ratio <= self.cost_threshold:
                visited[row, col] = True
                region_cells.append((row, col))
                
                # Add neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                        stack.append((nr, nc))
        
        return region_cells
    
    def _find_neighbors(self, regions: Dict[int, Region], height: int, width: int):
        """Find neighboring regions"""
        # Create a grid mapping cells to regions
        cell_to_region = {}
        for region_id, region in regions.items():
            for cell in region.cells:
                cell_to_region[cell] = region_id
        
        # Find neighbors
        for region_id, region in regions.items():
            neighbor_ids = set()
            
            # Check boundary cells
            for row, col in region.cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if (nr, nc) in cell_to_region:
                            neighbor_region_id = cell_to_region[(nr, nc)]
                            if neighbor_region_id != region_id:
                                neighbor_ids.add(neighbor_region_id)
            
            region.neighbors = neighbor_ids


def balanced_compress_search_space(slope_degrees: np.ndarray, cost_surface: np.ndarray,
                                 obstacle_mask: np.ndarray = None) -> Tuple[Dict[int, Region], np.ndarray]:
    """Balanced compression for better coverage"""
    compressor = BalancedCompressedPathfinder()
    regions = compressor.create_regions(slope_degrees, cost_surface, obstacle_mask)
    
    # Create region map
    region_map = np.full(slope_degrees.shape, -1, dtype=int)
    for region_id, region in regions.items():
        for row, col in region.cells:
            region_map[row, col] = region_id
    
    # Print compression statistics
    total_cells = slope_degrees.size
    compressed_cells = len(regions)
    cells_in_regions = sum(len(r.cells) for r in regions.values())
    obstacle_regions = sum(1 for r in regions.values() if r.has_obstacles)
    
    print(f"Balanced compression statistics:")
    print(f"  Original cells: {total_cells}")
    print(f"  Regions created: {compressed_cells}")
    print(f"  Cells in regions: {cells_in_regions} ({100*cells_in_regions/total_cells:.1f}%)")
    print(f"  Regions with obstacles: {obstacle_regions}")
    if compressed_cells > 0:
        print(f"  Average region size: {cells_in_regions/compressed_cells:.1f} cells")
        print(f"  Overall compression: {total_cells/compressed_cells:.1f}:1")
    
    return regions, region_map