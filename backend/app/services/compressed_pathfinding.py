"""
Compressed pathfinding using region-based nodes to reduce search space
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
    cells: Set[Tuple[int, int]]  # Set of (row, col) cells in this region
    avg_slope: float
    avg_cost: float
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    neighbors: Set[int] = None  # IDs of neighboring regions
    path_type: int = 0  # Path type ID (0 = off-path)
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()
    
    def contains(self, row: int, col: int) -> bool:
        """Check if a cell is in this region"""
        return (row, col) in self.cells
    
    def distance_to(self, other: 'Region') -> float:
        """Approximate distance between region centers"""
        center1_row = (self.min_row + self.max_row) / 2
        center1_col = (self.min_col + self.max_col) / 2
        center2_row = (other.min_row + other.max_row) / 2
        center2_col = (other.min_col + other.max_col) / 2
        
        return np.sqrt((center2_row - center1_row)**2 + (center2_col - center1_col)**2)


class CompressedPathfinder:
    """Pathfinder that groups similar terrain into regions"""
    
    def __init__(self, slope_threshold=4.0, cost_threshold=2.0, min_region_size=12):
        """
        Args:
            slope_threshold: Max slope difference within a region (degrees)
            cost_threshold: Max cost ratio within a region
            min_region_size: Minimum cells to form a region
        """
        self.slope_threshold = slope_threshold
        self.cost_threshold = cost_threshold
        self.min_region_size = min_region_size
        
    def create_regions(self, slope_degrees: np.ndarray, cost_surface: np.ndarray, 
                      obstacle_mask: np.ndarray = None, path_raster: np.ndarray = None) -> Dict[int, Region]:
        """Create regions from slope, cost, obstacle, and path type data"""
        height, width = slope_degrees.shape
        visited = np.zeros((height, width), dtype=bool)
        regions = {}
        region_id = 0
        
        # Pre-mark obstacles as visited so they won't be included in regions
        if obstacle_mask is not None:
            visited[obstacle_mask] = True
        
        # Region growing algorithm
        for i in range(height):
            for j in range(width):
                if not visited[i, j]:
                    # Start new region
                    region_cells = self._grow_region(i, j, slope_degrees, cost_surface, visited, 
                                                   obstacle_mask, path_raster)
                    
                    if len(region_cells) >= self.min_region_size:
                        # Create region
                        rows = [c[0] for c in region_cells]
                        cols = [c[1] for c in region_cells]
                        slopes = [slope_degrees[r, c] for r, c in region_cells]
                        costs = [cost_surface[r, c] for r, c in region_cells]
                        
                        # Determine region type from path raster
                        region_type = 'off_path'
                        if path_raster is not None:
                            path_types = [path_raster[r, c] for r, c in region_cells]
                            # Use most common path type in region (excluding 0 which means no path)
                            non_zero_types = [t for t in path_types if t > 0]
                            if non_zero_types:
                                from collections import Counter
                                region_type = Counter(non_zero_types).most_common(1)[0][0]
                        
                        region = Region(
                            id=region_id,
                            cells=set(region_cells),
                            avg_slope=np.mean(slopes),
                            avg_cost=np.mean(costs),
                            min_row=min(rows),
                            max_row=max(rows),
                            min_col=min(cols),
                            max_col=max(cols)
                        )
                        region.path_type = region_type  # Add path type to region
                        regions[region_id] = region
                        region_id += 1
                    else:
                        # Mark as visited but don't create region
                        # These cells will be handled individually
                        pass
        
        # Find neighbors
        self._find_neighbors(regions, height, width)
        
        return regions
    
    def _grow_region(self, start_row: int, start_col: int, 
                     slope_degrees: np.ndarray, cost_surface: np.ndarray,
                     visited: np.ndarray, obstacle_mask: np.ndarray = None,
                     path_raster: np.ndarray = None) -> List[Tuple[int, int]]:
        """Grow a region from a starting cell using flood fill"""
        height, width = slope_degrees.shape
        stack = [(start_row, start_col)]
        region_cells = []
        
        base_slope = slope_degrees[start_row, start_col]
        base_cost = cost_surface[start_row, start_col]
        base_path_type = path_raster[start_row, start_col] if path_raster is not None else 0
        
        while stack:
            row, col = stack.pop()
            
            if visited[row, col]:
                continue
            
            # Skip if it's an obstacle
            if obstacle_mask is not None and obstacle_mask[row, col]:
                continue
                
            # Check if this cell is similar enough
            slope_diff = abs(slope_degrees[row, col] - base_slope)
            cost_ratio = max(cost_surface[row, col] / base_cost, 
                           base_cost / cost_surface[row, col]) if base_cost > 0 else float('inf')
            
            # Check path type similarity
            path_type_match = True
            if path_raster is not None:
                current_path_type = path_raster[row, col]
                # Allow grouping of similar path types
                # 0 = off-path, can group with other off-path
                # Non-zero = on some kind of path, don't mix with off-path
                if base_path_type == 0:
                    path_type_match = (current_path_type == 0)
                elif current_path_type == 0:
                    path_type_match = False
                else:
                    # Both are paths, allow grouping (trails with trails, roads with roads, etc)
                    path_type_match = True
            
            if (slope_diff <= self.slope_threshold and 
                cost_ratio <= self.cost_threshold and 
                path_type_match):
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
    
    def _find_nearest_region(self, cell: Tuple[int, int], regions: Dict[int, Region]) -> Region:
        """Find the nearest region to a cell"""
        row, col = cell
        min_distance = float('inf')
        nearest_region = None
        
        for region in regions.values():
            # Calculate distance to region center
            center_row = (region.min_row + region.max_row) / 2
            center_col = (region.min_col + region.max_col) / 2
            distance = np.sqrt((row - center_row)**2 + (col - center_col)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_region = region
        
        return nearest_region
    
    def find_path_compressed(self, regions: Dict[int, Region], 
                           start_cell: Tuple[int, int], 
                           end_cell: Tuple[int, int],
                           cost_surface: np.ndarray) -> List[Tuple[int, int]]:
        """Find path using compressed regions"""
        # Find start and end regions
        start_region = None
        end_region = None
        
        for region in regions.values():
            if start_cell in region.cells:
                start_region = region
            if end_cell in region.cells:
                end_region = region
                
        # If start/end not in regions, find nearest regions
        if not start_region:
            start_region = self._find_nearest_region(start_cell, regions)
            print(f"Start cell not in region, using nearest region {start_region.id}")
            
        if not end_region:
            end_region = self._find_nearest_region(end_cell, regions)
            print(f"End cell not in region, using nearest region {end_region.id}")
            
        if not start_region or not end_region:
            print("Could not find regions for start/end points")
            return None
            
        print(f"Start region: {start_region.id}, End region: {end_region.id}")
        print(f"Start region has {len(start_region.neighbors)} neighbors")
        print(f"End region has {len(end_region.neighbors)} neighbors")
        
        # A* on region graph
        open_set = [(0, start_region.id)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_region.id] = 0
        
        iterations = 0
        max_iterations = len(regions) * 10  # Reasonable limit
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_cost, current_id = heapq.heappop(open_set)
            
            if current_id == end_region.id:
                # Reconstruct path through regions
                region_path = []
                while current_id in came_from:
                    region_path.append(current_id)
                    current_id = came_from[current_id]
                region_path.append(start_region.id)
                region_path.reverse()
                
                # Convert to cell path
                return self._regions_to_cells(region_path, regions, start_cell, end_cell, cost_surface)
            
            current_region = regions[current_id]
            
            for neighbor_id in current_region.neighbors:
                neighbor_region = regions[neighbor_id]
                
                # Cost is based on distance and terrain difficulty
                movement_cost = (current_region.distance_to(neighbor_region) * 
                               (current_region.avg_cost + neighbor_region.avg_cost) / 2)
                
                tentative_g = g_score[current_id] + movement_cost
                
                if tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g
                    
                    # Heuristic: distance to end region
                    h_score = neighbor_region.distance_to(end_region) * 0.8
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score, neighbor_id))
        
        print(f"Compressed pathfinding failed after {iterations} iterations")
        print(f"Explored {len(g_score)} regions out of {len(regions)} total")
        return None  # No path found
    
    def _regions_to_cells(self, region_path: List[int], 
                         regions: Dict[int, Region],
                         start_cell: Tuple[int, int],
                         end_cell: Tuple[int, int],
                         cost_surface: np.ndarray) -> List[Tuple[int, int]]:
        """Convert region path to cell path"""
        if not region_path:
            return [start_cell, end_cell]
            
        cell_path = []
        
        # Connect start cell to first region
        first_region = regions[region_path[0]]
        if start_cell not in first_region.cells:
            # Find closest cell in first region
            closest_cell = self._find_closest_cell_in_region(start_cell, first_region)
            cell_path.append(start_cell)
            cell_path.append(closest_cell)
        else:
            cell_path.append(start_cell)
        
        # Connect regions
        for i in range(len(region_path) - 1):
            current_region = regions[region_path[i]]
            next_region = regions[region_path[i + 1]]
            
            # Use region centers as waypoints (simplified)
            current_center = ((current_region.min_row + current_region.max_row) // 2,
                            (current_region.min_col + current_region.max_col) // 2)
            if current_center not in cell_path:
                cell_path.append(current_center)
        
        # Connect last region to end cell
        last_region = regions[region_path[-1]]
        if end_cell not in last_region.cells:
            closest_cell = self._find_closest_cell_in_region(end_cell, last_region)
            if closest_cell not in cell_path:
                cell_path.append(closest_cell)
        
        if end_cell not in cell_path:
            cell_path.append(end_cell)
            
        return cell_path
    
    def _find_closest_cell_in_region(self, target_cell: Tuple[int, int], region: Region) -> Tuple[int, int]:
        """Find the closest cell in a region to a target cell"""
        min_distance = float('inf')
        closest_cell = None
        
        for cell in region.cells:
            distance = np.sqrt((cell[0] - target_cell[0])**2 + (cell[1] - target_cell[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell
                
        return closest_cell if closest_cell else next(iter(region.cells))


def compress_search_space(slope_degrees: np.ndarray, cost_surface: np.ndarray,
                         obstacle_mask: np.ndarray = None, path_raster: np.ndarray = None) -> Tuple[Dict[int, Region], np.ndarray]:
    """
    Compress the search space by grouping similar terrain
    
    Args:
        slope_degrees: Array of slope values in degrees
        cost_surface: Array of traversal costs
        obstacle_mask: Boolean array where True indicates obstacles
        path_raster: Array of path type IDs (0 = no path)
    
    Returns:
        regions: Dictionary of region ID to Region objects
        region_map: Array mapping each cell to its region ID (-1 if no region)
    """
    compressor = CompressedPathfinder()
    regions = compressor.create_regions(slope_degrees, cost_surface, obstacle_mask, path_raster)
    
    # Create region map
    region_map = np.full(slope_degrees.shape, -1, dtype=int)
    for region_id, region in regions.items():
        for row, col in region.cells:
            region_map[row, col] = region_id
    
    # Print compression statistics
    total_cells = slope_degrees.size
    compressed_cells = len(regions)
    cells_in_regions = sum(len(r.cells) for r in regions.values())
    
    # Count cells by type
    obstacle_cells = np.sum(obstacle_mask) if obstacle_mask is not None else 0
    path_cells = np.sum(path_raster > 0) if path_raster is not None else 0
    
    print(f"Compression statistics:")
    print(f"  Original cells: {total_cells}")
    print(f"  Obstacle cells excluded: {obstacle_cells}")
    print(f"  Path cells: {path_cells}")
    print(f"  Regions created: {compressed_cells}")
    print(f"  Cells in regions: {cells_in_regions} ({100*cells_in_regions/total_cells:.1f}%)")
    if compressed_cells > 0:
        print(f"  Compression ratio: {cells_in_regions/compressed_cells:.1f}:1 average region size")
        print(f"  Overall compression: {total_cells/compressed_cells:.1f}:1")
    
    return regions, region_map