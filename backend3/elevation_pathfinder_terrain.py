#!/usr/bin/env python3
"""
A* pathfinding on elevation data with terrain type awareness.
Incorporates both elevation changes and terrain preferences.
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
from elevation import TwoLayerElevationLibrary, Bounds
from path_layer import PathLayer, PathType

@dataclass
class TerrainNode:
    """Node in the pathfinding graph with terrain awareness"""
    row: int
    col: int
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Optional['TerrainNode'] = None
    elevation: float = 0
    terrain_type: int = PathType.UNKNOWN
    consecutive_steep_distance: float = 0  # For sustained slope tracking
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class TerrainAwarePathfinder:
    """A* pathfinder that considers both elevation and terrain type"""
    
    def __init__(self, 
                 elevation_lib: TwoLayerElevationLibrary,
                 path_layer: PathLayer,
                 bounds: Bounds,
                 resolution: float = 10.0):
        self.elevation_lib = elevation_lib
        self.path_layer = path_layer
        self.bounds = bounds
        self.resolution = resolution
        
        # Get elevation data
        result = elevation_lib.get_elevation_array(bounds)
        if isinstance(result, tuple):
            self.elevation, transform_data = result
        else:
            self.elevation = result
            # Create transform from bounds
            from rasterio.transform import from_bounds
            transform_data = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north,
                                       self.elevation.shape[1], self.elevation.shape[0])
        
        # Clean up open files if using FD-safe wrapper
        if hasattr(elevation_lib, 'close_all'):
            elevation_lib.close_all()
        
        # Handle transform format
        if isinstance(transform_data, dict):
            # Transform data is in the format from elevation library
            from rasterio.transform import from_bounds
            self.transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north,
                                       self.elevation.shape[1], self.elevation.shape[0])
        else:
            self.transform = transform_data
            
        self.rows, self.cols = self.elevation.shape
        
        # Load path layer data
        tile_x = int(bounds.west * 100)
        tile_y = int(bounds.south * 100)
        self.terrain_types = path_layer.load_tile(tile_x, tile_y)
        
        if self.terrain_types is None:
            print("Warning: No terrain data available, using unknown terrain")
            self.terrain_types = np.full(self.elevation.shape, PathType.UNKNOWN, dtype=np.uint8)
        elif self.terrain_types.shape != self.elevation.shape:
            # Resize terrain types to match elevation
            from scipy.ndimage import zoom
            zoom_y = self.elevation.shape[0] / self.terrain_types.shape[0]
            zoom_x = self.elevation.shape[1] / self.terrain_types.shape[1]
            self.terrain_types = zoom(self.terrain_types, (zoom_y, zoom_x), order=0)
        
        # Cost parameters
        self.elevation_weight = 1.0
        self.elevation_exponent = 2.0
        self.terrain_weight = 1.0
        self.obstacle_cost = 10000.0
        self.off_path_penalty = 1.5
        self.max_slope_degrees = 45.0
        
        # Terrain cost multipliers (lower is better)
        self.terrain_costs = {
            PathType.UNKNOWN: 1.0,
            PathType.OBSTACLE: self.obstacle_cost,
            PathType.HIKING_PATH: 0.3,  # Strong preference for trails
            PathType.STREET: 0.7,        # Streets are OK but not preferred
            PathType.NATURAL: 0.8        # Natural areas are good
        }
        
        # Sustained slope parameters
        self.steep_threshold = 15.0  # degrees
        self.fatigue_distance = 100.0  # meters
        self.fatigue_exponent = 2.0
        self.sustained_slope_weight = 0.5
    
    def set_parameters(self, **kwargs):
        """Update pathfinder parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter {key}")
    
    def lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to grid coordinates"""
        col, row = ~self.transform * (lon, lat)
        return int(row), int(col)
    
    def grid_to_lat_lon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon"""
        lon, lat = self.transform * (col, row)
        return lat, lon
    
    def calculate_move_cost(self, from_row: int, from_col: int, 
                          to_row: int, to_col: int,
                          straight_line_distance: float,
                          current_distance: float,
                          from_steep_distance: float = 0) -> Tuple[float, float]:
        """
        Calculate cost of moving between two cells considering terrain and elevation.
        Returns: (cost, new_steep_distance)
        """
        # Get elevations
        elev_from = self.elevation[from_row, from_col]
        elev_to = self.elevation[to_row, to_col]
        
        # Get terrain types
        terrain_from = self.terrain_types[from_row, from_col]
        terrain_to = self.terrain_types[to_row, to_col]
        
        # Calculate physical distance
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        horiz_distance = self.resolution * np.sqrt(row_diff**2 + col_diff**2)
        
        # Base movement cost (distance)
        base_cost = horiz_distance
        
        # Terrain cost multiplier
        terrain_multiplier = self.terrain_costs.get(terrain_to, 1.0)
        
        # Add transition penalty when leaving a good path
        if (terrain_from == PathType.HIKING_PATH and 
            terrain_to not in [PathType.HIKING_PATH, PathType.NATURAL]):
            terrain_multiplier *= 1.5  # Penalty for leaving trails
        
        # Calculate slope
        elevation_change = abs(elev_to - elev_from)
        slope_radians = np.arctan2(elevation_change, horiz_distance)
        slope_degrees = np.degrees(slope_radians)
        
        # Check max slope constraint
        if slope_degrees > self.max_slope_degrees:
            return float('inf'), from_steep_distance  # Impassable
        
        # Elevation penalty (exponential based on slope)
        elevation_penalty = self.elevation_weight * (slope_degrees / 10.0) ** self.elevation_exponent
        
        # Additional penalty for going uphill
        if elev_to > elev_from:
            elevation_penalty *= 1.5
        
        # Track sustained steep sections
        new_steep_distance = from_steep_distance
        sustained_penalty = 0
        
        if slope_degrees > self.steep_threshold:
            new_steep_distance += horiz_distance
            # Exponential fatigue penalty
            fatigue_factor = (new_steep_distance / self.fatigue_distance) ** self.fatigue_exponent
            sustained_penalty = self.sustained_slope_weight * fatigue_factor * horiz_distance
        else:
            # Recovery when on gentler slopes
            new_steep_distance = max(0, new_steep_distance - horiz_distance * 0.5)
        
        # Path deviation penalty
        deviation_ratio = current_distance / max(straight_line_distance, 1.0)
        deviation_penalty = 0.1 * max(0, deviation_ratio - 1.5) ** 2
        
        # Total cost combines all factors
        total_cost = base_cost * terrain_multiplier * (1 + elevation_penalty + sustained_penalty + deviation_penalty)
        
        return total_cost, new_steep_distance
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connected)"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    # Check if terrain is passable
                    if self.terrain_types[new_row, new_col] != PathType.OBSTACLE:
                        neighbors.append((new_row, new_col))
        return neighbors
    
    def heuristic(self, row: int, col: int, goal_row: int, goal_col: int) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return self.resolution * np.sqrt((row - goal_row)**2 + (col - goal_col)**2)
    
    def find_path(self, start_lat: float, start_lon: float, 
                  end_lat: float, end_lon: float) -> Optional[List[Tuple[float, float, float]]]:
        """
        Find optimal path from start to end considering terrain and elevation.
        Returns list of (lat, lon, elevation) tuples.
        """
        # Convert to grid coordinates
        start_row, start_col = self.lat_lon_to_grid(start_lat, start_lon)
        end_row, end_col = self.lat_lon_to_grid(end_lat, end_lon)
        
        # Validate coordinates
        if not (0 <= start_row < self.rows and 0 <= start_col < self.cols):
            print(f"Start point ({start_lat}, {start_lon}) is outside bounds")
            return None
        if not (0 <= end_row < self.rows and 0 <= end_col < self.cols):
            print(f"End point ({end_lat}, {end_lon}) is outside bounds")
            return None
        
        # Check if start/end are on obstacles
        if self.terrain_types[start_row, start_col] == PathType.OBSTACLE:
            print(f"Warning: Start point is on an obstacle!")
        if self.terrain_types[end_row, end_col] == PathType.OBSTACLE:
            print(f"Warning: End point is on an obstacle!")
        
        # Calculate straight-line distance
        straight_line_distance = self.heuristic(start_row, start_col, end_row, end_col)
        
        print(f"\nPathfinding from ({start_lat:.6f}, {start_lon:.6f}) to ({end_lat:.6f}, {end_lon:.6f})")
        print(f"Grid: ({start_row}, {start_col}) to ({end_row}, {end_col})")
        print(f"Start terrain: {self.path_layer.get_path_type_name(self.terrain_types[start_row, start_col])}")
        print(f"End terrain: {self.path_layer.get_path_type_name(self.terrain_types[end_row, end_col])}")
        print(f"Straight-line distance: {straight_line_distance:.1f}m")
        
        # Initialize A* algorithm
        start_node = TerrainNode(
            row=start_row, 
            col=start_col, 
            g_cost=0,
            h_cost=self.heuristic(start_row, start_col, end_row, end_col),
            elevation=self.elevation[start_row, start_col],
            terrain_type=self.terrain_types[start_row, start_col]
        )
        
        open_set = [start_node]
        closed_set = set()
        best_g_cost = {}
        best_g_cost[(start_row, start_col)] = 0
        
        nodes_explored = 0
        start_time = time.time()
        
        while open_set:
            current = heapq.heappop(open_set)
            
            if (current.row, current.col) in closed_set:
                continue
            
            nodes_explored += 1
            if nodes_explored % 1000 == 0:
                print(f"  Explored {nodes_explored} nodes...")
            
            # Check if we reached the goal
            if current.row == end_row and current.col == end_col:
                print(f"\nPath found! Explored {nodes_explored} nodes in {time.time() - start_time:.2f}s")
                return self.reconstruct_path(current)
            
            closed_set.add((current.row, current.col))
            
            # Explore neighbors
            for next_row, next_col in self.get_neighbors(current.row, current.col):
                if (next_row, next_col) in closed_set:
                    continue
                
                # Calculate cost to move to this neighbor
                move_cost, new_steep_distance = self.calculate_move_cost(
                    current.row, current.col, next_row, next_col,
                    straight_line_distance, current.g_cost,
                    current.consecutive_steep_distance
                )
                
                # Skip if move is impossible (infinite cost)
                if move_cost == float('inf'):
                    continue
                
                new_g_cost = current.g_cost + move_cost
                
                # Skip if we've found a better path to this cell
                if (next_row, next_col) in best_g_cost and new_g_cost >= best_g_cost[(next_row, next_col)]:
                    continue
                
                best_g_cost[(next_row, next_col)] = new_g_cost
                
                # Create new node
                new_node = TerrainNode(
                    row=next_row,
                    col=next_col,
                    g_cost=new_g_cost,
                    h_cost=self.heuristic(next_row, next_col, end_row, end_col),
                    parent=current,
                    elevation=self.elevation[next_row, next_col],
                    terrain_type=self.terrain_types[next_row, next_col],
                    consecutive_steep_distance=new_steep_distance
                )
                
                heapq.heappush(open_set, new_node)
        
        print(f"\nNo path found after exploring {nodes_explored} nodes")
        return None
    
    def reconstruct_path(self, end_node: TerrainNode) -> List[Tuple[float, float, float]]:
        """Reconstruct path from end node back to start"""
        path = []
        current = end_node
        total_distance = 0
        total_elevation_gain = 0
        terrain_stats = {t: 0 for t in PathType.__dict__.values() if isinstance(t, int)}
        
        while current:
            lat, lon = self.grid_to_lat_lon(current.row, current.col)
            path.append((lat, lon, current.elevation))
            
            # Collect statistics
            terrain_stats[current.terrain_type] = terrain_stats.get(current.terrain_type, 0) + 1
            
            if current.parent:
                # Calculate distance
                row_diff = current.row - current.parent.row
                col_diff = current.col - current.parent.col
                segment_distance = self.resolution * np.sqrt(row_diff**2 + col_diff**2)
                total_distance += segment_distance
                
                # Calculate elevation gain
                elev_diff = current.elevation - current.parent.elevation
                if elev_diff > 0:
                    total_elevation_gain += elev_diff
            
            current = current.parent
        
        path.reverse()
        
        # Print statistics
        print(f"\nPath statistics:")
        print(f"  Total distance: {total_distance:.1f}m")
        print(f"  Total elevation gain: {total_elevation_gain:.1f}m")
        print(f"  Average grade: {total_elevation_gain/total_distance*100:.1f}%")
        
        print(f"\nTerrain breakdown:")
        total_points = sum(terrain_stats.values())
        for terrain_type, count in terrain_stats.items():
            if count > 0:
                percentage = count / total_points * 100
                name = self.path_layer.get_path_type_name(terrain_type)
                print(f"  {name}: {percentage:.1f}%")
        
        return path