#!/usr/bin/env python3
"""
A* pathfinding algorithm using elevation data.
Finds paths that minimize elevation changes while staying reasonably direct.
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds


@dataclass
class PathNode:
    """Node in the pathfinding grid"""
    row: int
    col: int
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0  # Heuristic cost to goal
    f_cost: float = float('inf')  # Total cost (g + h)
    parent: Optional['PathNode'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __hash__(self):
        return hash((self.row, self.col))


class ElevationPathfinder:
    """
    A* pathfinder that uses elevation data to find energy-efficient paths.
    """
    
    def __init__(self, 
                 elevation_array: np.ndarray,
                 pixel_size_meters: float,
                 elevation_weight: float = 1.0,
                 elevation_exponent: float = 2.0,
                 distance_weight: float = 1.0,
                 max_slope_degrees: float = 45.0):
        """
        Initialize the pathfinder.
        
        Args:
            elevation_array: 2D array of elevations
            pixel_size_meters: Size of each pixel in meters
            elevation_weight: Weight for elevation change penalty
            elevation_exponent: Exponent for elevation penalty (higher = more penalty for big changes)
            distance_weight: Weight for distance from straight line
            max_slope_degrees: Maximum traversable slope
        """
        self.elevation = elevation_array
        self.rows, self.cols = elevation_array.shape
        self.pixel_size = pixel_size_meters
        self.elevation_weight = elevation_weight
        self.elevation_exponent = elevation_exponent
        self.distance_weight = distance_weight
        self.max_slope = np.tan(np.radians(max_slope_degrees))
        
        # 8-connected neighbors (including diagonals)
        self.neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Distance multipliers for diagonal vs cardinal moves
        self.neighbor_distances = [
            np.sqrt(2), 1.0, np.sqrt(2),
            1.0,              1.0,
            np.sqrt(2), 1.0, np.sqrt(2)
        ]
    
    def calculate_move_cost(self, from_row: int, from_col: int, 
                          to_row: int, to_col: int,
                          straight_line_distance: float,
                          current_distance: float) -> float:
        """
        Calculate the cost of moving from one cell to another.
        
        Returns:
            Cost of the move, or inf if move is impossible
        """
        # Get elevations
        elev_from = self.elevation[from_row, from_col]
        elev_to = self.elevation[to_row, to_col]
        
        # Calculate horizontal distance
        dr = to_row - from_row
        dc = to_col - from_col
        horiz_distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        # Check slope constraint
        slope = abs(elev_to - elev_from) / horiz_distance
        if slope > self.max_slope:
            return float('inf')
        
        # Base movement cost (distance)
        base_cost = horiz_distance
        
        # Elevation change penalty
        elevation_change = abs(elev_to - elev_from)
        # Exponential penalty for large elevation changes
        elevation_penalty = self.elevation_weight * (elevation_change ** self.elevation_exponent)
        
        # Additional penalty for going uphill (asymmetric cost)
        if elev_to > elev_from:
            elevation_penalty *= 1.5  # 50% more costly to go uphill
        
        # Distance from straight line penalty
        # Calculate perpendicular distance from straight line path
        straight_line_position = current_distance / straight_line_distance
        expected_row = self.start_row + (self.goal_row - self.start_row) * straight_line_position
        expected_col = self.start_col + (self.goal_col - self.start_col) * straight_line_position
        
        deviation = np.sqrt((to_row - expected_row)**2 + (to_col - expected_col)**2) * self.pixel_size
        deviation_penalty = self.distance_weight * deviation
        
        return base_cost + elevation_penalty + deviation_penalty
    
    def heuristic(self, row: int, col: int) -> float:
        """
        A* heuristic function - estimated cost to goal.
        Uses Euclidean distance with minimum elevation change estimate.
        """
        # Euclidean distance
        dr = self.goal_row - row
        dc = self.goal_col - col
        distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        # Estimate minimum elevation change (optimistic)
        elev_current = self.elevation[row, col]
        elev_goal = self.elevation[self.goal_row, self.goal_col]
        min_elev_change = abs(elev_goal - elev_current)
        
        # Combined heuristic
        return distance + self.elevation_weight * min_elev_change
    
    def find_path(self, start_row: int, start_col: int, 
                  goal_row: int, goal_col: int) -> Optional[List[Tuple[int, int]]]:
        """
        Find optimal path from start to goal using A*.
        
        Returns:
            List of (row, col) tuples representing the path, or None if no path exists
        """
        self.start_row, self.start_col = start_row, start_col
        self.goal_row, self.goal_col = goal_row, goal_col
        
        # Calculate straight line distance for deviation penalty
        dr = goal_row - start_row
        dc = goal_col - start_col
        self.straight_line_distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        # Initialize start node
        start_node = PathNode(start_row, start_col, g_cost=0)
        start_node.h_cost = self.heuristic(start_row, start_col)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # Priority queue and visited set
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        
        # Track best g_cost for each node
        g_costs: Dict[Tuple[int, int], float] = {(start_row, start_col): 0}
        
        # Node lookup for updating parents
        all_nodes: Dict[Tuple[int, int], PathNode] = {(start_row, start_col): start_node}
        
        iterations = 0
        max_iterations = self.rows * self.cols * 4  # Prevent infinite loops
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            
            # Skip if already processed
            if (current.row, current.col) in closed_set:
                continue
            
            # Check if we reached the goal
            if current.row == goal_row and current.col == goal_col:
                # Reconstruct path
                path = []
                node = current
                while node:
                    path.append((node.row, node.col))
                    node = node.parent
                return list(reversed(path))
            
            closed_set.add((current.row, current.col))
            
            # Check all neighbors
            for (dr, dc), base_dist in zip(self.neighbors, self.neighbor_distances):
                neighbor_row = current.row + dr
                neighbor_col = current.col + dc
                
                # Check bounds
                if (neighbor_row < 0 or neighbor_row >= self.rows or
                    neighbor_col < 0 or neighbor_col >= self.cols):
                    continue
                
                # Skip if already processed
                if (neighbor_row, neighbor_col) in closed_set:
                    continue
                
                # Calculate move cost
                move_cost = self.calculate_move_cost(
                    current.row, current.col,
                    neighbor_row, neighbor_col,
                    self.straight_line_distance,
                    current.g_cost
                )
                
                # Skip if move is impossible
                if move_cost == float('inf'):
                    continue
                
                # Calculate new g_cost
                new_g_cost = current.g_cost + move_cost
                
                # Skip if we already found a better path to this node
                if (neighbor_row, neighbor_col) in g_costs:
                    if new_g_cost >= g_costs[(neighbor_row, neighbor_col)]:
                        continue
                
                # Update or create neighbor node
                if (neighbor_row, neighbor_col) in all_nodes:
                    neighbor = all_nodes[(neighbor_row, neighbor_col)]
                else:
                    neighbor = PathNode(neighbor_row, neighbor_col)
                    all_nodes[(neighbor_row, neighbor_col)] = neighbor
                
                # Update costs and parent
                neighbor.g_cost = new_g_cost
                neighbor.h_cost = self.heuristic(neighbor_row, neighbor_col)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current
                
                g_costs[(neighbor_row, neighbor_col)] = new_g_cost
                heapq.heappush(open_set, neighbor)
        
        # No path found
        return None
    
    def calculate_path_stats(self, path: List[Tuple[int, int]]) -> Dict:
        """Calculate statistics about a path."""
        if not path or len(path) < 2:
            return {}
        
        total_distance = 0
        total_ascent = 0
        total_descent = 0
        elevations = []
        
        for i in range(len(path)):
            row, col = path[i]
            elev = self.elevation[row, col]
            elevations.append(elev)
            
            if i > 0:
                prev_row, prev_col = path[i-1]
                
                # Distance
                dr = row - prev_row
                dc = col - prev_col
                distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
                total_distance += distance
                
                # Elevation change
                prev_elev = self.elevation[prev_row, prev_col]
                elev_change = elev - prev_elev
                
                if elev_change > 0:
                    total_ascent += elev_change
                else:
                    total_descent += abs(elev_change)
        
        # Straight line distance
        start_row, start_col = path[0]
        end_row, end_col = path[-1]
        dr = end_row - start_row
        dc = end_col - start_col
        straight_distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        return {
            'total_distance': total_distance,
            'straight_distance': straight_distance,
            'distance_ratio': total_distance / straight_distance if straight_distance > 0 else 1,
            'total_ascent': total_ascent,
            'total_descent': total_descent,
            'net_elevation': elevations[-1] - elevations[0],
            'elevations': elevations,
            'min_elevation': min(elevations),
            'max_elevation': max(elevations)
        }


def visualize_path(elevation_array: np.ndarray, path: List[Tuple[int, int]], 
                   bounds: Bounds, stats: Dict, output_file: str = "path_result.png"):
    """Visualize the found path on the elevation map."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot 1: Path on elevation map
    im = ax1.imshow(elevation_array, cmap='terrain', origin='upper',
                    extent=[bounds.west, bounds.east, bounds.south, bounds.north])
    
    # Convert path to coordinates
    if path:
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        
        # Convert to lat/lon
        lons = bounds.west + (cols / elevation_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elevation_array.shape[0]) * (bounds.north - bounds.south)
        
        # Plot path
        ax1.plot(lons, lats, 'r-', linewidth=3, label='Found Path')
        ax1.plot(lons[0], lats[0], 'go', markersize=10, label='Start')
        ax1.plot(lons[-1], lats[-1], 'ro', markersize=10, label='Goal')
        
        # Plot straight line
        ax1.plot([lons[0], lons[-1]], [lats[0], lats[-1]], 
                'k--', linewidth=2, alpha=0.5, label='Direct Path')
    
    ax1.set_title('A* Path on Elevation Map', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Elevation (m)')
    
    # Plot 2: Elevation profile
    if path and 'elevations' in stats:
        distances = [0]
        for i in range(1, len(path)):
            dr = path[i][0] - path[i-1][0]
            dc = path[i][1] - path[i-1][1]
            dist = np.sqrt(dr*dr + dc*dc) * stats.get('pixel_size', 10)
            distances.append(distances[-1] + dist)
        
        ax2.plot(distances, stats['elevations'], 'b-', linewidth=2)
        ax2.fill_between(distances, stats['elevations'], alpha=0.3)
        ax2.set_xlabel('Distance along path (m)')
        ax2.set_ylabel('Elevation (m)')
        ax2.set_title('Elevation Profile', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Total Distance: {stats['total_distance']:.1f}m\n"
        stats_text += f"Direct Distance: {stats['straight_distance']:.1f}m\n"
        stats_text += f"Distance Ratio: {stats['distance_ratio']:.2f}\n"
        stats_text += f"Total Ascent: {stats['total_ascent']:.1f}m\n"
        stats_text += f"Total Descent: {stats['total_descent']:.1f}m\n"
        stats_text += f"Net Elevation: {stats['net_elevation']:.1f}m"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def find_elevation_path(data_dir: str,
                       start_lat: float, start_lon: float,
                       goal_lat: float, goal_lon: float,
                       resolution: int = 10,
                       elevation_weight: float = 0.5,
                       elevation_exponent: float = 2.0,
                       distance_weight: float = 0.1,
                       max_slope_degrees: float = 45.0,
                       margin: float = 0.002,
                       output_file: str = "elevation_path.png") -> Optional[str]:
    """
    Find and visualize an optimal path between two points.
    
    Args:
        data_dir: Directory containing elevation data
        start_lat, start_lon: Starting coordinates
        goal_lat, goal_lon: Goal coordinates
        resolution: DEM resolution in meters
        elevation_weight: Weight for elevation penalty (0-1, higher = avoid elevation changes)
        elevation_exponent: Exponent for elevation penalty (1-3, higher = strongly avoid big changes)
        distance_weight: Weight for deviation penalty (0-1, higher = stay closer to direct path)
        max_slope_degrees: Maximum traversable slope
        margin: Extra bounds around start/goal for loading data
        output_file: Output visualization file
    
    Returns:
        Path to output file if successful, None otherwise
    """
    # Calculate bounds with margin
    min_lat = min(start_lat, goal_lat) - margin
    max_lat = max(start_lat, goal_lat) + margin
    min_lon = min(start_lon, goal_lon) - margin
    max_lon = max(start_lon, goal_lon) + margin
    
    bounds = Bounds(north=max_lat, south=min_lat, east=max_lon, west=min_lon)
    
    # Load elevation data
    print(f"Loading elevation data...")
    elev_lib = TwoLayerElevationLibrary(data_dir, resolution)
    result = elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print(f"Failed to load elevation data: {result.get('message', 'Unknown error')}")
        return None
    
    # Get elevation array
    elev_array, metadata = elev_lib.get_elevation_array(bounds)
    
    # Calculate pixel size
    lat_center = (bounds.north + bounds.south) / 2
    lon_deg_to_m = 111000 * np.cos(np.radians(lat_center))
    pixel_size = abs(metadata['transform']['a']) * lon_deg_to_m
    
    # Convert lat/lon to pixel coordinates
    def latlon_to_pixel(lat, lon):
        col = int((lon - bounds.west) / (bounds.east - bounds.west) * elev_array.shape[1])
        row = int((bounds.north - lat) / (bounds.north - bounds.south) * elev_array.shape[0])
        return min(max(row, 0), elev_array.shape[0]-1), min(max(col, 0), elev_array.shape[1]-1)
    
    start_row, start_col = latlon_to_pixel(start_lat, start_lon)
    goal_row, goal_col = latlon_to_pixel(goal_lat, goal_lon)
    
    print(f"Start: ({start_lat:.4f}, {start_lon:.4f}) -> pixel ({start_row}, {start_col})")
    print(f"Goal: ({goal_lat:.4f}, {goal_lon:.4f}) -> pixel ({goal_row}, {goal_col})")
    print(f"Start elevation: {elev_array[start_row, start_col]:.1f}m")
    print(f"Goal elevation: {elev_array[goal_row, goal_col]:.1f}m")
    
    # Create pathfinder
    print(f"\nFinding path with parameters:")
    print(f"  Elevation weight: {elevation_weight}")
    print(f"  Elevation exponent: {elevation_exponent}")
    print(f"  Distance weight: {distance_weight}")
    print(f"  Max slope: {max_slope_degrees}°")
    
    pathfinder = ElevationPathfinder(
        elev_array, pixel_size,
        elevation_weight=elevation_weight,
        elevation_exponent=elevation_exponent,
        distance_weight=distance_weight,
        max_slope_degrees=max_slope_degrees
    )
    
    # Find path
    path = pathfinder.find_path(start_row, start_col, goal_row, goal_col)
    
    if path:
        print(f"\nPath found with {len(path)} waypoints!")
        stats = pathfinder.calculate_path_stats(path)
        stats['pixel_size'] = pixel_size
        
        print(f"\nPath statistics:")
        print(f"  Total distance: {stats['total_distance']:.1f}m")
        print(f"  Direct distance: {stats['straight_distance']:.1f}m")
        print(f"  Distance ratio: {stats['distance_ratio']:.2f}x")
        print(f"  Total ascent: {stats['total_ascent']:.1f}m")
        print(f"  Total descent: {stats['total_descent']:.1f}m")
        print(f"  Net elevation change: {stats['net_elevation']:.1f}m")
        
        # Visualize
        output = visualize_path(elev_array, path, bounds, stats, output_file)
        print(f"\nVisualization saved to: {output}")
        return output
    else:
        print("\nNo path found!")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find elevation-aware paths using A*')
    parser.add_argument('--start', type=str, required=True,
                       help='Start coordinates as lat,lon')
    parser.add_argument('--goal', type=str, required=True,
                       help='Goal coordinates as lat,lon')
    parser.add_argument('--resolution', type=int, default=10,
                       help='DEM resolution in meters (default: 10)')
    parser.add_argument('--elevation-weight', type=float, default=0.5,
                       help='Elevation penalty weight 0-1 (default: 0.5)')
    parser.add_argument('--elevation-exponent', type=float, default=2.0,
                       help='Elevation penalty exponent 1-3 (default: 2.0)')
    parser.add_argument('--distance-weight', type=float, default=0.1,
                       help='Path deviation penalty weight 0-1 (default: 0.1)')
    parser.add_argument('--max-slope', type=float, default=45.0,
                       help='Maximum slope in degrees (default: 45)')
    parser.add_argument('--data-dir', type=str, default='./elevation_data',
                       help='Directory for elevation data')
    parser.add_argument('--output', type=str, default='elevation_path.png',
                       help='Output filename')
    
    args = parser.parse_args()
    
    # Parse coordinates
    start_parts = args.start.split(',')
    goal_parts = args.goal.split(',')
    
    if len(start_parts) != 2 or len(goal_parts) != 2:
        parser.error("Coordinates must be in format: lat,lon")
    
    try:
        start_lat, start_lon = float(start_parts[0]), float(start_parts[1])
        goal_lat, goal_lon = float(goal_parts[0]), float(goal_parts[1])
    except ValueError:
        parser.error("Invalid coordinate format")
    
    find_elevation_path(
        args.data_dir,
        start_lat, start_lon,
        goal_lat, goal_lon,
        resolution=args.resolution,
        elevation_weight=args.elevation_weight,
        elevation_exponent=args.elevation_exponent,
        distance_weight=args.distance_weight,
        max_slope_degrees=args.max_slope,
        output_file=args.output
    )