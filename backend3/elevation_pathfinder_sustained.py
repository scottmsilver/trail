#!/usr/bin/env python3
"""
Enhanced A* pathfinding with sustained slope penalty.
Penalizes staying at high slopes for too long, encouraging rest breaks on gentler terrain.
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds


@dataclass
class PathNodeWithFatigue:
    """Node that tracks accumulated slope exposure"""
    row: int
    col: int
    g_cost: float = float('inf')
    h_cost: float = 0
    f_cost: float = float('inf')
    parent: Optional['PathNodeWithFatigue'] = None
    
    # New attributes for sustained slope tracking
    consecutive_steep_distance: float = 0  # Distance on steep slopes without break
    total_steep_distance: float = 0  # Total distance on steep slopes
    steep_segments: List[float] = field(default_factory=list)  # List of steep segment lengths
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __hash__(self):
        return hash((self.row, self.col))


class SustainedSlopePathfinder:
    """
    Enhanced pathfinder that penalizes sustained steep slopes.
    """
    
    def __init__(self, 
                 elevation_array: np.ndarray,
                 pixel_size_meters: float,
                 elevation_weight: float = 1.0,
                 elevation_exponent: float = 2.0,
                 distance_weight: float = 1.0,
                 sustained_slope_weight: float = 1.0,
                 steep_threshold_degrees: float = 15.0,
                 rest_threshold_degrees: float = 8.0,
                 fatigue_distance: float = 100.0,
                 fatigue_exponent: float = 2.0,
                 max_slope_degrees: float = 45.0):
        """
        Initialize the pathfinder with sustained slope penalty.
        
        Args:
            sustained_slope_weight: Weight for sustained slope penalty
            steep_threshold_degrees: Slope above which is considered "steep"
            rest_threshold_degrees: Slope below which resets the steep counter
            fatigue_distance: Distance scale for fatigue accumulation (meters)
            fatigue_exponent: How quickly fatigue accumulates (higher = faster)
        """
        self.elevation = elevation_array
        self.rows, self.cols = elevation_array.shape
        self.pixel_size = pixel_size_meters
        self.elevation_weight = elevation_weight
        self.elevation_exponent = elevation_exponent
        self.distance_weight = distance_weight
        self.sustained_slope_weight = sustained_slope_weight
        self.steep_threshold = np.tan(np.radians(steep_threshold_degrees))
        self.rest_threshold = np.tan(np.radians(rest_threshold_degrees))
        self.fatigue_distance = fatigue_distance
        self.fatigue_exponent = fatigue_exponent
        self.max_slope = np.tan(np.radians(max_slope_degrees))
        
        # 8-connected neighbors
        self.neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        self.neighbor_distances = [
            np.sqrt(2), 1.0, np.sqrt(2),
            1.0,              1.0,
            np.sqrt(2), 1.0, np.sqrt(2)
        ]
    
    def calculate_move_cost_with_fatigue(self, from_node: PathNodeWithFatigue,
                                       to_row: int, to_col: int,
                                       straight_line_distance: float,
                                       current_distance: float) -> Tuple[float, PathNodeWithFatigue]:
        """
        Calculate move cost including sustained slope penalty.
        
        Returns:
            Tuple of (cost, new_node_state)
        """
        from_row, from_col = from_node.row, from_node.col
        
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
            return float('inf'), None
        
        # Base movement cost
        base_cost = horiz_distance
        
        # Elevation change penalty
        elevation_change = abs(elev_to - elev_from)
        elevation_penalty = self.elevation_weight * (elevation_change ** self.elevation_exponent)
        
        # Uphill penalty
        if elev_to > elev_from:
            elevation_penalty *= 1.5
        
        # Distance from straight line penalty
        straight_line_position = current_distance / straight_line_distance
        expected_row = self.start_row + (self.goal_row - self.start_row) * straight_line_position
        expected_col = self.start_col + (self.goal_col - self.start_col) * straight_line_position
        
        deviation = np.sqrt((to_row - expected_row)**2 + (to_col - expected_col)**2) * self.pixel_size
        deviation_penalty = self.distance_weight * deviation
        
        # Create new node state
        new_node = PathNodeWithFatigue(
            row=to_row,
            col=to_col,
            consecutive_steep_distance=from_node.consecutive_steep_distance,
            total_steep_distance=from_node.total_steep_distance,
            steep_segments=from_node.steep_segments.copy()
        )
        
        # Calculate sustained slope penalty
        sustained_penalty = 0
        
        if slope > self.steep_threshold:
            # On steep terrain - accumulate fatigue
            new_node.consecutive_steep_distance += horiz_distance
            new_node.total_steep_distance += horiz_distance
            
            # Exponential penalty for sustained steep sections
            fatigue_factor = (new_node.consecutive_steep_distance / self.fatigue_distance) ** self.fatigue_exponent
            sustained_penalty = self.sustained_slope_weight * fatigue_factor * horiz_distance
            
            # Additional penalty for very long steep sections
            if new_node.consecutive_steep_distance > self.fatigue_distance * 2:
                sustained_penalty *= 2  # Double penalty for extremely long steep sections
                
        elif slope < self.rest_threshold:
            # On gentle terrain - reset consecutive counter
            if new_node.consecutive_steep_distance > 0:
                # Record the completed steep segment
                new_node.steep_segments.append(new_node.consecutive_steep_distance)
                new_node.consecutive_steep_distance = 0
        
        # Penalty for multiple steep segments (encourages fewer, longer segments over many short ones)
        segment_penalty = len(new_node.steep_segments) * 10  # Small penalty per segment
        
        total_cost = base_cost + elevation_penalty + deviation_penalty + sustained_penalty + segment_penalty
        
        return total_cost, new_node
    
    def heuristic(self, row: int, col: int) -> float:
        """A* heuristic function."""
        dr = self.goal_row - row
        dc = self.goal_col - col
        distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        elev_current = self.elevation[row, col]
        elev_goal = self.elevation[self.goal_row, self.goal_col]
        min_elev_change = abs(elev_goal - elev_current)
        
        return distance + self.elevation_weight * min_elev_change
    
    def find_path(self, start_row: int, start_col: int, 
                  goal_row: int, goal_col: int) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path considering sustained slope fatigue."""
        self.start_row, self.start_col = start_row, start_col
        self.goal_row, self.goal_col = goal_row, goal_col
        
        # Calculate straight line distance
        dr = goal_row - start_row
        dc = goal_col - start_col
        self.straight_line_distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        # Initialize start node
        start_node = PathNodeWithFatigue(start_row, start_col, g_cost=0)
        start_node.h_cost = self.heuristic(start_row, start_col)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # Priority queue and visited set
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        
        # Track best g_cost for each position
        g_costs: Dict[Tuple[int, int], float] = {(start_row, start_col): 0}
        
        # Node lookup
        all_nodes: Dict[Tuple[int, int], PathNodeWithFatigue] = {(start_row, start_col): start_node}
        
        iterations = 0
        max_iterations = self.rows * self.cols * 4
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_set)
            
            if (current.row, current.col) in closed_set:
                continue
            
            if current.row == goal_row and current.col == goal_col:
                # Reconstruct path
                path = []
                node = current
                while node:
                    path.append((node.row, node.col))
                    node = node.parent
                
                # Store fatigue info for analysis
                self.final_node_state = current
                return list(reversed(path))
            
            closed_set.add((current.row, current.col))
            
            # Check all neighbors
            for (dr, dc), base_dist in zip(self.neighbors, self.neighbor_distances):
                neighbor_row = current.row + dr
                neighbor_col = current.col + dc
                
                if (neighbor_row < 0 or neighbor_row >= self.rows or
                    neighbor_col < 0 or neighbor_col >= self.cols):
                    continue
                
                if (neighbor_row, neighbor_col) in closed_set:
                    continue
                
                # Calculate move cost with fatigue
                move_cost, new_node_state = self.calculate_move_cost_with_fatigue(
                    current, neighbor_row, neighbor_col,
                    self.straight_line_distance, current.g_cost
                )
                
                if move_cost == float('inf') or new_node_state is None:
                    continue
                
                new_g_cost = current.g_cost + move_cost
                
                if (neighbor_row, neighbor_col) in g_costs:
                    if new_g_cost >= g_costs[(neighbor_row, neighbor_col)]:
                        continue
                
                # Update neighbor with new state
                new_node_state.g_cost = new_g_cost
                new_node_state.h_cost = self.heuristic(neighbor_row, neighbor_col)
                new_node_state.f_cost = new_g_cost + new_node_state.h_cost
                new_node_state.parent = current
                
                g_costs[(neighbor_row, neighbor_col)] = new_g_cost
                all_nodes[(neighbor_row, neighbor_col)] = new_node_state
                heapq.heappush(open_set, new_node_state)
        
        return None
    
    def calculate_path_stats_with_fatigue(self, path: List[Tuple[int, int]]) -> Dict:
        """Calculate statistics including fatigue analysis."""
        if not path or len(path) < 2:
            return {}
        
        # Standard stats
        total_distance = 0
        total_ascent = 0
        total_descent = 0
        elevations = []
        slopes = []
        
        # Fatigue tracking
        steep_segments = []
        current_steep_start = None
        current_steep_distance = 0
        
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
                
                # Slope
                slope = abs(elev_change) / distance if distance > 0 else 0
                slopes.append(np.degrees(np.arctan(slope)))
                
                # Track steep segments
                if slope > self.steep_threshold:
                    if current_steep_start is None:
                        current_steep_start = i - 1
                    current_steep_distance += distance
                else:
                    if current_steep_start is not None and slope < self.rest_threshold:
                        # End of steep segment
                        steep_segments.append({
                            'start_idx': current_steep_start,
                            'end_idx': i - 1,
                            'distance': current_steep_distance
                        })
                        current_steep_start = None
                        current_steep_distance = 0
        
        # Handle final steep segment
        if current_steep_start is not None:
            steep_segments.append({
                'start_idx': current_steep_start,
                'end_idx': len(path) - 1,
                'distance': current_steep_distance
            })
        
        # Straight line distance
        start_row, start_col = path[0]
        end_row, end_col = path[-1]
        dr = end_row - start_row
        dc = end_col - start_col
        straight_distance = np.sqrt(dr*dr + dc*dc) * self.pixel_size
        
        # Calculate longest steep segment
        longest_steep = max(steep_segments, key=lambda x: x['distance'])['distance'] if steep_segments else 0
        
        return {
            'total_distance': total_distance,
            'straight_distance': straight_distance,
            'distance_ratio': total_distance / straight_distance if straight_distance > 0 else 1,
            'total_ascent': total_ascent,
            'total_descent': total_descent,
            'net_elevation': elevations[-1] - elevations[0],
            'elevations': elevations,
            'slopes': slopes,
            'min_elevation': min(elevations),
            'max_elevation': max(elevations),
            'steep_segments': steep_segments,
            'num_steep_segments': len(steep_segments),
            'longest_steep_segment': longest_steep,
            'total_steep_distance': sum(s['distance'] for s in steep_segments),
            'avg_slope': np.mean(slopes) if slopes else 0,
            'max_slope': max(slopes) if slopes else 0
        }


def visualize_path_with_fatigue(elevation_array: np.ndarray, path: List[Tuple[int, int]], 
                               bounds: Bounds, stats: Dict, pathfinder: SustainedSlopePathfinder,
                               output_file: str = "path_fatigue_result.png"):
    """Visualize path with fatigue analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    
    # Plot 1: Path on elevation map with steep segments highlighted
    im = ax1.imshow(elevation_array, cmap='terrain', origin='upper',
                    extent=[bounds.west, bounds.east, bounds.south, bounds.north])
    
    if path:
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        
        lons = bounds.west + (cols / elevation_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elevation_array.shape[0]) * (bounds.north - bounds.south)
        
        # Plot full path
        ax1.plot(lons, lats, 'k-', linewidth=4, alpha=0.3, label='Full Path')
        
        # Highlight steep segments
        for seg in stats.get('steep_segments', []):
            seg_lons = lons[seg['start_idx']:seg['end_idx']+1]
            seg_lats = lats[seg['start_idx']:seg['end_idx']+1]
            ax1.plot(seg_lons, seg_lats, 'r-', linewidth=3, alpha=0.8)
        
        ax1.plot(lons[0], lats[0], 'go', markersize=10, label='Start')
        ax1.plot(lons[-1], lats[-1], 'ro', markersize=10, label='Goal')
    
    ax1.set_title('Path with Steep Segments (Red)', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Elevation (m)')
    
    # Plot 2: Slope along path
    if path and 'slopes' in stats:
        distances = [0]
        for i in range(1, len(path)):
            dr = path[i][0] - path[i-1][0]
            dc = path[i][1] - path[i-1][1]
            dist = np.sqrt(dr*dr + dc*dc) * pathfinder.pixel_size
            distances.append(distances[-1] + dist)
        
        ax2.plot(distances[1:], stats['slopes'], 'b-', linewidth=2)
        ax2.axhline(y=np.degrees(np.arctan(pathfinder.steep_threshold)), 
                   color='r', linestyle='--', label=f'Steep threshold ({np.degrees(np.arctan(pathfinder.steep_threshold)):.1f}°)')
        ax2.axhline(y=np.degrees(np.arctan(pathfinder.rest_threshold)), 
                   color='g', linestyle='--', label=f'Rest threshold ({np.degrees(np.arctan(pathfinder.rest_threshold)):.1f}°)')
        
        # Shade steep segments
        for seg in stats.get('steep_segments', []):
            start_dist = distances[seg['start_idx']]
            end_dist = distances[seg['end_idx']]
            ax2.axvspan(start_dist, end_dist, alpha=0.3, color='red')
        
        ax2.set_xlabel('Distance along path (m)')
        ax2.set_ylabel('Slope (degrees)')
        ax2.set_title('Slope Profile with Steep Segments', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Elevation profile
    if path and 'elevations' in stats:
        distances = [0]
        for i in range(1, len(path)):
            dr = path[i][0] - path[i-1][0]
            dc = path[i][1] - path[i-1][1]
            dist = np.sqrt(dr*dr + dc*dc) * pathfinder.pixel_size
            distances.append(distances[-1] + dist)
        
        ax3.plot(distances, stats['elevations'], 'b-', linewidth=2)
        ax3.fill_between(distances, stats['elevations'], alpha=0.3)
        
        # Mark steep segments
        for i, seg in enumerate(stats.get('steep_segments', [])):
            start_dist = distances[seg['start_idx']]
            end_dist = distances[seg['end_idx']]
            ax3.axvspan(start_dist, end_dist, alpha=0.2, color='red', 
                       label='Steep segment' if i == 0 else '')
        
        ax3.set_xlabel('Distance along path (m)')
        ax3.set_ylabel('Elevation (m)')
        ax3.set_title('Elevation Profile', fontsize=14)
        ax3.grid(True, alpha=0.3)
        if stats.get('steep_segments'):
            ax3.legend()
    
    # Plot 4: Statistics
    ax4.axis('off')
    
    stats_text = f"Path Statistics:\n"
    stats_text += f"Total Distance: {stats['total_distance']:.1f}m\n"
    stats_text += f"Direct Distance: {stats['straight_distance']:.1f}m\n"
    stats_text += f"Distance Ratio: {stats['distance_ratio']:.2f}\n"
    stats_text += f"Total Ascent: {stats['total_ascent']:.1f}m\n"
    stats_text += f"Total Descent: {stats['total_descent']:.1f}m\n"
    stats_text += f"Average Slope: {stats.get('avg_slope', 0):.1f}°\n"
    stats_text += f"Maximum Slope: {stats.get('max_slope', 0):.1f}°\n"
    stats_text += f"\nSteep Segment Analysis:\n"
    stats_text += f"Number of Steep Segments: {stats.get('num_steep_segments', 0)}\n"
    stats_text += f"Total Steep Distance: {stats.get('total_steep_distance', 0):.1f}m\n"
    stats_text += f"Longest Steep Segment: {stats.get('longest_steep_segment', 0):.1f}m\n"
    stats_text += f"Steep Distance %: {100 * stats.get('total_steep_distance', 0) / stats['total_distance']:.1f}%"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add steep segment details
    if stats.get('steep_segments'):
        seg_text = "\nSteep Segment Details:\n"
        for i, seg in enumerate(stats['steep_segments'][:5]):  # Show first 5
            seg_text += f"  Segment {i+1}: {seg['distance']:.1f}m\n"
        if len(stats['steep_segments']) > 5:
            seg_text += f"  ... and {len(stats['steep_segments'])-5} more"
        
        ax4.text(0.6, 0.9, seg_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Path Analysis with Sustained Slope Fatigue', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def compare_pathfinding_methods(start_lat: float, start_lon: float,
                              goal_lat: float, goal_lon: float,
                              resolution: int = 10,
                              data_dir: str = './elevation_data',
                              output_file: str = 'fatigue_comparison.png'):
    """Compare standard pathfinding with fatigue-aware pathfinding."""
    
    # Setup bounds
    margin = 0.005
    min_lat = min(start_lat, goal_lat) - margin
    max_lat = max(start_lat, goal_lat) + margin
    min_lon = min(start_lon, goal_lon) - margin
    max_lon = max(start_lon, goal_lon) + margin
    bounds = Bounds(north=max_lat, south=min_lat, east=max_lon, west=min_lon)
    
    # Load elevation data
    print("Loading elevation data...")
    elev_lib = TwoLayerElevationLibrary(data_dir, resolution)
    result = elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print(f"Failed to load elevation data")
        return
    
    # Get elevation array
    elev_array, metadata = elev_lib.get_elevation_array(bounds)
    
    # Calculate pixel size
    lat_center = (bounds.north + bounds.south) / 2
    lon_deg_to_m = 111000 * np.cos(np.radians(lat_center))
    pixel_size = abs(metadata['transform']['a']) * lon_deg_to_m
    
    # Convert coordinates
    def latlon_to_pixel(lat, lon):
        col = int((lon - bounds.west) / (bounds.east - bounds.west) * elev_array.shape[1])
        row = int((bounds.north - lat) / (bounds.north - bounds.south) * elev_array.shape[0])
        return min(max(row, 0), elev_array.shape[0]-1), min(max(col, 0), elev_array.shape[1]-1)
    
    start_row, start_col = latlon_to_pixel(start_lat, start_lon)
    goal_row, goal_col = latlon_to_pixel(goal_lat, goal_lon)
    
    # Find path WITHOUT fatigue penalty
    print("\nFinding standard path (no fatigue penalty)...")
    pathfinder_standard = SustainedSlopePathfinder(
        elev_array, pixel_size,
        elevation_weight=0.5,
        elevation_exponent=2.0,
        distance_weight=0.1,
        sustained_slope_weight=0.0,  # No fatigue penalty
        steep_threshold_degrees=15.0,
        rest_threshold_degrees=8.0
    )
    
    path_standard = pathfinder_standard.find_path(start_row, start_col, goal_row, goal_col)
    if path_standard:
        stats_standard = pathfinder_standard.calculate_path_stats_with_fatigue(path_standard)
        print(f"Standard path: {stats_standard['total_distance']:.0f}m, "
              f"longest steep: {stats_standard['longest_steep_segment']:.0f}m")
    
    # Find path WITH fatigue penalty
    print("\nFinding fatigue-aware path...")
    pathfinder_fatigue = SustainedSlopePathfinder(
        elev_array, pixel_size,
        elevation_weight=0.5,
        elevation_exponent=2.0,
        distance_weight=0.1,
        sustained_slope_weight=2.0,  # Strong fatigue penalty
        steep_threshold_degrees=15.0,
        rest_threshold_degrees=8.0,
        fatigue_distance=50.0,  # Fatigue after 50m of steep terrain
        fatigue_exponent=2.5
    )
    
    path_fatigue = pathfinder_fatigue.find_path(start_row, start_col, goal_row, goal_col)
    if path_fatigue:
        stats_fatigue = pathfinder_fatigue.calculate_path_stats_with_fatigue(path_fatigue)
        print(f"Fatigue-aware path: {stats_fatigue['total_distance']:.0f}m, "
              f"longest steep: {stats_fatigue['longest_steep_segment']:.0f}m")
    
    # Visualize comparison
    if path_standard and path_fatigue:
        visualize_path_with_fatigue(elev_array, path_fatigue, bounds, stats_fatigue, 
                                   pathfinder_fatigue, output_file)
        print(f"\nVisualization saved to: {output_file}")
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        for ax, path, stats, title in [(ax1, path_standard, stats_standard, "Standard Path"),
                                       (ax2, path_fatigue, stats_fatigue, "Fatigue-Aware Path")]:
            ax.imshow(elev_array, cmap='terrain', origin='upper',
                     extent=[bounds.west, bounds.east, bounds.south, bounds.north])
            
            path_array = np.array(path)
            rows = path_array[:, 0]
            cols = path_array[:, 1]
            lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
            lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
            
            # Plot path with steep segments
            ax.plot(lons, lats, 'k-', linewidth=4, alpha=0.3)
            for seg in stats.get('steep_segments', []):
                seg_lons = lons[seg['start_idx']:seg['end_idx']+1]
                seg_lats = lats[seg['start_idx']:seg['end_idx']+1]
                ax.plot(seg_lons, seg_lats, 'r-', linewidth=3, alpha=0.8)
            
            ax.plot(lons[0], lats[0], 'go', markersize=10)
            ax.plot(lons[-1], lats[-1], 'ro', markersize=10)
            
            ax.set_title(f"{title}\nDistance: {stats['total_distance']:.0f}m, "
                        f"Longest steep: {stats['longest_steep_segment']:.0f}m")
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Pathfinding Comparison: Standard vs Fatigue-Aware', fontsize=16)
        plt.tight_layout()
        plt.savefig('path_comparison.png', dpi=150, bbox_inches='tight')
        print("Comparison saved to: path_comparison.png")


if __name__ == "__main__":
    # Compare methods on the requested coordinates
    compare_pathfinding_methods(40.6599, -111.5662, 40.6483, -111.5648)