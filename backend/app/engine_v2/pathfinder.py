# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""
Terrain-aware A* pathfinder for engine v2.
Ported from backend3/elevation_pathfinder_terrain.py with: direct array inputs,
logging instead of prints, (path, stats) return, new PathType vocabulary, and
weighted A* via heuristic_weight.
"""
import heapq
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.engine_v2.path_layer import PathType, get_path_type_name

logger = logging.getLogger(__name__)

DEFAULT_TERRAIN_COSTS = {
    PathType.UNKNOWN: 0.5,
    PathType.OBSTACLE: 10000.0,
    PathType.TRAIL: 0.2,
    PathType.PATH: 0.3,
    PathType.FOOTWAY: 0.6,
    PathType.RESIDENTIAL: 0.85,
    PathType.NATURAL: 0.4,
}


@dataclass
class TerrainNode:
    """Node in the pathfinding graph with terrain awareness"""

    row: int
    col: int
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Optional["TerrainNode"] = None
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

    def __init__(
        self,
        elevation: np.ndarray,
        transform,
        terrain_types: np.ndarray,
        resolution: float = 10.0,
        heuristic_weight: float = 1.0,
    ):
        if elevation.shape != terrain_types.shape:
            raise ValueError(f"elevation {elevation.shape} != terrain {terrain_types.shape}")
        self.elevation = elevation
        self.transform = transform
        self.terrain_types = terrain_types
        self.resolution = resolution
        self.heuristic_weight = heuristic_weight
        self.rows, self.cols = elevation.shape

        # Cost parameters (same defaults as backend3 original)
        self.elevation_weight = 1.0
        self.elevation_exponent = 2.0
        self.obstacle_cost = 10000.0
        self.max_slope_degrees = 45.0
        self.terrain_costs = dict(DEFAULT_TERRAIN_COSTS)
        self.steep_threshold = 15.0
        self.fatigue_distance = 100.0
        self.fatigue_exponent = 2.0
        self.sustained_slope_weight = 0.5

    def _euclidean(self, row, col, goal_row, goal_col) -> float:
        return self.resolution * math.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2)

    def heuristic(self, row, col, goal_row, goal_col) -> float:
        return self.heuristic_weight * self._euclidean(row, col, goal_row, goal_col)

    def set_parameters(self, **kwargs):
        """Update pathfinder parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter {key}")

    def lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to grid coordinates"""
        col, row = ~self.transform * (lon, lat)
        return int(row), int(col)

    def grid_to_lat_lon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon"""
        lon, lat = self.transform * (col, row)
        return lat, lon

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

    def calculate_move_cost(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        straight_line_distance: float,
        current_distance: float,
        from_steep_distance: float = 0,
    ) -> Tuple[float, float]:
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
        horiz_distance = self.resolution * math.sqrt(row_diff**2 + col_diff**2)

        # Base movement cost (distance)
        base_cost = horiz_distance

        # Terrain cost multiplier
        terrain_multiplier = self.terrain_costs.get(terrain_to, 1.0)

        # Add transition penalty when leaving a good path
        if terrain_from == PathType.TRAIL and terrain_to not in (PathType.TRAIL, PathType.PATH, PathType.NATURAL):
            terrain_multiplier *= 1.5  # Penalty for leaving trails

        # Calculate slope
        elevation_change = abs(elev_to - elev_from)
        slope_radians = math.atan2(elevation_change, horiz_distance)
        slope_degrees = math.degrees(slope_radians)

        # Check max slope constraint
        if slope_degrees > self.max_slope_degrees:
            return float("inf"), from_steep_distance  # Impassable

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

    def find_path(
        self, start_lat: float, start_lon: float, end_lat: float, end_lon: float
    ) -> Optional[Tuple[List[Tuple[float, float, float]], dict]]:
        """
        Find optimal path from start to end considering terrain and elevation.
        Returns (path, stats) where path is a list of (lat, lon, elevation) tuples.
        """
        # Convert to grid coordinates
        start_row, start_col = self.lat_lon_to_grid(start_lat, start_lon)
        end_row, end_col = self.lat_lon_to_grid(end_lat, end_lon)

        # Validate coordinates
        if not (0 <= start_row < self.rows and 0 <= start_col < self.cols):
            logger.debug(f"Start point ({start_lat}, {start_lon}) is outside bounds")
            return None
        if not (0 <= end_row < self.rows and 0 <= end_col < self.cols):
            logger.debug(f"End point ({end_lat}, {end_lon}) is outside bounds")
            return None

        # Check if start/end are on obstacles
        if self.terrain_types[start_row, start_col] == PathType.OBSTACLE:
            logger.debug("Warning: Start point is on an obstacle!")
        if self.terrain_types[end_row, end_col] == PathType.OBSTACLE:
            logger.debug("Warning: End point is on an obstacle!")

        # Calculate straight-line distance (unweighted, used for deviation penalty)
        straight_line_distance = self._euclidean(start_row, start_col, end_row, end_col)

        logger.debug(f"\nPathfinding from ({start_lat:.6f}, {start_lon:.6f}) to ({end_lat:.6f}, {end_lon:.6f})")
        logger.debug(f"Grid: ({start_row}, {start_col}) to ({end_row}, {end_col})")
        logger.debug(f"Start terrain: {get_path_type_name(self.terrain_types[start_row, start_col])}")
        logger.debug(f"End terrain: {get_path_type_name(self.terrain_types[end_row, end_col])}")
        logger.debug(f"Straight-line distance: {straight_line_distance:.1f}m")

        # Initialize A* algorithm
        start_node = TerrainNode(
            row=start_row,
            col=start_col,
            g_cost=0,
            h_cost=self.heuristic(start_row, start_col, end_row, end_col),
            elevation=self.elevation[start_row, start_col],
            terrain_type=self.terrain_types[start_row, start_col],
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
                logger.debug(f"  Explored {nodes_explored} nodes...")

            # Check if we reached the goal
            if current.row == end_row and current.col == end_col:
                logger.debug(f"\nPath found! Explored {nodes_explored} nodes in {time.time() - start_time:.2f}s")
                return self.reconstruct_path(current, nodes_explored, time.time() - start_time)

            closed_set.add((current.row, current.col))

            # Explore neighbors
            for next_row, next_col in self.get_neighbors(current.row, current.col):
                if (next_row, next_col) in closed_set:
                    continue

                # Calculate cost to move to this neighbor
                move_cost, new_steep_distance = self.calculate_move_cost(
                    current.row,
                    current.col,
                    next_row,
                    next_col,
                    straight_line_distance,
                    current.g_cost,
                    current.consecutive_steep_distance,
                )

                # Skip if move is impossible (infinite cost)
                if move_cost == float("inf"):
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
                    consecutive_steep_distance=new_steep_distance,
                )

                heapq.heappush(open_set, new_node)

        logger.debug(f"\nNo path found after exploring {nodes_explored} nodes")
        return None

    def reconstruct_path(self, end_node, nodes_explored, elapsed_s):
        path = []
        current = end_node
        total_distance = 0.0
        total_elevation_gain = 0.0
        terrain_counts: Dict[int, int] = {}

        while current:
            lat, lon = self.grid_to_lat_lon(current.row, current.col)
            path.append((lat, lon, float(current.elevation)))
            terrain_counts[current.terrain_type] = terrain_counts.get(current.terrain_type, 0) + 1
            if current.parent:
                row_diff = current.row - current.parent.row
                col_diff = current.col - current.parent.col
                total_distance += self.resolution * math.sqrt(row_diff**2 + col_diff**2)
                elev_diff = current.elevation - current.parent.elevation
                if elev_diff > 0:
                    total_elevation_gain += elev_diff
            current = current.parent

        path.reverse()
        total_points = max(sum(terrain_counts.values()), 1)
        stats = {
            "distance_m": round(total_distance, 1),
            "elevation_gain_m": round(float(total_elevation_gain), 1),
            "nodes_explored": nodes_explored,
            "time_s": round(elapsed_s, 3),
            "terrain_breakdown": {get_path_type_name(t): round(c / total_points, 3) for t, c in terrain_counts.items()},
        }
        return path, stats
