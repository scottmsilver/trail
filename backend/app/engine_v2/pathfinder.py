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
import os
import time
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


class TerrainNode:
    """Node in the pathfinding graph with terrain awareness.

    Plain __slots__ class (not a dataclass) with f_cost precomputed at
    construction: the A* heap comparison used to invoke the f_cost @property
    twice per comparison, and heapq does O(log n) comparisons per push/pop over
    hundreds of thousands of nodes. __slots__ also speeds attribute access and
    cuts per-node memory.
    """

    __slots__ = (
        "row",
        "col",
        "g_cost",
        "h_cost",
        "f_cost",
        "parent",
        "elevation",
        "terrain_type",
        "consecutive_steep_distance",
        "consecutive_steep_climb",
    )

    def __init__(
        self,
        row,
        col,
        g_cost,
        h_cost,
        parent=None,
        elevation=0,
        terrain_type=PathType.UNKNOWN,
        consecutive_steep_distance=0,
        consecutive_steep_climb=0.0,
    ):
        self.row = row
        self.col = col
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
        self.elevation = elevation
        self.terrain_type = terrain_type
        self.consecutive_steep_distance = consecutive_steep_distance
        # Vertical meters climbed on the current unbroken steep run (resets at a
        # bench). The extent-aware passability gate blocks when this exceeds the
        # expertise-level scramble budget. Unused when scramble_budget_m is None.
        self.consecutive_steep_climb = consecutive_steep_climb

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
        # Extent-aware passability (opt-in). When scramble_budget_m is set, the
        # memoryless slope>max_slope gate is replaced by a stateful one: block a
        # move only when the *continuous* vertical climb on steep ground
        # (slope > extent_threshold_degrees, reset at any bench) exceeds the
        # budget. The budget is an expertise level — ~1.5 m casual .. ~15 m
        # alpinist (see docs/terrain-passability-extent-aware.md). None => the
        # classic memoryless gate (unchanged golden/native behavior).
        self.scramble_budget_m = None
        self.extent_threshold_degrees = 45.0
        # Optional per-cell cost multiplier grid (same shape as elevation).
        # Used by the alternative-routes search to tax a corridor around
        # already-found routes so the next search prefers a different line.
        # None => no overlay (unchanged behavior).
        self.cost_overlay = None
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
        return_breakdown: bool = False,
    ):
        """
        Calculate cost of moving between two cells considering terrain and elevation.

        Returns ``(cost, new_steep_distance)`` normally. When ``return_breakdown``
        is True, returns a dict ``{"cost", "factors", "new_steep_distance"}`` where
        ``factors`` (base/terrain/slope/sustained/deviation) sum to ``cost`` — used
        by the eval scorer to attribute why a path is expensive.
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

        # Alternative-routes corridor tax (see find_path hot loop)
        if self.cost_overlay is not None:
            terrain_multiplier *= float(self.cost_overlay[to_row, to_col])

        # Calculate slope
        elevation_change = abs(elev_to - elev_from)
        slope_radians = math.atan2(elevation_change, horiz_distance)
        slope_degrees = math.degrees(slope_radians)

        # Check max slope constraint
        if slope_degrees > self.max_slope_degrees:
            if return_breakdown:
                inf = float("inf")
                return {
                    "cost": inf,
                    "factors": {"base": base_cost, "terrain": 0.0, "slope": inf, "sustained": 0.0, "deviation": 0.0},
                    "new_steep_distance": from_steep_distance,
                }
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

        if return_breakdown:
            # Attribute the multiplicative cost additively so components sum to
            # total_cost: base carries distance, terrain the multiplier excess,
            # each penalty is (base*terrain)*penalty.
            bt = base_cost * terrain_multiplier
            return {
                "cost": total_cost,
                "factors": {
                    "base": base_cost,
                    "terrain": bt - base_cost,
                    "slope": bt * elevation_penalty,
                    "sustained": bt * sustained_penalty,
                    "deviation": bt * deviation_penalty,
                },
                "new_steep_distance": new_steep_distance,
            }

        return total_cost, new_steep_distance

    def find_path(
        self, start_lat: float, start_lon: float, end_lat: float, end_lon: float
    ) -> Optional[Tuple[List[Tuple[float, float, float]], dict]]:
        """
        Find optimal path from start to end considering terrain and elevation.
        Returns (path, stats) where path is a list of (lat, lon, elevation) tuples.
        """
        # Fast path: the native C kernel is a byte-identical port (same paths and
        # stats, verified in benchmarks/v2_tuning). Use it when available; fall
        # back to the pure-Python loop below on any failure or when disabled via
        # TRAIL_V2_DISABLE_NATIVE=1.
        if (
            self.scramble_budget_m is None
            and self.cost_overlay is None
            and os.environ.get("TRAIL_V2_DISABLE_NATIVE") not in ("1", "true", "True")
        ):
            try:
                from app.engine_v2 import pathfinder_native

                if pathfinder_native.available():
                    return pathfinder_native.find_path_native(self, start_lat, start_lon, end_lat, end_lon)
            except Exception:
                logger.warning("native A* kernel failed; falling back to pure Python", exc_info=True)

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

        # --- hot loop: everything below is an inlined, locals-bound version of
        # get_neighbors() + calculate_move_cost() + heuristic(). The standalone
        # methods are kept intact for the public API / tests; this duplicates
        # their exact arithmetic to avoid per-edge Python call + attribute
        # lookup + scalar-numpy overhead. Any change here must keep byte-for-byte
        # identical results (see benchmarks/v2_tuning). ---
        rows = self.rows
        cols = self.cols
        res = self.resolution
        hw = self.heuristic_weight
        ew = self.elevation_weight
        eexp = self.elevation_exponent
        max_slope = self.max_slope_degrees
        budget = self.scramble_budget_m  # None => classic memoryless slope gate
        ext_thr = self.extent_threshold_degrees
        steep_thr = self.steep_threshold
        fat_dist = self.fatigue_distance
        fat_exp = self.fatigue_exponent
        sust_w = self.sustained_slope_weight
        tcost_get = self.terrain_costs.get
        sld = max(straight_line_distance, 1.0)
        degrees = math.degrees
        atan2 = math.atan2
        sqrt = math.sqrt
        push = heapq.heappush
        pop = heapq.heappop

        # Native-Python views: nested-list indexing returns plain float/int
        # (no np.float64 boxing) and the values are bit-identical to arr[r, c].
        elev = self.elevation.tolist()
        terr = self.terrain_types.tolist()
        ovl = self.cost_overlay.tolist() if self.cost_overlay is not None else None

        OBSTACLE = int(PathType.OBSTACLE)
        TRAIL = int(PathType.TRAIL)
        good_after_trail = frozenset((int(PathType.TRAIL), int(PathType.PATH), int(PathType.NATURAL)))

        # 8-connected offsets in the SAME order get_neighbors yields them
        # (dr: -1,0,1 outer, dc: -1,0,1 inner, skipping 0,0), with the per-step
        # horizontal distance precomputed (res * sqrt(dr^2 + dc^2)).
        offsets = [
            (dr, dc, res * sqrt(dr * dr + dc * dc))
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]

        start_node = TerrainNode(
            row=start_row,
            col=start_col,
            g_cost=0,
            h_cost=hw * (res * sqrt((start_row - end_row) ** 2 + (start_col - end_col) ** 2)),
            elevation=elev[start_row][start_col],
            terrain_type=terr[start_row][start_col],
        )

        # closed_set / best_g_cost are keyed by a flat integer (row*cols+col)
        # rather than a (row, col) tuple: int hashing/equality is cheaper than
        # tuple, and these structures see hundreds of thousands of ops.
        open_set = [start_node]
        closed_set = set()
        best_g_cost = {start_row * cols + start_col: 0}

        nodes_explored = 0
        start_time = time.time()

        while open_set:
            current = pop(open_set)
            cr = current.row
            cc = current.col
            ckey = cr * cols + cc

            if ckey in closed_set:
                continue

            nodes_explored += 1

            if cr == end_row and cc == end_col:
                logger.debug(f"Path found! Explored {nodes_explored} nodes in {time.time() - start_time:.2f}s")
                return self.reconstruct_path(current, nodes_explored, time.time() - start_time)

            closed_set.add(ckey)

            elev_from = elev[cr][cc]
            terrain_from = terr[cr][cc]
            g_from = current.g_cost
            steep_from = current.consecutive_steep_distance
            climb_from = current.consecutive_steep_climb
            deviation_ratio = g_from / sld
            deviation_penalty = 0.1 * max(0, deviation_ratio - 1.5) ** 2
            from_is_trail = terrain_from == TRAIL

            for dr, dc, horiz_distance in offsets:
                next_row = cr + dr
                next_col = cc + dc
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                terrain_to = terr[next_row][next_col]
                if terrain_to == OBSTACLE:
                    continue
                nkey = next_row * cols + next_col
                if nkey in closed_set:
                    continue

                elev_to = elev[next_row][next_col]

                # slope
                rise = elev_to - elev_from
                elevation_change = rise if rise >= 0 else -rise
                slope_degrees = degrees(atan2(elevation_change, horiz_distance))
                if budget is None:
                    if slope_degrees > max_slope:
                        continue  # impassable (classic memoryless gate)
                    new_climb = 0.0
                else:
                    # Extent-aware gate: accumulate continuous vertical climb on
                    # steep ground, reset at a bench, block when it exceeds the
                    # expertise-level scramble budget. A short scramble-able step
                    # passes; a sustained wall does not.
                    if slope_degrees > ext_thr:
                        new_climb = climb_from + (rise if rise > 0 else 0.0)
                    else:
                        new_climb = 0.0
                    if new_climb > budget:
                        continue  # impassable at this expertise level

                terrain_multiplier = tcost_get(terrain_to, 1.0)
                if from_is_trail and terrain_to not in good_after_trail:
                    terrain_multiplier *= 1.5
                if ovl is not None:
                    terrain_multiplier *= ovl[next_row][next_col]

                elevation_penalty = ew * (slope_degrees / 10.0) ** eexp
                if elev_to > elev_from:
                    elevation_penalty *= 1.5

                if slope_degrees > steep_thr:
                    new_steep_distance = steep_from + horiz_distance
                    fatigue_factor = (new_steep_distance / fat_dist) ** fat_exp
                    sustained_penalty = sust_w * fatigue_factor * horiz_distance
                else:
                    new_steep_distance = max(0, steep_from - horiz_distance * 0.5)
                    sustained_penalty = 0

                move_cost = (
                    horiz_distance
                    * terrain_multiplier
                    * (1 + elevation_penalty + sustained_penalty + deviation_penalty)
                )

                new_g_cost = g_from + move_cost

                prev = best_g_cost.get(nkey)
                if prev is not None and new_g_cost >= prev:
                    continue
                best_g_cost[nkey] = new_g_cost

                h = hw * (res * sqrt((next_row - end_row) ** 2 + (next_col - end_col) ** 2))
                push(
                    open_set,
                    TerrainNode(
                        row=next_row,
                        col=next_col,
                        g_cost=new_g_cost,
                        h_cost=h,
                        parent=current,
                        elevation=elev_to,
                        terrain_type=terrain_to,
                        consecutive_steep_distance=new_steep_distance,
                        consecutive_steep_climb=new_climb,
                    ),
                )

        logger.debug(f"No path found after exploring {nodes_explored} nodes")
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
