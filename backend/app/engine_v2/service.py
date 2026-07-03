# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""TrailFinderServiceV2: v1-compatible find_route contract over the v2 engine."""
import asyncio
import logging
import math
from typing import List, Optional, Tuple

from app.engine_v2.elevation import Bounds, TwoLayerElevationLibrary
from app.engine_v2.path_layer import PathLayer, PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder
from app.models.route import Coordinate
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


class TrailFinderServiceV2:
    def __init__(
        self,
        data_dir: str = "dem_data_v2",
        cache_dir: str = "path_cache_v2",
        resolution: int = 10,
        buffer: float = 0.02,
        max_distance_km: float = 50.0,
        elevation_lib=None,
        path_layer=None,
    ):
        self.buffer = buffer
        self.resolution = resolution
        self.max_distance_km = max_distance_km
        self.elevation_lib = elevation_lib or TwoLayerElevationLibrary(data_dir=data_dir, resolution=resolution)
        self.path_layer = path_layer or PathLayer(cache_dir=cache_dir)

    # --- validation (same semantics as v1) -------------------------------
    @staticmethod
    def _haversine_km(a: Coordinate, b: Coordinate) -> float:
        r = 6371.0
        dlat = math.radians(b.lat - a.lat)
        dlon = math.radians(b.lon - a.lon)
        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(a.lat)) * math.cos(math.radians(b.lat)) * math.sin(dlon / 2) ** 2
        )
        return 2 * r * math.asin(math.sqrt(h))

    def validate_route_request(self, start: Coordinate, end: Coordinate) -> bool:
        if start.lat == end.lat and start.lon == end.lon:
            return False
        return self._haversine_km(start, end) <= self.max_distance_km

    def calculate_bounding_box(self, start: Coordinate, end: Coordinate, buffer: Optional[float] = None) -> Bounds:
        b = self.buffer if buffer is None else buffer
        return Bounds(
            south=min(start.lat, end.lat) - b,
            north=max(start.lat, end.lat) + b,
            west=min(start.lon, end.lon) - b,
            east=max(start.lon, end.lon) + b,
        )

    # --- pathfinder construction (seam for tests) -------------------------
    # Profile presets translated from v1's PathPreferencePresets
    # (urban_walker / trail_seeker / flexible_hiker path_costs tables).
    _PROFILE_COSTS = {
        "easy": {  # urban_walker: prefer built paths, avoid off-path
            PathType.TRAIL: 0.25,
            PathType.PATH: 0.3,
            PathType.FOOTWAY: 0.4,
            PathType.RESIDENTIAL: 0.7,
            PathType.NATURAL: 0.3,
            PathType.UNKNOWN: 1.5,
        },
        "experienced": {  # trail_seeker: strong natural preference
            PathType.TRAIL: 0.15,
            PathType.PATH: 0.2,
            PathType.FOOTWAY: 0.7,
            PathType.RESIDENTIAL: 0.9,
            PathType.NATURAL: 0.3,
            PathType.UNKNOWN: 0.4,
        },
        "trail_runner": {  # flexible_hiker: mild preference, off-path ok
            PathType.TRAIL: 0.5,
            PathType.PATH: 0.6,
            PathType.FOOTWAY: 0.6,
            PathType.RESIDENTIAL: 0.8,
            PathType.NATURAL: 0.8,
            PathType.UNKNOWN: 1.0,
        },
        "accessibility": {  # urban_walker, stricter off-path avoidance
            PathType.TRAIL: 0.6,
            PathType.PATH: 0.4,
            PathType.FOOTWAY: 0.2,
            PathType.RESIDENTIAL: 0.6,
            PathType.NATURAL: 0.8,
            PathType.UNKNOWN: 3.0,
        },
    }  # "default" and unknown profiles keep DEFAULT_TERRAIN_COSTS

    def _make_pathfinder(self, elevation, transform, terrain_grid, options: dict) -> TerrainAwarePathfinder:
        pf = TerrainAwarePathfinder(
            elevation,
            transform,
            terrain_grid,
            resolution=self.resolution,
            heuristic_weight=options.get("heuristicWeight") or 1.0,
        )

        profile_costs = self._PROFILE_COSTS.get(options.get("userProfile"))
        if profile_costs:
            pf.terrain_costs.update(profile_costs)

        if options.get("maxSlope") is not None:
            pf.set_parameters(max_slope_degrees=options["maxSlope"])
        if options.get("gradientPreference") is not None:
            pf.set_parameters(elevation_weight=options["gradientPreference"])

        t = options.get("trailPreference") or 1.0
        if t != 1.0:
            for pt in (PathType.TRAIL, PathType.PATH, PathType.NATURAL, PathType.UNKNOWN):
                pf.terrain_costs[pt] /= t
            for pt in (PathType.FOOTWAY, PathType.RESIDENTIAL):
                pf.terrain_costs[pt] *= t

        custom = options.get("customPathCosts") or {}
        mapping = {
            "trail": PathType.TRAIL,
            "path": PathType.PATH,
            "footway": PathType.FOOTWAY,
            "residential": PathType.RESIDENTIAL,
            "off_path": PathType.UNKNOWN,
        }
        for key, ptype in mapping.items():
            if custom.get(key) is not None:
                pf.terrain_costs[ptype] = custom[key]
        return pf

    # --- main entry --------------------------------------------------------
    async def find_route(self, start: Coordinate, end: Coordinate, options: dict) -> Tuple[List[Coordinate], dict]:
        options = options or {}
        if not self.validate_route_request(start, end):
            return [], {"error": "Invalid route request: coordinates too far apart or identical"}
        try:
            return await asyncio.to_thread(self._find_route_sync, start, end, options)
        except Exception as e:
            logger.exception("v2 route failed")
            return [], {"error": f"v2 engine error: {e}"}

    def _find_route_sync(self, start, end, options):
        warnings: List[str] = []
        bounds = self.calculate_bounding_box(start, end, options.get("buffer"))

        self.elevation_lib.load_area(bounds)
        elevation, meta = self.elevation_lib.get_elevation_array(bounds)
        transform = meta.get("transform") if isinstance(meta, dict) else None
        if transform is None:
            transform = from_bounds(
                bounds.west, bounds.south, bounds.east, bounds.north, elevation.shape[1], elevation.shape[0]
            )
        if hasattr(self.elevation_lib, "close_all"):
            self.elevation_lib.close_all()

        terrain_grid = self.path_layer.get_grid(bounds, elevation.shape, transform)
        if not (terrain_grid != PathType.UNKNOWN).any():
            warnings.append("OSM data unavailable — terrain-only routing")

        pf = self._make_pathfinder(elevation, transform, terrain_grid, options)
        result = pf.find_path(start.lat, start.lon, end.lat, end.lon)
        if result is None:
            stats = {"error": "No route found", "engine": "v2"}
            if warnings:
                stats["warnings"] = warnings
            return [], stats

        raw_path, stats = result
        stats["engine"] = "v2"
        if warnings:
            stats["warnings"] = warnings
        path = [Coordinate(lat=lat, lon=lon) for (lat, lon, _elev) in raw_path]
        return path, stats
