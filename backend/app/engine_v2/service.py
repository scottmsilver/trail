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
import os
import threading
from typing import List, Optional, Tuple

from rasterio.transform import Affine, from_bounds

from app.engine_v2.elevation import Bounds, TwoLayerElevationLibrary
from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary
from app.engine_v2.path_layer import PathLayer, PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder
from app.engine_v2.scoring import dominant_factor, rasterize_segment, score_polyline_cells
from app.models.eval import ScoredPath, ScoredSegment
from app.models.route import Coordinate

logger = logging.getLogger(__name__)


class TrailFinderServiceV2:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        resolution: int = 10,
        buffer: float = 0.02,
        max_distance_km: float = 50.0,
        elevation_lib=None,
        path_layer=None,
    ):
        self.buffer = buffer
        self.resolution = resolution
        self.max_distance_km = max_distance_km
        # Cache dirs are shared, derived-data artifacts (DEM tiles, OSM path
        # grids) -- identical for any caller, so they resolve from an
        # explicit constructor arg, then an env var, then the legacy
        # CWD-relative default. This lets any client/eval/CI point at one
        # shared cache instead of cold-starting a private copy per CWD.
        self._data_dir = data_dir if data_dir is not None else os.environ.get("TRAIL_V2_DEM_DIR", "dem_data_v2")
        self._cache_dir = (
            cache_dir if cache_dir is not None else os.environ.get("TRAIL_V2_PATH_CACHE_DIR", "path_cache_v2")
        )
        logger.info("v2 engine cache locations: data_dir=%s cache_dir=%s", self._data_dir, self._cache_dir)
        # Bound open file descriptors by default: the plain library grows
        # _open_datasets unboundedly on a long-lived module-level service.
        self.elevation_lib = elevation_lib or FDManagedElevationLibrary(
            TwoLayerElevationLibrary(data_dir=self._data_dir, resolution=resolution), max_open_files=50
        )
        self.path_layer = path_layer or PathLayer(cache_dir=self._cache_dir)
        # Shared library + close_all() clears all open datasets, so data
        # loading must be serialized per service instance.
        self._data_lock = threading.Lock()

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

    def _load_pathfinder(self, bounds, options):
        """Load elevation + terrain for ``bounds`` and build a configured
        pathfinder. Shared by routing and eval scoring so both observe identical
        cost inputs. Returns ``(pathfinder, warnings)``.

        Shared library + close_all() clears all open datasets, so loading is
        serialized per service instance; the lock also serializes PathLayer's
        non-atomic npy cache write.
        """
        warnings: List[str] = []
        with self._data_lock:
            self.elevation_lib.load_area(bounds)
            elevation, meta = self.elevation_lib.get_elevation_array(bounds)
            # Metadata carries the transform as a coefficient dict {a..f};
            # injected fakes may pass a real Affine. Pathfinder needs an Affine.
            raw_transform = meta.get("transform") if isinstance(meta, dict) else None
            if isinstance(raw_transform, Affine):
                transform = raw_transform
            elif isinstance(raw_transform, dict) and all(k in raw_transform for k in "abcdef"):
                transform = Affine(
                    raw_transform["a"],
                    raw_transform["b"],
                    raw_transform["c"],
                    raw_transform["d"],
                    raw_transform["e"],
                    raw_transform["f"],
                )
            else:
                transform = from_bounds(
                    bounds.west, bounds.south, bounds.east, bounds.north, elevation.shape[1], elevation.shape[0]
                )
            if hasattr(self.elevation_lib, "close_all"):
                self.elevation_lib.close_all()

            terrain_grid = self.path_layer.get_grid(bounds, elevation.shape, transform)
        if not (terrain_grid != PathType.UNKNOWN).any():
            warnings.append("OSM data unavailable — terrain-only routing")

        return self._make_pathfinder(elevation, transform, terrain_grid, options), warnings

    async def score_path(self, path: List[Coordinate], options: dict) -> ScoredPath:
        """Score an arbitrary polyline with the engine's cost function (async)."""
        return await asyncio.to_thread(self._score_path_sync, path, options)

    def _score_path_sync(self, path: List[Coordinate], options: dict) -> ScoredPath:
        options = options or {}
        if len(path) < 2:
            raise ValueError("path needs at least two points")
        # Bounding box covering the whole polyline (plus the usual buffer).
        sw = Coordinate(lat=min(p.lat for p in path), lon=min(p.lon for p in path))
        ne = Coordinate(lat=max(p.lat for p in path), lon=max(p.lon for p in path))
        bounds = self.calculate_bounding_box(sw, ne, options.get("buffer"))
        pf, _warnings = self._load_pathfinder(bounds, options)

        verts = [pf.lat_lon_to_grid(p.lat, p.lon) for p in path]
        # Deviation penalty is measured against the whole-path straight line —
        # the same reference the engine's optimal path is scored against.
        sld = pf.resolution * math.sqrt((verts[0][0] - verts[-1][0]) ** 2 + (verts[0][1] - verts[-1][1]) ** 2)
        segments: List[ScoredSegment] = []
        total = 0.0
        dist = 0.0
        egain = 0.0
        steep = 0.0
        for (a, b), pa, pb in zip(zip(verts, verts[1:]), path, path[1:]):
            cells = rasterize_segment(a[0], a[1], b[0], b[1])
            seg = score_polyline_cells(pf, cells, sld, running_distance=dist, running_steep=steep)
            segments.append(
                ScoredSegment(
                    **{
                        "from": pa,
                        "to": pb,
                        "cost": seg["total"],
                        "factors": seg["factors"],
                        "dominantFactor": dominant_factor(seg["factors"]),
                    }
                )
            )
            total += seg["total"]
            dist += seg["distance"]
            egain += seg["egain"]
            steep = seg["steep"]
        return ScoredPath(
            path=path,
            snapped=False,
            totalCost=total,
            distanceM=dist,
            elevationGainM=egain,
            segments=segments,
        )

    def _find_route_sync(self, start, end, options):
        bounds = self.calculate_bounding_box(start, end, options.get("buffer"))
        pf, warnings = self._load_pathfinder(bounds, options)
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
        self._augment_client_stats(raw_path, stats)
        path = [Coordinate(lat=lat, lon=lon) for (lat, lon, _elev) in raw_path]
        return path, stats

    def _augment_client_stats(self, raw_path, stats: dict) -> dict:
        """Add v1-compatible stat keys so the frontend and GPX generator — which
        read the v1 schema (distance_km, estimated_time_min, difficulty,
        path_with_slopes) — render v2 routes correctly instead of showing
        NaN/blank. The native v2 keys (distance_m, terrain_breakdown, ...) are
        left intact."""
        distance_km = round(stats.get("distance_m", 0.0) / 1000.0, 2)

        path_with_slopes = []
        max_slope = 0.0
        for i, (lat, lon, elev) in enumerate(raw_path):
            slope_deg = 0.0
            if i > 0:
                p_lat, p_lon, p_elev = raw_path[i - 1]
                horiz_m = self._haversine_km(Coordinate(lat=p_lat, lon=p_lon), Coordinate(lat=lat, lon=lon)) * 1000.0
                if horiz_m > 0:
                    slope_deg = round(math.degrees(math.atan2(elev - p_elev, horiz_m)), 1)
            max_slope = max(max_slope, abs(slope_deg))
            path_with_slopes.append({"lat": lat, "lon": lon, "elevation": round(float(elev), 2), "slope": slope_deg})

        stats["distance_km"] = distance_km
        stats["estimated_time_min"] = int(distance_km * 15)  # ~4 km/h, matches v1
        stats["max_slope"] = round(max_slope, 1)
        stats["difficulty"] = self._difficulty(distance_km, max_slope)
        stats["waypoints"] = len(raw_path)
        stats["path_with_slopes"] = path_with_slopes
        return stats

    @staticmethod
    def _difficulty(distance_km: float, max_slope: float) -> str:
        """Distance+slope difficulty, matching v1's _estimate_difficulty_with_slope."""
        base = 1 if distance_km < 3 else 2 if distance_km < 8 else 3
        slope = 1 if max_slope < 10 else 2 if max_slope < 20 else 3
        return {1: "easy", 2: "moderate", 3: "hard"}[max(base, slope)]
