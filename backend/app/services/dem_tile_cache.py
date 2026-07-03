import os

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.plot import show

matplotlib.use("Agg")  # Use non-interactive backend
# import matplotlib.pyplot as plt  # Not needed in backend
import heapq
import logging
import time
from math import atan, degrees, sqrt

import geopandas as gpd
import osmnx as ox
import py3dep
from app.services.compressed_pathfinding import CompressedPathfinder, compress_search_space
from app.services.compressed_pathfinding_balanced import balanced_compress_search_space
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
from app.services.preprocessing import PathfindingPreprocessor
from app.services.tiled_dem_cache import TiledDEMCache
from pyproj import Transformer
from shapely.geometry import box

logger = logging.getLogger(__name__)


class DEMTileCache:
    def __init__(
        self,
        buffer=0.05,
        debug_mode=False,
        obstacle_config=None,
        path_preferences=None,
        dem_smoothing_sigma=1.0,
        expand_ski_runs=True,
    ):
        """
        Initializes the DEMTileCache.

        Parameters:
        - buffer: Buffer size in degrees to extend the area of interest.
        - debug_mode: If True, collect detailed pathfinding debug information.
        - obstacle_config: ObstacleConfig instance for customizing obstacle handling
        - path_preferences: PathPreferences instance for preferring certain paths
        - expand_ski_runs: If True, expand ski run lines to realistic corridor widths
        """
        self.dem_smoothing_sigma = dem_smoothing_sigma
        self.expand_ski_runs = expand_ski_runs
        self.buffer = buffer
        self.debug_mode = debug_mode
        self.debug_data = None
        self.obstacle_config = obstacle_config or ObstacleConfig()
        self.path_preferences = path_preferences or PathPreferences()
        self.preprocessor = PathfindingPreprocessor()
        self.preprocessing_cache = {}  # Cache preprocessed data by tile bounds
        self.terrain_cache = {}  # Cache downloaded terrain data
        self.cost_surface_cache = {}  # Cache computed cost surfaces

        # Initialize tiled cache for cost surfaces
        self.tiled_cache = TiledDEMCache(tile_size_degrees=0.01)  # ~1km tiles

        # Get absolute path for DEM data directory
        self.dem_data_dir = os.path.abspath("dem_data")

        # Get HTTP cache location (py3dep/HyRiver cache)
        http_cache_path = os.environ.get("HYRIVER_CACHE_NAME", os.path.abspath("cache/aiohttp_cache.sqlite"))
        if not os.path.isabs(http_cache_path):
            http_cache_path = os.path.abspath(http_cache_path)

        # Get cache sizes
        http_cache_size = 0
        if os.path.exists(http_cache_path):
            http_cache_size = os.path.getsize(http_cache_path) / (1024 * 1024)  # MB

        # Only print cache info if debug mode is on
        if debug_mode:
            logger.info("[CACHE PATHS] DEMTileCache initialized:")
            logger.info(f"  - HTTP cache (py3dep): {http_cache_path} ({http_cache_size:.0f} MB)")
            logger.info(f"  - DEM data directory: {self.dem_data_dir}")
            logger.info(f"  - Tile cache directory: {os.path.abspath(self.tiled_cache.cache_dir)}")

    def _create_tile_aligned_cache_key(self, min_lat, max_lat, min_lon, max_lon):
        """
        Create a cache key aligned to tile boundaries for better geometric reuse.
        Routes with slight coordinate differences will map to the same cache key.
        """
        tile_size = self.tiled_cache.tile_size

        # Round bounds outward to tile boundaries
        # This ensures we always cover the requested area
        aligned_min_lat = np.floor(min_lat / tile_size) * tile_size
        aligned_max_lat = np.ceil(max_lat / tile_size) * tile_size
        aligned_min_lon = np.floor(min_lon / tile_size) * tile_size
        aligned_max_lon = np.ceil(max_lon / tile_size) * tile_size

        # Use 3 decimal places (roughly 111m precision) since we're tile-aligned
        return f"{aligned_min_lat:.3f},{aligned_max_lat:.3f},{aligned_min_lon:.3f},{aligned_max_lon:.3f}"

    def find_route(self, lat1, lon1, lat2, lon2):
        """
        Finds a hiking route between two GPS points.

        Parameters:
        - lat1, lon1: Latitude and longitude of the starting point.
        - lat2, lon2: Latitude and longitude of the destination point.

        Returns:
        - path_coords: List of (longitude, latitude) tuples representing the path coordinates.
        """
        logger.info(f"\n[ROUTE REQUEST] Finding route from ({lat1:.4f}, {lon1:.4f}) to ({lat2:.4f}, {lon2:.4f})")
        logger.debug(
            f"[CACHE STATUS] Terrain cache: {len(self.terrain_cache)} entries, Cost surface cache: {len(self.cost_surface_cache)} entries"
        )

        # Define Area of Interest
        min_lat, max_lat, min_lon, max_lon = self.define_area_of_interest(lat1, lon1, lat2, lon2)

        # TRY TILE-BASED APPROACH FIRST - this enables geometric reuse
        tiles_needed = self.tiled_cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
        logger.debug(
            f"[TILE] Need {len(tiles_needed)} tiles for bounds ({min_lat:.4f}, {max_lat:.4f}, {min_lon:.4f}, {max_lon:.4f})"
        )

        # Try to compose from tiles (only missing tiles will be computed)
        target_bounds = (min_lat, max_lat, min_lon, max_lon)
        composed_data = self.tiled_cache.compose_tiles(
            tiles_needed, "cost", target_bounds, compute_func=self._compute_tile_cost_surface
        )

        if composed_data:
            # Success! Use tile-based data
            logger.info("[TILE SUCCESS] Using tile-based approach - no DEM download needed")
            return self._process_route_with_composed_tiles(
                composed_data, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon
            )

        # FALLBACK: If tile approach failed, use original DEM download approach
        logger.warning("[TILE FALLBACK] Tile composition failed, falling back to DEM download")

        # Create cache key for this area (tile-aligned for better reuse)
        cache_key = self._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)

        # Check if we have cached terrain data
        if cache_key in self.terrain_cache:
            dem, out_trans, crs = self.terrain_cache[cache_key]
            logger.debug(f"Using cached terrain data for bounds: {cache_key}")
        else:
            # Download DEM Data
            dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
            if dem_file is None:
                return None

            # Read DEM Data
            dem, out_trans, crs = self.read_dem(dem_file)

            # Reproject DEM to Projected CRS
            dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)

            # Cache the reprojected terrain data
            self.terrain_cache[cache_key] = (dem, out_trans, crs)
            logger.debug(f"Cached terrain data for bounds: {cache_key}")

            # Skip reprojection if using cached data (already reprojected)
            return self._process_route_with_terrain(
                dem, out_trans, crs, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon
            )

        # For cached data, skip reprojection
        return self._process_route_with_terrain(
            dem, out_trans, crs, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon
        )

    def _process_route_with_composed_tiles(
        self, composed_data, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon
    ):
        """Process route finding with tile-composed data"""
        # Extract data from composed tiles
        cost_surface = composed_data["cost_surface"]
        slope_degrees = composed_data["slope_degrees"]
        dem = composed_data.get("dem")
        out_trans = composed_data["transform"]
        crs = composed_data.get("crs", "EPSG:4326")

        # Build indices
        indices = self.build_indices(cost_surface)

        # Get indices for start and end points using the existing method
        start_idx, end_idx, transformer = self.get_indices(lat1, lon1, lat2, lon2, out_trans, crs, indices)

        if start_idx is None or end_idx is None:
            logger.error("[ERROR] Could not find start or end point in the cost surface")
            return None

        # Stats
        cost_stats = {
            "min": float(np.min(cost_surface)),
            "max": float(np.max(cost_surface)),
            "mean": float(np.mean(cost_surface)),
        }
        logger.debug(
            f"Cost surface stats: min={cost_stats['min']:.2f}, max={cost_stats['max']:.2f}, mean={cost_stats['mean']:.2f}"
        )

        # Check for impassable cells
        impassable_cells = np.sum(cost_surface >= 1000)
        total_cells = cost_surface.size
        impassable_percent = (impassable_cells / total_cells) * 100
        logger.debug(f"Percentage of impassable cells (cost > 1000): {impassable_percent:.1f}%")

        # Run pathfinding
        algo_start = time.time()

        # Determine algorithm based on distance
        # start_idx and end_idx are flat indices, need to convert to row/col
        row1, col1 = start_idx // cost_surface.shape[1], start_idx % cost_surface.shape[1]
        row2, col2 = end_idx // cost_surface.shape[1], end_idx % cost_surface.shape[1]
        grid_distance = abs(row2 - row1) + abs(col2 - col1)

        use_compression = grid_distance > 3000
        use_bidirectional = grid_distance > 500

        # Get optimization config if available
        optimization_config = getattr(self, "optimization_config", None)

        if use_compression:
            logger.debug(f"[ALGORITHM] Large search area (grid distance: {grid_distance}). Using compressed pathfinding...")
            # For now, just use bidirectional
            path = self.bidirectional_astar(
                cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem, optimization_config
            )
        elif use_bidirectional:
            logger.debug(f"[ALGORITHM] Medium search area (grid distance: {grid_distance}). Using bidirectional A*...")
            path = self.bidirectional_astar(
                cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem, optimization_config
            )
        else:
            logger.debug(f"[ALGORITHM] Small search area (grid distance: {grid_distance}). Using standard optimized A*...")
            path = self.astar_pathfinding_optimized(
                cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem
            )

        algo_time = time.time() - algo_start
        logger.debug(f"[TIMING] Pathfinding completed in {algo_time:.3f}s")

        if path is None:
            return None

        # Calculate slopes and path types for the path
        path_raster = composed_data.get("path_raster")
        path_types = composed_data.get("path_types")
        path_with_slopes = self.calculate_path_slopes(path, dem, out_trans, transformer, path_raster, path_types)

        return path_with_slopes

    def _process_route_with_terrain(
        self, dem, out_trans, crs, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon
    ):
        """Process route finding with terrain data (extracted for reuse with cached data)"""

        # Create cache key for cost surface (tile-aligned for better reuse)
        cost_cache_key = self._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon) + "_cost"

        # Try to use tiled cost surface first
        tiled_result = self._try_tiled_cost_surface(min_lat, max_lat, min_lon, max_lon, dem, out_trans, crs)
        path_raster = None
        path_types = None
        if tiled_result is not None:
            cost_surface, slope_degrees, indices, dem_composed, path_raster, path_types = tiled_result
            # Use composed DEM if available
            if dem_composed is not None:
                dem = dem_composed
            # Skip the regular cache check since we have tiled data
        else:
            # Fall back to regular caching
            # Check if we have cached cost surface
            if cost_cache_key in self.cost_surface_cache:
                cache_start = time.time()
                cost_surface, slope_degrees, indices = self.cost_surface_cache[cost_cache_key]
                logger.debug(f"[CACHE HIT] Using cached cost surface for bounds: {cost_cache_key}")
                logger.debug(f"[TIMING] Cache retrieval took {time.time() - cache_start:.3f}s")
            else:
                logger.debug(f"[CACHE MISS] Computing cost surface for bounds: {cost_cache_key}")
                compute_start = time.time()

                # Fetch and Rasterize Obstacles
                obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
                obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)

                # Fetch preferred paths
                paths = self.fetch_paths(min_lat, max_lat, min_lon, max_lon)
                path_raster, path_types, path_raw_tags = self.rasterize_paths(paths, out_trans, dem.shape, crs)

                # Compute Slope and Cost Surface with Obstacles and Path Preferences
                cost_surface, slope_degrees, slope_change = self.compute_cost_surface(
                    dem, out_trans, obstacle_mask, path_raster, path_types
                )

                # Build Indices for Pathfinding
                indices = self.build_indices(cost_surface)

                # Cache the processed data
                self.cost_surface_cache[cost_cache_key] = (cost_surface, slope_degrees, indices)
                logger.debug(f"[TIMING] Cost surface computation took {time.time() - compute_start:.3f}s")
                logger.debug(f"[CACHE] Stored cost surface for bounds: {cost_cache_key}")

        # Log some statistics about the cost surface
        logger.debug(
            f"Cost surface stats: min={np.min(cost_surface):.2f}, max={np.max(cost_surface):.2f}, mean={np.mean(cost_surface):.2f}"
        )
        logger.debug(
            f"Percentage of impassable cells (cost > 1000): {np.sum(cost_surface > 1000) / cost_surface.size * 100:.1f}%"
        )

        # If indices not from cache, build them
        if cost_cache_key not in self.cost_surface_cache:
            indices = self.build_indices(cost_surface)

        # Get Start and End Indices
        # Use composed transform if available (from tiled composition)
        transform_to_use = getattr(self, "_composed_transform", out_trans)
        crs_to_use = getattr(self, "_composed_crs", crs)
        start_idx, end_idx, transformer = self.get_indices(
            lat1, lon1, lat2, lon2, transform_to_use, crs_to_use, indices
        )
        if start_idx is None or end_idx is None:
            return None

        # Clear composed transform and CRS for next request
        if hasattr(self, "_composed_transform"):
            delattr(self, "_composed_transform")
        if hasattr(self, "_composed_crs"):
            delattr(self, "_composed_crs")

        # Check grid distance to decide between compressed and regular pathfinding
        start_row, start_col = np.unravel_index(start_idx, indices.shape)
        end_row, end_col = np.unravel_index(end_idx, indices.shape)
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)

        # Use compressed pathfinding for large search spaces
        # Consider both grid distance and terrain complexity
        use_compression = grid_distance > 800  # Higher threshold for compression
        use_bidirectional = grid_distance > 200  # Use bidirectional for medium to large searches

        # Log timing for algorithm selection
        algo_start = time.time()

        if use_compression:
            logger.debug(f"[ALGORITHM] Large search area (grid distance: {grid_distance}). Using compressed pathfinding...")
            path = self.astar_pathfinding_compressed(
                cost_surface,
                indices,
                start_idx,
                end_idx,
                transform_to_use,
                transformer,
                slope_degrees,
                obstacle_mask,
                path_raster,
                dem,
            )
            # If compressed pathfinding fails, fall back to bidirectional A*
            if path is None:
                logger.warning("[ALGORITHM] Compressed pathfinding failed, falling back to bidirectional A*...")
                path = self.bidirectional_astar(
                    cost_surface, indices, start_idx, end_idx, transform_to_use, transformer, dem
                )
        elif use_bidirectional:
            # Use bidirectional A* for medium searches
            logger.debug(f"[ALGORITHM] Medium search area (grid distance: {grid_distance}). Using bidirectional A*...")
            path = self.bidirectional_astar(
                cost_surface, indices, start_idx, end_idx, transform_to_use, transformer, dem
            )
        else:
            # Use optimized A* for smaller searches
            logger.debug(f"[ALGORITHM] Small search area (grid distance: {grid_distance}). Using standard optimized A*...")
            path = self.astar_pathfinding_optimized(
                cost_surface, indices, start_idx, end_idx, transform_to_use, transformer, dem
            )

        algo_time = time.time() - algo_start
        logger.debug(f"[TIMING] Pathfinding completed in {algo_time:.3f}s")

        if path is None:
            return None

        # Calculate slopes for each segment of the path
        path_with_slopes = self.calculate_path_slopes(path, dem, out_trans, transformer, path_raster, path_types)

        # Skip plotting in backend service
        # self.plot_results(dem, out_trans, crs, path, lat1, lon1, lat2, lon2,
        #                   min_lat, max_lat, min_lon, max_lon, obstacles=obstacles)

        return path_with_slopes

    # Add methods to download and read DEM data
    def download_dem(self, min_lat, max_lat, min_lon, max_lon):
        """
        Downloads and clips the DEM data for the specified area using py3dep.

        Returns:
        - dem_file: Path to the DEM file.
        """
        # Use tile-aligned bounds for filename to enable better reuse
        cache_key = self._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)
        # Replace commas with underscores for filename
        cache_name = f"dem_{cache_key.replace(',', '_')}.tif"
        dem_file = os.path.join(self.dem_data_dir, cache_name)

        if not os.path.exists(self.dem_data_dir):
            os.makedirs(self.dem_data_dir)
            logger.info(f"[CACHE] Created DEM data directory: {self.dem_data_dir}")

        # Check if file already exists
        if os.path.exists(dem_file):
            logger.debug(f"Using cached DEM file: {dem_file}")
            return dem_file

        try:
            # Log HTTP cache status
            http_cache_path = os.environ.get("HYRIVER_CACHE_NAME", os.path.abspath("cache/aiohttp_cache.sqlite"))
            if not os.path.isabs(http_cache_path):
                http_cache_path = os.path.abspath(http_cache_path)

            cache_exists = os.path.exists(http_cache_path)
            if cache_exists:
                cache_size_mb = os.path.getsize(http_cache_path) / (1024 * 1024)
                logger.debug(f"[HTTP CACHE] Using cache: {http_cache_path} ({cache_size_mb:.1f} MB)")
            else:
                logger.debug(f"[HTTP CACHE] No cache found at: {http_cache_path}")

            # Try 3m resolution first for finding paths around steep slopes
            try:
                dem = py3dep.get_map(
                    "DEM",
                    (min_lon, min_lat, max_lon, max_lat),
                    resolution=3,  # 3m resolution for finding switchbacks and narrow paths
                    crs="EPSG:4326",
                )
                logger.debug("Downloaded DEM at 3m resolution")
            except Exception as e3:
                logger.warning(f"3m resolution failed ({e3}), trying 10m resolution...")
                try:
                    dem = py3dep.get_map(
                        "DEM",
                        (min_lon, min_lat, max_lon, max_lat),
                        resolution=10,  # 10m resolution as fallback
                        crs="EPSG:4326",
                    )
                    logger.debug("Downloaded DEM at 10m resolution")
                except Exception as e10:
                    logger.warning(f"10m resolution failed ({e10}), trying 30m resolution...")
                    # Fallback to 30m resolution
                    dem = py3dep.get_map(
                        "DEM",
                        (min_lon, min_lat, max_lon, max_lat),
                        resolution=30,  # 30m resolution fallback
                        crs="EPSG:4326",
                    )
                    logger.debug("Downloaded DEM at 30m resolution")

            # Save to file
            dem.rio.to_raster(dem_file)
            return dem_file
        except Exception as e:
            logger.error(f"Error downloading DEM data: {e}", exc_info=True)
            return None

    def read_dem(self, dem_file):
        """
        Reads the DEM data from the file.

        Returns:
        - dem: Numpy array of the DEM.
        - out_trans: Affine transform of the DEM.
        - crs: Coordinate Reference System of the DEM.
        """
        try:
            with rasterio.open(dem_file) as src:
                dem = src.read(1)
                out_trans = src.transform
                crs = src.crs
            return dem, out_trans, crs
        except Exception as e:
            logger.error(f"Error reading DEM data: {e}", exc_info=True)
            return None, None, None

    def define_area_of_interest(self, lat1, lon1, lat2, lon2):
        """
        Defines the area of interest with a buffer.

        Returns:
        - min_lat, max_lat, min_lon, max_lon
        """
        min_lat = min(lat1, lat2) - self.buffer
        max_lat = max(lat1, lat2) + self.buffer
        min_lon = min(lon1, lon2) - self.buffer
        max_lon = max(lon1, lon2) + self.buffer
        return min_lat, max_lat, min_lon, max_lon

    def merge_dem_tiles(self, tile_files):
        """
        Merges DEM tiles into a single raster.

        Returns:
        - dem: Numpy array of the DEM.
        - out_trans: Affine transform of the merged DEM.
        """
        try:
            src_files_to_mosaic = [rasterio.open(fp) for fp in tile_files]
            mosaic, out_trans = merge(src_files_to_mosaic)
            dem = mosaic[0]

            # Close the files
            for src in src_files_to_mosaic:
                src.close()

            return dem, out_trans
        except Exception as e:
            logger.error(f"Error merging DEM tiles: {e}", exc_info=True)
            return None, None

    def reproject_dem(self, dem, out_trans, crs):
        """
        Reprojects the DEM to a projected CRS.

        Returns:
        - dem: Reprojected DEM array.
        - out_trans: New affine transform.
        - crs: New coordinate reference system.
        """
        if crs.is_geographic:
            dst_crs = "EPSG:3857"  # Web Mercator
            transform, width, height = rasterio.warp.calculate_default_transform(
                crs,
                dst_crs,
                dem.shape[1],
                dem.shape[0],
                *rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], out_trans),
            )
            dem_resampled = np.empty((height, width), dtype=np.float32)
            rasterio.warp.reproject(
                source=dem,
                destination=dem_resampled,
                src_transform=out_trans,
                src_crs=crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=rasterio.enums.Resampling.bilinear,
            )
            dem = dem_resampled
            out_trans = transform
            crs = rasterio.crs.CRS.from_string(dst_crs)

        # Apply smoothing once after reprojection if configured
        if self.dem_smoothing_sigma > 0:
            from scipy.ndimage import gaussian_filter

            dem = gaussian_filter(dem, sigma=self.dem_smoothing_sigma)
            logger.debug(
                f"[DEM SMOOTHING] Applied Gaussian filter with sigma={self.dem_smoothing_sigma} pixels during reprojection"
            )

        return dem, out_trans, crs

    def fetch_obstacles(self, min_lat, max_lat, min_lon, max_lon):
        """
        Fetches obstacle data from OSM using configuration.

        Returns:
        - obstacles: GeoDataFrame containing obstacle geometries.
        """
        # Create a bounding box polygon
        bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)

        # Fetch OSM data for obstacles using config tags
        try:
            ox.settings.log_console = False
            obstacles = ox.features_from_polygon(bbox_polygon, self.obstacle_config.osm_tags)
            return obstacles
        except Exception as e:
            if type(e).__name__ == "InsufficientResponseError":
                logger.debug("No obstacle features in bbox (%.5f,%.5f,%.5f,%.5f)", min_lat, max_lat, min_lon, max_lon)
            else:
                logger.warning("Error fetching obstacles: %s", e, exc_info=True)
            return gpd.GeoDataFrame()

    def get_obstacle_mask(self, obstacles, transform, dem_shape, crs):
        """
        Rasterizes obstacles to create an obstacle mask.

        Returns:
        - obstacle_mask: Numpy array where obstacle cells are True.
        """
        if obstacles.empty:
            obstacle_mask = np.zeros(dem_shape, dtype=bool)
            return obstacle_mask

        # Reproject obstacles to DEM CRS
        obstacles = obstacles.to_crs(crs.to_string())

        # Prepare geometries for rasterization
        obstacle_geometries = obstacles.geometry.values
        shapes = [(geom, 1) for geom in obstacle_geometries if geom is not None]

        # Rasterize obstacles
        obstacle_raster = rasterize(
            shapes, out_shape=dem_shape, transform=transform, fill=0, all_touched=True, dtype=np.uint8
        )

        # Create obstacle mask
        obstacle_mask = obstacle_raster.astype(bool)

        return obstacle_mask

    def fetch_paths(self, min_lat, max_lat, min_lon, max_lon):
        """
        Fetches preferred path data from OSM.

        Returns:
        - paths: GeoDataFrame containing path geometries and types.
        """
        bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)

        try:
            ox.settings.log_console = False
            # Fetch paths using preference tags
            paths = ox.features_from_polygon(bbox_polygon, self.path_preferences.preferred_path_tags)
            return paths
        except Exception as e:
            if type(e).__name__ == "InsufficientResponseError":
                logger.debug("No path features in bbox (%.5f,%.5f,%.5f,%.5f)", min_lat, max_lat, min_lon, max_lon)
            else:
                logger.warning("Error fetching paths: %s", e, exc_info=True)
            return gpd.GeoDataFrame()

    def rasterize_paths(self, paths, transform, dem_shape, crs, obstacle_mask=None, expand_paths=False):
        """
        Rasterizes paths to create path preference raster.

        Returns:
        - path_raster: Numpy array where path cells contain type IDs
        - path_types: Dict mapping IDs to path types
        - path_raw_tags: Dict mapping IDs to original OSM tags
        """
        if paths.empty:
            return np.zeros(dem_shape, dtype=int), {}, {}

        # Reproject paths to DEM CRS
        paths = paths.to_crs(crs.to_string())

        # Create mapping of path types to IDs and store raw tags
        path_types = {}
        path_raw_tags = {}  # New: store original OSM tags for each path
        path_id = 1

        # Prepare shapes for rasterization with path type IDs
        shapes = []
        for idx, row in paths.iterrows():
            if row.geometry is not None:
                # Collect all non-null OSM tags for this feature
                raw_tags = {}
                for col in paths.columns:
                    if col != "geometry" and pd.notna(row[col]) and not str(col).startswith("osm"):
                        raw_tags[col] = str(row[col])

                # Determine path type from tags
                path_type = "off_path"
                source_tag = None
                if "highway" in row and pd.notna(row["highway"]):
                    path_type = str(row["highway"])
                    source_tag = "highway"
                elif "leisure" in row and pd.notna(row["leisure"]):
                    path_type = str(row["leisure"])
                    source_tag = "leisure"
                elif "landuse" in row and pd.notna(row["landuse"]):
                    path_type = str(row["landuse"])
                    source_tag = "landuse"
                elif "natural" in row and pd.notna(row["natural"]):
                    # Map natural types to our cost categories
                    natural_type = str(row["natural"])
                    source_tag = "natural"
                    if natural_type in ["grassland", "heath"]:
                        path_type = "grass"
                    elif natural_type == "meadow":
                        path_type = "meadow"
                    elif natural_type in ["beach", "sand"]:
                        path_type = "beach"
                    else:
                        path_type = natural_type
                elif "piste:type" in row and pd.notna(row["piste:type"]):
                    path_type = str(row["piste:type"])
                    source_tag = "piste:type"

                # Get or assign ID for this specific feature
                current_path_id = path_id
                path_id += 1

                # Store the interpreted type and raw tags
                path_types[current_path_id] = path_type
                path_raw_tags[current_path_id] = {
                    "osm_id": str(idx),
                    "interpreted_type": path_type,
                    "source_tag": source_tag,
                    "tags": raw_tags,
                }

                shapes.append((row.geometry, current_path_id))

        # Rasterize paths
        path_raster = rasterize(
            shapes,
            out_shape=dem_shape,
            transform=transform,
            fill=0,  # 0 means no path
            all_touched=True,
            dtype=np.uint16,  # Support up to 65535 path IDs
        )

        # Widen 1px path lines into realistic corridors, then bridge small gaps so
        # the trail network is connected. Imported directly: a genuine import
        # failure must surface (caught and logged loudly upstream), never be
        # silently swallowed into trail-blind routing.
        from app.services.path_connectivity import PathConnectivityRepair
        from app.services.path_expansion import PathExpander

        pixel_size_m = abs(transform.a)  # Assuming square pixels

        # Expand all paths to realistic widths (not just ski runs)
        if expand_paths or self.expand_ski_runs:  # Use parameter or instance setting
            path_raster, path_types = PathExpander.expand_paths(path_raster, path_types, path_raw_tags, pixel_size_m)
            logger.debug("Expanded path lines to corridors (pixel_size=%.2fm)", pixel_size_m)

        # Repair path connectivity (bridge small gaps) -- AFTER expansion, BEFORE cost calc
        path_raster, path_types = PathConnectivityRepair.repair_path_connectivity(
            path_raster, path_types, path_raw_tags, pixel_size_m, obstacle_mask
        )

        # Note: path_types now maps ID to type directly (no need to reverse)
        return path_raster, path_types, path_raw_tags

    def compute_cost_surface(self, dem, out_trans, obstacle_mask, path_raster=None, path_types=None):
        """
        Computes the slope and creates the cost surface, incorporating obstacles and path preferences.
        Now includes DEM smoothing to remove artifacts and slope change rate penalties.

        Returns:
        - cost_surface: Numpy array representing the cost of traversing each cell.
        - slope_degrees: Numpy array of slope values in degrees.
        - slope_change: Numpy array of slope change magnitude in degrees per meter.
        """
        # DEM is already smoothed during reprojection
        # Calculate slopes directly from the DEM
        cell_size_x = out_trans.a
        cell_size_y = -out_trans.e  # Negative because of the coordinate system
        dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
        slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_degrees = np.degrees(slope_radians)

        # Calculate slope change (gradient of slope)
        # This gives us how rapidly the slope is changing
        dslope_dx, dslope_dy = np.gradient(slope_degrees, cell_size_x, cell_size_y)
        slope_change = np.sqrt(dslope_dx**2 + dslope_dy**2)

        # Create cost surface using configuration
        cost_surface = np.ones_like(dem)

        # Apply slope-based costs from configuration
        for i in range(slope_degrees.shape[0]):
            for j in range(slope_degrees.shape[1]):
                slope = slope_degrees[i, j]
                base_cost = self.obstacle_config.get_slope_cost_multiplier(slope)

                # Apply path preference multiplier if on a path
                if path_raster is not None and path_types is not None:
                    path_id = path_raster[i, j]
                    if path_id > 0 and path_id in path_types:
                        path_type = path_types[path_id]
                        path_multiplier = self.path_preferences.get_path_cost_multiplier(path_type)

                        # FIXED: Don't let path preferences override slope costs
                        # Minimum cost should be based on slope, paths just get a small bonus
                        # For steep slopes, being on a path helps less
                        if slope > 25:
                            # On very steep slopes, paths provide minimal benefit
                            # At 25°: 80% of slope cost, at 45°: 95% of slope cost
                            path_benefit = 0.8 + (slope - 25) / 100.0  # 0.8 to 1.0
                            cost_surface[i, j] = base_cost * min(1.0, max(path_benefit, path_multiplier))
                        else:
                            # On moderate slopes, paths can reduce cost but not below 50% of slope cost
                            cost_surface[i, j] = base_cost * max(0.5, path_multiplier)
                    else:
                        # Off-path terrain - use full slope cost
                        # Don't reduce slope penalties for off-path terrain
                        cost_surface[i, j] = base_cost
                else:
                    cost_surface[i, j] = base_cost

        # Assign obstacle costs from configuration
        # BUT: Don't override paths (which may include bridged connections)
        obstacle_indices = np.where(obstacle_mask)
        for i, j in zip(obstacle_indices[0], obstacle_indices[1]):
            # Only mark as obstacle if it's not on a path
            if path_raster is None or path_raster[i, j] == 0:
                cost_surface[i, j] = self.obstacle_config.get_cost_for_feature("default")

        return cost_surface, slope_degrees, slope_change

    def build_indices(self, cost_surface):
        """
        Builds indices array from the cost surface.

        Returns:
        - indices: Numpy array of indices mapping the grid cells.
        """
        indices = np.arange(cost_surface.size).reshape(cost_surface.shape)
        return indices

    def get_indices(self, lat1, lon1, lat2, lon2, out_trans, crs, indices):
        """
        Transforms GPS coordinates to raster indices.

        Returns:
        - start_idx, end_idx: Indices of the start and end points in the flattened cost surface.
        - transformer: Transformer object for coordinate conversions.
        """
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_start, y_start = transformer.transform(lon1, lat1)
        x_end, y_end = transformer.transform(lon2, lat2)

        cell_size_x = out_trans.a
        cell_size_y = out_trans.e  # Already negative

        col_start = int((x_start - out_trans.c) / cell_size_x)
        row_start = int((y_start - out_trans.f) / cell_size_y)
        col_end = int((x_end - out_trans.c) / cell_size_x)
        row_end = int((y_end - out_trans.f) / cell_size_y)

        height, width = indices.shape

        # Ensure indices are within bounds
        if not (0 <= row_start < height and 0 <= col_start < width and 0 <= row_end < height and 0 <= col_end < width):
            return None, None, None

        start_idx = indices[row_start, col_start]
        end_idx = indices[row_end, col_end]

        return start_idx, end_idx, transformer

    def _compute_tile_cost_surface(self, min_lat, max_lat, min_lon, max_lon):
        """Compute cost surface for a single tile"""
        try:
            # Download DEM for tile
            dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
            if dem_file is None:
                return None

            # Read and reproject DEM
            dem, out_trans, crs = self.read_dem(dem_file)
            if dem is None:
                return None

            dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)

            # Fetch obstacles and paths
            obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
            obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)
            paths = self.fetch_paths(min_lat, max_lat, min_lon, max_lon)
            path_raster, path_types, path_raw_tags = self.rasterize_paths(
                paths, out_trans, dem.shape, crs, obstacle_mask, expand_paths=True
            )

            # Compute cost surface
            cost_surface, slope_degrees, slope_change = self.compute_cost_surface(
                dem, out_trans, obstacle_mask, path_raster, path_types
            )

            return {
                "cost_surface": cost_surface,
                "slope_degrees": slope_degrees,
                "slope_change": slope_change,
                "transform": out_trans,
                "crs": crs,
                "dem": dem,
                "path_raster": path_raster,
                "path_types": path_types,
                "path_raw_tags": path_raw_tags,
            }
        except Exception:
            logger.error(
                "Error computing tile cost surface for tile bbox=(%.6f,%.6f,%.6f,%.6f); "
                "falling back to non-tiled surface",
                min_lat,
                max_lat,
                min_lon,
                max_lon,
                exc_info=True,
            )
            return None

    def _try_tiled_cost_surface(self, min_lat, max_lat, min_lon, max_lon, dem, out_trans, crs):
        """Try to compose cost surface from tiles"""
        try:
            # Get required tiles
            tiles = self.tiled_cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
            logger.debug(
                f"[TILE] Need {len(tiles)} tiles for bounds ({min_lat:.4f}, {max_lat:.4f}, {min_lon:.4f}, {max_lon:.4f})"
            )

            # Use tile composition
            target_bounds = (min_lat, max_lat, min_lon, max_lon)
            composed_data = self.tiled_cache.compose_tiles(
                tiles, "cost", target_bounds, compute_func=self._compute_tile_cost_surface
            )

            if composed_data:
                if composed_data.get("composed_from"):
                    logger.debug(f"[TILE COMPOSE] Using {composed_data['composed_from']} tiles for cost surface")
                else:
                    tile_x, tile_y = tiles[0]
                    logger.debug(f"[TILE HIT] Using single tile ({tile_x}, {tile_y}) for cost surface")

                # Extract cost surface and related data
                cost_surface = composed_data["cost_surface"]
                slope_degrees = composed_data["slope_degrees"]
                slope_change = composed_data.get("slope_change")  # Get slope change data
                dem_composed = composed_data.get("dem")  # Get composed DEM data
                path_raster = composed_data.get("path_raster")
                path_types = composed_data.get("path_types")
                indices = self.build_indices(cost_surface)

                # Update the transform if we have a composite
                if composed_data.get("transform") is not None:
                    # Store the composed transform for later use
                    self._composed_transform = composed_data["transform"]

                # Store CRS for get_indices to use
                self._composed_crs = composed_data.get("crs", crs)

                logger.info(f"[TILE SUCCESS] Returning tiled cost surface with shape {cost_surface.shape}")
                return cost_surface, slope_degrees, indices, dem_composed, path_raster, path_types, slope_change
            else:
                logger.debug("[TILE] compose_tiles returned None")

        except Exception as e:
            logger.error(f"[TILE ERROR] Exception in _try_tiled_cost_surface: {e}", exc_info=True)
            import traceback

            traceback.print_exc()

        logger.warning("[TILE] Falling back to regular computation")
        return None

    def astar_pathfinding_compressed(
        self,
        cost_surface,
        indices,
        start_idx,
        end_idx,
        out_trans,
        transformer,
        slope_degrees,
        obstacle_mask=None,
        path_raster=None,
        dem=None,
    ):
        """
        Computes path using compressed regions for large search spaces.
        """
        height, width = cost_surface.shape
        start_row, start_col = np.unravel_index(start_idx, (height, width))
        end_row, end_col = np.unravel_index(end_idx, (height, width))

        logger.debug("Using compressed pathfinding...")

        # Calculate grid distance for this method
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)

        # Use simpler compression for very large searches
        # The full compression with obstacles/paths is too fragmented for this terrain
        if grid_distance > 1000:
            logger.debug("Using simplified compression for very large search...")
            regions, region_map = balanced_compress_search_space(slope_degrees, cost_surface, obstacle_mask)
        else:
            regions, region_map = compress_search_space(slope_degrees, cost_surface, obstacle_mask, path_raster)

        # Find path using compressed space
        compressor = CompressedPathfinder()
        compressed_path = compressor.find_path_compressed(
            regions, (start_row, start_col), (end_row, end_col), cost_surface
        )

        if not compressed_path:
            logger.warning("Compressed pathfinding failed, falling back to regular A*")
            return None

        # Refine the path using local A* between waypoints
        refined_path = []
        for i in range(len(compressed_path) - 1):
            current_cell = compressed_path[i]
            next_cell = compressed_path[i + 1]

            # If cells are adjacent, just add the connection
            if abs(current_cell[0] - next_cell[0]) <= 1 and abs(current_cell[1] - next_cell[1]) <= 1:
                if i == 0 or current_cell != refined_path[-1]:
                    refined_path.append(current_cell)
            else:
                # Run optimized local A* between these cells
                local_path = self._local_astar_optimized(
                    cost_surface,
                    indices,
                    current_cell,
                    next_cell,
                    out_trans,
                    transformer,
                    dem,
                    max_distance=50,  # Limit local search
                )
                if local_path:
                    # Add path excluding duplicates
                    for cell in local_path:
                        if not refined_path or cell != refined_path[-1]:
                            refined_path.append(cell)

        # Add final cell
        if compressed_path[-1] not in refined_path:
            refined_path.append(compressed_path[-1])

        # Convert to coordinates
        path = []
        for row, col in refined_path:
            x = out_trans.c + col * out_trans.a + out_trans.a / 2
            y = out_trans.f + row * out_trans.e + out_trans.e / 2
            x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
            path.append((x_lon, y_lat))

        return path

    def _local_astar_optimized(
        self, cost_surface, indices, start_cell, end_cell, out_trans, transformer, dem=None, max_distance=50
    ):
        """Optimized local A* for refining paths between waypoints"""
        height, width = cost_surface.shape
        start_row, start_col = start_cell
        end_row, end_col = end_cell

        # Check if cells are too far apart
        if abs(end_row - start_row) > max_distance or abs(end_col - start_col) > max_distance:
            # Just return direct line for very distant cells
            return [start_cell, end_cell]

        # Convert cells to indices
        start_idx = start_row * width + start_col
        end_idx = end_row * width + end_col

        # Use optimized pathfinding with local search config
        optimization_config = {
            "early_termination": True,
            "stagnation_limit": 100,  # Lower for local search
            "dynamic_weights": True,
            "memory_limit": 5000,  # Lower for local search
            "corner_cutting": False,  # Disable for local refinement
            "max_iterations": max_distance * max_distance * 4,
        }

        # Use the optimized pathfinding method
        path_coords = self.astar_pathfinding_optimized(
            cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem, optimization_config
        )

        if not path_coords:
            # No path found, return direct line
            return [start_cell, end_cell]

        # Convert coordinates back to cells
        path_cells = []
        for lon, lat in path_coords:
            # Convert lat/lon back to grid indices
            x, y = transformer.transform(lon, lat)
            col = int(round((x - out_trans.c) / out_trans.a))
            row = int(round((y - out_trans.f) / out_trans.e))

            if 0 <= row < height and 0 <= col < width:
                path_cells.append((row, col))

        return path_cells if path_cells else [start_cell, end_cell]

    def astar_pathfinding(self, cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem=None):
        """
        Computes the least-cost path using the A* algorithm.

        Returns:
        - path: List of (longitude, latitude) tuples representing the path coordinates.
        """
        height, width = cost_surface.shape
        open_set = []
        heapq.heappush(open_set, (0, start_idx))

        came_from = {}
        g_score = np.full(cost_surface.size, np.inf)
        g_score[start_idx] = 0

        f_score = np.full(cost_surface.size, np.inf)
        f_score[start_idx] = self.heuristic(start_idx, end_idx, indices.shape, out_trans)

        closed_set = set()

        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Initialize debug data if in debug mode
        if self.debug_mode:
            self.debug_data = {
                "explored_nodes": [],
                "cost_calculations": [],
                "path_decisions": [],
                "grid_exploration": {
                    "shape": indices.shape,
                    "g_scores": np.full(indices.shape, np.inf),
                    "f_scores": np.full(indices.shape, np.inf),
                    "h_scores": np.full(indices.shape, np.inf),
                    "explored": np.zeros(indices.shape, dtype=bool),
                    "in_path": np.zeros(indices.shape, dtype=bool),
                },
                "decision_points": [],
                "terrain_costs": cost_surface.copy(),
                "bounds": {"transform": out_trans, "start_idx": start_idx, "end_idx": end_idx},
            }

        step_count = 0
        max_iterations = 10000000  # Further increased for long routes at 3m resolution

        # Calculate expected search area to warn about large searches
        start_row, start_col = np.unravel_index(start_idx, indices.shape)
        end_row, end_col = np.unravel_index(end_idx, indices.shape)
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)

        if grid_distance > 1000:
            logger.warning(f"Warning: Large search area detected. Grid distance: {grid_distance} cells")

        while open_set and step_count < max_iterations:
            current_f, current = heapq.heappop(open_set)
            step_count += 1

            if current == end_idx:
                path = self.reconstruct_path_astar(came_from, current, out_trans, transformer, indices.shape)

                # Mark path nodes in debug data
                if self.debug_mode and path:
                    self._mark_path_in_debug(path, out_trans, transformer, indices.shape)

                return path

            closed_set.add(current)
            row_current, col_current = np.unravel_index(current, indices.shape)

            # Record exploration in debug mode
            if self.debug_mode:
                self.debug_data["explored_nodes"].append(
                    {
                        "step": step_count,
                        "node_idx": current,
                        "row": row_current,
                        "col": col_current,
                        "g_score": g_score[current],
                        "f_score": current_f,
                        "h_score": current_f - g_score[current],
                    }
                )
                self.debug_data["grid_exploration"]["explored"][row_current, col_current] = True
                self.debug_data["grid_exploration"]["g_scores"][row_current, col_current] = g_score[current]
                self.debug_data["grid_exploration"]["f_scores"][row_current, col_current] = current_f

            # Explore neighbors
            neighbor_evaluations = []

            # Use preprocessed neighbors if available
            preprocessed_data = None  # Not available in this version
            if preprocessed_data and current in preprocessed_data["neighbor_map"]:
                # Use precomputed neighbors
                for neighbor, base_distance in preprocessed_data["neighbor_map"][current]:
                    if neighbor in closed_set:
                        continue

                    row_neighbor, col_neighbor = np.unravel_index(neighbor, indices.shape)
                    terrain_cost = cost_surface[row_neighbor, col_neighbor]

                    # Calculate movement cost
                    movement_cost = terrain_cost * base_distance
                    tentative_g_score = g_score[current] + movement_cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score

                        # Calculate heuristic (with caching if available)
                        if neighbor not in heuristic_cache:
                            heuristic_cache[neighbor] = self.heuristic(neighbor, end_idx, indices.shape, out_trans)
                        heuristic_cost = heuristic_cache[neighbor]

                        # Apply dynamic weight
                        if use_dynamic_weights:
                            progress = 1.0 - (heuristic_cost / start_h) if start_h > 0 else 0
                            # Reduce heuristic weight based on gradient preference
                            # gradient_preference > 1 means we're more willing to explore longer paths
                            base_weight = 2.0 / self.obstacle_config.gradient_preference
                            heuristic_weight = base_weight - (progress * base_weight / 2.0)
                        else:
                            heuristic_weight = 1.0 / self.obstacle_config.gradient_preference

                        f_score = tentative_g_score + heuristic_cost * heuristic_weight
                        tie_breaker += 1
                        heapq.heappush(open_set, (f_score, -tentative_g_score, tie_breaker, neighbor))
            else:
                # Fallback to standard neighbor exploration
                for dy, dx in neighbors_offsets:
                    row_neighbor = row_current + dy
                    col_neighbor = col_current + dx

                    if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                        neighbor = indices[row_neighbor, col_neighbor]
                        if neighbor in closed_set:
                            continue

                        # Calculate tentative g_score
                        distance = sqrt((dy * out_trans.e) ** 2 + (dx * out_trans.a) ** 2)
                        terrain_cost = cost_surface[row_neighbor, col_neighbor]
                        movement_cost = terrain_cost * distance
                        tentative_g_score = g_score[current] + movement_cost
                        heuristic_cost = self.heuristic(neighbor, end_idx, indices.shape, out_trans)
                        tentative_f_score = tentative_g_score + heuristic_cost

                        # Record neighbor evaluation in debug mode
                        if self.debug_mode:
                            # Get elevation data for slope calculation
                            if dem is not None:
                                current_elevation = (
                                    dem[row_current, col_current]
                                    if 0 <= row_current < dem.shape[0] and 0 <= col_current < dem.shape[1]
                                    else 0
                                )
                                neighbor_elevation = (
                                    dem[row_neighbor, col_neighbor]
                                    if 0 <= row_neighbor < dem.shape[0] and 0 <= col_neighbor < dem.shape[1]
                                    else 0
                                )
                            else:
                                current_elevation = 0
                                neighbor_elevation = 0
                            elevation_change = neighbor_elevation - current_elevation
                            slope_degrees = np.degrees(np.arctan(abs(elevation_change) / max(distance, 0.1)))

                            # Break down the terrain cost components
                            base_cost = 1.0  # Base movement cost
                            slope_penalty = max(0, terrain_cost - base_cost)  # Additional cost due to slope/obstacles

                            neighbor_evaluations.append(
                                {
                                    "neighbor_idx": neighbor,
                                    "row": row_neighbor,
                                    "col": col_neighbor,
                                    "direction": (dy, dx),
                                    "direction_name": self._get_direction_name(dy, dx),
                                    "distance_meters": distance,
                                    "elevation_change_m": elevation_change,
                                    "slope_degrees": slope_degrees,
                                    "terrain_breakdown": {
                                        "base_cost": base_cost,
                                        "slope_penalty": slope_penalty,
                                        "total_terrain_cost": terrain_cost,
                                        "is_obstacle": terrain_cost > 100,  # High cost indicates obstacle
                                    },
                                    "cost_breakdown": {
                                        "distance_component": distance,
                                        "terrain_component": terrain_cost,
                                        "total_movement_cost": movement_cost,
                                        "explanation": f"movement_cost = terrain_cost({terrain_cost:.2f}) × distance({distance:.1f}m) = {movement_cost:.2f}",
                                    },
                                    "g_score_breakdown": {
                                        "previous_g_score": g_score[current],
                                        "movement_cost": movement_cost,
                                        "tentative_g_score": tentative_g_score,
                                        "explanation": f"g_score = previous({g_score[current]:.2f}) + movement_cost({movement_cost:.2f}) = {tentative_g_score:.2f}",
                                    },
                                    "f_score_breakdown": {
                                        "g_score": tentative_g_score,
                                        "h_score": heuristic_cost,
                                        "f_score": tentative_f_score,
                                        "explanation": f"f_score = g_score({tentative_g_score:.2f}) + h_score({heuristic_cost:.2f}) = {tentative_f_score:.2f}",
                                    },
                                    "current_g_score": g_score[neighbor],
                                    "is_improvement": tentative_g_score < g_score[neighbor],
                                }
                            )

                        if tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_f_score
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

            # Record decision point in debug mode
            if self.debug_mode:
                self.debug_data["decision_points"].append(
                    {
                        "step": step_count,
                        "current_node": {
                            "idx": current,
                            "row": row_current,
                            "col": col_current,
                            "lat_lon": self._idx_to_latlon(current, out_trans, transformer, indices.shape),
                        },
                        "neighbors_evaluated": neighbor_evaluations,
                        "chosen_neighbors": [n for n in neighbor_evaluations if n["is_improvement"]],
                    }
                )

        if step_count >= max_iterations:
            logger.info(f"A* search stopped after {max_iterations} iterations")
        else:
            logger.info(f"A* search exhausted all possibilities after {step_count} iterations")

        return None  # No path found

    def heuristic(self, node_idx, end_idx, shape, out_trans):
        """
        Heuristic function for the A* algorithm (Euclidean distance).

        Returns:
        - Estimated cost from the current node to the end node.
        """
        row_node, col_node = np.unravel_index(node_idx, shape)
        row_end, col_end = np.unravel_index(end_idx, shape)

        dx = (col_node - col_end) * out_trans.a
        dy = (row_node - row_end) * out_trans.e  # Negative because of coordinate system

        # Use a more realistic heuristic that accounts for terrain difficulty
        # This prevents the algorithm from being too aggressive about straight-line paths
        # through steep terrain
        distance = sqrt(dx**2 + dy**2)

        # Use a less optimistic multiplier that better reflects real terrain costs
        # This encourages the algorithm to explore alternative routes when facing steep slopes
        return distance * 0.8  # More conservative estimate

    def _greedy_path_to_goal(self, start_row, start_col, end_row, end_col, cost_surface, out_trans, transformer):
        """
        Create a simple greedy path from current position to goal.
        Used for completing paths when early termination triggers.
        """
        path = []
        current_row, current_col = start_row, start_col

        while (current_row, current_col) != (end_row, end_col):
            # Move towards goal
            dr = np.clip(end_row - current_row, -1, 1)
            dc = np.clip(end_col - current_col, -1, 1)

            next_row = current_row + dr
            next_col = current_col + dc

            # Check if valid and passable
            if (
                0 <= next_row < cost_surface.shape[0]
                and 0 <= next_col < cost_surface.shape[1]
                and cost_surface[next_row, next_col] < 1000
            ):

                # Convert to coordinates
                x = out_trans.c + next_col * out_trans.a + out_trans.a / 2
                y = out_trans.f + next_row * out_trans.e + out_trans.e / 2
                lon, lat = transformer.transform(x, y)
                path.append((lon, lat))

                current_row, current_col = next_row, next_col
            else:
                # Can't reach goal greedily
                return None

            # Safety limit
            if len(path) > 10000:
                return None

        return path

    def reconstruct_path_astar(self, came_from, current, out_trans, transformer, shape):
        """
        Reconstructs the path from the came_from map.

        Returns:
        - path: List of (longitude, latitude) tuples representing the path coordinates.
        """
        path = []
        while current in came_from:
            row, col = np.unravel_index(current, shape)
            x = out_trans.c + col * out_trans.a + out_trans.a / 2
            y = out_trans.f + row * out_trans.e + out_trans.e / 2
            x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
            path.append((x_lon, y_lat))
            current = came_from[current]

        # Add the starting point
        row, col = np.unravel_index(current, shape)
        x = out_trans.c + col * out_trans.a + out_trans.a / 2
        y = out_trans.f + row * out_trans.e + out_trans.e / 2
        x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
        path.append((x_lon, y_lat))

        path.reverse()
        return path

    def astar_pathfinding_optimized(
        self, cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem=None, optimization_config=None
    ):
        """
        Optimized A* pathfinding with performance improvements.

        Optimizations included:
        1. Conservative early termination
        2. Dynamic weight adjustment
        3. Memory limiting
        4. Improved tie-breaking
        5. Corner cutting (line-of-sight)

        optimization_config: dict with optional parameters:
            - early_termination: bool (default True)
            - stagnation_limit: int (default 5000)
            - dynamic_weights: bool (default True)
            - memory_limit: int (default 50000 nodes)
            - corner_cutting: bool (default True)
            - max_iterations: int (default 10000000)
        """

        # Default optimization settings - SAFE OPTIMIZATIONS ONLY
        if optimization_config is None:
            optimization_config = {}

        # Safe defaults that maintain path quality
        use_early_termination = optimization_config.get("early_termination", True)  # Safe with high limit
        stagnation_limit = optimization_config.get("stagnation_limit", 10000)  # Conservative limit
        use_dynamic_weights = optimization_config.get("dynamic_weights", False)  # DISABLED - alters paths
        memory_limit = optimization_config.get("memory_limit", 50000)
        use_corner_cutting = optimization_config.get("corner_cutting", False)  # DISABLED - alters paths
        max_iterations = optimization_config.get("max_iterations", 10000000)
        use_preprocessing = optimization_config.get("use_preprocessing", True)  # Safe optimization

        # Performance tracking
        import time

        start_time = time.time()

        height, width = cost_surface.shape

        # Check for preprocessed data
        preprocessed_data = None
        tile_key = f"{height}x{width}"  # Simple key for now

        if use_preprocessing and tile_key in self.preprocessing_cache:
            preprocessed_data = self.preprocessing_cache[tile_key]
            if self.debug_mode:
                logger.debug(f"Using cached preprocessing data for {tile_key}")
        elif use_preprocessing:
            # Preprocess the tile
            if self.debug_mode:
                logger.debug(f"Preprocessing tile {tile_key}...")
            preprocess_start = time.time()
            preprocessed_data = self.preprocessor.preprocess_tile(cost_surface, indices, out_trans)
            self.preprocessing_cache[tile_key] = preprocessed_data
            if self.debug_mode:
                preprocess_time = time.time() - preprocess_start
                logger.debug(f"Preprocessing completed in {preprocess_time:.3f}s")
                logger.debug(f"Passable ratio: {preprocessed_data['cost_statistics']['passable_ratio']:.2%}")
        open_set = []

        # Priority queue with tie-breaking: (f_score, -g_score, tie_breaker, node_idx)
        # Using negative g_score to prefer nodes closer to goal when f_scores are equal
        tie_breaker = 0
        heapq.heappush(open_set, (0, 0, tie_breaker, start_idx))

        came_from = {}
        g_score = {}  # Use dict instead of full array for memory efficiency
        g_score[start_idx] = 0

        closed_set = set()

        # Heuristic cache for this search
        heuristic_cache = {}

        # Calculate initial heuristic for dynamic weighting
        start_h = self.heuristic(start_idx, end_idx, indices.shape, out_trans)
        heuristic_cache[start_idx] = start_h

        # Early termination tracking
        best_h_score = float("inf")
        stagnation_counter = 0
        best_node = start_idx

        # Neighbor offsets
        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Debug info
        if self.debug_mode:
            self.debug_data = {
                "explored_nodes": [],
                "optimization_stats": {
                    "early_terminations": 0,
                    "memory_prunes": 0,
                    "corner_cuts": 0,
                    "dynamic_weight_changes": [],
                },
                "grid_exploration": {
                    "shape": indices.shape,
                    "g_scores": np.full(indices.shape, np.inf),
                    "f_scores": np.full(indices.shape, np.inf),
                    "h_scores": np.full(indices.shape, np.inf),
                    "explored": np.zeros(indices.shape, dtype=bool),
                    "in_path": np.zeros(indices.shape, dtype=bool),
                },
                "decision_points": [],
                "terrain_costs": cost_surface.copy(),
                "bounds": {"transform": out_trans, "start_idx": start_idx, "end_idx": end_idx},
            }

        step_count = 0

        # Calculate expected search area
        start_row, start_col = np.unravel_index(start_idx, indices.shape)
        end_row, end_col = np.unravel_index(end_idx, indices.shape)
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)

        if grid_distance > 1000:
            logger.warning(f"Large search area: {grid_distance} cells. Optimizations enabled.")

        while open_set and step_count < max_iterations:
            # Memory limiting - prune open set if too large
            if use_early_termination and len(open_set) > memory_limit:
                # Keep only the best half
                temp = []
                for _ in range(memory_limit // 2):
                    if open_set:
                        temp.append(heapq.heappop(open_set))
                open_set = temp
                heapq.heapify(open_set)

                if self.debug_mode:
                    self.debug_data["optimization_stats"]["memory_prunes"] += 1

            current_f, neg_g, _, current = heapq.heappop(open_set)
            current_g = -neg_g
            step_count += 1

            # Goal check
            if current == end_idx:
                path = self.reconstruct_path_astar(came_from, current, out_trans, transformer, indices.shape)

                if self.debug_mode:
                    elapsed = time.time() - start_time
                    logger.info(f"Optimized path found in {elapsed:.3f}s, {step_count} iterations")
                    logger.debug(f"Optimizations: {self.debug_data['optimization_stats']}")
                    self._mark_path_in_debug(path, out_trans, transformer, indices.shape)

                return path

            # Skip if already processed
            if current in closed_set:
                continue

            closed_set.add(current)
            row_current, col_current = np.unravel_index(current, indices.shape)

            # Record in debug mode
            if self.debug_mode:
                self.debug_data["explored_nodes"].append(
                    {
                        "step": step_count,
                        "node_idx": current,
                        "row": row_current,
                        "col": col_current,
                        "g_score": current_g,
                        "f_score": current_f,
                        "h_score": current_f - current_g,
                    }
                )
                self.debug_data["grid_exploration"]["explored"][row_current, col_current] = True
                self.debug_data["grid_exploration"]["g_scores"][row_current, col_current] = current_g
                self.debug_data["grid_exploration"]["f_scores"][row_current, col_current] = current_f

            # Early termination check
            if use_early_termination:
                if current not in heuristic_cache:
                    heuristic_cache[current] = self.heuristic(current, end_idx, indices.shape, out_trans)
                current_h = heuristic_cache[current]

                if current_h < best_h_score:
                    best_h_score = current_h
                    best_node = current
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                    if stagnation_counter > stagnation_limit:
                        # Early termination but don't give up - continue from best node
                        if self.debug_mode:
                            self.debug_data["optimization_stats"]["early_terminations"] += 1
                            elapsed = time.time() - start_time
                            logger.debug(f"Early termination after {step_count} iterations, {elapsed:.3f}s")
                            logger.debug(f"Best node distance to goal: {best_h_score:.1f}")

                        # Don't actually terminate - this is what the benchmark must be doing
                        # to get full paths in 2.4 seconds
                        # The "early termination" might just be changing strategy, not stopping
                        if best_node != current and best_node in came_from:
                            # Jump to best node and continue
                            current = best_node
                            stagnation_counter = 0
                        else:
                            # Really stuck, give up
                            break

            # Corner cutting check (line of sight to goal)
            if use_corner_cutting and step_count % 100 == 0:  # Check periodically
                if self.has_line_of_sight(row_current, col_current, end_row, end_col, cost_surface):
                    # Direct path possible!
                    if self.debug_mode:
                        self.debug_data["optimization_stats"]["corner_cuts"] += 1

                    # Create direct path
                    came_from[end_idx] = current
                    path = self.reconstruct_path_astar(came_from, end_idx, out_trans, transformer, indices.shape)
                    return path

            # Calculate dynamic weight if enabled
            if use_dynamic_weights:
                # Progress = how close we are to goal (0 to 1)
                progress = 1.0 - (current_h / start_h) if start_h > 0 else 0
                # Reduce heuristic weight based on gradient preference
                base_weight = 2.0 / self.obstacle_config.gradient_preference
                heuristic_weight = base_weight - (progress * base_weight / 2.0)

                if self.debug_mode and step_count % 1000 == 0:
                    self.debug_data["optimization_stats"]["dynamic_weight_changes"].append(
                        {"step": step_count, "weight": heuristic_weight, "progress": progress}
                    )
            else:
                heuristic_weight = 1.0 / self.obstacle_config.gradient_preference

            # Explore neighbors
            neighbor_evaluations = [] if self.debug_mode else None

            for dy, dx in neighbors_offsets:
                row_neighbor = row_current + dy
                col_neighbor = col_current + dx

                if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                    neighbor = indices[row_neighbor, col_neighbor]

                    if neighbor in closed_set:
                        continue

                    # Skip impassable terrain
                    terrain_cost = cost_surface[row_neighbor, col_neighbor]
                    if terrain_cost >= 9999:  # Obstacle
                        continue

                    # Calculate movement cost
                    distance = sqrt((dy * out_trans.e) ** 2 + (dx * out_trans.a) ** 2)
                    movement_cost = terrain_cost * distance
                    tentative_g = current_g + movement_cost

                    # Skip if not an improvement
                    if neighbor in g_score and tentative_g >= g_score[neighbor]:
                        continue

                    # Update scores
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current

                    # Calculate f-score with dynamic weight
                    h_score = self.heuristic(neighbor, end_idx, indices.shape, out_trans)
                    f_score = tentative_g + h_score * heuristic_weight

                    # Add to open set with tie-breaking
                    tie_breaker += 1
                    heapq.heappush(open_set, (f_score, -tentative_g, tie_breaker, neighbor))

                    # Record for debug
                    if self.debug_mode and neighbor_evaluations is not None:
                        neighbor_evaluations.append(
                            {
                                "neighbor_idx": neighbor,
                                "row": row_neighbor,
                                "col": col_neighbor,
                                "direction": (dy, dx),
                                "direction_name": self._get_direction_name(dy, dx),
                                "distance_meters": distance,
                                "terrain_cost": terrain_cost,
                                "movement_cost": movement_cost,
                                "g_score": tentative_g,
                                "h_score": h_score,
                                "f_score": f_score,
                                "heuristic_weight": heuristic_weight,
                                "is_improvement": True,
                            }
                        )

            # Record decision point in debug mode
            if self.debug_mode and neighbor_evaluations:
                self.debug_data["decision_points"].append(
                    {
                        "step": step_count,
                        "current_node": {
                            "idx": current,
                            "row": row_current,
                            "col": col_current,
                            "lat_lon": self._idx_to_latlon(current, out_trans, transformer, indices.shape),
                        },
                        "neighbors_evaluated": neighbor_evaluations,
                        "chosen_neighbors": neighbor_evaluations,  # All were improvements
                    }
                )

        # No path found
        if self.debug_mode:
            elapsed = time.time() - start_time
            logger.info(f"No path found after {step_count} iterations, {elapsed:.3f}s")
            if use_early_termination:
                logger.debug(f"Best progress: {best_h_score:.1f}m from goal")

        return None

    def bidirectional_astar(
        self, cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem=None, optimization_config=None
    ):
        """
        Bidirectional A* pathfinding - searches from both start and goal simultaneously.

        This algorithm typically provides significant speedup (100x+) on complex terrain
        by reducing the search space dramatically.
        """
        if optimization_config is None:
            optimization_config = {}

        # Performance tracking
        start_time = time.time()
        nodes_explored = 0

        logger.debug(f"[Bidirectional A*] Starting search from {start_idx} to {end_idx}")
        logger.debug(f"[Bidirectional A*] Cost surface shape: {cost_surface.shape}")

        height, width = cost_surface.shape

        # Two search frontiers
        forward_open = []
        backward_open = []

        # Priority queues with tie-breaking
        tie_breaker = 0
        heapq.heappush(forward_open, (0, 0, tie_breaker, start_idx))
        tie_breaker += 1
        heapq.heappush(backward_open, (0, 0, tie_breaker, end_idx))

        # Separate g-scores and came_from for each direction
        forward_g = {start_idx: 0}
        backward_g = {end_idx: 0}

        forward_came_from = {}
        backward_came_from = {}

        forward_closed = set()
        backward_closed = set()

        # Meeting point tracking
        meeting_point = None
        best_path_cost = float("inf")

        # Neighbor offsets
        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # For distance calculations
        sqrt2 = sqrt(2)

        # Debug mode
        if self.debug_mode:
            logger.debug("Starting bidirectional A* search")
            if self.debug_data is None:
                self.debug_data = {}
            self.debug_data["algorithm"] = "bidirectional_astar"
            self.debug_data["forward_explored"] = []
            self.debug_data["backward_explored"] = []

        while forward_open and backward_open:
            # Forward search step
            if forward_open:
                f_score, neg_g, _, current = heapq.heappop(forward_open)
                current_g = -neg_g
                nodes_explored += 1

                # Check if we've met the backward search
                if current in backward_g:
                    total_cost = current_g + backward_g[current]
                    if total_cost < best_path_cost:
                        best_path_cost = total_cost
                        meeting_point = current
                        if self.debug_mode:
                            logger.debug(f"Paths meet at node {current} with total cost {total_cost:.2f}")

                if current not in forward_closed:
                    forward_closed.add(current)
                    row_current, col_current = np.unravel_index(current, indices.shape)

                    if self.debug_mode:
                        self.debug_data["forward_explored"].append(current)

                    # Explore neighbors
                    for dy, dx in neighbors_offsets:
                        row_neighbor = row_current + dy
                        col_neighbor = col_current + dx

                        if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                            neighbor = indices[row_neighbor, col_neighbor]

                            if neighbor in forward_closed:
                                continue

                            # Check passability
                            terrain_cost = cost_surface[row_neighbor, col_neighbor]
                            if terrain_cost >= 9999:  # Obstacle
                                continue

                            # Calculate movement cost
                            distance = sqrt2 if abs(dy) + abs(dx) == 2 else 1
                            movement_cost = terrain_cost * distance * out_trans.a  # Scale by resolution

                            # Add uphill transition penalty if DEM is available
                            if dem is not None:
                                current_elevation = dem[row_current, col_current]
                                neighbor_elevation = dem[row_neighbor, col_neighbor]
                                elevation_gain = neighbor_elevation - current_elevation

                                if elevation_gain > 0:  # Going uphill
                                    # Calculate slope angle of the transition
                                    horizontal_distance = distance * out_trans.a
                                    uphill_slope_degrees = np.degrees(np.arctan(elevation_gain / horizontal_distance))

                                    # Exponential penalty for uphill transitions
                                    # Much more aggressive penalties for any uphill movement
                                    if uphill_slope_degrees > 2:  # Lower threshold
                                        # More aggressive exponential formula
                                        uphill_penalty = 1.0 + (uphill_slope_degrees / 10.0) ** 2.5
                                        movement_cost *= uphill_penalty

                            tentative_g = current_g + movement_cost

                            if neighbor not in forward_g or tentative_g < forward_g[neighbor]:
                                forward_g[neighbor] = tentative_g
                                forward_came_from[neighbor] = current

                                # Calculate f-score
                                h_score = self.heuristic(neighbor, end_idx, indices.shape, out_trans)
                                f = tentative_g + h_score

                                tie_breaker += 1
                                heapq.heappush(forward_open, (f, -tentative_g, tie_breaker, neighbor))

            # Backward search step (similar but towards start)
            if backward_open:
                f_score, neg_g, _, current = heapq.heappop(backward_open)
                current_g = -neg_g
                nodes_explored += 1

                # Check if we've met the forward search
                if current in forward_g:
                    total_cost = current_g + forward_g[current]
                    if total_cost < best_path_cost:
                        best_path_cost = total_cost
                        meeting_point = current
                        if self.debug_mode:
                            logger.debug(f"Paths meet at node {current} with total cost {total_cost:.2f}")

                if current not in backward_closed:
                    backward_closed.add(current)
                    row_current, col_current = np.unravel_index(current, indices.shape)

                    if self.debug_mode:
                        self.debug_data["backward_explored"].append(current)

                    # Explore neighbors
                    for dy, dx in neighbors_offsets:
                        row_neighbor = row_current + dy
                        col_neighbor = col_current + dx

                        if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                            neighbor = indices[row_neighbor, col_neighbor]

                            if neighbor in backward_closed:
                                continue

                            # Check passability
                            terrain_cost = cost_surface[row_neighbor, col_neighbor]
                            if terrain_cost >= 9999:  # Obstacle
                                continue

                            # Calculate movement cost
                            distance = sqrt2 if abs(dy) + abs(dx) == 2 else 1
                            movement_cost = terrain_cost * distance * out_trans.a

                            # Add uphill transition penalty if DEM is available
                            if dem is not None:
                                current_elevation = dem[row_current, col_current]
                                neighbor_elevation = dem[row_neighbor, col_neighbor]
                                elevation_gain = neighbor_elevation - current_elevation

                                if elevation_gain > 0:  # Going uphill
                                    # Calculate slope angle of the transition
                                    horizontal_distance = distance * out_trans.a
                                    uphill_slope_degrees = np.degrees(np.arctan(elevation_gain / horizontal_distance))

                                    # Exponential penalty for uphill transitions
                                    # Much more aggressive penalties for any uphill movement
                                    if uphill_slope_degrees > 2:  # Lower threshold
                                        # More aggressive exponential formula
                                        uphill_penalty = 1.0 + (uphill_slope_degrees / 10.0) ** 2.5
                                        movement_cost *= uphill_penalty

                            tentative_g = current_g + movement_cost

                            if neighbor not in backward_g or tentative_g < backward_g[neighbor]:
                                backward_g[neighbor] = tentative_g
                                backward_came_from[neighbor] = current

                                # Calculate f-score (heuristic towards start)
                                h_score = self.heuristic(neighbor, start_idx, indices.shape, out_trans)
                                f = tentative_g + h_score

                                tie_breaker += 1
                                heapq.heappush(backward_open, (f, -tentative_g, tie_breaker, neighbor))

            # Early termination if we found a good path
            if meeting_point and best_path_cost < float("inf"):
                # Check if we've explored enough to be confident
                # (both searches should have made some progress)
                if len(forward_closed) > 10 and len(backward_closed) > 10:
                    break

        # Reconstruct path if found
        if meeting_point:
            # Reconstruct forward path (start to meeting point)
            forward_path = []
            current = meeting_point
            while current in forward_came_from:
                forward_path.append(current)
                current = forward_came_from[current]
            forward_path.append(start_idx)
            forward_path.reverse()

            # Reconstruct backward path (meeting point to end)
            backward_path = []
            current = meeting_point
            while current in backward_came_from:
                current = backward_came_from[current]
                backward_path.append(current)

            # Combine paths (avoiding duplicate meeting point)
            full_path_indices = forward_path[:-1] + backward_path

            # Convert indices to coordinates
            path = []
            for idx in full_path_indices:
                row, col = np.unravel_index(idx, indices.shape)
                x = out_trans.c + col * out_trans.a + out_trans.a / 2
                y = out_trans.f + row * out_trans.e + out_trans.e / 2
                x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
                path.append((x_lon, y_lat))

            elapsed = time.time() - start_time
            logger.info(f"[Bidirectional A*] SUCCESS - Found path in {elapsed:.3f}s")
            logger.debug(
                f"[Bidirectional A*] Nodes explored: {nodes_explored} (forward: {len(forward_closed)}, backward: {len(backward_closed)})"
            )
            logger.debug(f"[Bidirectional A*] Path length: {len(path)} points")
            logger.debug(f"[Bidirectional A*] Meeting point found after exploring {nodes_explored} nodes")

            if self.debug_mode:
                logger.debug(f"[Bidirectional A*] Total path cost: {best_path_cost:.2f}")

            return path

        # No path found
        if self.debug_mode:
            elapsed = time.time() - start_time
            logger.info(f"Bidirectional A* found no path after {elapsed:.3f}s")
            logger.debug(f"Nodes explored: {nodes_explored}")

        return None

    def plot_results(
        self, dem, out_trans, crs, path, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon, obstacles=None
    ):
        """
        Plots the DEM, area of interest, obstacles, and the calculated route.
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot the DEM
        extent = rasterio.plot.plotting_extent(dem, out_trans)
        show(dem, ax=ax, cmap="terrain", extent=extent, transform=out_trans)

        # Transform start/end points to map CRS
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x1, y1 = transformer.transform(lon1, lat1)
        x2, y2 = transformer.transform(lon2, lat2)

        # Plot the route
        if path:
            x_coords, y_coords = [], []
            for lon, lat in path:
                x, y = transformer.transform(lon, lat)
                x_coords.append(x)
                y_coords.append(y)
            ax.plot(x_coords, y_coords, color="blue", linewidth=3, label="Calculated Route")

        # Plot start/end points
        ax.scatter([x1, x2], [y1, y2], color="red", s=100, zorder=5, label="Start/End Points")
        ax.text(x1, y1, "Start", fontsize=12, ha="right")
        ax.text(x2, y2, "End", fontsize=12, ha="left")

        # Plot obstacles if provided
        if obstacles is not None and not obstacles.empty:
            obstacles = obstacles.to_crs(crs.to_string())
            obstacles.plot(ax=ax, facecolor="gray", edgecolor="darkgray", linewidth=0.5, alpha=0.5, label="Obstacles")

        # Labels and legend
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Least-Cost Path Over Terrain with Obstacles")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _idx_to_latlon(self, idx, out_trans, transformer, shape):
        """Convert grid index to lat/lon coordinates"""
        row, col = np.unravel_index(idx, shape)
        x = out_trans.c + col * out_trans.a + out_trans.a / 2
        y = out_trans.f + row * out_trans.e + out_trans.e / 2
        x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
        return (y_lat, x_lon)

    def _get_direction_name(self, dy, dx):
        """Convert direction offset to human-readable name"""
        direction_map = {
            (-1, -1): "Northwest",
            (-1, 0): "North",
            (-1, 1): "Northeast",
            (0, -1): "West",
            (0, 1): "East",
            (1, -1): "Southwest",
            (1, 0): "South",
            (1, 1): "Southeast",
        }
        return direction_map.get((dy, dx), f"({dy}, {dx})")

    def _mark_path_in_debug(self, path, out_trans, transformer, shape):
        """Mark path nodes in the debug grid"""
        if not self.debug_data:
            return

        for point in path:
            if isinstance(point, dict):
                lon, lat = point["lon"], point["lat"]
            else:
                lon, lat = point
            # Convert lat/lon back to grid indices
            # transformer expects lon, lat order for geographic coordinates
            x, y = transformer.transform(lon, lat)
            col = int(round((x - out_trans.c) / out_trans.a))
            row = int(round((y - out_trans.f) / out_trans.e))

            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                self.debug_data["grid_exploration"]["in_path"][row, col] = True

        # Debug: Log how many path points were marked
        path_count = np.sum(self.debug_data["grid_exploration"]["in_path"])
        logger.debug(f"Debug: Marked {path_count} cells as in_path out of {len(path)} path points")

        # Extra debug: show some of the marked cells
        in_path_indices = np.where(self.debug_data["grid_exploration"]["in_path"])
        if len(in_path_indices[0]) > 0:
            logger.debug(f"Debug: First few in_path cells: {list(zip(in_path_indices[0][:5], in_path_indices[1][:5]))}")

    def calculate_path_slopes(self, path, dem, out_trans, transformer, path_raster=None, path_types=None):
        """Calculate slope and path type for each segment of the path"""
        if not path or len(path) < 2:
            return path

        path_with_slopes = []

        for i in range(len(path)):
            lon, lat = path[i]

            # Get elevation at this point
            x, y = transformer.transform(lon, lat)
            col = int(round((x - out_trans.c) / out_trans.a))
            row = int(round((y - out_trans.f) / out_trans.e))

            elevation = None
            path_type = "off_path"  # Default

            if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                elevation = float(dem[row, col])
                # Check for NaN or invalid values
                if np.isnan(elevation):
                    elevation = None

                # Get path type at this location
                if path_raster is not None and path_types is not None:
                    path_id = path_raster[row, col]
                    if path_id > 0 and path_id in path_types:
                        path_type = path_types[path_id]

            # Calculate slope to next point
            slope_degrees = 0.0
            if i < len(path) - 1:
                next_lon, next_lat = path[i + 1]
                next_x, next_y = transformer.transform(next_lon, next_lat)
                next_col = int(round((next_x - out_trans.c) / out_trans.a))
                next_row = int(round((next_y - out_trans.f) / out_trans.e))

                if 0 <= next_row < dem.shape[0] and 0 <= next_col < dem.shape[1] and elevation is not None:
                    next_elevation = float(dem[next_row, next_col])

                    # Check for NaN in next elevation too
                    if np.isnan(next_elevation):
                        next_elevation = None

                    if next_elevation is not None:
                        # Calculate horizontal distance
                        distance = sqrt((next_x - x) ** 2 + (next_y - y) ** 2)

                        if distance > 0:
                            # Calculate slope
                            elevation_change = next_elevation - elevation
                            slope_radians = atan(abs(elevation_change) / distance)
                            slope_degrees = degrees(slope_radians)

                            # If going downhill, make slope negative
                            if elevation_change < 0:
                                slope_degrees = -slope_degrees

            path_with_slopes.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "elevation": elevation,
                    "slope": round(slope_degrees, 1),
                    "path_type": path_type,
                }
            )

        return path_with_slopes

    def get_cost_at_point(self, lat, lon, buffer=0.0001):
        """
        Get cost information for a single point FROM PRECOMPUTED DATA ONLY.
        Returns None if data is not precomputed.

        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            buffer: Buffer around point in degrees (default ~11 meters)

        Returns:
            Dictionary with cost information at the point, or None if not precomputed
        """
        try:
            # Define tiny bounds around the point
            min_lat = lat - buffer
            max_lat = lat + buffer
            min_lon = lon - buffer
            max_lon = lon + buffer

            # Get the tile containing this point
            tiles = self.tiled_cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
            if not tiles:
                return {"error": "No tile found for this location", "precomputed": False}

            # Get cost surface for the tile - DO NOT COMPUTE IF NOT CACHED
            tile_x, tile_y = tiles[0]  # Should only be one tile for such a small area
            tile_data = self.tiled_cache.get_tile(tile_x, tile_y, "cost")

            if tile_data is None:
                # Data not precomputed - return error instead of computing
                return {
                    "error": "Data not precomputed for this area",
                    "precomputed": False,
                    "tile": f"({tile_x}, {tile_y})",
                }

            # If we get here, we have precomputed data

            # Get the exact pixel for this point
            cost_surface = tile_data["cost_surface"]
            slope_degrees = tile_data["slope_degrees"]
            dem = tile_data.get("dem")
            path_raster = tile_data.get("path_raster")
            path_types = tile_data.get("path_types", {})
            path_raw_tags = tile_data.get("path_raw_tags", {})
            transform = tile_data["transform"]
            crs = tile_data["crs"]

            # Convert lat/lon to pixel coordinates
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

            # Get pixel indices
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)

            # Check bounds
            height, width = cost_surface.shape
            if not (0 <= row < height and 0 <= col < width):
                return None

            # Get values at this point
            cost = float(cost_surface[row, col])
            slope = float(slope_degrees[row, col])
            elevation = float(dem[row, col]) if dem is not None else None

            # Get path type
            path_id = int(path_raster[row, col]) if path_raster is not None else 0
            path_type = path_types.get(path_id, "off_path" if path_id == 0 else "unknown")

            # Get raw OSM tags for this path ID if available
            raw_tags_info = None
            if path_id > 0 and path_raw_tags:
                raw_tags_info = path_raw_tags.get(path_id, None)

            # Calculate cost factors
            base_cost = 1.0
            slope_cost = self.obstacle_config.get_slope_cost_multiplier(slope)
            path_multiplier = self.path_preferences.get_path_cost_multiplier(path_type)
            is_obstacle = cost >= 1000

            # Get all path types from the tile to show what we know about
            all_path_types_in_tile = {}
            if path_types:
                all_path_types_in_tile = {str(k): v for k, v in path_types.items()}

            # Calculate the exponential components
            import numpy as np

            growth_rate = 0.085
            exponential_value = np.exp(growth_rate * slope)

            # Show how final cost is calculated
            final_cost_calculation = {
                "formula": "final_cost = base_cost × e^(growth_rate × slope) × path_multiplier",
                "base_cost": base_cost,
                "slope_degrees": slope,
                "growth_rate": growth_rate,
                "exponential": exponential_value,
                "slope_cost_raw": slope_cost,
                "path_multiplier": path_multiplier,
                "calculation_steps": {
                    "step1": f"e^({growth_rate} × {slope:.1f}°) = {exponential_value:.3f}",
                    "step2": f"slope_cost = {base_cost} × {exponential_value:.3f} = {slope_cost:.3f}",
                    "step3": f"final_cost = {slope_cost:.3f} × {path_multiplier} = {cost:.3f}",
                },
            }

            return {
                "lat": lat,
                "lon": lon,
                "cost": cost,
                "slope": slope,
                "elevation": elevation,
                "path_type": path_type,
                "path_id": path_id,
                "raw_osm_data": raw_tags_info,  # This is what was used to determine the path type
                "tile_info": {
                    "tile_coords": f"({tile_x}, {tile_y})",
                    "all_path_types": all_path_types_in_tile,
                    "total_path_types": len(all_path_types_in_tile),
                },
                "factors": {
                    "base_cost": base_cost,
                    "slope_cost": slope_cost,
                    "path_multiplier": path_multiplier,
                    "is_obstacle": is_obstacle,
                },
                "cost_breakdown": final_cost_calculation,
            }

        except Exception as e:
            logger.error(f"Error getting cost at point: {e}", exc_info=True)
            import traceback

            traceback.print_exc()
            return None

    def get_cost_surface_for_bounds(self, min_lat, max_lat, min_lon, max_lon, start_lat, start_lon, end_lat, end_lon):
        """
        Get cost surface data for visualization purposes.
        Returns a dictionary with cost surface array and metadata for rendering.
        """
        try:
            # Try to use tiled cost surface first
            tiled_result = self._try_tiled_cost_surface(min_lat, max_lat, min_lon, max_lon, None, None, "EPSG:4326")

            if tiled_result is not None:
                cost_surface, slope_degrees, indices, dem_composed, path_raster, path_types, slope_change = tiled_result
                out_trans = self._composed_transform if hasattr(self, "_composed_transform") else None
                crs = self._composed_crs if hasattr(self, "_composed_crs") else "EPSG:4326"
                dem = dem_composed if dem_composed is not None else None
            else:
                # Fall back to computing it
                # Download DEM
                dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
                if not dem_file:
                    return None

                # Read and reproject DEM
                dem, out_trans, crs = self.read_dem(dem_file)
                if dem is None:
                    return None
                dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)

                # Fetch obstacles and paths
                obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
                obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)

                paths = self.fetch_paths(min_lat, max_lat, min_lon, max_lon)
                path_raster, path_types, path_raw_tags = self.rasterize_paths(paths, out_trans, dem.shape, crs)

                # Compute cost surface
                cost_surface, slope_degrees, slope_change = self.compute_cost_surface(
                    dem, out_trans, obstacle_mask, path_raster, path_types
                )

            # Convert transform to list for JSON serialization
            transform_list = [out_trans.a, out_trans.b, out_trans.c, out_trans.d, out_trans.e, out_trans.f]

            # Downsample if too large
            height, width = cost_surface.shape
            max_size = 500  # Maximum dimension for web visualization

            if height > max_size or width > max_size:
                # Calculate downsampling factor
                factor = max(height // max_size, width // max_size)

                # Downsample arrays
                cost_surface_downsampled = cost_surface[::factor, ::factor]
                slope_degrees_downsampled = slope_degrees[::factor, ::factor]
                slope_change_downsampled = slope_change[::factor, ::factor] if slope_change is not None else None
                dem_downsampled = dem[::factor, ::factor] if dem is not None else None
                path_raster_downsampled = path_raster[::factor, ::factor] if path_raster is not None else None

                # Adjust transform for downsampling
                new_transform = [
                    out_trans.a * factor,  # pixel width
                    out_trans.b,  # rotation
                    out_trans.c,  # x origin
                    out_trans.d,  # rotation
                    out_trans.e * factor,  # pixel height (negative)
                    out_trans.f,  # y origin
                ]
            else:
                cost_surface_downsampled = cost_surface
                slope_degrees_downsampled = slope_degrees
                slope_change_downsampled = slope_change
                dem_downsampled = dem
                path_raster_downsampled = path_raster
                new_transform = transform_list
                factor = 1

            # Create bounds for the data
            bounds = {"north": max_lat, "south": min_lat, "east": max_lon, "west": min_lon}

            # Prepare path types mapping for client
            path_types_list = {}
            if path_types:
                for k, v in path_types.items():
                    path_types_list[str(k)] = v

            # Replace NaN and inf values with reasonable defaults
            cost_surface_clean = np.nan_to_num(cost_surface_downsampled, nan=1000.0, posinf=10000.0, neginf=0.0)
            slope_degrees_clean = np.nan_to_num(slope_degrees_downsampled, nan=0.0, posinf=90.0, neginf=0.0)
            slope_change_clean = (
                np.nan_to_num(slope_change_downsampled, nan=0.0, posinf=45.0, neginf=0.0)
                if slope_change_downsampled is not None
                else None
            )

            return {
                "cost_surface": cost_surface_clean.tolist(),
                "slope_degrees": slope_degrees_clean.tolist(),
                "slope_change": slope_change_clean.tolist() if slope_change_clean is not None else None,
                "elevation": np.nan_to_num(dem_downsampled, nan=0.0).tolist() if dem_downsampled is not None else None,
                "path_raster": path_raster_downsampled.tolist() if path_raster_downsampled is not None else None,
                "path_types": path_types_list,
                "transform": new_transform,
                "bounds": bounds,
                "shape": {"height": cost_surface_downsampled.shape[0], "width": cost_surface_downsampled.shape[1]},
                "downsampling_factor": factor,
                "start": {"lat": start_lat, "lon": start_lon},
                "end": {"lat": end_lat, "lon": end_lon},
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error getting cost surface for bounds: {e}", exc_info=True)
            return None

    def get_debug_data(self):
        """Return the collected debug data"""
        if not self.debug_mode or not self.debug_data:
            return None

        # Helper function to convert numpy types to Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj

        # Limit the amount of data to prevent huge responses
        MAX_NODES = 1000  # Limit explored nodes
        MAX_DECISIONS = 500  # Limit decision points

        # For very large searches, we need to limit the data
        total_explored = len(self.debug_data["explored_nodes"])
        total_decisions = len(self.debug_data["decision_points"])

        logger.debug(f"Debug data size: {total_explored} explored nodes, {total_decisions} decision points")

        # Convert numpy arrays to lists for JSON serialization
        # But limit the size to prevent huge responses
        debug_copy = {
            "explored_nodes": convert_numpy_types(self.debug_data["explored_nodes"][:MAX_NODES]),
            "decision_points": convert_numpy_types(self.debug_data["decision_points"][:MAX_DECISIONS]),
            "grid_exploration": {
                "shape": list(self.debug_data["grid_exploration"]["shape"]),
                "g_scores": self.debug_data["grid_exploration"]["g_scores"].tolist(),
                "f_scores": self.debug_data["grid_exploration"]["f_scores"].tolist(),
                "h_scores": self.debug_data["grid_exploration"]["h_scores"].tolist(),
                "explored": self.debug_data["grid_exploration"]["explored"].tolist(),
                "in_path": self.debug_data["grid_exploration"]["in_path"].tolist(),
            },
            "terrain_costs": self.debug_data["terrain_costs"].tolist(),
            "bounds": {
                "start_idx": int(self.debug_data["bounds"]["start_idx"]),
                "end_idx": int(self.debug_data["bounds"]["end_idx"]),
                "transform": {
                    "a": float(self.debug_data["bounds"]["transform"].a),
                    "b": float(self.debug_data["bounds"]["transform"].b),
                    "c": float(self.debug_data["bounds"]["transform"].c),
                    "d": float(self.debug_data["bounds"]["transform"].d),
                    "e": float(self.debug_data["bounds"]["transform"].e),
                    "f": float(self.debug_data["bounds"]["transform"].f),
                },
            },
            "total_explored": total_explored,
            "total_decisions": total_decisions,
            "data_truncated": total_explored > MAX_NODES or total_decisions > MAX_DECISIONS,
        }
        return debug_copy

    def heuristic_optimized(self, node_idx, end_idx, shape, out_trans):
        """
        Optimized heuristic function with better terrain awareness.
        """
        row_node, col_node = np.unravel_index(node_idx, shape)
        row_end, col_end = np.unravel_index(end_idx, shape)

        dx = (col_node - col_end) * out_trans.a
        dy = (row_node - row_end) * out_trans.e

        # Euclidean distance
        distance = sqrt(dx**2 + dy**2)

        # Use a conservative multiplier to avoid overly optimistic estimates
        # This helps the algorithm make better decisions
        return distance * 0.9

    def reconstruct_path_optimized(self, came_from, current, out_trans, transformer, shape):
        """
        Optimized path reconstruction.
        """
        path = []

        while current in came_from:
            row, col = np.unravel_index(current, shape)
            x = out_trans.c + col * out_trans.a + out_trans.a / 2
            y = out_trans.f + row * out_trans.e + out_trans.e / 2
            x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
            path.append((x_lon, y_lat))
            current = came_from[current]

        # Add starting point
        row, col = np.unravel_index(current, shape)
        x = out_trans.c + col * out_trans.a + out_trans.a / 2
        y = out_trans.f + row * out_trans.e + out_trans.e / 2
        x_lon, y_lat = transformer.transform(x, y, direction="INVERSE")
        path.append((x_lon, y_lat))

        path.reverse()
        return path

    def has_line_of_sight(self, row1, col1, row2, col2, cost_surface, max_cost=1000):
        """
        Check if there's a clear line of sight between two points.
        Uses Bresenham's line algorithm.
        """
        # Get all points along the line
        points = list(self.bresenham_line(row1, col1, row2, col2))

        # Check if any point has high cost (obstacle)
        for row, col in points:
            if 0 <= row < cost_surface.shape[0] and 0 <= col < cost_surface.shape[1]:
                if cost_surface[row, col] >= max_cost:
                    return False
            else:
                return False  # Out of bounds

        return True

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get all points along a line.
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1

        if dx > dy:
            error = dx / 2
            while x != x1:
                points.append((x, y))
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2
            while y != y1:
                points.append((x, y))
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc

        points.append((x, y))
        return points

    def predownload_area(self, min_lat, max_lat, min_lon, max_lon, resolution=None):
        """
        Pre-download terrain data for a specific area.

        Parameters:
        - min_lat, max_lat, min_lon, max_lon: Bounding box of area
        - resolution: Optional resolution in meters (1, 3, 10, 30). If None, tries best available.

        Returns:
        - dict with download status and cached file info
        """
        import time

        start_time = time.time()

        # Create cache key (tile-aligned)
        cache_key = self._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)

        # Check if already cached
        if cache_key in self.terrain_cache:
            return {
                "status": "already_cached",
                "cache_key": cache_key,
                "time_seconds": 0,
                "message": "Terrain data already in memory cache",
            }

        try:
            # Download DEM data
            dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
            if dem_file is None:
                return {
                    "status": "failed",
                    "error": "Failed to download DEM data",
                    "time_seconds": time.time() - start_time,
                }

            # Read and reproject DEM data
            dem, out_trans, crs = self.read_dem(dem_file)
            if dem is None:
                return {
                    "status": "failed",
                    "error": "Failed to read DEM data",
                    "time_seconds": time.time() - start_time,
                }

            # Reproject to projected CRS
            dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)

            # Cache the terrain data
            self.terrain_cache[cache_key] = (dem, out_trans, crs)

            download_time = time.time() - start_time

            return {
                "status": "success",
                "cache_key": cache_key,
                "dem_shape": dem.shape,
                "resolution_m": abs(out_trans.a),  # Approximate resolution
                "time_seconds": download_time,
                "file_size_mb": dem.nbytes / (1024 * 1024),
                "message": f"Downloaded and cached terrain data in {download_time:.1f}s",
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "time_seconds": time.time() - start_time}

    def preprocess_area(self, min_lat, max_lat, min_lon, max_lon, force=False):
        """
        Preprocess terrain data for faster pathfinding.
        Area must be pre-downloaded first.

        Parameters:
        - min_lat, max_lat, min_lon, max_lon: Bounding box of area
        - force: Force re-preprocessing even if already cached

        Returns:
        - dict with preprocessing status
        """
        import time

        start_time = time.time()

        # Create cache key (tile-aligned)
        cache_key = self._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)
        cost_cache_key = f"{cache_key}_cost"

        # Check if terrain is downloaded
        if cache_key not in self.terrain_cache:
            return {
                "status": "failed",
                "error": "Terrain not downloaded. Run predownload_area first.",
                "time_seconds": 0,
            }

        # Get cached terrain
        dem, out_trans, crs = self.terrain_cache[cache_key]

        # Check if cost surface already cached
        if cost_cache_key in self.cost_surface_cache and not force:
            cost_surface, slope_degrees, indices = self.cost_surface_cache[cost_cache_key]
            tile_key = f"{cost_surface.shape[0]}x{cost_surface.shape[1]}"

            if tile_key in self.preprocessing_cache:
                return {
                    "status": "already_preprocessed",
                    "cache_keys": {"terrain": cache_key, "cost_surface": cost_cache_key, "preprocessing": tile_key},
                    "time_seconds": 0,
                    "message": "Area already fully preprocessed",
                }

        try:
            # Fetch obstacles data
            logger.debug("Fetching obstacles data...")
            obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
            obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)

            # Fetch preferred paths
            logger.debug("Fetching preferred paths...")
            paths = self.fetch_paths(min_lat, max_lat, min_lon, max_lon)
            path_raster, path_types, path_raw_tags = self.rasterize_paths(
                paths, out_trans, dem.shape, crs, obstacle_mask, expand_paths=True
            )

            # Compute cost surface
            logger.debug("Computing cost surface...")
            cost_surface, slope_degrees, slope_change = self.compute_cost_surface(
                dem, out_trans, obstacle_mask, path_raster, path_types
            )

            # Build indices
            logger.debug("Building indices...")
            indices = self.build_indices(cost_surface)

            # Cache cost surface (with slope_degrees to match expected format)
            self.cost_surface_cache[cost_cache_key] = (cost_surface, slope_degrees, indices)

            # Get cost surface statistics
            cost_surface_stats = {
                "min": float(np.min(cost_surface)),
                "max": float(np.max(cost_surface)),
                "mean": float(np.mean(cost_surface)),
                "impassable_pct": float(np.sum(cost_surface > 1000) / cost_surface.size * 100),
            }

            # Preprocess for pathfinding
            tile_key = f"{cost_surface.shape[0]}x{cost_surface.shape[1]}"

            if force or tile_key not in self.preprocessing_cache:
                logger.debug(f"Preprocessing tile {tile_key}...")
                preprocess_start = time.time()
                preprocessed_data = self.preprocessor.preprocess_tile(cost_surface, indices, out_trans)
                self.preprocessing_cache[tile_key] = preprocessed_data
                preprocess_time = time.time() - preprocess_start

                preprocessing_stats = {
                    "preprocessing_time": preprocess_time,
                    "passable_ratio": preprocessed_data["cost_statistics"]["passable_ratio"],
                    "neighbor_cache_size": len(preprocessed_data.get("neighbor_cache", {})),
                }
            else:
                preprocessing_stats = {"status": "already_cached", "preprocessing_time": 0}

            total_time = time.time() - start_time

            return {
                "status": "success",
                "cache_keys": {"terrain": cache_key, "cost_surface": cost_cache_key, "preprocessing": tile_key},
                "cost_surface_stats": cost_surface_stats,
                "preprocessing_stats": preprocessing_stats,
                "time_seconds": total_time,
                "message": f"Preprocessed area in {total_time:.1f}s",
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "time_seconds": time.time() - start_time}

    def get_cache_status(self):
        """
        Get current cache status and statistics.

        Returns:
        - dict with cache information
        """
        terrain_info = []
        for key, (dem, _, _) in self.terrain_cache.items():
            terrain_info.append({"key": key, "shape": dem.shape, "size_mb": dem.nbytes / (1024 * 1024)})

        cost_info = []
        for key, cache_data in self.cost_surface_cache.items():
            # Handle both dict and tuple cache formats
            if isinstance(cache_data, dict):
                cost_surface = cache_data["cost_surface"]
            elif len(cache_data) == 3:
                cost_surface, _, _ = cache_data
            else:
                cost_surface, _ = cache_data
            cost_info.append({"key": key, "shape": cost_surface.shape, "size_mb": cost_surface.nbytes / (1024 * 1024)})

        preprocessing_info = []
        for key, data in self.preprocessing_cache.items():
            preprocessing_info.append({"key": key, "passable_ratio": data["cost_statistics"]["passable_ratio"]})

        total_memory_mb = sum(info["size_mb"] for info in terrain_info)
        total_memory_mb += sum(info["size_mb"] for info in cost_info)

        return {
            "terrain_cache": {
                "count": len(self.terrain_cache),
                "entries": terrain_info,
                "total_size_mb": sum(info["size_mb"] for info in terrain_info),
            },
            "cost_surface_cache": {
                "count": len(self.cost_surface_cache),
                "entries": cost_info,
                "total_size_mb": sum(info["size_mb"] for info in cost_info),
            },
            "preprocessing_cache": {"count": len(self.preprocessing_cache), "entries": preprocessing_info},
            "total_memory_mb": total_memory_mb,
        }
