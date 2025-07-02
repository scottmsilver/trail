import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.features import rasterize
# import matplotlib.pyplot as plt  # Not needed in backend
from pyproj import Transformer
import requests
from math import radians, tan, cos, pi, sqrt, log, atan, degrees
from tqdm import tqdm
from shapely.geometry import box
import geopandas as gpd
import heapq
import osmnx as ox
import py3dep
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.services.path_preferences import PathPreferences
from app.services.compressed_pathfinding import compress_search_space, CompressedPathfinder
from app.services.compressed_pathfinding_balanced import balanced_compress_search_space

class DEMTileCache:
    def __init__(self, buffer=0.05, debug_mode=False, obstacle_config=None, path_preferences=None):
        """
        Initializes the DEMTileCache.

        Parameters:
        - buffer: Buffer size in degrees to extend the area of interest.
        - debug_mode: If True, collect detailed pathfinding debug information.
        - obstacle_config: ObstacleConfig instance for customizing obstacle handling
        - path_preferences: PathPreferences instance for preferring certain paths
        """
        self.buffer = buffer
        self.debug_mode = debug_mode
        self.debug_data = None
        self.obstacle_config = obstacle_config or ObstacleConfig()
        self.path_preferences = path_preferences or PathPreferences()
        if debug_mode:
            print(f"DEMTileCache initialized with debug_mode=True")

    def find_route(self, lat1, lon1, lat2, lon2):
        """
        Finds a hiking route between two GPS points.

        Parameters:
        - lat1, lon1: Latitude and longitude of the starting point.
        - lat2, lon2: Latitude and longitude of the destination point.

        Returns:
        - path_coords: List of (longitude, latitude) tuples representing the path coordinates.
        """
        # Define Area of Interest
        min_lat, max_lat, min_lon, max_lon = self.define_area_of_interest(lat1, lon1, lat2, lon2)

        # Download DEM Data
        dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
        if dem_file is None:
            return None

        # Read DEM Data
        dem, out_trans, crs = self.read_dem(dem_file)

        # Reproject DEM to Projected CRS
        dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)

        # Fetch and Rasterize Obstacles
        obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
        obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)

        # Fetch preferred paths
        paths = self.fetch_paths(min_lat, max_lat, min_lon, max_lon)
        path_raster, path_types = self.rasterize_paths(paths, out_trans, dem.shape, crs)

        # Compute Slope and Cost Surface with Obstacles and Path Preferences
        cost_surface, slope_degrees = self.compute_cost_surface(dem, out_trans, obstacle_mask, path_raster, path_types)
        
        # Log some statistics about the cost surface
        print(f"Cost surface stats: min={np.min(cost_surface):.2f}, max={np.max(cost_surface):.2f}, mean={np.mean(cost_surface):.2f}")
        print(f"Percentage of impassable cells (cost > 1000): {np.sum(cost_surface > 1000) / cost_surface.size * 100:.1f}%")

        # Build Indices for Pathfinding
        indices = self.build_indices(cost_surface)

        # Get Start and End Indices
        start_idx, end_idx, transformer = self.get_indices(lat1, lon1, lat2, lon2, out_trans, crs, indices)
        if start_idx is None or end_idx is None:
            return None

        # Check grid distance to decide between compressed and regular pathfinding
        start_row, start_col = np.unravel_index(start_idx, indices.shape)
        end_row, end_col = np.unravel_index(end_idx, indices.shape)
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)
        
        # Use compressed pathfinding for large search spaces
        # Consider both grid distance and terrain complexity
        use_compression = grid_distance > 800  # Higher threshold for compression
        
        if use_compression:
            print(f"Large search area detected (grid distance: {grid_distance}). Trying compressed pathfinding...")
            path = self.astar_pathfinding_compressed(
                cost_surface, indices, start_idx, end_idx, out_trans, transformer, 
                slope_degrees, obstacle_mask, path_raster, dem
            )
            # If compressed pathfinding fails, fall back to regular
            if path is None:
                print("Compressed pathfinding failed, falling back to regular A*...")
                path = self.astar_pathfinding(cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem)
        else:
            # Use regular A* for smaller searches
            print(f"Using regular A* (grid distance: {grid_distance})")
            path = self.astar_pathfinding(cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem)

        if path is None:
            return None

        # Calculate slopes for each segment of the path
        path_with_slopes = self.calculate_path_slopes(path, dem, out_trans, transformer)

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
        dem_dir = os.path.join('dem_data')
        dem_file = os.path.join(dem_dir, 'dem.tif')
        if not os.path.exists(dem_dir):
            os.makedirs(dem_dir)
        
        try:
            # Try 3m resolution first for finding paths around steep slopes
            try:
                dem = py3dep.get_map(
                    "DEM",
                    (min_lon, min_lat, max_lon, max_lat),
                    resolution=3,  # 3m resolution for finding switchbacks and narrow paths
                    crs="EPSG:4326"
                )
                print("Downloaded DEM at 3m resolution")
            except Exception as e3:
                print(f"3m resolution failed ({e3}), trying 10m resolution...")
                try:
                    dem = py3dep.get_map(
                        "DEM",
                        (min_lon, min_lat, max_lon, max_lat),
                        resolution=10,  # 10m resolution as fallback
                        crs="EPSG:4326"
                    )
                    print("Downloaded DEM at 10m resolution")
                except Exception as e10:
                    print(f"10m resolution failed ({e10}), trying 30m resolution...")
                    # Fallback to 30m resolution
                    dem = py3dep.get_map(
                        "DEM",
                        (min_lon, min_lat, max_lon, max_lat),
                        resolution=30,  # 30m resolution fallback
                        crs="EPSG:4326"
                    )
                    print("Downloaded DEM at 30m resolution")
            
            # Save to file
            dem.rio.to_raster(dem_file)
            return dem_file
        except Exception as e:
            print(f"Error downloading DEM data: {e}")
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
            print(f"Error reading DEM data: {e}")
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
            print(f"Error merging DEM tiles: {e}")
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
            dst_crs = 'EPSG:3857'  # Web Mercator
            transform, width, height = rasterio.warp.calculate_default_transform(
                crs, dst_crs, dem.shape[1], dem.shape[0],
                *rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], out_trans))
            dem_resampled = np.empty((height, width), dtype=np.float32)
            rasterio.warp.reproject(
                source=dem,
                destination=dem_resampled,
                src_transform=out_trans,
                src_crs=crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=rasterio.enums.Resampling.bilinear)
            dem = dem_resampled
            out_trans = transform
            crs = rasterio.crs.CRS.from_string(dst_crs)
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
            print(f"Error fetching obstacles: {e}")
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
            shapes,
            out_shape=dem_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
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
            print(f"Error fetching paths: {e}")
            return gpd.GeoDataFrame()

    def rasterize_paths(self, paths, transform, dem_shape, crs):
        """
        Rasterizes paths to create path preference raster.
        
        Returns:
        - path_raster: Numpy array where path cells contain type IDs
        - path_types: Dict mapping IDs to path types
        """
        if paths.empty:
            return np.zeros(dem_shape, dtype=int), {}
        
        # Reproject paths to DEM CRS
        paths = paths.to_crs(crs.to_string())
        
        # Create mapping of path types to IDs
        path_types = {}
        path_id = 1
        
        # Prepare shapes for rasterization with path type IDs
        shapes = []
        for idx, row in paths.iterrows():
            if row.geometry is not None:
                # Determine path type from tags
                path_type = 'off_path'
                if 'highway' in row and pd.notna(row['highway']):
                    path_type = str(row['highway'])
                elif 'leisure' in row and pd.notna(row['leisure']):
                    path_type = str(row['leisure'])
                elif 'landuse' in row and pd.notna(row['landuse']):
                    path_type = str(row['landuse'])
                elif 'natural' in row and pd.notna(row['natural']):
                    # Map natural types to our cost categories
                    natural_type = str(row['natural'])
                    if natural_type in ['grassland', 'heath']:
                        path_type = 'grass'
                    elif natural_type == 'meadow':
                        path_type = 'meadow'
                    elif natural_type in ['beach', 'sand']:
                        path_type = 'beach'
                    else:
                        path_type = natural_type
                
                # Get or assign ID for this path type
                if path_type not in path_types:
                    path_types[path_type] = path_id
                    path_id += 1
                
                shapes.append((row.geometry, path_types[path_type]))
        
        # Rasterize paths
        path_raster = rasterize(
            shapes,
            out_shape=dem_shape,
            transform=transform,
            fill=0,  # 0 means no path
            all_touched=True,
            dtype=np.uint8
        )
        
        # Reverse the path_types dict for easier lookup
        id_to_type = {v: k for k, v in path_types.items()}
        
        return path_raster, id_to_type

    def compute_cost_surface(self, dem, out_trans, obstacle_mask, path_raster=None, path_types=None):
        """
        Computes the slope and creates the cost surface, incorporating obstacles and path preferences.

        Returns:
        - cost_surface: Numpy array representing the cost of traversing each cell.
        - slope_degrees: Numpy array of slope values in degrees.
        """
        cell_size_x = out_trans.a
        cell_size_y = -out_trans.e  # Negative because of the coordinate system
        dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
        slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_degrees = np.degrees(slope_radians)

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
                        cost_surface[i, j] = base_cost * path_multiplier
                    else:
                        # Off-path terrain
                        off_path_multiplier = self.path_preferences.get_path_cost_multiplier(None)
                        cost_surface[i, j] = base_cost * off_path_multiplier
                else:
                    cost_surface[i, j] = base_cost

        # Assign obstacle costs from configuration (obstacles override path preferences)
        cost_surface[obstacle_mask] = self.obstacle_config.get_cost_for_feature('default')

        return cost_surface, slope_degrees

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
        if not (0 <= row_start < height and 0 <= col_start < width and
                0 <= row_end < height and 0 <= col_end < width):
            return None, None, None

        start_idx = indices[row_start, col_start]
        end_idx = indices[row_end, col_end]

        return start_idx, end_idx, transformer

    def astar_pathfinding_compressed(self, cost_surface, indices, start_idx, end_idx, out_trans, transformer, 
                                    slope_degrees, obstacle_mask=None, path_raster=None, dem=None):
        """
        Computes path using compressed regions for large search spaces.
        """
        height, width = cost_surface.shape
        start_row, start_col = np.unravel_index(start_idx, (height, width))
        end_row, end_col = np.unravel_index(end_idx, (height, width))
        
        print("Using compressed pathfinding...")
        
        # Calculate grid distance for this method
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)
        
        # Use simpler compression for very large searches
        # The full compression with obstacles/paths is too fragmented for this terrain
        if grid_distance > 1000:
            print("Using simplified compression for very large search...")
            regions, region_map = balanced_compress_search_space(slope_degrees, cost_surface, obstacle_mask)
        else:
            regions, region_map = compress_search_space(slope_degrees, cost_surface, obstacle_mask, path_raster)
        
        # Find path using compressed space
        compressor = CompressedPathfinder()
        compressed_path = compressor.find_path_compressed(
            regions, 
            (start_row, start_col), 
            (end_row, end_col),
            cost_surface
        )
        
        if not compressed_path:
            print("Compressed pathfinding failed, falling back to regular A*")
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
                # Run local A* between these cells
                local_path = self._local_astar(
                    cost_surface, indices, 
                    current_cell, next_cell,
                    max_distance=50  # Limit local search
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
            x_lon, y_lat = transformer.transform(x, y, direction='INVERSE')
            path.append((x_lon, y_lat))
        
        return path

    def _local_astar(self, cost_surface, indices, start_cell, end_cell, max_distance=50):
        """Local A* for refining paths between waypoints"""
        height, width = cost_surface.shape
        start_row, start_col = start_cell
        end_row, end_col = end_cell
        
        # Check if cells are too far apart
        if (abs(end_row - start_row) > max_distance or 
            abs(end_col - start_col) > max_distance):
            # Just return direct line for very distant cells
            return [start_cell, end_cell]
        
        # Standard A* but with limited search area
        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {}
        g_score[start_cell] = 0
        
        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]
        
        iterations = 0
        max_iterations = max_distance * max_distance * 4  # Reasonable limit
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == end_cell:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_cell)
                path.reverse()
                return path
            
            row, col = current
            
            for dr, dc in neighbors_offsets:
                nr, nc = row + dr, col + dc
                neighbor = (nr, nc)
                
                # Check bounds and distance limit
                if (0 <= nr < height and 0 <= nc < width and
                    abs(nr - start_row) <= max_distance and
                    abs(nc - start_col) <= max_distance):
                    
                    # Movement cost
                    diagonal = (dr != 0 and dc != 0)
                    distance = sqrt(2) if diagonal else 1
                    movement_cost = distance * cost_surface[nr, nc]
                    
                    tentative_g = g_score[current] + movement_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        
                        # Simple heuristic
                        h = sqrt((end_row - nr)**2 + (end_col - nc)**2)
                        f = tentative_g + h
                        
                        heapq.heappush(open_set, (f, neighbor))
        
        # No path found, return direct line
        return [start_cell, end_cell]

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

        neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                             (0, -1),         (0, 1),
                             (1, -1),  (1, 0),  (1, 1)]

        # Initialize debug data if in debug mode
        if self.debug_mode:
            self.debug_data = {
                'explored_nodes': [],
                'cost_calculations': [],
                'path_decisions': [],
                'grid_exploration': {
                    'shape': indices.shape,
                    'g_scores': np.full(indices.shape, np.inf),
                    'f_scores': np.full(indices.shape, np.inf),
                    'h_scores': np.full(indices.shape, np.inf),
                    'explored': np.zeros(indices.shape, dtype=bool),
                    'in_path': np.zeros(indices.shape, dtype=bool)
                },
                'decision_points': [],
                'terrain_costs': cost_surface.copy(),
                'bounds': {
                    'transform': out_trans,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
            }

        step_count = 0
        max_iterations = 10000000  # Further increased for long routes at 3m resolution
        
        # Calculate expected search area to warn about large searches
        start_row, start_col = np.unravel_index(start_idx, indices.shape)
        end_row, end_col = np.unravel_index(end_idx, indices.shape)
        grid_distance = abs(end_row - start_row) + abs(end_col - start_col)
        
        if grid_distance > 1000:
            print(f"Warning: Large search area detected. Grid distance: {grid_distance} cells")
        
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
                self.debug_data['explored_nodes'].append({
                    'step': step_count,
                    'node_idx': current,
                    'row': row_current,
                    'col': col_current,
                    'g_score': g_score[current],
                    'f_score': current_f,
                    'h_score': current_f - g_score[current]
                })
                self.debug_data['grid_exploration']['explored'][row_current, col_current] = True
                self.debug_data['grid_exploration']['g_scores'][row_current, col_current] = g_score[current]
                self.debug_data['grid_exploration']['f_scores'][row_current, col_current] = current_f

            # Explore neighbors
            neighbor_evaluations = []
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
                            current_elevation = dem[row_current, col_current] if 0 <= row_current < dem.shape[0] and 0 <= col_current < dem.shape[1] else 0
                            neighbor_elevation = dem[row_neighbor, col_neighbor] if 0 <= row_neighbor < dem.shape[0] and 0 <= col_neighbor < dem.shape[1] else 0
                        else:
                            current_elevation = 0
                            neighbor_elevation = 0
                        elevation_change = neighbor_elevation - current_elevation
                        slope_degrees = np.degrees(np.arctan(abs(elevation_change) / max(distance, 0.1)))
                        
                        # Break down the terrain cost components
                        base_cost = 1.0  # Base movement cost
                        slope_penalty = max(0, terrain_cost - base_cost)  # Additional cost due to slope/obstacles
                        
                        neighbor_evaluations.append({
                            'neighbor_idx': neighbor,
                            'row': row_neighbor,
                            'col': col_neighbor,
                            'direction': (dy, dx),
                            'direction_name': self._get_direction_name(dy, dx),
                            'distance_meters': distance,
                            'elevation_change_m': elevation_change,
                            'slope_degrees': slope_degrees,
                            'terrain_breakdown': {
                                'base_cost': base_cost,
                                'slope_penalty': slope_penalty,
                                'total_terrain_cost': terrain_cost,
                                'is_obstacle': terrain_cost > 100  # High cost indicates obstacle
                            },
                            'cost_breakdown': {
                                'distance_component': distance,
                                'terrain_component': terrain_cost,
                                'total_movement_cost': movement_cost,
                                'explanation': f"movement_cost = terrain_cost({terrain_cost:.2f}) × distance({distance:.1f}m) = {movement_cost:.2f}"
                            },
                            'g_score_breakdown': {
                                'previous_g_score': g_score[current],
                                'movement_cost': movement_cost,
                                'tentative_g_score': tentative_g_score,
                                'explanation': f"g_score = previous({g_score[current]:.2f}) + movement_cost({movement_cost:.2f}) = {tentative_g_score:.2f}"
                            },
                            'f_score_breakdown': {
                                'g_score': tentative_g_score,
                                'h_score': heuristic_cost,
                                'f_score': tentative_f_score,
                                'explanation': f"f_score = g_score({tentative_g_score:.2f}) + h_score({heuristic_cost:.2f}) = {tentative_f_score:.2f}"
                            },
                            'current_g_score': g_score[neighbor],
                            'is_improvement': tentative_g_score < g_score[neighbor]
                        })

                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_f_score
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

            # Record decision point in debug mode
            if self.debug_mode:
                self.debug_data['decision_points'].append({
                    'step': step_count,
                    'current_node': {
                        'idx': current,
                        'row': row_current,
                        'col': col_current,
                        'lat_lon': self._idx_to_latlon(current, out_trans, transformer, indices.shape)
                    },
                    'neighbors_evaluated': neighbor_evaluations,
                    'chosen_neighbors': [n for n in neighbor_evaluations if n['is_improvement']]
                })

        if step_count >= max_iterations:
            print(f"A* search stopped after {max_iterations} iterations")
        else:
            print(f"A* search exhausted all possibilities after {step_count} iterations")
            
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
            x_lon, y_lat = transformer.transform(x, y, direction='INVERSE')
            path.append((x_lon, y_lat))
            current = came_from[current]

        # Add the starting point
        row, col = np.unravel_index(current, shape)
        x = out_trans.c + col * out_trans.a + out_trans.a / 2
        y = out_trans.f + row * out_trans.e + out_trans.e / 2
        x_lon, y_lat = transformer.transform(x, y, direction='INVERSE')
        path.append((x_lon, y_lat))

        path.reverse()
        return path

    def plot_results(self, dem, out_trans, crs, path, lat1, lon1, lat2, lon2, min_lat, max_lat, min_lon, max_lon, obstacles=None):
        """
        Plots the DEM, area of interest, obstacles, and the calculated route.
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot the DEM
        extent = rasterio.plot.plotting_extent(dem, out_trans)
        show(dem, ax=ax, cmap='terrain', extent=extent, transform=out_trans)

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
            ax.plot(x_coords, y_coords, color='blue', linewidth=3, label='Calculated Route')

        # Plot start/end points
        ax.scatter([x1, x2], [y1, y2], color='red', s=100, zorder=5, label='Start/End Points')
        ax.text(x1, y1, 'Start', fontsize=12, ha='right')
        ax.text(x2, y2, 'End', fontsize=12, ha='left')

        # Plot obstacles if provided
        if obstacles is not None and not obstacles.empty:
            obstacles = obstacles.to_crs(crs.to_string())
            obstacles.plot(ax=ax, facecolor='gray', edgecolor='darkgray', linewidth=0.5, alpha=0.5, label='Obstacles')

        # Labels and legend
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Least-Cost Path Over Terrain with Obstacles')
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _idx_to_latlon(self, idx, out_trans, transformer, shape):
        """Convert grid index to lat/lon coordinates"""
        row, col = np.unravel_index(idx, shape)
        x = out_trans.c + col * out_trans.a + out_trans.a / 2
        y = out_trans.f + row * out_trans.e + out_trans.e / 2
        x_lon, y_lat = transformer.transform(x, y, direction='INVERSE')
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
            (1, 1): "Southeast"
        }
        return direction_map.get((dy, dx), f"({dy}, {dx})")

    def _mark_path_in_debug(self, path, out_trans, transformer, shape):
        """Mark path nodes in the debug grid"""
        if not self.debug_data:
            return
            
        for point in path:
            if isinstance(point, dict):
                lon, lat = point['lon'], point['lat']
            else:
                lon, lat = point
            # Convert lat/lon back to grid indices
            # transformer expects lon, lat order for geographic coordinates
            x, y = transformer.transform(lon, lat)
            col = int(round((x - out_trans.c) / out_trans.a))
            row = int(round((y - out_trans.f) / out_trans.e))
            
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                self.debug_data['grid_exploration']['in_path'][row, col] = True
                
        # Debug: Log how many path points were marked
        path_count = np.sum(self.debug_data['grid_exploration']['in_path'])
        print(f"Debug: Marked {path_count} cells as in_path out of {len(path)} path points")
        
        # Extra debug: show some of the marked cells
        in_path_indices = np.where(self.debug_data['grid_exploration']['in_path'])
        if len(in_path_indices[0]) > 0:
            print(f"Debug: First few in_path cells: {list(zip(in_path_indices[0][:5], in_path_indices[1][:5]))}")
    
    def calculate_path_slopes(self, path, dem, out_trans, transformer):
        """Calculate slope for each segment of the path"""
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
            if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                elevation = float(dem[row, col])
            
            # Calculate slope to next point
            slope_degrees = 0.0
            if i < len(path) - 1:
                next_lon, next_lat = path[i + 1]
                next_x, next_y = transformer.transform(next_lon, next_lat)
                next_col = int(round((next_x - out_trans.c) / out_trans.a))
                next_row = int(round((next_y - out_trans.f) / out_trans.e))
                
                if (0 <= next_row < dem.shape[0] and 0 <= next_col < dem.shape[1] and 
                    elevation is not None):
                    next_elevation = float(dem[next_row, next_col])
                    
                    # Calculate horizontal distance
                    distance = sqrt((next_x - x)**2 + (next_y - y)**2)
                    
                    if distance > 0:
                        # Calculate slope
                        elevation_change = next_elevation - elevation
                        slope_radians = atan(abs(elevation_change) / distance)
                        slope_degrees = degrees(slope_radians)
                        
                        # If going downhill, make slope negative
                        if elevation_change < 0:
                            slope_degrees = -slope_degrees
            
            path_with_slopes.append({
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'slope': round(slope_degrees, 1)
            })
        
        return path_with_slopes

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
        total_explored = len(self.debug_data['explored_nodes'])
        total_decisions = len(self.debug_data['decision_points'])
        
        print(f"Debug data size: {total_explored} explored nodes, {total_decisions} decision points")
        
        # Convert numpy arrays to lists for JSON serialization
        # But limit the size to prevent huge responses
        debug_copy = {
            'explored_nodes': convert_numpy_types(self.debug_data['explored_nodes'][:MAX_NODES]),
            'decision_points': convert_numpy_types(self.debug_data['decision_points'][:MAX_DECISIONS]),
            'grid_exploration': {
                'shape': list(self.debug_data['grid_exploration']['shape']),
                'g_scores': self.debug_data['grid_exploration']['g_scores'].tolist(),
                'f_scores': self.debug_data['grid_exploration']['f_scores'].tolist(),
                'h_scores': self.debug_data['grid_exploration']['h_scores'].tolist(),
                'explored': self.debug_data['grid_exploration']['explored'].tolist(),
                'in_path': self.debug_data['grid_exploration']['in_path'].tolist()
            },
            'terrain_costs': self.debug_data['terrain_costs'].tolist(),
            'bounds': {
                'start_idx': int(self.debug_data['bounds']['start_idx']),
                'end_idx': int(self.debug_data['bounds']['end_idx']),
                'transform': {
                    'a': float(self.debug_data['bounds']['transform'].a),
                    'b': float(self.debug_data['bounds']['transform'].b), 
                    'c': float(self.debug_data['bounds']['transform'].c),
                    'd': float(self.debug_data['bounds']['transform'].d),
                    'e': float(self.debug_data['bounds']['transform'].e),
                    'f': float(self.debug_data['bounds']['transform'].f)
                }
            },
            'total_explored': total_explored,
            'total_decisions': total_decisions,
            'data_truncated': total_explored > MAX_NODES or total_decisions > MAX_DECISIONS
        }
        return debug_copy

