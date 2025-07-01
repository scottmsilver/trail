import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.features import rasterize
# import matplotlib.pyplot as plt  # Not needed in backend
from pyproj import Transformer
import requests
from math import radians, tan, cos, pi, sqrt, log
from tqdm import tqdm
from shapely.geometry import box
import geopandas as gpd
import heapq
import osmnx as ox
import py3dep

class DEMTileCache:
    def __init__(self, buffer=0.05):
        """
        Initializes the DEMTileCache.

        Parameters:
        - buffer: Buffer size in degrees to extend the area of interest.
        """
        self.buffer = buffer

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
            print("Failed to download DEM data.")
            return None

        # Read DEM Data
        dem, out_trans, crs = self.read_dem(dem_file)

        # Reproject DEM to Projected CRS
        dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)

        # Fetch and Rasterize Obstacles
        obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
        obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)

        # Compute Slope and Cost Surface with Obstacles
        cost_surface = self.compute_cost_surface(dem, out_trans, obstacle_mask)

        # Build Indices for Pathfinding
        indices = self.build_indices(cost_surface)

        # Get Start and End Indices
        start_idx, end_idx, transformer = self.get_indices(lat1, lon1, lat2, lon2, out_trans, crs, indices)
        if start_idx is None or end_idx is None:
            print("Start or end point is outside the DEM area.")
            return None

        # Compute Least-Cost Path using A* Algorithm
        path = self.astar_pathfinding(cost_surface, indices, start_idx, end_idx, out_trans, transformer)

        if path is None:
            print("No path found.")
            return None

        # Skip plotting in backend service
        # self.plot_results(dem, out_trans, crs, path, lat1, lon1, lat2, lon2,
        #                   min_lat, max_lat, min_lon, max_lon, obstacles=obstacles)

        return path

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
        
        print("Downloading DEM data...")
        try:
            # Use py3dep to download DEM data
            dem = py3dep.get_map(
                "DEM",
                (min_lon, min_lat, max_lon, max_lat),
                resolution=30,  # 30m resolution
                crs="EPSG:4326"
            )
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
        Fetches obstacle data from OSM.

        Returns:
        - obstacles: GeoDataFrame containing obstacle geometries.
        """
        # Create a bounding box polygon
        bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
        
        # Fetch OSM data for obstacles
        print("Fetching obstacle data from OSM...")
        try:
            tags = {
                'natural': ['water', 'wetland'],
                'landuse': ['residential', 'industrial', 'retail', 'construction'],
                'building': True,
                'leisure': ['park', 'golf_course'],
                'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary'],
                'barrier': True
            }
            ox.settings.log_console = False
            obstacles = ox.features_from_polygon(bbox_polygon, tags)
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
            print("No obstacles found in the area.")
            obstacle_mask = np.zeros(dem_shape, dtype=bool)
            return obstacle_mask

        # Reproject obstacles to DEM CRS
        obstacles = obstacles.to_crs(crs.to_string())

        # Prepare geometries for rasterization
        obstacle_geometries = obstacles.geometry.values
        shapes = [(geom, 1) for geom in obstacle_geometries if geom is not None]

        # Rasterize obstacles
        print("Rasterizing obstacle data...")
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

    def compute_cost_surface(self, dem, out_trans, obstacle_mask):
        """
        Computes the slope and creates the cost surface, incorporating obstacles.

        Returns:
        - cost_surface: Numpy array representing the cost of traversing each cell.
        """
        cell_size_x = out_trans.a
        cell_size_y = -out_trans.e  # Negative because of the coordinate system
        dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
        slope = np.sqrt(dzdx**2 + dzdy**2)

        # Create cost surface
        cost_surface = 1 + slope * 10  # Adjust the multiplier as needed

        # Assign high cost to obstacle cells
        cost_surface[obstacle_mask] = np.inf  # Alternatively, use a very high cost

        return cost_surface

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

    def astar_pathfinding(self, cost_surface, indices, start_idx, end_idx, out_trans, transformer):
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

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == end_idx:
                return self.reconstruct_path_astar(came_from, current, out_trans, transformer, indices.shape)

            closed_set.add(current)

            row_current, col_current = np.unravel_index(current, indices.shape)

            for dy, dx in neighbors_offsets:
                row_neighbor = row_current + dy
                col_neighbor = col_current + dx

                if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                    neighbor = indices[row_neighbor, col_neighbor]
                    if neighbor in closed_set:
                        continue

                    # Calculate tentative g_score
                    distance = sqrt((dy * out_trans.e) ** 2 + (dx * out_trans.a) ** 2)
                    tentative_g_score = g_score[current] + cost_surface[row_neighbor, col_neighbor] * distance

                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end_idx, indices.shape, out_trans)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

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

        return sqrt(dx**2 + dy**2)

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

# Example usage
if __name__ == "__main__":
    # Use the same coordinates from your other files (Utah area)
    lat1, lon1 = 40.630, -111.580  # Start point
    lat2, lon2 = 40.650, -111.560  # End point
    
    print(f"Finding route from ({lat1}, {lon1}) to ({lat2}, {lon2})")
    
    route_finder = DEMTileCache(buffer=0.02)  # Smaller buffer for testing
    path_coords = route_finder.find_route(lat1, lon1, lat2, lon2)
    
    if path_coords:
        print(f"Hiking route found with {len(path_coords)} waypoints!")
        # Save to GPX file
        import gpxpy
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        
        for lon, lat in path_coords:
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))
        
        with open('path_t7.gpx', 'w') as f:
            f.write(gpx.to_xml())
        print("Route saved to path_t7.gpx")
    else:
        print("No route could be found.")