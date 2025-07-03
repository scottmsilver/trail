"""
Tiled DEM Cache - Divides terrain into tiles for efficient caching and composition
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from pyproj import Transformer
import pickle
from typing import Dict, List, Tuple, Optional
import hashlib
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Tile:
    """Represents a single tile of terrain/cost data"""
    tile_x: int
    tile_y: int
    bounds: Tuple[float, float, float, float]  # min_lat, max_lat, min_lon, max_lon
    data: np.ndarray
    transform: rasterio.Affine
    
class TiledDEMCache:
    """
    Manages terrain data in tiles for efficient caching and retrieval.
    Each tile covers a fixed geographic area (e.g., 0.01 degrees).
    """
    
    def __init__(self, tile_size_degrees: float = 0.01, cache_dir: str = "tile_cache"):
        """
        Initialize the tiled cache.
        
        Args:
            tile_size_degrees: Size of each tile in degrees (default 0.01 = ~1.1km at equator)
            cache_dir: Directory to store cached tiles
        """
        self.tile_size = tile_size_degrees
        self.cache_dir = os.path.abspath(cache_dir)
        self.memory_cache: Dict[str, Dict] = {}  # In-memory cache of tiles
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "terrain"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "cost"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "obstacles"), exist_ok=True)
        
        logger.info(f"TiledDEMCache initialized with tile size {tile_size_degrees}°")
        logger.info(f"[CACHE PATH] Tile cache directory: {self.cache_dir}")
    
    def _get_tile_coords(self, lat: float, lon: float) -> Tuple[int, int]:
        """Get tile coordinates for a given lat/lon"""
        tile_x = int(np.floor(lon / self.tile_size))
        tile_y = int(np.floor(lat / self.tile_size))
        return tile_x, tile_y
    
    def _get_tile_bounds(self, tile_x: int, tile_y: int) -> Tuple[float, float, float, float]:
        """Get geographic bounds for a tile"""
        min_lon = tile_x * self.tile_size
        max_lon = (tile_x + 1) * self.tile_size
        min_lat = tile_y * self.tile_size
        max_lat = (tile_y + 1) * self.tile_size
        return min_lat, max_lat, min_lon, max_lon
    
    def _get_tile_key(self, tile_x: int, tile_y: int, data_type: str) -> str:
        """Generate a unique key for a tile"""
        return f"{data_type}_{tile_x}_{tile_y}"
    
    def _get_tile_filename(self, tile_x: int, tile_y: int, data_type: str) -> str:
        """Get the filename for a cached tile"""
        return os.path.join(self.cache_dir, data_type, f"tile_{tile_x}_{tile_y}.pkl")
    
    def get_tiles_for_bounds(self, min_lat: float, max_lat: float, 
                           min_lon: float, max_lon: float) -> List[Tuple[int, int]]:
        """Get all tile coordinates that overlap with the given bounds"""
        min_tile_x, min_tile_y = self._get_tile_coords(min_lat, min_lon)
        max_tile_x, max_tile_y = self._get_tile_coords(max_lat, max_lon)
        
        tiles = []
        for ty in range(min_tile_y, max_tile_y + 1):
            for tx in range(min_tile_x, max_tile_x + 1):
                tiles.append((tx, ty))
        
        return tiles
    
    def load_tile_from_disk(self, tile_x: int, tile_y: int, data_type: str) -> Optional[Dict]:
        """Load a tile from disk cache"""
        filename = self._get_tile_filename(tile_x, tile_y, data_type)
        
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    tile_data = pickle.load(f)
                logger.info(f"[TILE CACHE HIT] Loaded {data_type} tile ({tile_x}, {tile_y}) from: {filename}")
                return tile_data
            except Exception as e:
                logger.error(f"Error loading tile from {filename}: {e}")
                return None
        return None
    
    def save_tile_to_disk(self, tile_x: int, tile_y: int, data_type: str, tile_data: Dict):
        """Save a tile to disk cache"""
        filename = self._get_tile_filename(tile_x, tile_y, data_type)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(tile_data, f)
            logger.info(f"[TILE CACHE] Saved {data_type} tile ({tile_x}, {tile_y}) to: {filename}")
        except Exception as e:
            logger.error(f"Error saving tile to {filename}: {e}")
    
    def get_tile(self, tile_x: int, tile_y: int, data_type: str, 
                 compute_func=None, **compute_kwargs) -> Optional[Dict]:
        """
        Get a tile from cache or compute it if necessary.
        
        Args:
            tile_x, tile_y: Tile coordinates
            data_type: Type of data (terrain, cost, obstacles)
            compute_func: Function to compute tile data if not cached
            **compute_kwargs: Additional arguments for compute_func
        
        Returns:
            Dictionary with tile data and metadata
        """
        # Check memory cache first
        tile_key = self._get_tile_key(tile_x, tile_y, data_type)
        
        if tile_key in self.memory_cache:
            logger.info(f"[TILE CACHE HIT] Memory cache hit for {data_type} tile ({tile_x}, {tile_y})")
            return self.memory_cache[tile_key]
        
        # Check disk cache
        tile_data = self.load_tile_from_disk(tile_x, tile_y, data_type)
        
        if tile_data is not None:
            # Store in memory cache
            self.memory_cache[tile_key] = tile_data
            return tile_data
        
        # Compute tile if not cached
        if compute_func is not None:
            logger.info(f"[TILE CACHE MISS] Computing {data_type} tile ({tile_x}, {tile_y})")
            
            min_lat, max_lat, min_lon, max_lon = self._get_tile_bounds(tile_x, tile_y)
            tile_data = compute_func(min_lat, max_lat, min_lon, max_lon, **compute_kwargs)
            
            if tile_data is not None:
                # Add metadata
                tile_data['tile_x'] = tile_x
                tile_data['tile_y'] = tile_y
                tile_data['bounds'] = (min_lat, max_lat, min_lon, max_lon)
                
                # Cache the result
                self.memory_cache[tile_key] = tile_data
                self.save_tile_to_disk(tile_x, tile_y, data_type, tile_data)
                
                return tile_data
        
        return None
    
    def compose_tiles(self, tiles: List[Tuple[int, int]], data_type: str, 
                     target_bounds: Tuple[float, float, float, float],
                     compute_func=None, **compute_kwargs) -> Optional[Dict]:
        """
        Compose multiple tiles into a single dataset covering the target bounds.
        
        Args:
            tiles: List of (tile_x, tile_y) coordinates
            data_type: Type of data to compose
            target_bounds: (min_lat, max_lat, min_lon, max_lon) of desired area
            compute_func: Function to compute missing tiles
            
        Returns:
            Dictionary with composed data and transform
        """
        logger.info(f"[TILE COMPOSE] Composing {len(tiles)} tiles for {data_type}")
        
        # Get all tile data
        tile_data_list = []
        for tile_x, tile_y in tiles:
            tile_data = self.get_tile(tile_x, tile_y, data_type, compute_func, **compute_kwargs)
            if tile_data is not None:
                tile_data_list.append(tile_data)
        
        if not tile_data_list:
            logger.error(f"No tiles could be loaded for composition")
            return None
        
        # Sort tiles by position for consistent ordering
        tile_data_list.sort(key=lambda t: (t['tile_y'], t['tile_x']))
        
        # If only one tile and it covers the target bounds, return it directly
        if len(tile_data_list) == 1:
            tile = tile_data_list[0]
            t_bounds = tile['bounds']
            if (t_bounds[0] <= target_bounds[0] and t_bounds[1] >= target_bounds[1] and
                t_bounds[2] <= target_bounds[2] and t_bounds[3] >= target_bounds[3]):
                logger.info(f"[TILE COMPOSE] Single tile covers entire area")
                return tile
        
        # Compose multiple tiles into a single array
        logger.info(f"[TILE COMPOSE] Composing {len(tile_data_list)} tiles")
        
        # Find the grid arrangement
        min_tile_x = min(t['tile_x'] for t in tile_data_list)
        max_tile_x = max(t['tile_x'] for t in tile_data_list)
        min_tile_y = min(t['tile_y'] for t in tile_data_list)
        max_tile_y = max(t['tile_y'] for t in tile_data_list)
        
        tiles_x = max_tile_x - min_tile_x + 1
        tiles_y = max_tile_y - min_tile_y + 1
        
        # Get first tile for reference
        first_tile = tile_data_list[0]
        
        # Find the most appropriate tile size
        # Collect all tile sizes and find the maximum to ensure no data is cut off
        tile_sizes = []
        for tile_data in tile_data_list:
            h, w = tile_data['cost_surface'].shape
            tile_sizes.append((h, w))
        
        # Use maximum dimensions to ensure all data fits
        tile_height = max(h for h, w in tile_sizes)
        tile_width = max(w for h, w in tile_sizes)
        
        # Log size variations if they exist
        unique_sizes = set(tile_sizes)
        if len(unique_sizes) > 1:
            size_counts = {}
            for size in tile_sizes:
                size_counts[size] = size_counts.get(size, 0) + 1
            logger.info(f"[TILE COMPOSE] Tile size variations detected: {size_counts}")
            logger.info(f"[TILE COMPOSE] This is normal - tiles at different latitudes or edges have slightly different pixel counts")
            logger.info(f"[TILE COMPOSE] Using size {tile_height}x{tile_width} for composition to accommodate all tiles")
        
        # Create composite array
        composite_height = tiles_y * tile_height
        composite_width = tiles_x * tile_width
        
        # Initialize arrays
        # Use a high but finite cost for uninitialized areas (obstacles have cost 1000)
        cost_surface = np.full((composite_height, composite_width), 1000.0, dtype=np.float32)
        slope_degrees = np.zeros((composite_height, composite_width), dtype=np.float32)
        # Initialize DEM with NaN to indicate missing data (not zero!)
        dem = np.full((composite_height, composite_width), np.nan, dtype=np.float32)
        
        # Fill in tile data
        outlier_tiles = 0
        for tile_data in tile_data_list:
            # Calculate position in composite
            tile_offset_x = tile_data['tile_x'] - min_tile_x
            tile_offset_y = max_tile_y - tile_data['tile_y']  # Flip Y for array indexing
            
            y_start = tile_offset_y * tile_height
            y_end = y_start + tile_height
            x_start = tile_offset_x * tile_width
            x_end = x_start + tile_width
            
            # Get actual tile dimensions
            tile_h, tile_w = tile_data['cost_surface'].shape
            
            # Check if this tile is significantly smaller (likely an edge case)
            if tile_h < tile_height * 0.5 or tile_w < tile_width * 0.5:
                outlier_tiles += 1
                logger.info(f"[TILE COMPOSE] Edge/partial tile at ({tile_data['tile_x']}, {tile_data['tile_y']}): {tile_h}x{tile_w}")
                logger.info(f"[TILE COMPOSE] This tile likely covers an area with limited elevation data availability")
            elif tile_h != tile_height or tile_w != tile_width:
                # Small variation (±1-2 pixels)
                logger.debug(f"[TILE COMPOSE] Normal size variation at ({tile_data['tile_x']}, {tile_data['tile_y']}): {tile_h}x{tile_w} (expected ~{tile_height}x{tile_width})")
            
            # Copy tile data (handle size variations gracefully)
            actual_y_end = min(y_end, y_start + tile_h)
            actual_x_end = min(x_end, x_start + tile_w)
            
            cost_surface[y_start:actual_y_end, x_start:actual_x_end] = tile_data['cost_surface'][:actual_y_end-y_start, :actual_x_end-x_start]
            slope_degrees[y_start:actual_y_end, x_start:actual_x_end] = tile_data['slope_degrees'][:actual_y_end-y_start, :actual_x_end-x_start]
            if 'dem' in tile_data:
                dem[y_start:actual_y_end, x_start:actual_x_end] = tile_data['dem'][:actual_y_end-y_start, :actual_x_end-x_start]  # Copy DEM data
        
        if outlier_tiles > 0:
            logger.info(f"[TILE COMPOSE] Handled {outlier_tiles} outlier tiles")
        
        # Calculate composite transform based on tile arrangement
        # Use the actual tile transforms to calculate the composite bounds
        
        # Find the top-left tile (minimum x, maximum y in tile coordinates)
        top_left_tile = None
        for tile_data in tile_data_list:
            if tile_data['tile_x'] == min_tile_x and tile_data['tile_y'] == max_tile_y:
                top_left_tile = tile_data
                break
        
        if top_left_tile:
            # Use the actual transform from the top-left tile
            tl_transform = top_left_tile['transform']
            # The composite origin is the top-left corner of the top-left tile
            composite_west = tl_transform.c
            composite_north = tl_transform.f
            pixel_width = tl_transform.a
            pixel_height = tl_transform.e
            logger.info(f"[TILE COMPOSE] Using top-left tile transform origin: ({composite_west}, {composite_north})")
        else:
            # Fallback: calculate from first tile
            first_transform = first_tile['transform']
            pixel_width = first_transform.a
            pixel_height = first_transform.e
            
            # Calculate offset from first tile to top-left of composite
            x_offset = (first_tile['tile_x'] - min_tile_x) * tile_width * pixel_width
            y_offset = (max_tile_y - first_tile['tile_y']) * tile_height * (-pixel_height)
            
            composite_west = first_transform.c - x_offset
            composite_north = first_transform.f + y_offset
            logger.info(f"[TILE COMPOSE] Calculated composite origin from first tile: ({composite_west}, {composite_north})")
        
        from rasterio.transform import from_origin
        composite_transform = from_origin(composite_west, composite_north, pixel_width, -pixel_height)
        logger.info(f"[TILE COMPOSE] Created transform with origin ({composite_west}, {composite_north}), pixel size ({pixel_width}, {-pixel_height})")
        
        logger.info(f"[TILE COMPOSE] Created composite {composite_height}x{composite_width}")
        
        return {
            'cost_surface': cost_surface,
            'slope_degrees': slope_degrees,
            'dem': dem,  # Include composed DEM data
            'transform': composite_transform,
            'crs': first_tile.get('crs'),
            'bounds': (min_tile_y * self.tile_size, (max_tile_y + 1) * self.tile_size,
                      min_tile_x * self.tile_size, (max_tile_x + 1) * self.tile_size),
            'composed_from': len(tile_data_list)
        }
    
    def clear_memory_cache(self):
        """Clear the in-memory cache"""
        self.memory_cache.clear()
        logger.info("Cleared in-memory tile cache")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache"""
        stats = {
            'memory_tiles': len(self.memory_cache),
            'tile_size_degrees': self.tile_size,
            'cache_dir': self.cache_dir
        }
        
        # Count disk tiles
        for data_type in ['terrain', 'cost', 'obstacles']:
            dir_path = os.path.join(self.cache_dir, data_type)
            if os.path.exists(dir_path):
                stats[f'disk_{data_type}_tiles'] = len([f for f in os.listdir(dir_path) if f.endswith('.pkl')])
            else:
                stats[f'disk_{data_type}_tiles'] = 0
        
        return stats