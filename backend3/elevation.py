#!/usr/bin/env python3
"""
Production-ready two-layer elevation system.
Eliminates tile boundary artifacts while providing all necessary APIs.

Layer 1: Large seamless regions downloaded from py3dep
Layer 2: Small tiles created from Layer 1 for efficient queries
"""

import os
import json
import shutil
import numpy as np

# Fix aiohttp import issue before importing py3dep
import aiohttp
if not hasattr(aiohttp, 'ClientConnectorDNSError'):
    class ClientConnectorDNSError(aiohttp.ClientConnectorError):
        """DNS resolution error."""
        pass
    aiohttp.ClientConnectorDNSError = ClientConnectorDNSError

import py3dep
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.windows import Window
from typing import Dict, Tuple, Optional, List, Any, Set
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Bounds:
    """Geographic bounds"""
    north: float
    south: float
    east: float
    west: float
    
    def __post_init__(self):
        if self.north <= self.south:
            raise ValueError("North must be greater than south")
        if self.east <= self.west:
            raise ValueError("East must be greater than west")
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (west, south, east, north) for rasterio/py3dep"""
        return (self.west, self.south, self.east, self.north)
    
    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within these bounds"""
        return (self.south <= lat <= self.north and 
                self.west <= lon <= self.east)
    
    def overlaps(self, other: 'Bounds') -> bool:
        """Check if bounds overlap with another"""
        return not (self.north <= other.south or self.south >= other.north or
                   self.east <= other.west or self.west >= other.east)
    
    def union(self, other: 'Bounds') -> 'Bounds':
        """Return union of two bounds"""
        return Bounds(
            north=max(self.north, other.north),
            south=min(self.south, other.south),
            east=max(self.east, other.east),
            west=min(self.west, other.west)
        )


class TwoLayerElevationLibrary:
    """
    Production-ready two-layer elevation library with explicit data management.
    
    IMPORTANT: No automatic downloading or caching!
    - Data must be explicitly loaded with load_area()
    - All queries (get_elevation, get_elevation_array) fail if data not preloaded
    - No hidden downloads or automatic fetching
    
    Features:
    - No tile boundary artifacts
    - All APIs from single-layer system
    - Multiple resolution support
    - Explicit load/remove operations only
    
    Note: _open_datasets is NOT a cache - it's just keeping file handles open
    for performance. Data is either loaded (via load_area) or not available.
    """
    
    # Supported resolutions and their tile sizes
    RESOLUTIONS = {
        1: 0.001,    # 1m resolution, ~0.001° tiles
        3: 0.005,    # 3m resolution, 0.005° tiles  
        5: 0.005,    # 5m resolution, 0.005° tiles
        10: 0.01,    # 10m resolution, 0.01° tiles
        30: 0.01,    # 30m resolution, 0.01° tiles
        60: 0.02     # 60m resolution, 0.02° tiles
    }
    
    # Layer 1 region size (larger for seamless coverage)
    LAYER1_REGION_SIZE = 0.1  # 0.1° regions
    
    def __init__(self, data_dir: str, resolution: int = 10):
        """
        Initialize the two-layer elevation library.
        
        Args:
            data_dir: Directory to store elevation data
            resolution: Resolution in meters (1, 3, 5, 10, 30, or 60)
        """
        # Initialize cache first to avoid __del__ issues
        self._open_datasets = {}
        
        if resolution not in self.RESOLUTIONS:
            raise ValueError(f"Invalid resolution {resolution}. Must be one of {list(self.RESOLUTIONS.keys())}")
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        self.data_dir = data_dir
        self.resolution = resolution
        self.tile_size = self.RESOLUTIONS[resolution]
        
        # Create directory structure
        self.resolution_dir = os.path.join(data_dir, f"{resolution}m")
        self.layer1_dir = os.path.join(self.resolution_dir, "layer1_seamless")
        self.layer2_dir = os.path.join(self.resolution_dir, "layer2_tiles")
        
        os.makedirs(self.layer1_dir, exist_ok=True)
        os.makedirs(self.layer2_dir, exist_ok=True)
        
        # Index files
        self.layer1_index_file = os.path.join(self.layer1_dir, "index.json")
        self.layer2_index_file = os.path.join(self.layer2_dir, "index.json")
        
        # Load or create indexes
        self.layer1_index = self._load_json(self.layer1_index_file, {"regions": {}})
        self.layer2_index = self._load_json(self.layer2_index_file, {"tiles": {}})
        
        logger.info(f"Initialized TwoLayerElevationLibrary: {resolution}m resolution, "
                   f"tile size {self.tile_size}°")
    
    def _load_json(self, filepath: str, default: dict) -> dict:
        """Load JSON file or return default"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    
    def _save_json(self, data: dict, filepath: str):
        """Save JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_region_key(self, lat: float, lon: float) -> str:
        """Get Layer 1 region key for a coordinate"""
        region_row = int(np.floor(lat / self.LAYER1_REGION_SIZE))
        region_col = int(np.floor(lon / self.LAYER1_REGION_SIZE))
        return f"r{region_row}_{region_col}"
    
    def _get_region_bounds(self, region_key: str) -> Bounds:
        """Get bounds for a Layer 1 region"""
        parts = region_key.split('_')
        row = int(parts[0][1:])  # Remove 'r' prefix
        col = int(parts[1])
        return Bounds(
            south=row * self.LAYER1_REGION_SIZE,
            north=(row + 1) * self.LAYER1_REGION_SIZE,
            west=col * self.LAYER1_REGION_SIZE,
            east=(col + 1) * self.LAYER1_REGION_SIZE
        )
    
    def _get_tile_key(self, lat: float, lon: float) -> str:
        """Get Layer 2 tile key for a coordinate"""
        # Small epsilon for boundary handling
        epsilon = 1e-10
        tile_row = int(np.floor((lat - epsilon) / self.tile_size))
        tile_col = int(np.floor((lon - epsilon) / self.tile_size))
        return f"{tile_row}_{tile_col}"
    
    def _get_tile_bounds(self, tile_key: str) -> Bounds:
        """Get bounds for a Layer 2 tile"""
        row, col = map(int, tile_key.split('_'))
        return Bounds(
            south=row * self.tile_size,
            north=(row + 1) * self.tile_size,
            west=col * self.tile_size,
            east=(col + 1) * self.tile_size
        )
    
    def _ensure_layer1_coverage(self, bounds: Bounds) -> List[str]:
        """
        Ensure Layer 1 has seamless coverage for the requested bounds.
        Downloads any missing regions.
        
        Returns:
            List of region keys that cover the bounds
        """
        # Calculate required regions
        min_row = int(np.floor(bounds.south / self.LAYER1_REGION_SIZE))
        max_row = int(np.floor(bounds.north / self.LAYER1_REGION_SIZE))
        min_col = int(np.floor(bounds.west / self.LAYER1_REGION_SIZE))
        max_col = int(np.floor(bounds.east / self.LAYER1_REGION_SIZE))
        
        required_regions = []
        regions_to_download = []
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                region_key = f"r{row}_{col}"
                required_regions.append(region_key)
                
                if region_key not in self.layer1_index["regions"]:
                    regions_to_download.append(region_key)
        
        # Download missing regions
        for region_key in regions_to_download:
            region_bounds = self._get_region_bounds(region_key)
            logger.info(f"Layer 1: Downloading region {region_key}")
            
            try:
                # Download seamless data from py3dep
                dem = py3dep.get_dem(
                    region_bounds.to_tuple(),
                    resolution=self.resolution,
                    crs="EPSG:4326"
                )
                
                # Reproject if needed
                if dem.rio.crs != "EPSG:4326":
                    logger.info(f"Reprojecting from {dem.rio.crs} to EPSG:4326")
                    dem = dem.rio.reproject("EPSG:4326")
                
                # Save to Layer 1
                filename = f"region_{region_key}.tif"
                filepath = os.path.join(self.layer1_dir, filename)
                
                dem_array = dem.to_numpy()
                transform = from_bounds(
                    float(dem.x.min()), float(dem.y.min()),
                    float(dem.x.max()), float(dem.y.max()),
                    dem_array.shape[1], dem_array.shape[0]
                )
                
                with rasterio.open(
                    filepath, 'w',
                    driver='GTiff',
                    height=dem_array.shape[0],
                    width=dem_array.shape[1],
                    count=1,
                    dtype=dem_array.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    compress='lzw'
                ) as dst:
                    dst.write(dem_array, 1)
                
                # Update index
                self.layer1_index["regions"][region_key] = {
                    "file": filename,
                    "bounds": {
                        "north": float(dem.y.max()),
                        "south": float(dem.y.min()),
                        "east": float(dem.x.max()),
                        "west": float(dem.x.min())
                    },
                    "shape": list(dem_array.shape),
                    "resolution_m": self.resolution,
                    "download_time": datetime.now().isoformat(),
                    "size_bytes": os.path.getsize(filepath)
                }
                
                self._save_json(self.layer1_index, self.layer1_index_file)
                
            except Exception as e:
                logger.error(f"Failed to download region {region_key}: {e}")
                raise
        
        return required_regions
    
    def _create_layer2_tiles(self, bounds: Bounds) -> List[str]:
        """
        Create Layer 2 tiles from Layer 1 data for the requested bounds.
        
        Returns:
            List of tile keys created
        """
        # Ensure Layer 1 coverage
        regions = self._ensure_layer1_coverage(bounds)
        
        # Calculate required tiles
        min_row = int(np.floor(bounds.south / self.tile_size))
        max_row = int(np.floor(bounds.north / self.tile_size))
        min_col = int(np.floor(bounds.west / self.tile_size))
        max_col = int(np.floor(bounds.east / self.tile_size))
        
        tiles_created = []
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                
                # Skip if tile already exists
                if tile_key in self.layer2_index["tiles"]:
                    continue
                
                tile_bounds = self._get_tile_bounds(tile_key)
                
                # Find which Layer 1 region contains this tile
                region_key = self._get_region_key(
                    (tile_bounds.north + tile_bounds.south) / 2,
                    (tile_bounds.east + tile_bounds.west) / 2
                )
                
                if region_key not in self.layer1_index["regions"]:
                    logger.warning(f"Region {region_key} not found for tile {tile_key}")
                    continue
                
                # Extract tile from Layer 1 region
                region_file = os.path.join(self.layer1_dir, 
                                         self.layer1_index["regions"][region_key]["file"])
                
                with rasterio.open(region_file) as src:
                    # Calculate the exact window to extract
                    # We need to be very careful about pixel alignment
                    west_px = (tile_bounds.west - src.bounds.left) / src.res[0]
                    east_px = (tile_bounds.east - src.bounds.left) / src.res[0]
                    north_px = (src.bounds.top - tile_bounds.north) / src.res[1]
                    south_px = (src.bounds.top - tile_bounds.south) / src.res[1]
                    
                    # Round to integer pixels
                    col_start = int(np.round(west_px))
                    col_stop = int(np.round(east_px))
                    row_start = int(np.round(north_px))
                    row_stop = int(np.round(south_px))
                    
                    # Ensure valid bounds
                    col_start = max(0, min(col_start, src.width))
                    col_stop = max(0, min(col_stop, src.width))
                    row_start = max(0, min(row_start, src.height))
                    row_stop = max(0, min(row_stop, src.height))
                    
                    if row_stop <= row_start or col_stop <= col_start:
                        continue
                    
                    # Read subset
                    window = Window(col_start, row_start,
                                  col_stop - col_start, row_stop - row_start)
                    data = src.read(1, window=window)
                    
                    # Save tile
                    tile_file = os.path.join(self.layer2_dir, f"tile_{tile_key}.tif")
                    
                    tile_transform = from_bounds(
                        tile_bounds.west, tile_bounds.south,
                        tile_bounds.east, tile_bounds.north,
                        data.shape[1], data.shape[0]
                    )
                    
                    with rasterio.open(
                        tile_file, 'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs='EPSG:4326',
                        transform=tile_transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(data, 1)
                    
                    # Update index
                    self.layer2_index["tiles"][tile_key] = {
                        "bounds": {
                            "north": tile_bounds.north,
                            "south": tile_bounds.south,
                            "east": tile_bounds.east,
                            "west": tile_bounds.west
                        },
                        "source_region": region_key,
                        "resolution_m": self.resolution,
                        "shape": list(data.shape),
                        "file": f"tile_{tile_key}.tif",
                        "size_bytes": os.path.getsize(tile_file),
                        "creation_time": datetime.now().isoformat()
                    }
                    
                    tiles_created.append(tile_key)
        
        if tiles_created:
            self._save_json(self.layer2_index, self.layer2_index_file)
            logger.info(f"Layer 2: Created {len(tiles_created)} tiles")
        
        return tiles_created
    
    def load_area(self, bounds: Bounds) -> Dict[str, Any]:
        """
        Load elevation data for an area.
        
        Returns:
            Dictionary with status and statistics
        """
        logger.info(f"Loading area: {bounds.south:.4f},{bounds.west:.4f} to "
                   f"{bounds.north:.4f},{bounds.east:.4f}")
        
        try:
            # Create Layer 2 tiles (which ensures Layer 1 coverage)
            tiles_created = self._create_layer2_tiles(bounds)
            
            # Count existing tiles in the area
            min_row = int(np.floor(bounds.south / self.tile_size))
            max_row = int(np.floor(bounds.north / self.tile_size))
            min_col = int(np.floor(bounds.west / self.tile_size))
            max_col = int(np.floor(bounds.east / self.tile_size))
            
            total_tiles = 0
            existing_tiles = 0
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    tile_key = f"{row}_{col}"
                    total_tiles += 1
                    if tile_key in self.layer2_index["tiles"]:
                        existing_tiles += 1
            
            return {
                "status": "success",
                "message": "Area loaded successfully with no tile boundary artifacts",
                "tiles_created": len(tiles_created),
                "tiles_existing": existing_tiles - len(tiles_created),
                "total_tiles": total_tiles,
                "resolution_m": self.resolution
            }
            
        except Exception as e:
            logger.error(f"Failed to load area: {e}")
            return {
                "status": "error",
                "message": str(e),
                "tiles_created": 0
            }
    
    def remove_area(self, bounds: Bounds) -> Dict[str, Any]:
        """
        Remove elevation data for an area.
        Note: Only removes Layer 2 tiles. Layer 1 regions are kept for efficiency.
        
        Returns:
            Dictionary with status and statistics
        """
        logger.info(f"Removing area: {bounds.south:.4f},{bounds.west:.4f} to "
                   f"{bounds.north:.4f},{bounds.east:.4f}")
        
        # Calculate tiles to remove
        min_row = int(np.floor(bounds.south / self.tile_size))
        max_row = int(np.floor(bounds.north / self.tile_size))
        min_col = int(np.floor(bounds.west / self.tile_size))
        max_col = int(np.floor(bounds.east / self.tile_size))
        
        tiles_removed = 0
        tiles_not_found = 0
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                
                if tile_key in self.layer2_index["tiles"]:
                    # Remove file
                    tile_file = os.path.join(self.layer2_dir, f"tile_{tile_key}.tif")
                    try:
                        if os.path.exists(tile_file):
                            os.remove(tile_file)
                            logger.debug(f"Removed tile {tile_key}")
                        
                        # Remove from index
                        del self.layer2_index["tiles"][tile_key]
                        tiles_removed += 1
                    except Exception as e:
                        logger.error(f"Failed to remove tile {tile_key}: {e}")
                else:
                    tiles_not_found += 1
        
        # Save updated index
        self._save_json(self.layer2_index, self.layer2_index_file)
        
        # Clean up any cached datasets
        self._cleanup_cache()
        
        return {
            "status": "success",
            "tiles_removed": tiles_removed,
            "tiles_not_found": tiles_not_found,
            "total_tiles_checked": tiles_removed + tiles_not_found
        }
    
    def remove_all(self) -> Dict[str, Any]:
        """
        Remove all loaded elevation data.
        Removes both Layer 1 and Layer 2 data.
        
        Returns:
            Dictionary with status and statistics
        """
        logger.info("Removing all loaded elevation data...")
        
        tiles_removed = 0
        regions_removed = 0
        
        try:
            # Remove Layer 2 tiles
            for tile_key in list(self.layer2_index["tiles"].keys()):
                tile_file = os.path.join(self.layer2_dir, f"tile_{tile_key}.tif")
                if os.path.exists(tile_file):
                    os.remove(tile_file)
                del self.layer2_index["tiles"][tile_key]
                tiles_removed += 1
            
            # Remove Layer 1 regions
            for region_key in list(self.layer1_index["regions"].keys()):
                region_info = self.layer1_index["regions"][region_key]
                region_file = os.path.join(self.layer1_dir, region_info["file"])
                if os.path.exists(region_file):
                    os.remove(region_file)
                del self.layer1_index["regions"][region_key]
                regions_removed += 1
            
            # Save empty indexes
            self._save_json(self.layer1_index, self.layer1_index_file)
            self._save_json(self.layer2_index, self.layer2_index_file)
            
            # Clean up cache
            self._cleanup_cache()
            
            return {
                "status": "success",
                "tiles_removed": tiles_removed,
                "regions_removed": regions_removed
            }
            
        except Exception as e:
            logger.error(f"Error during remove_all: {e}")
            return {
                "status": "error",
                "message": str(e),
                "tiles_removed": tiles_removed,
                "regions_removed": regions_removed
            }
    
    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get elevation at a single point.
        Fails if data not preloaded.
        
        Raises:
            ValueError: If elevation data not available at the point
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.layer2_index["tiles"]:
            raise ValueError(
                f"Elevation data not available at ({lat:.4f}, {lon:.4f}). "
                f"Please load this area first using load_area()."
            )
        
        tile_file = os.path.join(self.layer2_dir, f"tile_{tile_key}.tif")
        
        # Use cached dataset if available
        if tile_file in self._open_datasets:
            src = self._open_datasets[tile_file]
        else:
            src = rasterio.open(tile_file)
            self._open_datasets[tile_file] = src
        
        # Convert lat/lon to pixel coordinates
        row, col = src.index(lon, lat)
        
        # Clamp indices to valid bounds
        row = min(max(0, row), src.height - 1)
        col = min(max(0, col), src.width - 1)
        
        # Read single pixel
        elevation = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
        
        return float(elevation)
    
    def get_elevation_array(self, bounds: Bounds) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get elevation array for a bounding box.
        Fails if any required data not preloaded.
        
        Returns:
            Tuple of (elevation_array, metadata_dict)
        
        Raises:
            ValueError: If elevation data not available for the bounds
        """
        # Calculate required tiles
        min_row = int(np.floor(bounds.south / self.tile_size))
        max_row = int(np.floor(bounds.north / self.tile_size))
        min_col = int(np.floor(bounds.west / self.tile_size))
        max_col = int(np.floor(bounds.east / self.tile_size))
        
        # Check if all tiles are available
        missing_tiles = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                if tile_key not in self.layer2_index["tiles"]:
                    missing_tiles.append(tile_key)
        
        if missing_tiles:
            raise ValueError(
                f"Elevation data not available for requested bounds. "
                f"Missing tiles: {missing_tiles[:5]}{'...' if len(missing_tiles) > 5 else ''}. "
                f"Please load this area first using load_area()."
            )
        
        # Collect all tiles
        tiles_to_merge = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                if tile_key in self.layer2_index["tiles"]:
                    tile_file = os.path.join(self.layer2_dir, f"tile_{tile_key}.tif")
                    if os.path.exists(tile_file):
                        tiles_to_merge.append(tile_file)
        
        if not tiles_to_merge:
            raise ValueError("No tiles available for requested bounds")
        
        if len(tiles_to_merge) == 1:
            # Single tile - simple case
            with rasterio.open(tiles_to_merge[0]) as src:
                # Read the exact subset needed
                row_start, col_start = src.index(bounds.west, bounds.north)
                row_stop, col_stop = src.index(bounds.east, bounds.south)
                
                # Clamp to valid bounds
                row_start = max(0, min(row_start, src.height))
                row_stop = max(0, min(row_stop + 1, src.height))
                col_start = max(0, min(col_start, src.width))
                col_stop = max(0, min(col_stop + 1, src.width))
                
                window = Window(col_start, row_start,
                              col_stop - col_start, row_stop - row_start)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
        else:
            # Multiple tiles - merge them
            datasets = []
            for f in tiles_to_merge:
                if f in self._open_datasets:
                    datasets.append(self._open_datasets[f])
                else:
                    ds = rasterio.open(f)
                    self._open_datasets[f] = ds
                    datasets.append(ds)
            
            # Merge tiles - use first method to avoid duplication at boundaries
            data, transform = merge(
                datasets, 
                bounds=bounds.to_tuple(),
                method='first',  # Use first valid pixel, don't blend overlaps
                nodata=0  # Treat 0 as nodata
            )
            data = data[0]  # Get first band
        
        # Build metadata
        metadata = {
            "bounds": {
                "north": bounds.north,
                "south": bounds.south,
                "east": bounds.east,
                "west": bounds.west
            },
            "shape": list(data.shape),
            "resolution_m": self.resolution,
            "tiles_used": len(tiles_to_merge),
            "transform": {
                "a": transform.a,
                "b": transform.b,
                "c": transform.c,
                "d": transform.d,
                "e": transform.e,
                "f": transform.f
            },
            "crs": "EPSG:4326"
        }
        
        return data, metadata
    
    def list_loaded_areas(self) -> Dict[str, Any]:
        """
        List all loaded areas, intelligently merging adjacent tiles.
        
        Returns:
            Dictionary with loaded areas and statistics
        """
        if not self.layer2_index["tiles"]:
            return {
                "areas": [],
                "total_tiles": 0,
                "total_regions": len(self.layer1_index["regions"]),
                "total_size_mb": 0
            }
        
        # Group tiles into contiguous areas
        tiles_by_key = {}
        for tile_key, tile_info in self.layer2_index["tiles"].items():
            row, col = map(int, tile_key.split('_'))
            tiles_by_key[(row, col)] = tile_info
        
        # Find connected components
        visited = set()
        areas = []
        
        def get_neighbors(row, col):
            """Get valid neighboring tile coordinates"""
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if (nr, nc) in tiles_by_key and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            return neighbors
        
        def bfs_area(start_row, start_col):
            """Find all connected tiles starting from a point"""
            queue = [(start_row, start_col)]
            area_tiles = []
            min_row = max_row = start_row
            min_col = max_col = start_col
            
            while queue:
                row, col = queue.pop(0)
                if (row, col) in visited:
                    continue
                
                visited.add((row, col))
                area_tiles.append((row, col))
                
                min_row = min(min_row, row)
                max_row = max(max_row, row)
                min_col = min(min_col, col)
                max_col = max(max_col, col)
                
                for nr, nc in get_neighbors(row, col):
                    if (nr, nc) not in visited:
                        queue.append((nr, nc))
            
            return area_tiles, min_row, max_row, min_col, max_col
        
        # Find all contiguous areas
        for (row, col) in tiles_by_key:
            if (row, col) not in visited:
                area_tiles, min_row, max_row, min_col, max_col = bfs_area(row, col)
                
                area_bounds = Bounds(
                    south=min_row * self.tile_size,
                    north=(max_row + 1) * self.tile_size,
                    west=min_col * self.tile_size,
                    east=(max_col + 1) * self.tile_size
                )
                
                areas.append({
                    "bounds": {
                        "north": area_bounds.north,
                        "south": area_bounds.south,
                        "east": area_bounds.east,
                        "west": area_bounds.west
                    },
                    "tiles": len(area_tiles),
                    "area_sq_km": ((area_bounds.north - area_bounds.south) * 111) *
                                 ((area_bounds.east - area_bounds.west) * 111 * 
                                  np.cos(np.radians((area_bounds.north + area_bounds.south) / 2)))
                })
        
        # Calculate total size
        total_size = 0
        for tile_info in self.layer2_index["tiles"].values():
            total_size += tile_info.get("size_bytes", 0)
        
        # Add Layer 1 size
        for region_info in self.layer1_index["regions"].values():
            total_size += region_info.get("size_bytes", 0)
        
        return {
            "areas": areas,
            "total_tiles": len(self.layer2_index["tiles"]),
            "total_regions": len(self.layer1_index["regions"]),
            "total_size_mb": total_size / (1024 * 1024),
            "resolution_m": self.resolution
        }
    
    def get_tile_info(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about the tile containing a point.
        
        Returns:
            None if tile not loaded, otherwise tile information
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.layer2_index["tiles"]:
            return None
        
        info = self.layer2_index["tiles"][tile_key].copy()
        info["tile_key"] = tile_key
        
        # Add Layer 1 region info
        if "source_region" in info:
            region_key = info["source_region"]
            if region_key in self.layer1_index["regions"]:
                info["layer1_region"] = self.layer1_index["regions"][region_key].copy()
        
        return info
    
    def get_all_tiles_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded tiles"""
        tiles_info = []
        for tile_key, tile_data in self.layer2_index["tiles"].items():
            info = tile_data.copy()
            info["tile_key"] = tile_key
            tiles_info.append(info)
        return tiles_info
    
    def _cleanup_cache(self):
        """Clean up cached rasterio datasets"""
        for ds in self._open_datasets.values():
            ds.close()
        self._open_datasets.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup_cache()

