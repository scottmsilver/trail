#!/usr/bin/env python3
"""
Elevation data library with explicit preloading.
Data is stored as tiles on disk. Must preload before accessing.
"""

import os
import json
import hashlib
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import py3dep

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
        """Return as (west, south, east, north) for rasterio"""
        return (self.west, self.south, self.east, self.north)
    
    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within these bounds"""
        return (self.south <= lat <= self.north and 
                self.west <= lon <= self.east)


class ElevationLibrary:
    """
    Manages elevation data with explicit preloading.
    No automatic fetching - user must preload areas first.
    """
    
    def __init__(self, data_dir: str, resolution: int = 10):
        """
        Initialize elevation library.
        
        Args:
            data_dir: Directory to store elevation tiles (required)
            resolution: Resolution in meters (1m, 3m, 5m, 10m, 30m, 60m)
                - 1m: Only available in limited US areas
                - 3m: Available in most of continental US (from USGS 3DEP)
                - 5m: Available in Alaska
                - 10m: Available in most of US (from USGS 3DEP)
                - 30m: Global coverage (from USGS National Map)
                - 60m: Alaska coverage
        """
        if not data_dir:
            raise ValueError("data_dir is required. Please specify where to store elevation data.")
        valid_resolutions = [1, 3, 5, 10, 30, 60]
        if resolution not in valid_resolutions:
            raise ValueError(f"Invalid resolution {resolution}m. Valid options: {valid_resolutions}")
        
        self.data_dir = data_dir
        self.resolution = resolution
        
        # Adjust tile size based on resolution
        # Smaller tiles for higher resolution data to keep file sizes reasonable
        if resolution <= 3:
            self.tile_size = 0.005  # ~0.5km tiles for 1m/3m data
        elif resolution <= 10:
            self.tile_size = 0.01   # ~1km tiles for 5m/10m data
        else:
            self.tile_size = 0.02   # ~2km tiles for 30m/60m data
        
        # Create data directory with resolution subdirectory
        self.resolution_dir = os.path.join(self.data_dir, f"{resolution}m")
        os.makedirs(self.resolution_dir, exist_ok=True)
        
        # Load tile index for this resolution
        self.index_file = os.path.join(self.resolution_dir, "tile_index.json")
        self.tile_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load tile index from disk"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"tiles": {}, "version": "1.0"}
    
    def _save_index(self):
        """Save tile index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump(self.tile_index, f, indent=2)
    
    def _get_tile_key(self, lat: float, lon: float) -> str:
        """Get tile key for a coordinate"""
        tile_row = int(np.floor(lat / self.tile_size))
        tile_col = int(np.floor(lon / self.tile_size))
        return f"{tile_row}_{tile_col}"
    
    def _get_tile_bounds(self, tile_key: str) -> Bounds:
        """Get bounds for a tile key"""
        row, col = map(int, tile_key.split('_'))
        return Bounds(
            south=row * self.tile_size,
            north=(row + 1) * self.tile_size,
            west=col * self.tile_size,
            east=(col + 1) * self.tile_size
        )
    
    def _get_tile_filename(self, tile_key: str) -> str:
        """Get filename for a tile"""
        return os.path.join(self.resolution_dir, f"tile_{tile_key}.tif")
    
    def load_area(self, bounds: Bounds) -> Dict[str, Any]:
        """
        Load elevation data for an area.
        Downloads and stores as tiles on disk.
        
        Returns:
            Dictionary with status and statistics
        """
        print(f"Loading area: {bounds.south:.4f},{bounds.west:.4f} to {bounds.north:.4f},{bounds.east:.4f}")
        
        # Calculate required tiles
        min_row = int(np.floor(bounds.south / self.tile_size))
        max_row = int(np.floor(bounds.north / self.tile_size))
        min_col = int(np.floor(bounds.west / self.tile_size))
        max_col = int(np.floor(bounds.east / self.tile_size))
        
        tiles_needed = []
        tiles_existing = []
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                if tile_key in self.tile_index["tiles"]:
                    tiles_existing.append(tile_key)
                else:
                    tiles_needed.append(tile_key)
        
        print(f"Tiles needed: {len(tiles_needed)}, already loaded: {len(tiles_existing)}")
        
        if not tiles_needed:
            return {
                "status": "success",
                "message": "All tiles already loaded",
                "tiles_downloaded": 0,
                "tiles_existing": len(tiles_existing)
            }
        
        # Download tiles
        downloaded = 0
        failed = []
        
        for i, tile_key in enumerate(tiles_needed):
            print(f"Downloading tile {i+1}/{len(tiles_needed)}: {tile_key}")
            
            tile_bounds = self._get_tile_bounds(tile_key)
            tile_file = self._get_tile_filename(tile_key)
            
            try:
                # Download elevation data using py3dep
                dem = py3dep.get_dem(
                    tile_bounds.to_tuple(),
                    resolution=self.resolution,
                    crs="EPSG:4326"
                )
                
                # Save to file
                transform = from_bounds(
                    tile_bounds.west, tile_bounds.south,
                    tile_bounds.east, tile_bounds.north,
                    dem.shape[1], dem.shape[0]
                )
                
                with rasterio.open(
                    tile_file, 'w',
                    driver='GTiff',
                    height=dem.shape[0],
                    width=dem.shape[1],
                    count=1,
                    dtype=dem.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    compress='lzw'
                ) as dst:
                    dst.write(dem, 1)
                
                # Update index with comprehensive metadata
                self.tile_index["tiles"][tile_key] = {
                    "bounds": {
                        "north": tile_bounds.north,
                        "south": tile_bounds.south,
                        "east": tile_bounds.east,
                        "west": tile_bounds.west
                    },
                    "resolution_m": self.resolution,
                    "shape": list(dem.shape),  # [height, width]
                    "transform": {
                        "a": transform.a,  # pixel width in degrees
                        "b": transform.b,  # rotation (usually 0)
                        "c": transform.c,  # x coordinate of upper left corner
                        "d": transform.d,  # rotation (usually 0)
                        "e": transform.e,  # pixel height in degrees (negative)
                        "f": transform.f   # y coordinate of upper left corner
                    },
                    "crs": "EPSG:4326",
                    "pixel_size_degrees": {
                        "x": transform.a,
                        "y": abs(transform.e)
                    },
                    "pixel_size_meters": {
                        "x": transform.a * 111000 * np.cos(np.radians((tile_bounds.north + tile_bounds.south) / 2)),
                        "y": abs(transform.e) * 111000
                    },
                    "file": os.path.basename(tile_file),
                    "size_bytes": os.path.getsize(tile_file) if os.path.exists(tile_file) else 0,
                    "download_time": datetime.now().isoformat()
                }
                
                downloaded += 1
                
            except Exception as e:
                print(f"Failed to download tile {tile_key}: {e}")
                failed.append(tile_key)
        
        # Save updated index
        self._save_index()
        
        return {
            "status": "success" if not failed else "partial",
            "tiles_downloaded": downloaded,
            "tiles_existing": len(tiles_existing),
            "tiles_failed": failed,
            "total_tiles": len(tiles_needed) + len(tiles_existing)
        }
    
    def remove_area(self, bounds: Bounds) -> Dict[str, Any]:
        """
        Remove elevation data for an area.
        Deletes tiles from disk and updates index.
        
        Returns:
            Dictionary with status and statistics
        """
        print(f"Removing area: {bounds.south:.4f},{bounds.west:.4f} to {bounds.north:.4f},{bounds.east:.4f}")
        
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
                
                if tile_key in self.tile_index["tiles"]:
                    # Remove file
                    tile_file = self._get_tile_filename(tile_key)
                    try:
                        if os.path.exists(tile_file):
                            os.remove(tile_file)
                            print(f"  Removed tile {tile_key}")
                        
                        # Remove from index
                        del self.tile_index["tiles"][tile_key]
                        tiles_removed += 1
                    except Exception as e:
                        print(f"  Failed to remove tile {tile_key}: {e}")
                else:
                    tiles_not_found += 1
        
        # Save updated index
        self._save_index()
        
        return {
            "status": "success",
            "tiles_removed": tiles_removed,
            "tiles_not_found": tiles_not_found,
            "total_tiles_checked": tiles_removed + tiles_not_found
        }
    
    def remove_all(self) -> Dict[str, Any]:
        """
        Remove all loaded elevation data.
        Deletes all tiles and clears index.
        
        Returns:
            Dictionary with status and statistics
        """
        print("Removing all loaded elevation data...")
        
        tiles_removed = 0
        failed_removals = []
        
        # Get all tiles
        all_tiles = list(self.tile_index["tiles"].keys())
        
        for tile_key in all_tiles:
            tile_file = self._get_tile_filename(tile_key)
            try:
                if os.path.exists(tile_file):
                    os.remove(tile_file)
                    print(f"  Removed tile {tile_key}")
                
                del self.tile_index["tiles"][tile_key]
                tiles_removed += 1
            except Exception as e:
                print(f"  Failed to remove tile {tile_key}: {e}")
                failed_removals.append(tile_key)
        
        # Save empty index
        self._save_index()
        
        return {
            "status": "success" if not failed_removals else "partial",
            "tiles_removed": tiles_removed,
            "failed_removals": failed_removals
        }
    
    def get_tile_info(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about the tile containing a point.
        Returns None if tile not loaded.
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.tile_index["tiles"]:
            return None
        
        return self.tile_index["tiles"][tile_key].copy()
    
    def get_all_tiles_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded tiles"""
        tiles_info = []
        for tile_key, tile_data in self.tile_index["tiles"].items():
            info = tile_data.copy()
            info["tile_key"] = tile_key
            tiles_info.append(info)
        return tiles_info
    
    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get elevation at a single point.
        Fails if data not preloaded.
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.tile_index["tiles"]:
            raise ValueError(
                f"Elevation data not available at ({lat:.4f}, {lon:.4f}). "
                f"Please load this area first using load_area()."
            )
        
        tile_file = self._get_tile_filename(tile_key)
        
        with rasterio.open(tile_file) as src:
            # Convert lat/lon to pixel coordinates
            row, col = src.index(lon, lat)
            
            # Ensure indices are within bounds
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                raise ValueError(
                    f"Coordinates ({lat:.4f}, {lon:.4f}) are outside the loaded tile bounds"
                )
            
            # Read single pixel
            elevation = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            
            return float(elevation)
    
    def get_elevation_array(self, bounds: Bounds) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get elevation array for a bounding box.
        Fails if any required data not preloaded.
        
        Returns:
            Tuple of (elevation_array, metadata_dict)
            metadata_dict contains:
                - transform: rasterio transform
                - bounds: actual bounds of returned data
                - shape: array shape
                - resolution_m: resolution in meters
                - pixel_size_degrees: pixel size in degrees
                - pixel_size_meters: pixel size in meters
                - crs: coordinate reference system
        """
        # Check all required tiles exist
        min_row = int(np.floor(bounds.south / self.tile_size))
        max_row = int(np.floor(bounds.north / self.tile_size))
        min_col = int(np.floor(bounds.west / self.tile_size))
        max_col = int(np.floor(bounds.east / self.tile_size))
        
        missing_tiles = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                if tile_key not in self.tile_index["tiles"]:
                    missing_tiles.append(tile_key)
        
        if missing_tiles:
            raise ValueError(
                f"Elevation data not available for {len(missing_tiles)} tiles. "
                f"Missing tiles: {missing_tiles[:5]}{'...' if len(missing_tiles) > 5 else ''}. "
                f"Please load this area first using load_area()."
            )
        
        # Merge tiles
        # For simplicity, we'll load each tile and stitch them together
        # In production, you might use rasterio.merge
        
        tiles_data = []
        
        for row in range(min_row, max_row + 1):
            row_tiles = []
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                tile_file = self._get_tile_filename(tile_key)
                
                with rasterio.open(tile_file) as src:
                    data = src.read(1)
                    if row == min_row and col == min_col:
                        # Save reference transform from first tile
                        base_transform = src.transform
                    
                    row_tiles.append(data)
            
            # Stack columns horizontally
            tiles_data.append(np.hstack(row_tiles))
        
        # Stack rows vertically (flip because of coordinate system)
        merged_data = np.vstack(tiles_data[::-1])
        
        # Calculate transform for merged data
        merged_transform = from_bounds(
            min_col * self.tile_size,
            min_row * self.tile_size,
            (max_col + 1) * self.tile_size,
            (max_row + 1) * self.tile_size,
            merged_data.shape[1],
            merged_data.shape[0]
        )
        
        # Calculate actual bounds
        actual_bounds = Bounds(
            south=min_row * self.tile_size,
            north=(max_row + 1) * self.tile_size,
            west=min_col * self.tile_size,
            east=(max_col + 1) * self.tile_size
        )
        
        # Calculate pixel sizes
        pixel_width_deg = merged_transform.a
        pixel_height_deg = abs(merged_transform.e)
        
        # Approximate pixel size in meters (at center latitude)
        center_lat = (actual_bounds.north + actual_bounds.south) / 2
        pixel_width_m = pixel_width_deg * 111000 * np.cos(np.radians(center_lat))
        pixel_height_m = pixel_height_deg * 111000
        
        metadata = {
            "transform": merged_transform,
            "bounds": {
                "north": actual_bounds.north,
                "south": actual_bounds.south,
                "east": actual_bounds.east,
                "west": actual_bounds.west
            },
            "shape": merged_data.shape,
            "resolution_m": self.resolution,
            "pixel_size_degrees": {
                "x": pixel_width_deg,
                "y": pixel_height_deg
            },
            "pixel_size_meters": {
                "x": pixel_width_m,
                "y": pixel_height_m
            },
            "crs": "EPSG:4326",
            "tiles_used": (max_row - min_row + 1) * (max_col - min_col + 1)
        }
        
        return merged_data, metadata
    
    def list_loaded_areas(self) -> Dict[str, Any]:
        """List all loaded areas with statistics"""
        if not self.tile_index["tiles"]:
            return {
                "areas": [],
                "total_tiles": 0,
                "total_size_mb": 0
            }
        
        # Group tiles by general area
        # This is a simple implementation - in practice you might want
        # to track actual load requests
        
        tiles = self.tile_index["tiles"]
        total_size = 0
        
        for tile_key, tile_info in tiles.items():
            tile_file = self._get_tile_filename(tile_key)
            if os.path.exists(tile_file):
                total_size += os.path.getsize(tile_file)
        
        # Find bounding box of all tiles
        all_bounds = [t["bounds"] for t in tiles.values()]
        overall_bounds = {
            "north": max(b["north"] for b in all_bounds),
            "south": min(b["south"] for b in all_bounds),
            "east": max(b["east"] for b in all_bounds),
            "west": min(b["west"] for b in all_bounds)
        }
        
        return {
            "areas": [overall_bounds],
            "total_tiles": len(tiles),
            "total_size_mb": total_size / (1024 * 1024),
            "tile_size_degrees": self.tile_size,
            "resolution_meters": self.resolution
        }


# Example command-line interface
if __name__ == "__main__":
    import sys
    
    # Check for data-dir and resolution flags
    data_dir = None
    resolution = 10  # default
    args = sys.argv[1:]
    
    # Extract data-dir (required)
    if "--data-dir" in args:
        idx = args.index("--data-dir")
        if idx + 1 < len(args):
            data_dir = args[idx + 1]
            args.pop(idx + 1)  # remove value
            args.pop(idx)  # remove flag
    
    # Extract resolution (optional)
    if "--resolution" in args:
        idx = args.index("--resolution")
        if idx + 1 < len(args):
            resolution = int(args[idx + 1])
            args.pop(idx + 1)  # remove value
            args.pop(idx)  # remove flag
    
    if not data_dir or len(args) < 1:
        print("Usage:")
        print("  python elevation.py --data-dir <path> [--resolution <meters>] <command> [args]")
        print("\nCommands:")
        print("  load <south> <north> <west> <east>  - Load elevation data for area")
        print("  remove <south> <north> <west> <east> - Remove elevation data for area")
        print("  remove-all                           - Remove all loaded data")
        print("  get <lat> <lon>                      - Get elevation at point")
        print("  list [--detailed]                    - List loaded areas")
        print("\nRequired:")
        print("  --data-dir <path>    Directory to store elevation data")
        print("\nOptional:")
        print("  --resolution <m>     Resolution in meters: 1, 3, 5, 10 (default), 30, 60")
        print("\nExamples:")
        print("  python elevation.py --data-dir ./elev_data --resolution 3 load 40.65 40.66 -111.51 -111.50")
        print("  python elevation.py --data-dir ./elev_data get 40.655 -111.505")
        print("  python elevation.py --data-dir ./elev_data list --detailed")
        sys.exit(1)
    
    try:
        lib = ElevationLibrary(data_dir=data_dir, resolution=resolution)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    command = args[0]
    
    if command == "load":
        if len(args) != 5:
            print("Usage: python elevation.py [--resolution <meters>] load <south> <north> <west> <east>")
            sys.exit(1)
        
        bounds = Bounds(
            south=float(args[1]),
            north=float(args[2]),
            west=float(args[3]),
            east=float(args[4])
        )
        
        result = lib.load_area(bounds)
        print(f"\nResult: {result}")
    
    elif command == "remove":
        if len(args) != 5:
            print("Usage: python elevation.py [--resolution <meters>] remove <south> <north> <west> <east>")
            sys.exit(1)
        
        bounds = Bounds(
            south=float(args[1]),
            north=float(args[2]),
            west=float(args[3]),
            east=float(args[4])
        )
        
        result = lib.remove_area(bounds)
        print(f"\nResult: {result}")
    
    elif command == "remove-all":
        result = lib.remove_all()
        print(f"\nResult: {result}")
        
    elif command == "get":
        if len(args) != 3:
            print("Usage: python elevation.py [--resolution <meters>] get <lat> <lon>")
            sys.exit(1)
        
        lat = float(args[1])
        lon = float(args[2])
        
        try:
            elevation = lib.get_elevation(lat, lon)
            print(f"Elevation at ({lat}, {lon}): {elevation:.1f} meters")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif command == "list":
        info = lib.list_loaded_areas()
        print(f"\nLoaded elevation data ({resolution}m resolution):")
        print(f"  Total tiles: {info['total_tiles']}")
        print(f"  Total size: {info['total_size_mb']:.1f} MB")
        print(f"  Resolution: {info['resolution_meters']}m")
        if info['areas']:
            for area in info['areas']:
                print(f"  Area: {area['south']:.3f},{area['west']:.3f} to {area['north']:.3f},{area['east']:.3f}")
        
        # Show detailed tile info if requested
        if "--detailed" in args:
            print("\nDetailed tile information:")
            tiles = lib.get_all_tiles_info()
            for tile in tiles:
                print(f"\n  Tile {tile['tile_key']}:")
                print(f"    Bounds: {tile['bounds']['south']:.4f},{tile['bounds']['west']:.4f} to {tile['bounds']['north']:.4f},{tile['bounds']['east']:.4f}")
                print(f"    Shape: {tile['shape'][0]}x{tile['shape'][1]} pixels")
                print(f"    Pixel size: {tile['pixel_size_meters']['x']:.1f}m x {tile['pixel_size_meters']['y']:.1f}m")
                print(f"    File size: {tile['size_bytes'] / 1024:.1f} KB")
                print(f"    Downloaded: {tile['download_time']}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)