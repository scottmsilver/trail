#!/usr/bin/env python3
"""
Slope Layer - Provides slope and slope change (curvature) analysis
Built on top of the two-layer elevation system.

This layer computes and stores:
- Slope (first derivative) - gradient magnitude in degrees
- Slope change (second derivative) - rate of change of slope
- Aspect (direction of slope) - compass bearing in degrees

Like the elevation system, this follows explicit load/fail behavior.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from scipy import ndimage

from elevation import TwoLayerElevationLibrary, Bounds


@dataclass
class SlopeData:
    """Container for slope analysis data"""
    slope: np.ndarray  # Slope in degrees
    slope_change: np.ndarray  # Rate of change of slope
    aspect: np.ndarray  # Direction of slope (0-360 degrees, 0=North)
    bounds: Bounds
    resolution_m: float


class SlopeLayer:
    """
    Slope analysis layer that builds on elevation data.
    
    Provides:
    - Slope (gradient magnitude) in degrees
    - Slope change (curvature/second derivative)
    - Aspect (slope direction)
    
    All data is tile-based and follows explicit load/fail pattern.
    """
    
    def __init__(self, elevation_lib: TwoLayerElevationLibrary, data_dir: str = None):
        """
        Initialize slope layer.
        
        Args:
            elevation_lib: The elevation library to use for source data
            data_dir: Directory for slope data (defaults to elevation_dir/slope)
        """
        self.elevation_lib = elevation_lib
        self.resolution = elevation_lib.resolution
        self.tile_size = elevation_lib.tile_size
        
        # Set up data directory
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(elevation_lib.resolution_dir), 
                "slope"
            )
        self.data_dir = data_dir
        self.resolution_dir = os.path.join(data_dir, f"{self.resolution}m")
        self.tiles_dir = os.path.join(self.resolution_dir, "tiles")
        
        # Create directories
        os.makedirs(self.tiles_dir, exist_ok=True)
        
        # Index for tracking loaded tiles
        self.index_file = os.path.join(self.resolution_dir, "slope_index.json")
        self.index = self._load_index()
        
    def _load_index(self) -> Dict[str, Any]:
        """Load or create the slope tile index"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"tiles": {}, "version": "1.0"}
    
    def _save_index(self):
        """Save the slope tile index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def compute_area(self, bounds: Bounds) -> Dict[str, Any]:
        """
        Compute slope data for an area from elevation data.
        
        This creates slope tiles for the specified bounds.
        Requires that elevation data is already loaded.
        
        Args:
            bounds: Area to compute slopes for
            
        Returns:
            Dict with status and statistics
        """
        # First check if elevation data is available
        try:
            elev_array, elev_meta = self.elevation_lib.get_elevation_array(bounds)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Elevation data not available: {str(e)}"
            }
        
        # Compute slopes for the entire area
        # Calculate pixel size in meters from the transform
        transform = elev_meta.get("transform", {})
        if transform:
            # Pixel width in degrees
            pixel_width_deg = abs(transform.get("a", 0))
            pixel_height_deg = abs(transform.get("e", 0))
            # Convert to meters (approximate at this latitude)
            lat_center = (bounds.north + bounds.south) / 2
            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat_center))
            pixel_width_m = pixel_width_deg * meters_per_degree_lon
            pixel_height_m = pixel_height_deg * meters_per_degree_lat
            pixel_size_m = (pixel_width_m + pixel_height_m) / 2  # Average
        else:
            # Fallback calculation
            pixel_size_m = self.resolution
        
        slope_data = self._compute_slopes(elev_array, bounds, pixel_size_m)
        
        # Now tile the results using same tiling as elevation Layer 2
        tiles_created = self._create_slope_tiles(slope_data, bounds)
        
        return {
            "status": "success",
            "tiles_created": tiles_created,
            "bounds": {
                "north": bounds.north,
                "south": bounds.south,
                "east": bounds.east,
                "west": bounds.west
            },
            "resolution_m": self.resolution
        }
    
    def _compute_slopes(self, elevation: np.ndarray, bounds: Bounds, 
                       pixel_size_m: float) -> SlopeData:
        """
        Compute slope metrics from elevation data.
        
        Args:
            elevation: Elevation array
            bounds: Geographic bounds
            pixel_size_m: Size of each pixel in meters
            
        Returns:
            SlopeData with computed metrics
        """
        # Compute gradients using Sobel operators
        # These give us dx/dy in elevation units per pixel
        grad_y, grad_x = np.gradient(elevation)
        
        # Convert to proper units (rise over run)
        # grad is in meters per pixel, divide by pixel size to get dimensionless gradient
        grad_x = grad_x / pixel_size_m
        grad_y = grad_y / pixel_size_m
        
        # Compute slope magnitude in degrees
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)
        
        # Compute aspect (direction of slope)
        # Note: grad_y is positive going north (up in geographic coordinates)
        # but array indices increase going south (down)
        # So we need to negate grad_y
        # atan2 takes (y, x) and returns angle from positive x-axis
        aspect_rad = np.arctan2(-grad_x, grad_y)  # Note the order and sign
        # Convert to compass bearing (0=North, 90=East, etc.)
        # atan2 gives: 0=East, π/2=North, π=West, -π/2=South
        # We want: 0=North, 90=East, 180=South, 270=West
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (90 - aspect_deg) % 360
        
        # Compute second derivatives for slope change
        # This gives us the curvature of the terrain
        grad2_xx = np.gradient(grad_x, axis=1) / pixel_size_m
        grad2_yy = np.gradient(grad_y, axis=0) / pixel_size_m
        grad2_xy = np.gradient(grad_x, axis=0) / pixel_size_m
        
        # Total curvature (Laplacian)
        curvature = grad2_xx + grad2_yy
        
        # Convert curvature to slope change in degrees per meter
        # This is an approximation but gives intuitive units
        slope_change = np.degrees(np.arctan(np.abs(curvature)))
        
        return SlopeData(
            slope=slope_deg,
            slope_change=slope_change,
            aspect=aspect_deg,
            bounds=bounds,
            resolution_m=self.resolution
        )
    
    def _create_slope_tiles(self, slope_data: SlopeData, bounds: Bounds) -> int:
        """Create tiles from computed slope data"""
        tiles_created = 0
        
        # Get the tile keys that would cover this area
        # We'll use the same tiling scheme as elevation Layer 2
        south_tiles = int(np.floor(bounds.south / self.tile_size))
        north_tiles = int(np.floor(bounds.north / self.tile_size))
        west_tiles = int(np.floor(bounds.west / self.tile_size))
        east_tiles = int(np.floor(bounds.east / self.tile_size))
        
        # Array dimensions
        height, width = slope_data.slope.shape
        
        # Geographic extent
        lat_range = bounds.north - bounds.south
        lon_range = bounds.east - bounds.west
        
        for lat_tile in range(south_tiles, north_tiles + 1):
            for lon_tile in range(west_tiles, east_tiles + 1):
                # Tile bounds
                tile_south = lat_tile * self.tile_size
                tile_north = (lat_tile + 1) * self.tile_size
                tile_west = lon_tile * self.tile_size
                tile_east = (lon_tile + 1) * self.tile_size
                
                tile_bounds = Bounds(
                    south=tile_south,
                    north=tile_north,
                    west=tile_west,
                    east=tile_east
                )
                
                # Skip if no overlap
                if not bounds.overlaps(tile_bounds):
                    continue
                
                # Extract the portion of data for this tile
                # Calculate pixel coordinates
                px_south = int((tile_south - bounds.south) / lat_range * height)
                px_north = int((tile_north - bounds.south) / lat_range * height)
                px_west = int((tile_west - bounds.west) / lon_range * width)
                px_east = int((tile_east - bounds.west) / lon_range * width)
                
                # Clamp to array bounds
                px_south = max(0, px_south)
                px_north = min(height, px_north)
                px_west = max(0, px_west)
                px_east = min(width, px_east)
                
                # Extract data
                tile_slope = slope_data.slope[px_south:px_north, px_west:px_east]
                tile_slope_change = slope_data.slope_change[px_south:px_north, px_west:px_east]
                tile_aspect = slope_data.aspect[px_south:px_north, px_west:px_east]
                
                if tile_slope.size == 0:
                    continue
                
                # Save tile
                tile_key = self._get_tile_key(
                    (tile_south + tile_north) / 2,
                    (tile_west + tile_east) / 2
                )
                
                self._save_tile(
                    tile_key, 
                    tile_slope, 
                    tile_slope_change,
                    tile_aspect,
                    tile_bounds
                )
                
                tiles_created += 1
        
        self._save_index()
        return tiles_created
    
    def _get_tile_key(self, lat: float, lon: float) -> str:
        """Get tile key for a coordinate (matches elevation Layer 2)"""
        lat_idx = int(np.floor(lat / self.tile_size))
        lon_idx = int(np.floor(lon / self.tile_size))
        return f"{lat_idx:04d}_{lon_idx:04d}"
    
    def _save_tile(self, tile_key: str, slope: np.ndarray, 
                   slope_change: np.ndarray, aspect: np.ndarray,
                   bounds: Bounds):
        """Save a slope tile to disk"""
        tile_path = os.path.join(self.tiles_dir, f"{tile_key}.tif")
        
        # Create transform
        height, width = slope.shape
        transform = Affine(
            (bounds.east - bounds.west) / width, 0, bounds.west,
            0, -(bounds.north - bounds.south) / height, bounds.north
        )
        
        # Save as multi-band GeoTIFF
        with rasterio.open(
            tile_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,  # 3 bands: slope, slope_change, aspect
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            compress='deflate'
        ) as dst:
            dst.write(slope.astype('float32'), 1)
            dst.write(slope_change.astype('float32'), 2)
            dst.write(aspect.astype('float32'), 3)
        
        # Update index
        self.index["tiles"][tile_key] = {
            "bounds": {
                "north": bounds.north,
                "south": bounds.south,
                "east": bounds.east,
                "west": bounds.west
            },
            "path": tile_path,
            "resolution_m": self.resolution
        }
    
    def get_slope(self, lat: float, lon: float) -> float:
        """
        Get slope in degrees at a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Slope in degrees
            
        Raises:
            ValueError: If slope data not available at this location
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.index["tiles"]:
            raise ValueError(
                f"Slope data not available at ({lat:.6f}, {lon:.6f}). "
                f"Please compute this area first using compute_area()."
            )
        
        # Load tile and extract value
        tile_info = self.index["tiles"][tile_key]
        with rasterio.open(tile_info["path"]) as src:
            # Get pixel coordinates
            py, px = src.index(lon, lat)
            
            # Bounds check
            if py < 0 or py >= src.height or px < 0 or px >= src.width:
                raise ValueError(f"Coordinates ({lat}, {lon}) outside tile bounds")
            
            # Read slope band (band 1)
            slope = src.read(1, window=((py, py+1), (px, px+1)))[0, 0]
            
        return float(slope)
    
    def get_slope_change(self, lat: float, lon: float) -> float:
        """
        Get slope change (curvature) at a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Slope change in degrees per meter
            
        Raises:
            ValueError: If slope data not available at this location
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.index["tiles"]:
            raise ValueError(
                f"Slope data not available at ({lat:.6f}, {lon:.6f}). "
                f"Please compute this area first using compute_area()."
            )
        
        # Load tile and extract value
        tile_info = self.index["tiles"][tile_key]
        with rasterio.open(tile_info["path"]) as src:
            # Get pixel coordinates
            py, px = src.index(lon, lat)
            
            # Bounds check
            if py < 0 or py >= src.height or px < 0 or px >= src.width:
                raise ValueError(f"Coordinates ({lat}, {lon}) outside tile bounds")
            
            # Read slope change band (band 2)
            slope_change = src.read(2, window=((py, py+1), (px, px+1)))[0, 0]
            
        return float(slope_change)
    
    def get_aspect(self, lat: float, lon: float) -> float:
        """
        Get aspect (slope direction) at a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Aspect in degrees (0=North, 90=East, 180=South, 270=West)
            
        Raises:
            ValueError: If slope data not available at this location
        """
        tile_key = self._get_tile_key(lat, lon)
        
        if tile_key not in self.index["tiles"]:
            raise ValueError(
                f"Slope data not available at ({lat:.6f}, {lon:.6f}). "
                f"Please compute this area first using compute_area()."
            )
        
        # Load tile and extract value
        tile_info = self.index["tiles"][tile_key]
        with rasterio.open(tile_info["path"]) as src:
            # Get pixel coordinates
            py, px = src.index(lon, lat)
            
            # Bounds check
            if py < 0 or py >= src.height or px < 0 or px >= src.width:
                raise ValueError(f"Coordinates ({lat}, {lon}) outside tile bounds")
            
            # Read aspect band (band 3)
            aspect = src.read(3, window=((py, py+1), (px, px+1)))[0, 0]
            
        return float(aspect)
    
    def get_slope_array(self, bounds: Bounds) -> Tuple[SlopeData, Dict[str, Any]]:
        """
        Get slope data arrays for a region.
        
        Args:
            bounds: Geographic bounds
            
        Returns:
            Tuple of (SlopeData, metadata dict)
            
        Raises:
            ValueError: If slope data not available for the full area
        """
        # Check which tiles we need
        required_tiles = []
        south_tiles = int(np.floor(bounds.south / self.tile_size))
        north_tiles = int(np.floor(bounds.north / self.tile_size))
        west_tiles = int(np.floor(bounds.west / self.tile_size))
        east_tiles = int(np.floor(bounds.east / self.tile_size))
        
        for lat_tile in range(south_tiles, north_tiles + 1):
            for lon_tile in range(west_tiles, east_tiles + 1):
                tile_key = f"{lat_tile:04d}_{lon_tile:04d}"
                if tile_key not in self.index["tiles"]:
                    raise ValueError(
                        f"Slope data not available for full area. Missing tile: {tile_key}. "
                        f"Please compute this area first using compute_area()."
                    )
                required_tiles.append(tile_key)
        
        # Load and merge tiles
        slope_array, slope_change_array, aspect_array = self._merge_tiles(
            required_tiles, bounds
        )
        
        slope_data = SlopeData(
            slope=slope_array,
            slope_change=slope_change_array,
            aspect=aspect_array,
            bounds=bounds,
            resolution_m=self.resolution
        )
        
        metadata = {
            "bounds": {
                "north": bounds.north,
                "south": bounds.south,
                "east": bounds.east,
                "west": bounds.west
            },
            "shape": slope_array.shape,
            "resolution_m": self.resolution,
            "tiles_used": len(required_tiles)
        }
        
        return slope_data, metadata
    
    def _merge_tiles(self, tile_keys: list, target_bounds: Bounds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge multiple tiles into arrays for the target bounds"""
        from rasterio.merge import merge
        from rasterio.windows import Window
        
        # Load all tile datasets
        datasets = []
        for tile_key in tile_keys:
            tile_info = self.index["tiles"][tile_key]
            datasets.append(rasterio.open(tile_info["path"]))
        
        try:
            # Merge all bands at once
            merged_data, merged_transform = merge(
                datasets,
                bounds=target_bounds.to_tuple(),
                method='first'
            )
            
            # Extract individual bands
            slope_data = merged_data[0]
            slope_change_data = merged_data[1]
            aspect_data = merged_data[2]
            
            return slope_data, slope_change_data, aspect_data
            
        finally:
            # Close datasets
            for ds in datasets:
                ds.close()
    
    def remove_area(self, bounds: Bounds) -> Dict[str, Any]:
        """
        Remove slope data for an area.
        
        Args:
            bounds: Area to remove
            
        Returns:
            Dict with removal statistics
        """
        tiles_removed = 0
        
        # Find tiles that overlap with bounds
        tiles_to_remove = []
        for tile_key, tile_info in self.index["tiles"].items():
            tile_bounds = Bounds(
                south=tile_info["bounds"]["south"],
                north=tile_info["bounds"]["north"],
                west=tile_info["bounds"]["west"],
                east=tile_info["bounds"]["east"]
            )
            
            if bounds.overlaps(tile_bounds):
                tiles_to_remove.append(tile_key)
        
        # Remove tiles
        for tile_key in tiles_to_remove:
            tile_path = self.index["tiles"][tile_key]["path"]
            if os.path.exists(tile_path):
                os.remove(tile_path)
                tiles_removed += 1
            del self.index["tiles"][tile_key]
        
        self._save_index()
        
        return {
            "status": "success",
            "tiles_removed": tiles_removed
        }
    
    def list_computed_areas(self) -> Dict[str, Any]:
        """List all areas with computed slope data"""
        if not self.index["tiles"]:
            return {
                "areas": [],
                "total_tiles": 0,
                "total_size_mb": 0
            }
        
        # Group tiles into contiguous areas
        # (Similar to elevation system's approach)
        areas = []
        total_size = 0
        
        # For now, just return overall bounds
        all_bounds = []
        for tile_info in self.index["tiles"].values():
            all_bounds.append(tile_info["bounds"])
            if os.path.exists(tile_info["path"]):
                total_size += os.path.getsize(tile_info["path"])
        
        if all_bounds:
            # Compute union of all bounds
            north = max(b["north"] for b in all_bounds)
            south = min(b["south"] for b in all_bounds)
            east = max(b["east"] for b in all_bounds)
            west = min(b["west"] for b in all_bounds)
            
            areas.append({
                "bounds": {
                    "north": north,
                    "south": south,
                    "east": east,
                    "west": west
                },
                "tiles": len(self.index["tiles"])
            })
        
        return {
            "areas": areas,
            "total_tiles": len(self.index["tiles"]),
            "total_size_mb": total_size / (1024 * 1024),
            "resolution_m": self.resolution
        }