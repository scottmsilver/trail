"""
File descriptor-safe elevation library wrapper.
Manages open file descriptors to prevent system limits.
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from collections import OrderedDict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class FDManagedElevationLibrary:
    """
    Wrapper around TwoLayerElevationLibrary that manages file descriptors.
    Uses an LRU cache to limit open files.
    """
    
    def __init__(self, base_library, max_open_files=50):
        """
        Initialize FD-managed elevation library.
        
        Args:
            base_library: The underlying TwoLayerElevationLibrary instance
            max_open_files: Maximum number of files to keep open
        """
        self.base_lib = base_library
        self.max_open_files = max_open_files
        self._open_datasets = OrderedDict()
        self._access_count = {}
        
    def _get_dataset(self, filepath):
        """Get a dataset, managing the cache"""
        if filepath in self._open_datasets:
            # Move to end (most recently used)
            self._open_datasets.move_to_end(filepath)
            self._access_count[filepath] += 1
            return self._open_datasets[filepath]
        
        # Need to open the file
        if len(self._open_datasets) >= self.max_open_files:
            # Close least recently used
            oldest_path, oldest_ds = self._open_datasets.popitem(last=False)
            oldest_ds.close()
            del self._access_count[oldest_path]
            logger.debug(f"Closed {oldest_path} (LRU)")
        
        # Open new dataset
        ds = rasterio.open(filepath)
        self._open_datasets[filepath] = ds
        self._access_count[filepath] = 1
        return ds
    
    def close_all(self):
        """Close all open datasets"""
        for ds in self._open_datasets.values():
            ds.close()
        self._open_datasets.clear()
        self._access_count.clear()
        
    def __del__(self):
        """Cleanup on deletion"""
        self.close_all()
    
    def load_area(self, bounds):
        """Delegate load_area to base library"""
        return self.base_lib.load_area(bounds)
    
    @contextmanager
    def batch_operation(self):
        """Context manager for batch operations - closes files after"""
        try:
            yield
        finally:
            # Close all files after batch operation
            self.close_all()
    
    def get_elevation_array_safe(self, bounds):
        """
        Get elevation array with proper FD management.
        Processes tiles in batches to limit open files.
        """
        # Get list of tiles needed
        min_row = int(np.floor(bounds.south / self.base_lib.tile_size))
        max_row = int(np.floor(bounds.north / self.base_lib.tile_size))
        min_col = int(np.floor(bounds.west / self.base_lib.tile_size))
        max_col = int(np.floor(bounds.east / self.base_lib.tile_size))
        
        tiles_to_merge = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                tile_key = f"{row}_{col}"
                if tile_key in self.base_lib.layer2_index["tiles"]:
                    tile_file = os.path.join(self.base_lib.layer2_dir, f"tile_{tile_key}.tif")
                    if os.path.exists(tile_file):
                        tiles_to_merge.append(tile_file)
        
        if not tiles_to_merge:
            raise ValueError("No elevation data available for this area")
        
        logger.info(f"Processing {len(tiles_to_merge)} tiles with FD management")
        
        if len(tiles_to_merge) == 1:
            # Single tile - simple case
            with rasterio.open(tiles_to_merge[0]) as src:
                # Calculate window
                col_start = int((bounds.west - src.bounds.left) / src.res[0])
                row_start = int((src.bounds.top - bounds.north) / src.res[1])
                col_stop = int((bounds.east - src.bounds.left) / src.res[0])
                row_stop = int((src.bounds.top - bounds.south) / src.res[1])
                
                col_start = max(0, col_start)
                row_start = max(0, row_start)
                col_stop = min(src.width, col_stop)
                row_stop = min(src.height, row_stop)
                
                window = rasterio.windows.Window(col_start, row_start,
                                                col_stop - col_start, row_stop - row_start)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
        else:
            # Multiple tiles - process in batches
            batch_size = max(10, self.max_open_files // 2)  # Use half the limit per batch
            
            # If we have few tiles, process all at once
            if len(tiles_to_merge) <= batch_size:
                datasets = []
                for f in tiles_to_merge:
                    datasets.append(self._get_dataset(f))
                
                # Merge tiles
                data, transform = merge(
                    datasets, 
                    bounds=bounds.to_tuple(),
                    method='first',
                    nodata=0
                )
                data = data[0]  # Get first band
            else:
                # Process in batches and merge results
                logger.info(f"Processing {len(tiles_to_merge)} tiles in batches of {batch_size}")
                
                # For very large areas, we need a different strategy
                # Create output array
                resolution = self.base_lib.resolution
                
                # Calculate output dimensions
                lat_extent = bounds.north - bounds.south
                lon_extent = bounds.east - bounds.west
                
                # Approximate pixels
                meters_per_deg_lat = 111320
                meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians((bounds.north + bounds.south) / 2))
                
                height = int(lat_extent * meters_per_deg_lat / resolution)
                width = int(lon_extent * meters_per_deg_lon / resolution)
                
                # Create output array
                output_data = np.full((height, width), np.nan, dtype=np.float32)
                
                # Process each tile individually
                from rasterio.transform import from_bounds
                output_transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, width, height)
                
                for tile_file in tiles_to_merge:
                    with rasterio.open(tile_file) as src:
                        # Read tile data
                        tile_data = src.read(1)
                        
                        # Calculate where this tile goes in output
                        # Get tile bounds
                        tile_bounds = src.bounds
                        
                        # Convert to output array indices
                        col_start = int((tile_bounds.left - bounds.west) / output_transform.a)
                        col_end = int((tile_bounds.right - bounds.west) / output_transform.a)
                        row_start = int((bounds.north - tile_bounds.top) / -output_transform.e)
                        row_end = int((bounds.north - tile_bounds.bottom) / -output_transform.e)
                        
                        # Clip to output bounds
                        col_start = max(0, col_start)
                        col_end = min(width, col_end)
                        row_start = max(0, row_start)
                        row_end = min(height, row_end)
                        
                        # Resample tile data to fit
                        if col_end > col_start and row_end > row_start:
                            from scipy.ndimage import zoom
                            
                            # Calculate zoom factors
                            zoom_y = (row_end - row_start) / tile_data.shape[0]
                            zoom_x = (col_end - col_start) / tile_data.shape[1]
                            
                            # Resample
                            if abs(zoom_y - 1.0) > 0.01 or abs(zoom_x - 1.0) > 0.01:
                                tile_resampled = zoom(tile_data, (zoom_y, zoom_x), order=1)
                            else:
                                tile_resampled = tile_data
                            
                            # Place in output
                            output_data[row_start:row_end, col_start:col_end] = tile_resampled
                
                data = output_data
                transform = output_transform
                
                # Replace NaN with 0
                data = np.nan_to_num(data, nan=0)
        
        # Build metadata
        metadata = {
            "bounds": {
                "north": bounds.north,
                "south": bounds.south,
                "east": bounds.east,
                "west": bounds.west
            },
            "shape": list(data.shape),
            "resolution_m": self.base_lib.resolution,
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