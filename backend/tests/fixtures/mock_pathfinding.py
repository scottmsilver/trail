"""
Mock pathfinding functionality for tests
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pyproj import Transformer


class MockPathfinder:
    """Mock pathfinder that returns realistic paths without computation"""
    
    @staticmethod
    def create_simple_path(start_idx: Tuple[int, int], end_idx: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a simple path between two points"""
        r1, c1 = start_idx
        r2, c2 = end_idx
        
        # Simple diagonal then straight path
        path = []
        
        # Move diagonally as much as possible
        r, c = r1, c1
        while r != r2 and c != c2:
            if r < r2:
                r += 1
            else:
                r -= 1
            if c < c2:
                c += 1
            else:
                c -= 1
            path.append((r, c))
        
        # Move straight for remaining distance
        while r != r2:
            if r < r2:
                r += 1
            else:
                r -= 1
            path.append((r, c))
            
        while c != c2:
            if c < c2:
                c += 1
            else:
                c -= 1
            path.append((r, c))
        
        # Add start and end if not already there
        if not path or path[0] != start_idx:
            path.insert(0, start_idx)
        if not path or path[-1] != end_idx:
            path.append(end_idx)
            
        return path
    
    @staticmethod
    def create_mock_stats(path: List[Tuple[int, int]], 
                         cost_surface: Optional[np.ndarray] = None) -> Dict:
        """Create realistic statistics for a path"""
        path_length = len(path)
        
        # Calculate approximate distance (assuming 3m per pixel)
        distance_pixels = 0
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            distance_pixels += np.sqrt((r2-r1)**2 + (c2-c1)**2)
        
        distance_m = distance_pixels * 3.0  # 3m resolution
        
        # Generate mock elevation changes
        elevation_gain = np.random.uniform(10, 50)  # 10-50m gain
        max_slope = np.random.uniform(5, 25)  # 5-25 degrees
        
        stats = {
            'nodes_explored': path_length * 10,
            'path_length': path_length,
            'computation_time': 0.05,
            'distance_m': distance_m,
            'elevation_gain_m': elevation_gain,
            'max_slope': max_slope,
            'algorithm': 'mock_pathfinder'
        }
        
        return stats


def mock_find_path(self, lat1: float, lon1: float, lat2: float, lon2: float, 
                   options: Optional[Dict] = None) -> Tuple[Optional[List[Tuple[float, float]]], Dict]:
    """Mock implementation of DEMTileCache.find_path"""
    # Check if we have the required bounds in cache
    required_bounds = f"{min(lat1,lat2):.4f},{max(lat1,lat2):.4f},{min(lon1,lon2):.4f},{max(lon1,lon2):.4f}"
    
    # For testing, we'll create a simple path
    # Convert lat/lon to mock indices
    start_idx = (int((lat1 - 40.6500) * 1000), int((lon1 + 111.5800) * 1000))
    end_idx = (int((lat2 - 40.6500) * 1000), int((lon2 + 111.5800) * 1000))
    
    # Create path in index space
    index_path = MockPathfinder.create_simple_path(start_idx, end_idx)
    
    # Convert back to lat/lon
    path = []
    for r, c in index_path:
        lat = 40.6500 + r / 1000.0
        lon = -111.5800 + c / 1000.0
        path.append((lat, lon))
    
    # Create stats
    stats = MockPathfinder.create_mock_stats(index_path)
    stats['bounds_checked'] = required_bounds
    stats['cache_hits'] = len(self.terrain_cache)
    
    return path, stats


def mock_get_indices(self, lat1: float, lon1: float, lat2: float, lon2: float,
                     out_trans, crs, indices=None):
    """Mock implementation of get_indices"""
    # Simple mock transformation
    start_idx = (int((lat1 - 40.6500) * 1000), int((lon1 + 111.5800) * 1000))
    end_idx = (int((lat2 - 40.6500) * 1000), int((lon2 + 111.5800) * 1000))
    
    # Mock transformer
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    return start_idx, end_idx, transformer