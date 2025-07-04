#!/usr/bin/env python3
"""
Pre-populated cache fixtures for testing
"""

import pytest
import numpy as np
import pickle
import os
import shutil
from pathlib import Path
import tempfile
from datetime import datetime
import rasterio
from rasterio.transform import Affine

from app.services.tiled_dem_cache import TiledDEMCache


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the session"""
    test_dir = tempfile.mkdtemp(prefix="trail_test_data_")
    yield test_dir
    # Cleanup after all tests
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_dem_data():
    """Create sample DEM (elevation) data"""
    # Create a 100x100 grid with realistic elevation patterns
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create elevation with hills and valleys
    elevation = 2000 + 100 * np.sin(X/2) * np.cos(Y/2) + 50 * np.sin(X) + 30 * np.cos(Y*2)
    
    # Add some noise
    elevation += np.random.normal(0, 5, elevation.shape)
    
    return elevation.astype(np.float32)


@pytest.fixture(scope="session")
def sample_cost_surface(sample_dem_data):
    """Create sample cost surface from DEM data"""
    # Calculate slopes
    dy, dx = np.gradient(sample_dem_data, 3.0, 3.0)  # 3m resolution
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    
    # Create cost surface based on slope
    cost_surface = np.ones_like(slope_degrees)
    cost_surface[slope_degrees < 5] = 1.0
    cost_surface[(slope_degrees >= 5) & (slope_degrees < 15)] = 1.5
    cost_surface[(slope_degrees >= 15) & (slope_degrees < 25)] = 3.0
    cost_surface[(slope_degrees >= 25) & (slope_degrees < 35)] = 10.0
    cost_surface[slope_degrees >= 35] = 1000.0  # Impassable
    
    return cost_surface.astype(np.float32)


@pytest.fixture(scope="session")
def sample_transform():
    """Create sample geospatial transform"""
    # Park City area approximate transform using rasterio.Affine
    from rasterio.transform import Affine
    
    # Create affine transform
    # Affine(a, b, c, d, e, f)
    # where:
    # a = pixel width in coordinate units
    # b = row rotation (typically 0)
    # c = x-coordinate of upper-left corner
    # d = column rotation (typically 0)
    # e = pixel height (negative for north-up)
    # f = y-coordinate of upper-left corner
    
    return Affine(
        3.0,            # pixel width in meters
        0.0,            # rotation
        -12426595.0,    # x origin (Web Mercator)
        0.0,            # rotation
        -3.0,           # pixel height (negative for north-up)
        4969664.0       # y origin (Web Mercator)
    )


@pytest.fixture(scope="session")
def populated_tile_cache(test_data_dir, sample_dem_data, sample_cost_surface, sample_transform):
    """Create a pre-populated tile cache"""
    cache_dir = Path(test_data_dir) / "tile_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create tile cache instance
    tile_cache = TiledDEMCache(
        tile_size_degrees=0.01,  # 0.01 degree tiles
        cache_dir=str(cache_dir)
    )
    
    # Pre-populate some tiles
    # These coordinates cover a small area in Park City
    base_lat, base_lon = 40.65, -111.57
    
    for i in range(3):  # 3x3 grid of tiles
        for j in range(3):
            tile_lat = base_lat + i * 0.01
            tile_lon = base_lon + j * 0.01
            
            # Create tile data
            tile_data = {
                'dem': sample_dem_data[i*30:(i+1)*30, j*30:(j+1)*30],
                'cost_surface': sample_cost_surface[i*30:(i+1)*30, j*30:(j+1)*30],
                'transform': sample_transform,
                'crs': 'EPSG:3857',  # Web Mercator
                'bounds': {
                    'min_lat': tile_lat,
                    'max_lat': tile_lat + 0.01,
                    'min_lon': tile_lon,
                    'max_lon': tile_lon + 0.01
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Save tile
            tile_x = int(tile_lon * 100)
            tile_y = int(tile_lat * 100)
            
            tile_path = cache_dir / "cost" / f"tile_{tile_x}_{tile_y}.pkl"
            tile_path.parent.mkdir(exist_ok=True)
            
            with open(tile_path, 'wb') as f:
                pickle.dump(tile_data, f)
    
    return tile_cache


@pytest.fixture(scope="session")
def populated_terrain_cache(test_data_dir, sample_dem_data, sample_transform):
    """Create pre-populated terrain cache"""
    cache_file = Path(test_data_dir) / "terrain_cache.pkl"
    
    # Terrain cache stores tuples of (dem, out_trans, crs)
    terrain_cache = {
        "40.6500,40.6600,-111.5800,-111.5700": (
            sample_dem_data,
            sample_transform,
            'EPSG:3857'
        ),
        "40.6520,40.6580,-111.5750,-111.5690": (
            sample_dem_data[10:70, 10:70],
            sample_transform,
            'EPSG:3857'
        )
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(terrain_cache, f)
    
    return cache_file


@pytest.fixture(scope="session")
def populated_cost_cache(test_data_dir, sample_cost_surface, sample_dem_data):
    """Create pre-populated cost surface cache"""
    cache_file = Path(test_data_dir) / "cost_cache.pkl"
    
    # Calculate slope degrees from DEM
    dy, dx = np.gradient(sample_dem_data, 3.0, 3.0)  # 3m resolution
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    
    # Create indices
    indices = np.ones_like(sample_cost_surface, dtype=np.int32)
    
    # Cost surface cache stores tuples of (cost_surface, slope_degrees, indices)
    cost_cache = {
        "40.6500,40.6600,-111.5800,-111.5700_cost": (
            sample_cost_surface,
            slope_degrees,
            indices
        )
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cost_cache, f)
    
    return cache_file


@pytest.fixture
def mock_dem_cache(populated_tile_cache, populated_terrain_cache, populated_cost_cache, test_data_dir, monkeypatch):
    """Mock DEM cache that uses pre-populated data"""
    from app.services.dem_tile_cache import DEMTileCache
    from tests.fixtures.mock_pathfinding import mock_find_path, mock_get_indices
    
    # Monkey patch environment for cache location
    monkeypatch.setenv("HYRIVER_CACHE_NAME", str(Path(test_data_dir) / "http_cache.sqlite"))
    
    # Create cache instance
    cache = DEMTileCache()
    
    # Override the dem data directory
    cache.dem_data_dir = test_data_dir
    
    # Pre-load the terrain and cost caches
    cache.terrain_cache = {}
    cache.cost_surface_cache = {}
    
    # Load pre-populated data
    with open(populated_terrain_cache, 'rb') as f:
        cache.terrain_cache = pickle.load(f)
    
    with open(populated_cost_cache, 'rb') as f:
        cache.cost_surface_cache = pickle.load(f)
    
    # Set the tile cache
    cache.tiled_cache = populated_tile_cache
    
    # Add mock methods
    cache.find_path = lambda lat1, lon1, lat2, lon2, options=None: mock_find_path(cache, lat1, lon1, lat2, lon2, options)
    cache.get_indices = lambda lat1, lon1, lat2, lon2, out_trans, crs, indices=None: mock_get_indices(cache, lat1, lon1, lat2, lon2, out_trans, crs, indices)
    
    return cache


@pytest.fixture
def sample_route_request():
    """Sample route request for testing"""
    return {
        "start": {"lat": 40.6560, "lon": -111.5708},
        "end": {"lat": 40.6520, "lon": -111.5688},
        "options": {
            "userProfile": "default",
            "resolution": "high"
        }
    }


@pytest.fixture
def expected_route_stats():
    """Expected statistics for sample route"""
    return {
        "distance_km": pytest.approx(0.5, rel=0.2),  # ~500m +/- 20%
        "elevation_gain_m": pytest.approx(50, rel=0.5),  # ~50m +/- 50%
        "waypoints": pytest.approx(100, rel=0.5),  # ~100 points +/- 50%
        "max_slope": pytest.approx(25.0, abs=10.0)  # ~25° +/- 10°
    }