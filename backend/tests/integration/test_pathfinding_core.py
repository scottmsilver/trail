"""
Core pathfinding integration tests without visualization
"""

import pytest
import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from pyproj import Transformer
from rasterio.transform import from_origin


@pytest.mark.integration
def test_basic_pathfinding():
    """Test basic pathfinding functionality with simple terrain"""
    # Initialize pathfinder
    pathfinder = DEMTileCache(debug_mode=False)
    
    # Create simple 10x10 grid with uniform cost
    rows, cols = 10, 10
    cost_surface = np.ones((rows, cols))
    
    # Create transform and transformer
    transform = from_origin(-111.5, 40.5, 30, 30)  # 30m resolution
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    
    # Test straight line path
    start_idx = indices[2, 2]
    end_idx = indices[7, 7]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    assert path is not None, "Should find a path"
    assert len(path) > 0, "Path should have points"
    assert len(path) >= 5, "Path should have reasonable length for diagonal movement"
    
    # Verify start and end points
    start_lon, start_lat = path[0]
    end_lon, end_lat = path[-1]
    
    # The path should start and end near the specified indices
    # Convert back to check
    start_x, start_y = transformer.transform(start_lon, start_lat)
    start_col = int((start_x - transform.c) / transform.a)
    start_row = int((start_y - transform.f) / transform.e)
    
    end_x, end_y = transformer.transform(end_lon, end_lat)
    end_col = int((end_x - transform.c) / transform.a)
    end_row = int((end_y - transform.f) / transform.e)
    
    assert abs(start_row - 2) <= 1 and abs(start_col - 2) <= 1, "Path should start near (2,2)"
    assert abs(end_row - 7) <= 1 and abs(end_col - 7) <= 1, "Path should end near (7,7)"


@pytest.mark.integration
def test_pathfinding_with_obstacles():
    """Test pathfinding avoids high-cost obstacles"""
    pathfinder = DEMTileCache(debug_mode=False)
    
    # Create grid with obstacle
    rows, cols = 10, 10
    cost_surface = np.ones((rows, cols))
    
    # Add vertical wall obstacle
    cost_surface[2:8, 5] = 1000.0  # Very high cost
    
    transform = from_origin(-111.5, 40.5, 30, 30)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    
    # Path from left to right of obstacle
    start_idx = indices[5, 2]
    end_idx = indices[5, 8]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    assert path is not None, "Should find a path around obstacle"
    
    # Convert path to grid coordinates and verify it avoids obstacle
    for lon, lat in path:
        x, y = transformer.transform(lon, lat)
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        
        if 0 <= row < rows and 0 <= col < cols:
            # Path should not go through high-cost cells
            assert cost_surface[row, col] < 1000, f"Path should avoid obstacles at ({row}, {col})"


@pytest.mark.integration  
def test_pathfinding_no_path():
    """Test pathfinding when no path exists"""
    pathfinder = DEMTileCache(debug_mode=False)
    
    # Create grid with complete barrier
    rows, cols = 10, 10
    cost_surface = np.ones((rows, cols))
    
    # Add horizontal wall that blocks all paths
    cost_surface[5, :] = np.inf  # Impassable barrier
    
    transform = from_origin(-111.5, 40.5, 30, 30)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    
    # Try to path from top to bottom across barrier
    start_idx = indices[2, 5]
    end_idx = indices[8, 5]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    # Should return None or empty path when no path exists
    assert path is None or len(path) == 0, "Should not find path across impassable barrier"


@pytest.mark.integration
def test_pathfinding_integration_components():
    """Test integration between DEMTileCache components"""
    # Test with obstacle config
    from app.services.obstacle_config import ObstacleConfig
    obstacle_config = ObstacleConfig(
        avoid_water=True,
        avoid_highways=True,
        allow_water_crossing=False
    )
    
    # Test with path preferences  
    from app.services.path_preferences import PathPreferences
    path_prefs = PathPreferences(
        stick_to_paths=True,
        path_transition_penalty=3.0
    )
    
    # Initialize with configurations
    pathfinder = DEMTileCache(
        debug_mode=False,
        obstacle_config=obstacle_config,
        path_preferences=path_prefs
    )
    
    # Verify components are integrated
    assert pathfinder.obstacle_config.avoid_water == True
    assert pathfinder.path_preferences.stick_to_paths == True
    assert pathfinder.path_preferences.path_transition_penalty == 3.0
    
    # Create simple test scenario
    rows, cols = 5, 5
    cost_surface = np.ones((rows, cols)) * 2.0  # Base cost of 2
    
    transform = from_origin(-111.5, 40.5, 30, 30)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    
    start_idx = indices[0, 0]
    end_idx = indices[4, 4]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    assert path is not None, "Should find path with configurations"
    assert len(path) >= 2, "Path should connect start to end"