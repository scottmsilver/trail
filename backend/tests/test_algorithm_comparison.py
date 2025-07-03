#!/usr/bin/env python3
"""
Direct comparison of pathfinding algorithms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


def load_precomputed_data():
    """Load the precomputed cost surface"""
    cache_file = "precomputed_cache/40.5986,40.7072,-111.6206,-111.5139_cost.pkl"
    
    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        return None
    
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def test_algorithms():
    """Compare algorithms directly"""
    
    # Load precomputed data
    print("Loading precomputed terrain data...")
    cache_data = load_precomputed_data()
    if not cache_data:
        return
    
    cost_surface = cache_data['cost_surface']
    indices = cache_data['indices']
    out_trans = cache_data['out_trans']
    crs = cache_data['crs']
    dem = cache_data.get('dem')
    
    # Test coordinates
    start_lat, start_lon = 40.6572, -111.5706
    end_lat, end_lon = 40.6486, -111.5639
    
    # Initialize cache for utility methods
    cache = DEMTileCache(
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences(),
        debug_mode=False
    )
    
    # Get indices
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    # Convert coordinates to indices
    x_start, y_start = transformer.transform(start_lon, start_lat)
    x_end, y_end = transformer.transform(end_lon, end_lat)
    
    cell_size_x = out_trans.a
    cell_size_y = out_trans.e
    
    col_start = int((x_start - out_trans.c) / cell_size_x)
    row_start = int((y_start - out_trans.f) / cell_size_y)
    col_end = int((x_end - out_trans.c) / cell_size_x)
    row_end = int((y_end - out_trans.f) / cell_size_y)
    
    start_idx = indices[row_start, col_start]
    end_idx = indices[row_end, col_end]
    
    print(f"\n{'='*70}")
    print("DIRECT ALGORITHM COMPARISON")
    print(f"{'='*70}")
    print(f"Route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    print(f"Grid: ({row_start}, {col_start}) → ({row_end}, {col_end})")
    print(f"Distance: {abs(row_end - row_start) + abs(col_end - col_start)} cells")
    print(f"{'='*70}\n")
    
    # Test 1: Standard A* with optimizations
    print("TEST 1: Standard A* (with optimizations)")
    print("-" * 70)
    
    optimization_config = {
        'early_termination': True,
        'stagnation_limit': 10000,
        'dynamic_weights': False,
        'corner_cutting': False
    }
    
    start_time = time.time()
    path_standard = cache.astar_pathfinding_optimized(
        cost_surface, indices, start_idx, end_idx,
        out_trans, transformer, dem, optimization_config
    )
    standard_time = time.time() - start_time
    
    print(f"Time: {standard_time:.3f}s")
    print(f"Path points: {len(path_standard) if path_standard else 0}")
    
    # Test 2: Bidirectional A*
    print("\nTEST 2: Bidirectional A*")
    print("-" * 70)
    
    start_time = time.time()
    path_bidirectional = cache.bidirectional_astar(
        cost_surface, indices, start_idx, end_idx,
        out_trans, transformer, dem
    )
    bidirectional_time = time.time() - start_time
    
    print(f"Time: {bidirectional_time:.3f}s")
    print(f"Path points: {len(path_bidirectional) if path_bidirectional else 0}")
    
    # Test 3: Standard A* with dynamic weights
    print("\nTEST 3: Standard A* (with dynamic weights)")
    print("-" * 70)
    
    optimization_config_dynamic = {
        'early_termination': True,
        'stagnation_limit': 10000,
        'dynamic_weights': True,
        'weight_start': 1.0,
        'weight_end': 1.2,
        'corner_cutting': False
    }
    
    start_time = time.time()
    path_dynamic = cache.astar_pathfinding_optimized(
        cost_surface, indices, start_idx, end_idx,
        out_trans, transformer, dem, optimization_config_dynamic
    )
    dynamic_time = time.time() - start_time
    
    print(f"Time: {dynamic_time:.3f}s")
    print(f"Path points: {len(path_dynamic) if path_dynamic else 0}")
    
    # Results summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Algorithm':<30} {'Time (s)':>10} {'Speedup':>10} {'Path Pts':>10}")
    print("-" * 70)
    
    baseline = standard_time
    results = [
        ("Standard A*", standard_time, len(path_standard) if path_standard else 0),
        ("Bidirectional A*", bidirectional_time, len(path_bidirectional) if path_bidirectional else 0),
        ("Standard A* + Dynamic Weights", dynamic_time, len(path_dynamic) if path_dynamic else 0)
    ]
    
    for name, time_taken, path_pts in results:
        speedup = baseline / time_taken if time_taken > 0 else 0
        print(f"{name:<30} {time_taken:>10.3f} {speedup:>10.1f}x {path_pts:>10}")
    
    # Best performer
    best_time = min(standard_time, bidirectional_time, dynamic_time)
    best_name = ""
    if best_time == bidirectional_time:
        best_name = "Bidirectional A*"
    elif best_time == dynamic_time:
        best_name = "Standard A* + Dynamic Weights"
    else:
        best_name = "Standard A*"
    
    print(f"\n{'='*70}")
    print(f"WINNER: {best_name} ({best_time:.3f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_algorithms()