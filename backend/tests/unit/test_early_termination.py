#!/usr/bin/env python3
"""
Test if early termination is working properly in the system.
"""

import time
from route_cli import VerboseDEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
import pytest


@pytest.mark.unit
def test_with_config(config_name, optimization_config):
    """Test pathfinding with specific config"""
    
    # Initialize cache with precomputed data
    cache = VerboseDEMTileCache(
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences()
    )
    
    # Coordinates
    start_lat, start_lon = 40.6572, -111.5706
    end_lat, end_lon = 40.6486, -111.5639
    
    # Get cached data directly
    min_lat, max_lat, min_lon, max_lon = cache.define_area_of_interest(start_lat, start_lon, end_lat, end_lon)
    cache_key = f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}"
    cost_cache_key = f"{cache_key}_cost"
    
    if cost_cache_key not in cache.cost_surface_cache:
        print(f"ERROR: No cached cost surface for {config_name}")
        return None
    
    # Get cached data
    cached_data = cache.cost_surface_cache[cost_cache_key]
    cost_surface = cached_data['cost_surface']
    indices = cached_data['indices']
    dem = cached_data['dem']
    out_trans = cached_data['out_trans']
    crs = cached_data['crs']
    
    # Get indices
    start_idx, end_idx, transformer = cache.get_indices(start_lat, start_lon, end_lat, end_lon, out_trans, crs, indices)
    
    print(f"\n{config_name}:")
    print(f"  Config: {optimization_config}")
    
    # Time pathfinding
    start_time = time.time()
    path = cache.astar_pathfinding_optimized(
        cost_surface, indices, start_idx, end_idx, 
        out_trans, transformer, dem, optimization_config
    )
    elapsed = time.time() - start_time
    
    print(f"  Time: {elapsed:.1f}s")
    if path:
        print(f"  Path: {len(path)} points")
    else:
        print(f"  Path: FAILED")
    
    return elapsed


def main():
    print("Testing Early Termination Impact")
    print("="*50)
    
    # Test configurations
    configs = [
        ("No optimizations", {
            'early_termination': False,
            'preprocessing': False,
            'dynamic_weights': False,
            'corner_cutting': False
        }),
        ("Early termination ON", {
            'early_termination': True,
            'stagnation_limit': 10000,
            'preprocessing': False,
            'dynamic_weights': False,
            'corner_cutting': False
        }),
        ("Current CLI config", {
            'early_termination': True,
            'stagnation_limit': 10000,
            'dynamic_weights': False,
            'corner_cutting': False
        })
    ]
    
    times = []
    for name, config in configs:
        t = test_with_config(name, config)
        if t:
            times.append((name, t))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    if times:
        baseline = times[0][1]
        for name, t in times:
            speedup = baseline / t
            print(f"{name:30} {t:6.1f}s (speedup: {speedup:.1f}x)")


if __name__ == "__main__":
    main()