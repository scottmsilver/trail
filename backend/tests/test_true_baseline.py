#!/usr/bin/env python3
"""
Test true baseline performance with ALL optimizations disabled
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


def test_baseline_vs_optimized():
    """Compare true baseline vs optimized pathfinding"""
    
    print("🧪 TRUE BASELINE VS OPTIMIZED TEST")
    print("="*60)
    
    # Very short route
    start_lat, start_lon = 37.7694, -122.4762
    end_lat, end_lon = 37.7704, -122.4752
    
    print(f"Route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    print("-"*60)
    
    # Initialize cache
    cache = DEMTileCache(
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences()
    )
    
    # Pre-cache terrain
    print("\n📦 Pre-caching terrain...")
    _ = cache.find_route(start_lat, start_lon, end_lat, end_lon)
    print("✓ Terrain cached")
    
    # Test 1: TRUE BASELINE (all optimizations OFF)
    print("\n1️⃣ TRUE BASELINE (all optimizations OFF):")
    
    original = cache.astar_pathfinding_optimized
    def baseline_wrapper(*args, **kwargs):
        kwargs['optimization_config'] = {
            'use_preprocessing': False,
            'early_termination': False,
            'dynamic_weights': False,
            'memory_limit': 100000,
            'corner_cutting': False,
            'max_iterations': 10000000
        }
        return original(*args, **kwargs)
    cache.astar_pathfinding_optimized = baseline_wrapper
    
    start_time = time.time()
    path1 = cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time1 = time.time() - start_time
    
    if path1:
        print(f"✓ Success! Time: {time1:.3f}s, Points: {len(path1)}")
    else:
        print(f"✗ Failed ({time1:.3f}s)")
    
    # Test 2: SAFE OPTIMIZATIONS (preprocessing only)
    print("\n2️⃣ SAFE OPTIMIZATIONS (preprocessing only):")
    
    def safe_wrapper(*args, **kwargs):
        kwargs['optimization_config'] = {
            'use_preprocessing': True,
            'early_termination': False,
            'dynamic_weights': False,
            'memory_limit': 50000,
            'corner_cutting': False,
            'max_iterations': 10000000
        }
        return original(*args, **kwargs)
    cache.astar_pathfinding_optimized = safe_wrapper
    
    start_time = time.time()
    path2 = cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time2 = time.time() - start_time
    
    if path2:
        print(f"✓ Success! Time: {time2:.3f}s, Points: {len(path2)}")
        
        # Check if paths identical
        identical = (len(path1) == len(path2) and 
                    all(p1['lat'] == p2['lat'] and p1['lon'] == p2['lon'] 
                        for p1, p2 in zip(path1, path2)))
        print(f"  Paths identical: {'Yes' if identical else 'No'}")
        print(f"  Speedup: {time1/time2:.1f}x")
    
    # Test 3: DEFAULT (what was being used)
    print("\n3️⃣ DEFAULT SETTINGS (current defaults):")
    
    def default_wrapper(*args, **kwargs):
        # Don't set optimization_config, use defaults
        return original(*args, **kwargs)
    cache.astar_pathfinding_optimized = default_wrapper
    
    start_time = time.time()
    path3 = cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time3 = time.time() - start_time
    
    if path3:
        print(f"✓ Success! Time: {time3:.3f}s, Points: {len(path3)}")
        
        identical = (len(path1) == len(path3) and 
                    all(p1['lat'] == p2['lat'] and p1['lon'] == p2['lon'] 
                        for p1, p2 in zip(path1, path3)))
        print(f"  Paths identical to baseline: {'Yes' if identical else 'No'}")
        print(f"  Speedup vs baseline: {time1/time3:.1f}x")
    
    # Summary
    print("\n📊 SUMMARY:")
    print("-"*60)
    print(f"True Baseline:       {time1:.3f}s (all optimizations OFF)")
    print(f"Safe Optimizations:  {time2:.3f}s ({time1/time2:.1f}x speedup)")
    print(f"Default Settings:    {time3:.3f}s ({time1/time3:.1f}x speedup)")
    
    print("\n⚠️  NOTE: The 'baseline' in benchmarks was using default settings")
    print("         which include many optimizations turned ON by default!")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    test_baseline_vs_optimized()