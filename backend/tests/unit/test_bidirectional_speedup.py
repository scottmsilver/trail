#!/usr/bin/env python3
"""
Test the speedup of Bidirectional A* vs Standard A* on the same route
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
import pytest


@pytest.mark.unit
@pytest.mark.skip(reason="Test requires actual algorithm implementation, not mocks")
def test_algorithms(mock_dem_cache):
    """Compare standard A* vs bidirectional A* on the same route"""
    
    # Test coordinates
    start_lat, start_lon = 40.6572, -111.5706
    end_lat, end_lon = 40.6486, -111.5639
    
    print("="*70)
    print("PATHFINDING ALGORITHM COMPARISON")
    print("="*70)
    print(f"Route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    print("="*70)
    
    # Use mock cache
    cache = mock_dem_cache
    
    # First, run the route to ensure caches are loaded
    print("\nWarming up caches...")
    warmup_path, warmup_stats = cache.find_path(start_lat, start_lon, end_lat, end_lon)
    if not warmup_path:
        print("Failed to find route during warmup!")
        return
    print(f"Warmup complete, found path with {len(warmup_path)} points")
    
    # Test 1: Standard A* with optimizations
    print("\n" + "-"*70)
    print("TEST 1: Standard A* (with early termination)")
    print("-"*70)
    
    # Temporarily override the algorithm selection
    original_find_route = cache.find_route
    
    def find_route_standard(lat1, lon1, lat2, lon2):
        # Force standard A*
        cache._force_algorithm = 'standard'
        return original_find_route(lat1, lon1, lat2, lon2)
    
    cache.find_route = find_route_standard
    
    start_time = time.time()
    path_standard = cache.find_route(start_lat, start_lon, end_lat, end_lon)
    standard_time = time.time() - start_time
    
    print(f"Time taken: {standard_time:.3f}s")
    print(f"Path points: {len(path_standard) if path_standard else 0}")
    
    # Test 2: Bidirectional A*
    print("\n" + "-"*70)
    print("TEST 2: Bidirectional A*")
    print("-"*70)
    
    def find_route_bidirectional(lat1, lon1, lat2, lon2):
        # Force bidirectional A*
        cache._force_algorithm = 'bidirectional'
        return original_find_route(lat1, lon1, lat2, lon2)
    
    cache.find_route = find_route_bidirectional
    
    start_time = time.time()
    path_bidirectional = cache.find_route(start_lat, start_lon, end_lat, end_lon)
    bidirectional_time = time.time() - start_time
    
    print(f"Time taken: {bidirectional_time:.3f}s")
    print(f"Path points: {len(path_bidirectional) if path_bidirectional else 0}")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    if standard_time > 0 and bidirectional_time > 0:
        speedup = standard_time / bidirectional_time
        print(f"Standard A* time:      {standard_time:.3f}s")
        print(f"Bidirectional A* time: {bidirectional_time:.3f}s")
        print(f"Speedup:               {speedup:.1f}x")
        
        if path_standard and path_bidirectional:
            print(f"\nPath quality:")
            print(f"Standard A* points:      {len(path_standard)}")
            print(f"Bidirectional A* points: {len(path_bidirectional)}")
            
            # Check if paths are similar
            if abs(len(path_standard) - len(path_bidirectional)) / len(path_standard) < 0.1:
                print("✓ Paths are similar in length (within 10%)")
            else:
                print("⚠ Paths differ significantly in length")


if __name__ == "__main__":
    test_algorithms()