#!/usr/bin/env python3
"""Test tile cache flow to understand why cost surface is being recomputed"""

import sys
sys.path.append('/home/ssilver/development/trail/backend')

from app.services.dem_tile_cache import DEMTileCache
import time
import pytest

# Test coordinates
lat1, lon1 = 40.6650, -111.4999
lat2, lon2 = 40.6699, -111.5065

# Initialize cache
cache = DEMTileCache()

print("\n=== Testing tile cache flow ===")
print(f"Route: ({lat1}, {lon1}) to ({lat2}, {lon2})")
print()

# First, check if tiles exist
min_lat = min(lat1, lat2) - 0.001
max_lat = max(lat1, lat2) + 0.001
min_lon = min(lon1, lon2) - 0.001
max_lon = max(lon2, lon2) + 0.001

tiles = cache.tiled_cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
print(f"Tiles needed: {tiles}")

# Check cache stats
stats = cache.tiled_cache.get_cache_stats()
print(f"\nCache stats: {stats}")

# Now test the route with timing
start_time = time.time()
print("\n=== Starting route computation ===")

# Add a custom print in compute_cost_surface to see if it's called
original_compute = cache.compute_cost_surface
call_count = 0

def wrapped_compute(*args, **kwargs):
    global call_count
    call_count += 1
    print(f"\n!!! compute_cost_surface called (call #{call_count}) !!!")
    print(f"!!! This is the slow operation taking 29.5s !!!\n")
    return original_compute(*args, **kwargs)

cache.compute_cost_surface = wrapped_compute

try:
    route = cache.find_route(lat1, lon1, lat2, lon2)
    elapsed = time.time() - start_time
    
    if route:
        print(f"\nRoute found with {len(route)} points in {elapsed:.1f}s")
    else:
        print(f"\nNo route found (took {elapsed:.1f}s)")
        
    print(f"\ncompute_cost_surface was called {call_count} times")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()