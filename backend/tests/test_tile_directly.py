#!/usr/bin/env python3
"""Test tiling directly"""

import sys
sys.path.append('.')

from app.services.tiled_dem_cache import TiledDEMCache

# Create a tiled cache
cache = TiledDEMCache(tile_size_degrees=0.01)

# Test getting tiles for a small area
min_lat, max_lat = 40.65, 40.66
min_lon, max_lon = -111.57, -111.56

print(f"Getting tiles for bounds: ({min_lat}, {max_lat}, {min_lon}, {max_lon})")
tiles = cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)

print(f"\nFound {len(tiles)} tiles:")
for tx, ty in tiles:
    bounds = cache._get_tile_bounds(tx, ty)
    print(f"  Tile ({tx}, {ty}): bounds = {bounds}")
    
# Test tile size
print(f"\nTile size: {cache.tile_size}° (~{cache.tile_size * 111}km at equator)")

# Show cache stats
stats = cache.get_cache_stats()
print(f"\nCache stats: {stats}")