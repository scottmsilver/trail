#!/usr/bin/env python3
"""
Test to verify that all route data (DEM, slopes, costs) is properly cached
and composed when tiles exist for an area.
"""

import os
import sys
import time
import numpy as np
import pickle

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tiled_dem_cache import TiledDEMCache
from app.services.dem_tile_cache import DEMTileCache
import pytest

@pytest.mark.integration
def test_tile_composition():
    """Test that tiles are properly composed with all data"""
    print("=" * 70)
    print("TILE CACHING COMPLETENESS TEST")
    print("=" * 70)
    
    # Initialize tiled cache
    tiled_cache = TiledDEMCache(tile_size_degrees=0.01)
    
    # Check existing tiles
    tile_dir = "tile_cache/cost"
    if os.path.exists(tile_dir):
        tiles = [f for f in os.listdir(tile_dir) if f.endswith('.pkl')]
        print(f"\nFound {len(tiles)} cached tiles")
        
        # Load a sample tile to verify contents
        if tiles:
            sample_tile_path = os.path.join(tile_dir, tiles[0])
            with open(sample_tile_path, 'rb') as f:
                sample_tile = pickle.load(f)
            
            print(f"\nSample tile '{tiles[0]}' contains:")
            for key in sample_tile.keys():
                if key in ['cost_surface', 'slope_degrees', 'dem']:
                    shape = sample_tile[key].shape if hasattr(sample_tile[key], 'shape') else 'N/A'
                    print(f"  - {key}: shape {shape}")
                else:
                    print(f"  - {key}")
            
            has_dem = 'dem' in sample_tile
            print(f"\nDEM data present in tiles: {'✓' if has_dem else '✗'}")
    
    # Test route that should use multiple tiles
    # Park City area coordinates
    test_routes = [
        {
            "name": "Small route (1-4 tiles)",
            "start": (40.653, -111.568),
            "end": (40.655, -111.566),
            "expected_tiles": 4
        },
        {
            "name": "Medium route (9-16 tiles)",
            "start": (40.650, -111.570),
            "end": (40.660, -111.560),
            "expected_tiles": 16
        }
    ]
    
    print("\n" + "-" * 70)
    print("TESTING TILE COMPOSITION")
    print("-" * 70)
    
    for route in test_routes:
        print(f"\n{route['name']}:")
        print(f"  Start: {route['start']}")
        print(f"  End: {route['end']}")
        
        # Calculate bounds
        min_lat = min(route['start'][0], route['end'][0]) - 0.001
        max_lat = max(route['start'][0], route['end'][0]) + 0.001
        min_lon = min(route['start'][1], route['end'][1]) - 0.001
        max_lon = max(route['start'][1], route['end'][1]) + 0.001
        
        # Get tiles needed
        tiles = tiled_cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
        print(f"  Tiles needed: {len(tiles)}")
        
        # Test composition
        print("  Testing composition...")
        start_time = time.time()
        
        # Compose tiles (this would normally include compute_func)
        # For testing, we'll check if tiles exist
        composed_data = None
        tile_data_list = []
        
        for tile_x, tile_y in tiles:
            tile_data = tiled_cache.get_tile(tile_x, tile_y, 'cost')
            if tile_data:
                tile_data_list.append(tile_data)
        
        if len(tile_data_list) == len(tiles):
            print(f"  ✓ All {len(tiles)} tiles available in cache")
            
            # Test actual composition
            if len(tile_data_list) > 1:
                composed_data = tiled_cache.compose_tiles(tiles, 'cost', (min_lat, max_lat, min_lon, max_lon))
                if composed_data:
                    composition_time = time.time() - start_time
                    print(f"  ✓ Composition successful in {composition_time:.3f}s")
                    
                    # Verify composed data
                    print("  Composed data contains:")
                    for key in ['cost_surface', 'slope_degrees', 'dem']:
                        if key in composed_data:
                            shape = composed_data[key].shape
                            print(f"    - {key}: shape {shape}")
                        else:
                            print(f"    - {key}: MISSING ⚠️")
                else:
                    print("  ✗ Composition failed")
            else:
                print("  Single tile - no composition needed")
        else:
            missing = len(tiles) - len(tile_data_list)
            print(f"  ⚠️  Missing {missing} tiles from cache")

@pytest.mark.integration
def test_route_with_cache():
    """Test an actual route calculation to verify caching"""
    print("\n" + "-" * 70)
    print("TESTING ROUTE CALCULATION WITH CACHE")
    print("-" * 70)
    
    # Initialize DEM cache
    dem_cache = DEMTileCache()
    
    # Test route
    start_lat, start_lon = 40.653, -111.568
    end_lat, end_lon = 40.655, -111.566
    
    print(f"\nRoute: ({start_lat}, {start_lon}) -> ({end_lat}, {end_lon})")
    
    # First run
    print("\nFirst run (may use disk cache):")
    start_time = time.time()
    path1 = dem_cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time1 = time.time() - start_time
    
    if path1:
        print(f"  ✓ Route found in {time1:.2f}s")
        print(f"  Path points: {len(path1)}")
    else:
        print("  ✗ Route not found")
    
    # Second run (should be faster)
    print("\nSecond run (should use memory cache):")
    start_time = time.time()
    path2 = dem_cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time2 = time.time() - start_time
    
    if path2:
        print(f"  ✓ Route found in {time2:.2f}s")
        print(f"  Speedup: {time1/time2:.1f}x")
    
    # Check cache status
    cache_status = dem_cache.get_cache_status()
    print(f"\nCache status:")
    print(f"  Terrain cache: {cache_status['terrain_cache']['count']} entries")
    print(f"  Cost surface cache: {cache_status['cost_surface_cache']['count']} entries")

def verify_park_city_coverage():
    """Verify that Park City area can be fully served from cache"""
    print("\n" + "-" * 70)
    print("PARK CITY AREA COVERAGE VERIFICATION")
    print("-" * 70)
    
    # Park City bounds (approximate)
    park_city_bounds = {
        "min_lat": 40.61,
        "max_lat": 40.69,
        "min_lon": -111.58,
        "max_lon": -111.48
    }
    
    tiled_cache = TiledDEMCache(tile_size_degrees=0.01)
    
    # Calculate tiles needed for full coverage
    tiles_needed = tiled_cache.get_tiles_for_bounds(
        park_city_bounds["min_lat"],
        park_city_bounds["max_lat"],
        park_city_bounds["min_lon"],
        park_city_bounds["max_lon"]
    )
    
    print(f"\nPark City area coverage:")
    print(f"  Bounds: {park_city_bounds}")
    print(f"  Total tiles needed: {len(tiles_needed)}")
    print(f"  Coverage area: ~{len(tiles_needed) * 1.21:.0f} km²")
    
    # Check how many tiles we have
    tile_dir = "tile_cache/cost"
    cached_tiles = set()
    if os.path.exists(tile_dir):
        for f in os.listdir(tile_dir):
            if f.endswith('.pkl'):
                parts = f.replace('tile_', '').replace('.pkl', '').split('_')
                if len(parts) == 2:
                    cached_tiles.add((int(parts[0]), int(parts[1])))
    
    # Check coverage
    tiles_covered = 0
    for tile in tiles_needed:
        if tile in cached_tiles:
            tiles_covered += 1
    
    coverage_percent = (tiles_covered / len(tiles_needed)) * 100 if tiles_needed else 0
    
    print(f"\nCurrent cache coverage:")
    print(f"  Tiles cached: {tiles_covered}/{len(tiles_needed)}")
    print(f"  Coverage: {coverage_percent:.1f}%")
    
    if coverage_percent < 100:
        missing_tiles = [t for t in tiles_needed if t not in cached_tiles]
        print(f"  Missing tiles: {len(missing_tiles)}")
        print(f"  Sample missing tiles: {missing_tiles[:5]}...")
    else:
        print("  ✓ Full coverage achieved!")

if __name__ == "__main__":
    # Run all tests
    test_tile_composition()
    test_route_with_cache()
    verify_park_city_coverage()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)