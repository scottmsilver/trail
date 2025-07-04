#!/usr/bin/env python3
"""
Prepopulate cache for a bounding box area defined by two corner points.

Usage:
    python prepopulate_area.py lat1,lon1 lat2,lon2
    
Example:
    python prepopulate_area.py 40.6500,-111.5700 40.6600,-111.5600
"""

import sys
import time
import os
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


def format_time(seconds):
    """Format time in human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def format_size_mb(size_mb):
    """Format size in MB or GB"""
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb/1024:.2f} GB"


def prepopulate_area(corner1, corner2):
    """Prepopulate cache for the bounding box defined by two corners"""
    
    # Parse corners
    lat1, lon1 = map(float, corner1.split(','))
    lat2, lon2 = map(float, corner2.split(','))
    
    # Calculate bounding box
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    
    # Calculate area size
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    area_km2 = lat_diff * 111 * lon_diff * 111 * 0.7  # Rough approximation
    
    print(f"\n🗺️  PREPOPULATING TRAIL MAP CACHE")
    print("="*60)
    print(f"Corner 1: {lat1:.4f}, {lon1:.4f}")
    print(f"Corner 2: {lat2:.4f}, {lon2:.4f}")
    print(f"Bounding box: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
    print(f"Area size: ~{area_km2:.1f} km² ({lat_diff:.4f}° × {lon_diff:.4f}°)")
    print("-"*60)
    
    # Initialize cache
    print("\n📦 Initializing cache system...")
    cache = DEMTileCache(
        buffer=0.0,  # No buffer needed for prepopulation
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences()
    )
    
    # Get initial cache status
    initial_status = cache.get_cache_status()
    print(f"Initial cache: {initial_status['terrain_cache']['count']} terrain entries, "
          f"{initial_status['cost_surface_cache']['count']} cost surfaces")
    
    overall_start = time.time()
    
    # Step 1: Download DEM data
    print(f"\n📥 Downloading elevation data...")
    download_start = time.time()
    
    dem_file = cache.download_dem(min_lat, max_lat, min_lon, max_lon)
    if not dem_file:
        print("❌ Failed to download DEM data")
        return False
    
    download_time = time.time() - download_start
    print(f"✓ Downloaded in {format_time(download_time)}")
    
    # Step 2: Read and process DEM
    print(f"\n🏔️  Processing elevation data...")
    process_start = time.time()
    
    dem, out_trans, crs = cache.read_dem(dem_file)
    if dem is None:
        print("❌ Failed to read DEM data")
        return False
        
    dem, out_trans, crs = cache.reproject_dem(dem, out_trans, crs)
    
    process_time = time.time() - process_start
    print(f"✓ DEM shape: {dem.shape} (processed in {format_time(process_time)})")
    
    # Cache terrain data
    cache_key = f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}"
    cache.terrain_cache[cache_key] = (dem, out_trans, crs)
    
    # Step 3: Fetch obstacles
    print(f"\n🚧 Fetching obstacle data...")
    obstacle_start = time.time()
    
    obstacles = cache.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
    obstacle_mask = cache.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)
    
    obstacle_time = time.time() - obstacle_start
    obstacle_count = sum(obstacle_mask.flatten())
    print(f"✓ Found {len(obstacles)} obstacles covering {obstacle_count:,} cells "
          f"(fetched in {format_time(obstacle_time)})")
    
    # Step 4: Fetch paths
    print(f"\n🥾 Fetching trail and path data...")
    path_start = time.time()
    
    paths = cache.fetch_paths(min_lat, max_lat, min_lon, max_lon)
    path_raster, path_types = cache.rasterize_paths(paths, out_trans, dem.shape, crs)
    
    path_time = time.time() - path_start
    path_count = sum(path_raster.flatten() > 0)
    print(f"✓ Found {len(paths)} path segments covering {path_count:,} cells "
          f"(fetched in {format_time(path_time)})")
    
    # Step 5: Compute cost surface
    print(f"\n💰 Computing cost surface...")
    cost_start = time.time()
    
    cost_surface, slope_degrees = cache.compute_cost_surface(
        dem, out_trans, obstacle_mask, path_raster, path_types
    )
    
    cost_time = time.time() - cost_start
    indices = cache.build_indices(cost_surface)
    
    import numpy as np
    print(f"✓ Cost surface computed in {format_time(cost_time)}")
    print(f"  Stats: min={np.min(cost_surface):.2f}, max={np.max(cost_surface):.2f}, "
          f"mean={np.mean(cost_surface):.2f}")
    impassable_pct = np.sum(cost_surface > 1000) / cost_surface.size * 100
    print(f"  Impassable cells: {impassable_pct:.1f}%")
    
    # Cache the cost surface
    cost_cache_key = f"{cache_key}_cost"
    cache.cost_surface_cache[cost_cache_key] = {
        'cost_surface': cost_surface,
        'indices': indices,
        'slope_degrees': slope_degrees,
        'obstacle_mask': obstacle_mask,
        'path_raster': path_raster,
        'dem': dem,
        'out_trans': out_trans,
        'crs': crs
    }
    
    # Step 6: Process tiles for tiled cache
    print(f"\n🗂️  Processing tiles for fast access...")
    tile_start = time.time()
    
    # The tiled cache will automatically process tiles as needed
    # We can trigger this by calling the tile methods
    tile_size = cache.tiled_cache.tile_size
    num_lat_tiles = int(np.ceil((max_lat - min_lat) / tile_size))
    num_lon_tiles = int(np.ceil((max_lon - min_lon) / tile_size))
    total_tiles = num_lat_tiles * num_lon_tiles
    
    print(f"  Area requires {total_tiles} tiles ({num_lat_tiles}×{num_lon_tiles})")
    
    # Process each tile
    processed_tiles = 0
    for i in range(num_lat_tiles):
        for j in range(num_lon_tiles):
            tile_min_lat = min_lat + i * tile_size
            tile_max_lat = min(tile_min_lat + tile_size, max_lat)
            tile_min_lon = min_lon + j * tile_size  
            tile_max_lon = min(tile_min_lon + tile_size, max_lon)
            
            # This will trigger tile processing and caching
            tiles_needed = cache.tiled_cache.get_tiles_for_bounds(
                tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon
            )
            processed_tiles += len(tiles_needed)
    
    tile_time = time.time() - tile_start
    print(f"✓ Processed {processed_tiles} tiles in {format_time(tile_time)}")
    
    # Final statistics
    total_time = time.time() - overall_start
    final_status = cache.get_cache_status()
    
    print(f"\n📊 PREPOPULATION COMPLETE")
    print("="*60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Cache growth:")
    print(f"  Terrain: {initial_status['terrain_cache']['count']} → "
          f"{final_status['terrain_cache']['count']} entries")
    print(f"  Cost surfaces: {initial_status['cost_surface_cache']['count']} → "
          f"{final_status['cost_surface_cache']['count']} entries")
    print(f"  Memory usage: {format_size_mb(initial_status['total_memory_mb'])} → "
          f"{format_size_mb(final_status['total_memory_mb'])}")
    
    # Check disk cache
    tile_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile_cache")
    if os.path.exists(tile_cache_dir):
        total_size = 0
        tile_count = 0
        for root, dirs, files in os.walk(tile_cache_dir):
            for f in files:
                if f.endswith('.pkl'):
                    tile_count += 1
                    total_size += os.path.getsize(os.path.join(root, f))
        
        print(f"\nDisk cache:")
        print(f"  Tiles: {tile_count} files")  
        print(f"  Size: {format_size_mb(total_size / (1024*1024))}")
    
    print("\n✅ Area is now prepopulated and ready for fast routing!")
    return True


def main():
    """Main CLI function"""
    if len(sys.argv) != 3:
        print("Usage: python prepopulate_area.py lat1,lon1 lat2,lon2")
        print("\nExample:")
        print("  python prepopulate_area.py 40.6500,-111.5700 40.6600,-111.5600")
        print("\nThis will prepopulate the rectangular area with corners at:")
        print("  - Corner 1: (40.6500, -111.5700)")
        print("  - Corner 2: (40.6600, -111.5600)")
        sys.exit(1)
    
    corner1 = sys.argv[1]
    corner2 = sys.argv[2]
    
    try:
        success = prepopulate_area(corner1, corner2)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()