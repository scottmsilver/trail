#!/usr/bin/env python3
"""Test route with 10m resolution as fallback for long routes"""

import asyncio
import time
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
from app.models.route import Coordinate
from app.services.dem_tile_cache import DEMTileCache
from geopy.distance import distance

async def test_with_resolution_fallback():
    # Failing route coordinates
    start = Coordinate(lat=40.6566, lon=-111.5701)
    end = Coordinate(lat=40.6286, lon=-111.5689)
    
    print(f"Testing route with resolution fallback")
    print(f"Start: {start.lat}, {start.lon}")
    print(f"End: {end.lat}, {end.lon}")
    
    # Calculate approximate distance
    direct_distance = distance((start.lat, start.lon), (end.lat, end.lon)).km
    print(f"Direct distance: {direct_distance:.2f} km")
    print("=" * 60)
    
    # First, override DEM download to use 10m for this test
    print("\nForcing 10m resolution for this long route...")
    
    # Monkey patch the download method temporarily
    original_download = DEMTileCache.download_dem
    
    def download_10m_only(self, min_lat, max_lat, min_lon, max_lon):
        """Download DEM at 10m resolution only"""
        import os
        import py3dep
        
        dem_dir = os.path.join('dem_data')
        dem_file = os.path.join(dem_dir, 'dem.tif')
        if not os.path.exists(dem_dir):
            os.makedirs(dem_dir)
        
        try:
            # Force 10m resolution
            dem = py3dep.get_map(
                "DEM",
                (min_lon, min_lat, max_lon, max_lat),
                resolution=10,
                crs="EPSG:4326"
            )
            print("Downloaded DEM at 10m resolution (forced)")
            dem.rio.to_raster(dem_file)
            return dem_file
        except Exception as e:
            print(f"Error downloading DEM: {e}")
            return None
    
    # Temporarily replace the method
    DEMTileCache.download_dem = download_10m_only
    
    try:
        # Use default configuration
        config = ObstacleConfig()
        config.use_continuous_slope = True
        config.slope_profile = 'default'
        
        path_prefs = PathPreferences()
        
        trail_finder = TrailFinderService(
            obstacle_config=config,
            path_preferences=path_prefs
        )
        
        print("\nStarting route search with 10m resolution...")
        start_time = time.time()
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        elapsed = time.time() - start_time
        print(f"\nSearch completed in {elapsed:.1f} seconds")
        
        if not path:
            error = stats.get('error', 'Unknown error')
            print(f"Route search failed: {error}")
        else:
            print(f"\nRoute found successfully!")
            print(f"  Total distance: {stats.get('distance_km', 0):.2f} km")
            print(f"  Path points: {len(path)}")
            
            # Analyze path
            if 'path_with_slopes' in stats:
                path_slopes = stats['path_with_slopes']
                slopes = [abs(p.get('slope', 0)) for p in path_slopes if 'slope' in p]
                
                if slopes:
                    import numpy as np
                    print(f"\nSlope statistics:")
                    print(f"  Max slope: {max(slopes):.1f}°")
                    print(f"  Average slope: {np.mean(slopes):.1f}°")
                    print(f"  Slopes > 30°: {sum(1 for s in slopes if s > 30)}")
                    print(f"  Slopes > 35°: {sum(1 for s in slopes if s > 35)}")
                    
    finally:
        # Restore original method
        DEMTileCache.download_dem = original_download
        print("\nRestored original DEM download method")

if __name__ == "__main__":
    asyncio.run(test_with_resolution_fallback())