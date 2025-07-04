#!/usr/bin/env python3
"""Test why steep slopes aren't being penalized enough - Fixed version"""

import asyncio
import numpy as np
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.services.dem_tile_cache import DEMTileCache
from app.models.route import Coordinate
import rasterio
import pytest

async def analyze_route_slopes():
    """Analyze the slopes along a problematic route"""
    
    # Test coordinates
    start = Coordinate(lat=40.6546, lon=-111.5705)
    end = Coordinate(lat=40.6485, lon=-111.5641)
    
    print(f"Analyzing route from {start.lat}, {start.lon} to {end.lat}, {end.lon}")
    print("=" * 60)
    
    # First, let's check the actual DEM and slope calculation
    dem_cache = DEMTileCache()
    
    # Get bounds for the area
    min_lat = min(start.lat, end.lat) - 0.01
    max_lat = max(start.lat, end.lat) + 0.01
    min_lon = min(start.lon, end.lon) - 0.01
    max_lon = max(start.lon, end.lon) + 0.01
    
    # Download and read DEM
    dem_file = dem_cache.download_dem(min_lat, max_lat, min_lon, max_lon)
    if dem_file:
        dem, out_trans, crs = dem_cache.read_dem(dem_file)
        if dem is not None:
            print(f"\nOriginal DEM Information:")
            print(f"  Shape: {dem.shape}")
            print(f"  CRS: {crs}")
            print(f"  Transform: {out_trans}")
            
            # Reproject if needed
            dem_reprojected, out_trans_reprojected, crs_reprojected = dem_cache.reproject_dem(dem, out_trans, crs)
            
            print(f"\nReprojected DEM Information:")
            print(f"  Shape: {dem_reprojected.shape}")
            print(f"  CRS: {crs_reprojected}")
            print(f"  Transform: {out_trans_reprojected}")
            print(f"  Cell size X: {out_trans_reprojected.a:.2f}m")
            print(f"  Cell size Y: {-out_trans_reprojected.e:.2f}m")
            
            # Calculate slopes properly
            cell_size_x = out_trans_reprojected.a
            cell_size_y = -out_trans_reprojected.e
            dzdx, dzdy = np.gradient(dem_reprojected, cell_size_x, cell_size_y)
            slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
            slope_degrees = np.degrees(slope_radians)
            
            print(f"\nSlope Statistics in Area:")
            print(f"  Max slope: {np.max(slope_degrees):.1f}°")
            print(f"  Mean slope: {np.mean(slope_degrees):.1f}°")
            print(f"  Median slope: {np.median(slope_degrees):.1f}°")
            print(f"  Pixels with slope > 20°: {np.sum(slope_degrees > 20)} ({100 * np.sum(slope_degrees > 20) / slope_degrees.size:.1f}%)")
            print(f"  Pixels with slope > 30°: {np.sum(slope_degrees > 30)} ({100 * np.sum(slope_degrees > 30) / slope_degrees.size:.1f}%)")
            print(f"  Pixels with slope > 40°: {np.sum(slope_degrees > 40)} ({100 * np.sum(slope_degrees > 40) / slope_degrees.size:.1f}%)")
    
    # Test with different slope penalties
    print("\n" + "=" * 60)
    print("Testing with different slope penalty configurations")
    print("=" * 60)
    
    # Create more aggressive slope configuration
    aggressive_config = ObstacleConfig()
    aggressive_config.use_continuous_slope = True
    aggressive_config.slope_profile = 'city_walker'  # Most aggressive slope penalties
    
    print("\nSlope cost multipliers for city_walker profile:")
    test_slopes = [5, 10, 15, 20, 25, 30, 35, 40]
    for slope in test_slopes:
        cost = aggressive_config.get_slope_cost_multiplier(slope)
        if cost == np.inf:
            print(f"  {slope}°: INFINITY (impassable)")
        else:
            print(f"  {slope}°: {cost:.2f}x")
    
    # Find route with aggressive penalties
    trail_finder = TrailFinderService(obstacle_config=aggressive_config)
    path, stats = await trail_finder.find_route(start, end, {})
    
    if not path:
        print(f"\nNo route found with city_walker profile: {stats.get('error', 'Unknown error')}")
    else:
        print(f"\nRoute found with city_walker profile:")
        print(f"  Points: {len(path)}")
        print(f"  Distance: {stats.get('distance_km', 0):.2f} km")
        
        if 'path_with_slopes' in stats:
            path_slopes = stats['path_with_slopes']
            slopes = [p.get('slope', 0) for p in path_slopes if 'slope' in p]
            
            if slopes:
                print(f"\nSlope statistics along path:")
                print(f"  Max slope: {max(slopes):.1f}°")
                print(f"  Average slope: {np.mean(np.abs(slopes)):.1f}°")
                print(f"  Slopes > 15°: {sum(1 for s in slopes if abs(s) > 15)}")
                print(f"  Slopes > 20°: {sum(1 for s in slopes if abs(s) > 20)}")
                print(f"  Slopes > 25°: {sum(1 for s in slopes if abs(s) > 25)}")
    
    # Try with wheelchair profile (extremely strict)
    print("\n" + "-" * 40)
    wheelchair_config = ObstacleConfig()
    wheelchair_config.use_continuous_slope = True
    wheelchair_config.slope_profile = 'wheelchair'
    
    print("\nSlope cost multipliers for wheelchair profile:")
    for slope in [5, 8, 10]:
        cost = wheelchair_config.get_slope_cost_multiplier(slope)
        if cost == np.inf:
            print(f"  {slope}°: INFINITY (impassable)")
        else:
            print(f"  {slope}°: {cost:.2f}x")
    
    trail_finder_wheelchair = TrailFinderService(obstacle_config=wheelchair_config)
    path_w, stats_w = await trail_finder_wheelchair.find_route(start, end, {})
    
    if not path_w:
        print(f"\nNo route found with wheelchair profile: {stats_w.get('error', 'Expected - terrain too steep')}")
    else:
        print(f"\nRoute found with wheelchair profile (unexpected!)")

if __name__ == "__main__":
    asyncio.run(analyze_route_slopes())