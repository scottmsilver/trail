#!/usr/bin/env python3
"""Test why steep slopes aren't being penalized enough"""

import asyncio
import numpy as np
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.services.dem_tile_cache import DEMTileCache
from app.models.route import Coordinate
import pytest

async def analyze_route_slopes():
    """Analyze the slopes along a problematic route"""
    
    # Test coordinates
    start = Coordinate(lat=40.6546, lon=-111.5705)
    end = Coordinate(lat=40.6485, lon=-111.5641)
    
    print(f"Analyzing route from {start.lat}, {start.lon} to {end.lat}, {end.lon}")
    print("=" * 60)
    
    # Test with different profiles
    profiles = [
        ("default", ObstacleConfig()),
        ("easy", ObstaclePresets.easy_hiker()),
        ("experienced", ObstaclePresets.experienced_hiker()),
    ]
    
    for profile_name, config in profiles:
        print(f"\nProfile: {profile_name}")
        print("-" * 40)
        
        # Enable continuous slope function
        config.use_continuous_slope = True
        config.slope_profile = profile_name if profile_name != "default" else "default"
        
        # Create trail finder
        trail_finder = TrailFinderService(obstacle_config=config)
        
        # Find route
        path, stats = await trail_finder.find_route(start, end, {})
        
        if not path:
            print(f"No route found: {stats.get('error', 'Unknown error')}")
            continue
            
        print(f"Route found with {len(path)} points")
        print(f"Total distance: {stats.get('distance_km', 0):.2f} km")
        
        # Get the path with slopes
        if 'path_with_slopes' in stats:
            path_slopes = stats['path_with_slopes']
            slopes = [p.get('slope', 0) for p in path_slopes if 'slope' in p]
            
            if slopes:
                print(f"\nSlope statistics:")
                print(f"  Max slope: {max(slopes):.1f}°")
                print(f"  Average slope: {np.mean(np.abs(slopes)):.1f}°")
                print(f"  Slopes > 20°: {sum(1 for s in slopes if abs(s) > 20)}")
                print(f"  Slopes > 30°: {sum(1 for s in slopes if abs(s) > 30)}")
                print(f"  Slopes > 40°: {sum(1 for s in slopes if abs(s) > 40)}")
                
                # Show steep segments
                steep_segments = []
                for i, p in enumerate(path_slopes[:-1]):
                    if abs(p.get('slope', 0)) > 30:
                        steep_segments.append({
                            'index': i,
                            'slope': p['slope'],
                            'lat': p['lat'],
                            'lon': p['lon'],
                            'elevation': p.get('elevation', 0)
                        })
                
                if steep_segments:
                    print(f"\nSteep segments (>30°):")
                    for seg in steep_segments[:10]:  # Show first 10
                        print(f"  Segment {seg['index']}: {seg['slope']:.1f}° at ({seg['lat']:.4f}, {seg['lon']:.4f}), elev: {seg['elevation']:.0f}m")
        
        # Check cost calculation for steep slopes
        print(f"\nSlope cost samples for {profile_name}:")
        test_slopes = [10, 20, 30, 40, 50]
        for slope in test_slopes:
            cost = config.get_slope_cost_multiplier(slope)
            print(f"  {slope}°: cost multiplier = {cost:.2f}")
    
    # Now let's check the actual A* behavior
    print("\n" + "=" * 60)
    print("Analyzing A* pathfinding behavior")
    print("=" * 60)
    
    # Create a debug trail finder
    debug_config = ObstacleConfig()
    debug_config.use_continuous_slope = True
    debug_trail_finder = TrailFinderService(debug_mode=True, obstacle_config=debug_config)
    
    # Get bounds for the area
    min_lat = min(start.lat, end.lat) - 0.01
    max_lat = max(start.lat, end.lat) + 0.01
    min_lon = min(start.lon, end.lon) - 0.01
    max_lon = max(start.lon, end.lon) + 0.01
    
    # Check DEM resolution
    dem_cache = DEMTileCache()
    dem_file = dem_cache.download_dem(min_lat, max_lat, min_lon, max_lon)
    if dem_file:
        dem, out_trans, crs = dem_cache.read_dem(dem_file)
        if dem is not None:
            print(f"\nDEM Information:")
            print(f"  Shape: {dem.shape}")
            print(f"  Cell size X: {out_trans.a:.2f}m")
            print(f"  Cell size Y: {-out_trans.e:.2f}m")
            print(f"  Min elevation: {np.min(dem):.0f}m")
            print(f"  Max elevation: {np.max(dem):.0f}m")
            
            # Calculate slopes in the DEM
            cell_size_x = out_trans.a
            cell_size_y = -out_trans.e
            dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
            slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
            slope_degrees = np.degrees(slope_radians)
            
            print(f"\nDEM Slope Statistics:")
            print(f"  Max slope in area: {np.max(slope_degrees):.1f}°")
            print(f"  Mean slope: {np.mean(slope_degrees):.1f}°")
            print(f"  Pixels with slope > 30°: {np.sum(slope_degrees > 30)} ({100 * np.sum(slope_degrees > 30) / slope_degrees.size:.1f}%)")
            print(f"  Pixels with slope > 40°: {np.sum(slope_degrees > 40)} ({100 * np.sum(slope_degrees > 40) / slope_degrees.size:.1f}%)")

if __name__ == "__main__":
    asyncio.run(analyze_route_slopes())