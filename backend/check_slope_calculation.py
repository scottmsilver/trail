#!/usr/bin/env python3
"""Check if slope calculation is correct for the problematic segment"""

import asyncio
import numpy as np
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.models.route import Coordinate
from geopy.distance import distance

async def check_slope_calculation():
    # Test coordinates
    start = Coordinate(lat=40.6546, lon=-111.5705)
    end = Coordinate(lat=40.6485, lon=-111.5641)
    
    # Get route with slopes
    config = ObstacleConfig()
    config.use_continuous_slope = True
    trail_finder = TrailFinderService(obstacle_config=config)
    path, stats = await trail_finder.find_route(start, end, {})
    
    if not path or 'path_with_slopes' not in stats:
        print("No path with slopes found")
        return
        
    path_slopes = stats['path_with_slopes']
    
    # Find the steep segment and verify calculation
    for i in range(len(path_slopes) - 1):
        p1 = path_slopes[i]
        p2 = path_slopes[i + 1]
        
        if abs(p1.get('slope', 0)) > 30:
            print(f"\nSteep segment found at index {i}:")
            print(f"Point 1: ({p1['lat']:.6f}, {p1['lon']:.6f}), elevation: {p1.get('elevation', 0):.1f}m")
            print(f"Point 2: ({p2['lat']:.6f}, {p2['lon']:.6f}), elevation: {p2.get('elevation', 0):.1f}m")
            print(f"Reported slope: {p1['slope']:.1f}°")
            
            # Manually calculate slope
            # Distance between points
            horizontal_dist = distance((p1['lat'], p1['lon']), (p2['lat'], p2['lon'])).meters
            vertical_dist = abs(p2.get('elevation', 0) - p1.get('elevation', 0))
            
            print(f"\nManual calculation:")
            print(f"Horizontal distance: {horizontal_dist:.1f}m")
            print(f"Vertical distance: {vertical_dist:.1f}m")
            
            if horizontal_dist > 0:
                slope_ratio = vertical_dist / horizontal_dist
                slope_degrees = np.degrees(np.arctan(slope_ratio))
                print(f"Calculated slope: {slope_degrees:.1f}°")
                
                # Check if this is reasonable
                print(f"\nAnalysis:")
                if slope_degrees > 30:
                    print("✗ This is indeed a very steep slope!")
                    print(f"  Rise/Run ratio: {slope_ratio:.3f}")
                    print(f"  For every meter forward, you climb {slope_ratio:.2f} meters")
                else:
                    print("✓ Slope calculation seems incorrect in the algorithm")
            else:
                print("Points are at same location horizontally")
    
    # Also check the resolution
    print(f"\n\nPath resolution check:")
    print(f"Total path points: {len(path_slopes)}")
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(path_slopes) - 1):
        p1 = path_slopes[i]
        p2 = path_slopes[i + 1]
        dist = distance((p1['lat'], p1['lon']), (p2['lat'], p2['lon'])).meters
        distances.append(dist)
    
    if distances:
        print(f"Average distance between points: {np.mean(distances):.1f}m")
        print(f"Min distance: {np.min(distances):.1f}m")
        print(f"Max distance: {np.max(distances):.1f}m")
        
        # Check if 10m resolution is being used correctly
        expected_resolution = 10.0  # meters
        close_to_resolution = sum(1 for d in distances if abs(d - expected_resolution) < 5) / len(distances)
        print(f"Points near {expected_resolution}m apart: {close_to_resolution*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(check_slope_calculation())