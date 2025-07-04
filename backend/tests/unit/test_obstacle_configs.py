"""
Test different obstacle configurations and their effects on pathfinding
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from pyproj import Transformer
from rasterio.transform import from_origin
import pytest


@pytest.mark.unit
def test_obstacle_configurations():
    """Test how different configurations handle obstacles"""
    
    # Create test terrain
    rows, cols = 20, 20
    dem = np.zeros((rows, cols))
    
    # Add some elevation features
    # Hill in the middle
    for i in range(8, 13):
        for j in range(8, 13):
            distance = np.sqrt((i-10)**2 + (j-10)**2)
            dem[i, j] = max(0, 5 - distance) * 10  # Up to 50m elevation
    
    # Create different obstacle scenarios
    scenarios = [
        {
            'name': 'Water Crossing',
            'description': 'Stream running north-south',
            'obstacles': [(i, 10) for i in range(5, 15)],  # Vertical stream
            'obstacle_type': 'stream',
            'start': (10, 2),
            'end': (10, 17)
        },
        {
            'name': 'Steep Terrain',
            'description': 'Path over steep hill',
            'obstacles': [],  # No explicit obstacles, just steep terrain
            'obstacle_type': None,
            'start': (5, 10),
            'end': (15, 10)
        },
        {
            'name': 'Building Avoidance',
            'description': 'Buildings blocking direct path',
            'obstacles': [(i, j) for i in range(8, 12) for j in range(8, 12)],
            'obstacle_type': 'building',
            'start': (10, 5),
            'end': (10, 15)
        }
    ]
    
    # Test with different user profiles
    profiles = [
        ('Easy Hiker', ObstaclePresets.easy_hiker()),
        ('Experienced Hiker', ObstaclePresets.experienced_hiker()),
        ('Trail Runner', ObstaclePresets.trail_runner()),
        ('Accessibility', ObstaclePresets.accessibility_focused())
    ]
    
    # Setup pathfinding
    transform = from_origin(-111.5, 40.5, 30, 30)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    pathfinder = DEMTileCache()
    
    results = []
    
    for scenario in scenarios:
        print(f"\n\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 50)
        
        for profile_name, config in profiles:
            # Create cost surface based on configuration
            cost_surface = np.ones((rows, cols))
            
            # Apply terrain slope costs
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    # Simple slope calculation
                    dz_y = abs(dem[i+1, j] - dem[i-1, j]) / 60  # 30m cells
                    dz_x = abs(dem[i, j+1] - dem[i, j-1]) / 60
                    slope_degrees = np.degrees(np.arctan(np.sqrt(dz_x**2 + dz_y**2)))
                    
                    cost_surface[i, j] = config.get_slope_cost_multiplier(slope_degrees)
            
            # Apply obstacle costs
            for obs_row, obs_col in scenario['obstacles']:
                if scenario['obstacle_type']:
                    cost = config.get_cost_for_feature(
                        scenario['obstacle_type'], 
                        scenario['obstacle_type']
                    )
                    cost_surface[obs_row, obs_col] = cost
            
            # Find path
            start_row, start_col = scenario['start']
            end_row, end_col = scenario['end']
            start_idx = indices[start_row, start_col]
            end_idx = indices[end_row, end_col]
            
            path = pathfinder.astar_pathfinding(
                cost_surface, indices, start_idx, end_idx,
                transform, transformer, dem
            )
            
            # Analyze results
            if path:
                path_cells = []
                total_cost = 0
                
                for lon, lat in path:
                    x, y = transformer.transform(lon, lat)
                    col = int((x - transform.c) / transform.a)
                    row = int((y - transform.f) / transform.e)
                    if 0 <= row < rows and 0 <= col < cols:
                        path_cells.append((row, col))
                        total_cost += cost_surface[row, col]
                
                # Check if path crosses obstacles
                crosses_obstacle = any(
                    (r, c) in scenario['obstacles'] 
                    for r, c in path_cells
                )
                
                # Calculate elevation gain
                elevation_gain = 0
                for i in range(1, len(path_cells)):
                    r1, c1 = path_cells[i-1]
                    r2, c2 = path_cells[i]
                    if dem[r2, c2] > dem[r1, c1]:
                        elevation_gain += dem[r2, c2] - dem[r1, c1]
                
                result = {
                    'profile': profile_name,
                    'path_length': len(path_cells),
                    'total_cost': total_cost,
                    'crosses_obstacle': crosses_obstacle,
                    'elevation_gain': elevation_gain,
                    'success': True
                }
            else:
                result = {
                    'profile': profile_name,
                    'path_length': 0,
                    'total_cost': np.inf,
                    'crosses_obstacle': False,
                    'elevation_gain': 0,
                    'success': False
                }
            
            results.append((scenario['name'], result))
            
            # Print results
            print(f"\n{profile_name}:")
            if result['success']:
                print(f"  ✓ Path found: {result['path_length']} steps")
                print(f"  Total cost: {result['total_cost']:.1f}")
                print(f"  Elevation gain: {result['elevation_gain']:.1f}m")
                if result['crosses_obstacle']:
                    print(f"  ⚠ Crosses obstacle (allowed by configuration)")
            else:
                print(f"  ✗ No path found (obstacles impassable)")
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY: How Different Users Handle Obstacles")
    print("="*60)
    
    # Analyze water crossing
    print("\nWater Crossing Scenario:")
    for profile_name, config in profiles:
        water_results = [r for s, r in results if s == 'Water Crossing' and r['profile'] == profile_name][0]
        if water_results['success']:
            if water_results['crosses_obstacle']:
                print(f"  {profile_name}: Crosses stream (cost={config.get_cost_for_feature('stream', 'stream')})")
            else:
                print(f"  {profile_name}: Goes around stream")
        else:
            print(f"  {profile_name}: Cannot cross - stream impassable")
    
    # Analyze steep terrain
    print("\nSteep Terrain Scenario:")
    for profile_name, config in profiles:
        steep_results = [r for s, r in results if s == 'Steep Terrain' and r['profile'] == profile_name][0]
        if steep_results['success']:
            print(f"  {profile_name}: Climbs hill (elevation gain={steep_results['elevation_gain']:.1f}m)")
        else:
            print(f"  {profile_name}: Cannot climb - too steep")
    
    return results


if __name__ == "__main__":
    print("Testing obstacle configurations...")
    test_obstacle_configurations()