#!/usr/bin/env python
"""Test pathfinding with more realistic varied terrain"""
import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstaclePresets
from rasterio.transform import from_origin
from pyproj import Transformer

def create_varied_terrain():
    """Create terrain with a steep hill and multiple paths"""
    rows, cols = 50, 50
    dem = np.zeros((rows, cols))
    
    # Create base elevation gradient
    for i in range(rows):
        for j in range(cols):
            dem[i, j] = i * 0.5  # Gentle overall slope
    
    # Add a steep hill in the middle
    center_row, center_col = 25, 25
    for i in range(rows):
        for j in range(cols):
            # Distance from center
            dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if dist < 15:
                # Add gaussian hill
                height = 30 * np.exp(-(dist**2) / 50)
                dem[i, j] += height
    
    # Create a valley path on the left
    for i in range(rows):
        dem[i, 10] = max(0, dem[i, 10] - 5)
        dem[i, 11] = max(0, dem[i, 11] - 3)
        dem[i, 12] = max(0, dem[i, 12] - 3)
        dem[i, 13] = max(0, dem[i, 13] - 5)
    
    # Create a ridge path on the right
    for i in range(rows):
        dem[i, 40] += 5
        dem[i, 41] += 7
        dem[i, 42] += 7  
        dem[i, 43] += 5
        
    return dem, rows, cols

def test_profile_pathfinding():
    """Test different profiles find different paths"""
    dem, rows, cols = create_varied_terrain()
    
    # Setup
    transform = from_origin(-111.5, 40.5, 30, 30)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    obstacle_mask = np.zeros((rows, cols), dtype=bool)
    
    # Add some obstacles
    obstacle_mask[20:30, 22:28] = True  # Block direct path through hill
    
    # Test profiles
    profiles = {
        'easy': ObstaclePresets.easy_hiker(),
        'experienced': ObstaclePresets.experienced_hiker(),
        'trail_runner': ObstaclePresets.trail_runner(),
        'accessibility': ObstaclePresets.accessibility_focused()
    }
    
    # Start and end points
    start_idx = indices[10, 25]  # Start near top center
    end_idx = indices[40, 25]    # End near bottom center
    
    results = {}
    
    for name, config in profiles.items():
        print(f"\nTesting {name} profile...")
        pathfinder = DEMTileCache(obstacle_config=config)
        cost_surface = pathfinder.compute_cost_surface(dem, transform, obstacle_mask)
        
        # Check cost variations
        valley_cost = cost_surface[25, 11]  # Valley path
        hill_cost = cost_surface[25, 25]    # Hill center
        ridge_cost = cost_surface[25, 41]   # Ridge path
        
        print(f"  Valley cost: {valley_cost:.2f}")
        print(f"  Hill cost: {hill_cost:.2f}")
        print(f"  Ridge cost: {ridge_cost:.2f}")
        
        # Find path
        path = pathfinder.astar_pathfinding(
            cost_surface, indices, start_idx, end_idx, transform, transformer, dem
        )
        
        if path:
            # Calculate average column (indicates left/right preference)
            cols_visited = []
            for lon, lat in path:
                x, y = transformer.transform(lon, lat)
                col = int((x - transform.c) / transform.a)
                cols_visited.append(col)
            
            avg_col = np.mean(cols_visited)
            results[name] = {
                'path_length': len(path),
                'avg_col': avg_col,
                'valley_cost': valley_cost,
                'hill_cost': hill_cost,
                'ridge_cost': ridge_cost
            }
            print(f"  Path found: {len(path)} points, avg column: {avg_col:.1f}")
            
            # Determine path preference
            if avg_col < 20:
                print(f"  → Prefers valley route (left)")
            elif avg_col > 35:
                print(f"  → Prefers ridge route (right)")
            else:
                print(f"  → Takes middle route")
        else:
            print(f"  No path found!")
            results[name] = None
    
    # Analyze differences
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    
    if all(results.values()):
        # Easy hiker should prefer valley (leftmost, gentlest)
        if results['easy']['avg_col'] < results['experienced']['avg_col']:
            print("✓ Easy hiker prefers gentler valley route")
        else:
            print("✗ Easy hiker does not prefer valley route as expected")
        
        # Trail runner might take more direct routes
        if results['trail_runner']['path_length'] <= results['easy']['path_length']:
            print("✓ Trail runner takes more direct route")
        else:
            print("✗ Trail runner does not optimize for distance")
        
        # Accessibility should strongly avoid steep areas
        if results['accessibility'] and results['accessibility']['hill_cost'] > 100:
            print("✓ Accessibility profile avoids steep terrain")
        else:
            print("✗ Accessibility profile does not properly penalize steep terrain")

if __name__ == "__main__":
    test_profile_pathfinding()