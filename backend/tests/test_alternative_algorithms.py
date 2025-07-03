#!/usr/bin/env python3
"""
Test alternative pathfinding algorithms on real terrain data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
from alternative_pathfinding_algorithms import PathfindingAlgorithms, benchmark_algorithms
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


def load_precomputed_cache():
    """Load the precomputed cost surface from disk"""
    cache_dir = "precomputed_cache"
    cache_file = os.path.join(cache_dir, "40.5986,40.7072,-111.6206,-111.5139_cost.pkl")
    
    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        print("Please run precompute_area.py first")
        return None
    
    print(f"Loading precomputed cache from {cache_file}")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    return cache_data


def test_on_real_terrain():
    """Test algorithms on real Park City terrain"""
    
    # Load precomputed data
    cache_data = load_precomputed_cache()
    if not cache_data:
        return
    
    cost_surface = cache_data['cost_surface']
    indices = cache_data['indices']
    out_trans = cache_data['out_trans']
    crs = cache_data['crs']
    
    print(f"Loaded cost surface: {cost_surface.shape}")
    print(f"Cost range: {np.min(cost_surface):.2f} to {np.max(cost_surface):.2f}")
    print(f"Impassable cells: {np.sum(cost_surface >= 1000)} ({np.sum(cost_surface >= 1000)/cost_surface.size*100:.1f}%)")
    
    # Test coordinates (same as CLI)
    start_lat, start_lon = 40.6572, -111.5706
    end_lat, end_lon = 40.6486, -111.5639
    
    # Get indices
    cache = DEMTileCache(
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences()
    )
    
    # Convert coordinates to indices
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_start, y_start = transformer.transform(start_lon, start_lat)
    x_end, y_end = transformer.transform(end_lon, end_lat)
    
    cell_size_x = out_trans.a
    cell_size_y = out_trans.e  # Already negative
    
    col_start = int((x_start - out_trans.c) / cell_size_x)
    row_start = int((y_start - out_trans.f) / cell_size_y)
    col_end = int((x_end - out_trans.c) / cell_size_x)
    row_end = int((y_end - out_trans.f) / cell_size_y)
    
    height, width = indices.shape
    
    # Ensure indices are within bounds
    if not (0 <= row_start < height and 0 <= col_start < width and
            0 <= row_end < height and 0 <= col_end < width):
        print("Coordinates out of bounds!")
        return
    
    start_idx = indices[row_start, col_start]
    end_idx = indices[row_end, col_end]
    
    print(f"\nRoute: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    print(f"Grid coordinates: ({row_start}, {col_start}) → ({row_end}, {col_end})")
    print(f"Manhattan distance: {abs(row_end - row_start) + abs(col_end - col_start)} cells")
    print(f"Euclidean distance: {np.sqrt((row_end - row_start)**2 + (col_end - col_start)**2):.1f} cells")
    
    # First, time the standard A* for comparison
    print("\n" + "="*60)
    print("Testing Standard A* (current implementation)...")
    print("="*60)
    
    start_time = time.time()
    path = cache.astar_pathfinding_optimized(
        cost_surface, indices, start_idx, end_idx,
        out_trans, transformer, cache_data.get('dem'),
        optimization_config={
            'early_termination': True,
            'stagnation_limit': 10000,
            'dynamic_weights': False,
            'corner_cutting': False
        }
    )
    standard_time = time.time() - start_time
    
    print(f"Standard A* time: {standard_time:.2f}s")
    print(f"Path points: {len(path) if path else 0}")
    
    # Now test alternative algorithms
    print("\n" + "="*60)
    print("Testing Alternative Algorithms...")
    print("="*60)
    
    results = benchmark_algorithms(cost_surface, indices, start_idx, end_idx)
    
    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Algorithm':<25} {'Time (s)':>10} {'Speedup':>10} {'Nodes':>10} {'Status':>10}")
    print("-"*70)
    
    # Add standard A* to comparison
    print(f"{'Standard A* (baseline)':<25} {standard_time:>10.2f} {1.0:>10.1f}x "
          f"{'N/A':>10} {'Success' if path else 'Failed':>10}")
    
    for name, result in results.items():
        if result.path:
            speedup = standard_time / result.time_taken if result.time_taken > 0 else 0
            print(f"{name:<25} {result.time_taken:>10.2f} {speedup:>10.1f}x "
                  f"{result.nodes_explored:>10} {'Success':>10}")
        else:
            print(f"{name:<25} {'N/A':>10} {'N/A':>10} "
                  f"{result.nodes_explored:>10} {'Failed':>10}")
    
    # Analyze why standard A* is slow
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Sample the cost surface along a straight line
    straight_line_costs = []
    for t in np.linspace(0, 1, 100):
        r = int(row_start + t * (row_end - row_start))
        c = int(col_start + t * (col_end - col_start))
        if 0 <= r < height and 0 <= c < width:
            straight_line_costs.append(cost_surface[r, c])
    
    print(f"Straight line analysis:")
    print(f"  Average cost: {np.mean(straight_line_costs):.2f}")
    print(f"  Max cost: {np.max(straight_line_costs):.2f}")
    print(f"  Impassable cells: {sum(c >= 1000 for c in straight_line_costs)}")
    
    # Check terrain complexity in the search area
    search_radius = int(np.sqrt((row_end - row_start)**2 + (col_end - col_start)**2) * 1.5)
    r_min = max(0, min(row_start, row_end) - search_radius)
    r_max = min(height, max(row_start, row_end) + search_radius)
    c_min = max(0, min(col_start, col_end) - search_radius)
    c_max = min(width, max(col_start, col_end) + search_radius)
    
    search_area = cost_surface[r_min:r_max, c_min:c_max]
    print(f"\nSearch area analysis ({r_max-r_min}x{c_max-c_min} cells):")
    print(f"  Average cost: {np.mean(search_area):.2f}")
    print(f"  Cost std dev: {np.std(search_area):.2f}")
    print(f"  Impassable cells: {np.sum(search_area >= 1000)} ({np.sum(search_area >= 1000)/search_area.size*100:.1f}%)")
    
    # Terrain variation (indicates complexity)
    gradients = np.gradient(search_area)
    roughness = np.sqrt(gradients[0]**2 + gradients[1]**2)
    print(f"  Terrain roughness: {np.mean(roughness):.2f} (std: {np.std(roughness):.2f})")


def test_grid_based_optimizations():
    """Test optimizations specifically for grid-based pathfinding"""
    
    print("\n" + "="*60)
    print("GRID-BASED OPTIMIZATION IDEAS")
    print("="*60)
    
    suggestions = [
        {
            "name": "Preprocessing-based approaches",
            "ideas": [
                "1. Contraction Hierarchies - precompute shortcuts",
                "2. Distance oracles - precompute key distances",
                "3. Reach-based pruning - identify important nodes",
                "4. Hub labeling - find common waypoints"
            ]
        },
        {
            "name": "Search space reduction",
            "ideas": [
                "1. Bounding box pruning - limit search area",
                "2. Dead-end detection - avoid exploring cul-de-sacs",
                "3. Symmetry breaking - avoid redundant paths",
                "4. Goal bounding - tighten heuristic bounds"
            ]
        },
        {
            "name": "Terrain-specific optimizations",
            "ideas": [
                "1. Slope-aware search ordering",
                "2. Contour following for steep terrain",
                "3. Valley/ridge detection and following",
                "4. Obstacle clustering and avoidance"
            ]
        },
        {
            "name": "Memory and cache optimizations",
            "ideas": [
                "1. Bit-packed data structures",
                "2. Cache-friendly node ordering",
                "3. SIMD operations for neighbor evaluation",
                "4. Memory pool allocation"
            ]
        }
    ]
    
    for category in suggestions:
        print(f"\n{category['name']}:")
        for idea in category['ideas']:
            print(f"  {idea}")


if __name__ == "__main__":
    print("Testing alternative pathfinding algorithms on real terrain...")
    print("="*60)
    
    test_on_real_terrain()
    test_grid_based_optimizations()
    
    print("\n" + "="*60)
    print("Recommendations based on terrain analysis:")
    print("="*60)
    print("1. The terrain is highly complex with many obstacles")
    print("2. Standard A* explores too many nodes due to terrain variation")
    print("3. Bidirectional search should help by reducing search space")
    print("4. Jump Point Search may struggle with non-uniform costs")
    print("5. Consider implementing region-based decomposition")
    print("6. Hierarchical approaches would benefit from preprocessing")