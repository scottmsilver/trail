#!/usr/bin/env python3
"""
Test path similarity using Discrete Fréchet Distance and other metrics
"""

import numpy as np
from math import sqrt
import time
import heapq
from typing import List, Tuple, Dict


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def discrete_frechet_distance(path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]) -> float:
    """
    Calculate the Discrete Fréchet Distance between two paths.
    
    The Fréchet distance is like walking two dogs on leashes - it's the minimum
    leash length needed where both can walk along their paths from start to end
    without backtracking.
    
    Returns the minimum maximum distance between paired points.
    """
    n = len(path1)
    m = len(path2)
    
    # Initialize the dynamic programming table
    dp = np.full((n, m), float('inf'))
    
    # Base case
    dp[0][0] = euclidean_distance(path1[0], path2[0])
    
    # Fill first row
    for j in range(1, m):
        dp[0][j] = max(dp[0][j-1], euclidean_distance(path1[0], path2[j]))
    
    # Fill first column
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], euclidean_distance(path1[i], path2[0]))
    
    # Fill the rest of the table
    for i in range(1, n):
        for j in range(1, m):
            d = euclidean_distance(path1[i], path2[j])
            dp[i][j] = max(d, min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
    
    return dp[n-1][m-1]


def hausdorff_distance(path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]) -> float:
    """
    Calculate the directed Hausdorff distance from path1 to path2.
    This is the maximum distance from any point in path1 to its nearest point in path2.
    """
    max_dist = 0
    for p1 in path1:
        min_dist = float('inf')
        for p2 in path2:
            dist = euclidean_distance(p1, p2)
            min_dist = min(min_dist, dist)
        max_dist = max(max_dist, min_dist)
    return max_dist


def calculate_comprehensive_path_metrics(path1: List[Tuple[int, int]], 
                                       path2: List[Tuple[int, int]], 
                                       grid_shape: Tuple[int, int]) -> Dict:
    """
    Calculate comprehensive path similarity metrics including Fréchet distance.
    
    Returns dict with:
    - frechet_distance: Discrete Fréchet distance
    - hausdorff_distance: Maximum deviation between paths
    - length_ratio: Ratio of path lengths
    - start_end_match: Whether paths have same endpoints
    - path_length_diff: Absolute difference in path lengths
    - is_identical: Whether paths are exactly the same
    """
    if not path1 or not path2:
        return {
            'is_identical': False,
            'reason': 'Empty path(s)',
            'frechet_distance': float('inf'),
            'hausdorff_distance': float('inf')
        }
    
    # Check if paths are identical
    is_identical = len(path1) == len(path2) and all(p1 == p2 for p1, p2 in zip(path1, path2))
    
    # Basic metrics
    length_diff = abs(len(path1) - len(path2))
    length_ratio = min(len(path1), len(path2)) / max(len(path1), len(path2))
    
    # Endpoint check
    start_match = path1[0] == path2[0]
    end_match = path1[-1] == path2[-1]
    
    # Calculate distances
    frechet_dist = discrete_frechet_distance(path1, path2)
    hausdorff_dist = hausdorff_distance(path1, path2)
    
    # Normalize by grid diagonal for scale-independent comparison
    grid_diagonal = sqrt(grid_shape[0]**2 + grid_shape[1]**2)
    
    # Determine if paths are "same enough"
    # Criteria for "same path":
    # 1. Fréchet distance < 2 cells
    # 2. Same start and end points
    # 3. Length difference < 2 cells
    same_path = (
        frechet_dist < 2.0 and
        start_match and end_match and
        length_diff < 2
    )
    
    return {
        'is_identical': is_identical,
        'is_same_path': same_path,
        'frechet_distance': frechet_dist,
        'frechet_normalized': frechet_dist / grid_diagonal,
        'hausdorff_distance': hausdorff_dist,
        'hausdorff_normalized': hausdorff_dist / grid_diagonal,
        'length_diff': length_diff,
        'length_ratio': length_ratio,
        'start_match': start_match,
        'end_match': end_match,
        'path1_length': len(path1),
        'path2_length': len(path2)
    }


def standard_astar(cost_surface, start_idx, end_idx, shape):
    """Standard A* implementation"""
    height, width = shape
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    came_from = {}
    g_score = np.full(cost_surface.size, np.inf)
    g_score[start_idx] = 0
    
    closed_set = set()
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == end_idx:
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            return path
        
        closed_set.add(current)
        row_current, col_current = np.unravel_index(current, shape)
        
        for dy, dx in neighbors_offsets:
            row_neighbor = row_current + dy
            col_neighbor = col_current + dx
            
            if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                neighbor = row_neighbor * width + col_neighbor
                
                if neighbor in closed_set or cost_surface[row_neighbor, col_neighbor] >= 9999:
                    continue
                
                distance = sqrt(dy**2 + dx**2)
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                movement_cost = terrain_cost * distance
                tentative_g = g_score[current] + movement_cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    end_row, end_col = np.unravel_index(end_idx, shape)
                    h = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    f = tentative_g + h
                    
                    heapq.heappush(open_set, (f, neighbor))
    
    return None


def optimized_with_dynamic_weights(cost_surface, start_idx, end_idx, shape):
    """A* with dynamic weight adjustment (affects path quality)"""
    height, width = shape
    open_set = []
    tie_breaker = 0
    heapq.heappush(open_set, (0, 0, tie_breaker, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}
    closed_set = set()
    
    end_row, end_col = np.unravel_index(end_idx, shape)
    start_row, start_col = np.unravel_index(start_idx, shape)
    start_h = sqrt((end_row - start_row)**2 + (end_col - start_col)**2)
    
    best_h_score = start_h
    
    neighbors_data = [
        (-1, -1, sqrt(2)), (-1, 0, 1.0), (-1, 1, sqrt(2)),
        (0, -1, 1.0),                     (0, 1, 1.0),
        (1, -1, sqrt(2)),  (1, 0, 1.0),  (1, 1, sqrt(2))
    ]
    
    while open_set:
        current_f, neg_g, _, current = heapq.heappop(open_set)
        current_g = -neg_g
        
        if current == end_idx:
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            return path
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        row_current, col_current = np.unravel_index(current, shape)
        
        current_h = sqrt((end_row - row_current)**2 + (end_col - col_current)**2)
        if current_h < best_h_score:
            best_h_score = current_h
        
        for dy, dx, base_dist in neighbors_data:
            row_neighbor = row_current + dy
            col_neighbor = col_current + dx
            
            if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                neighbor = row_neighbor * width + col_neighbor
                
                if neighbor in closed_set:
                    continue
                
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                if terrain_cost >= 9999:
                    continue
                
                movement_cost = terrain_cost * base_dist
                tentative_g = current_g + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    
                    h = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    
                    # Dynamic weight - this is what affects path quality
                    progress = 1.0 - (h / start_h) if start_h > 0 else 0
                    weight = 2.0 - progress
                    
                    f = tentative_g + h * weight
                    tie_breaker += 1
                    heapq.heappush(open_set, (f, -tentative_g, tie_breaker, neighbor))
    
    return None


def safe_optimized(cost_surface, start_idx, end_idx, shape):
    """A* with only safe optimizations"""
    height, width = shape
    open_set = []
    tie_breaker = 0
    heapq.heappush(open_set, (0, 0, tie_breaker, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}
    closed_set = set()
    h_cache = {}
    
    end_row, end_col = np.unravel_index(end_idx, shape)
    
    neighbors_data = [
        (-1, -1, sqrt(2)), (-1, 0, 1.0), (-1, 1, sqrt(2)),
        (0, -1, 1.0),                     (0, 1, 1.0),
        (1, -1, sqrt(2)),  (1, 0, 1.0),  (1, 1, sqrt(2))
    ]
    
    while open_set:
        current_f, neg_g, _, current = heapq.heappop(open_set)
        current_g = -neg_g
        
        if current == end_idx:
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            return path
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        row_current, col_current = np.unravel_index(current, shape)
        
        for dy, dx, base_dist in neighbors_data:
            row_neighbor = row_current + dy
            col_neighbor = col_current + dx
            
            if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                neighbor = row_neighbor * width + col_neighbor
                
                if neighbor in closed_set:
                    continue
                
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                if terrain_cost >= 9999:
                    continue
                
                movement_cost = terrain_cost * base_dist
                tentative_g = current_g + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    
                    # Cached heuristic
                    if neighbor not in h_cache:
                        h_cache[neighbor] = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    h = h_cache[neighbor]
                    
                    # Standard weight (no dynamic adjustment!)
                    f = tentative_g + h
                    tie_breaker += 1
                    heapq.heappush(open_set, (f, -tentative_g, tie_breaker, neighbor))
    
    return None


def run_frechet_analysis():
    """Test path similarity using Fréchet distance and other metrics"""
    print("PATH SIMILARITY ANALYSIS WITH FRÉCHET DISTANCE")
    print("="*80)
    print("Comparing paths using multiple metrics including Discrete Fréchet Distance")
    print("="*80)
    
    # Create test grid
    np.random.seed(42)
    size = 100
    cost_surface = np.random.random((size, size)) * 5 + 1
    obstacles = np.random.random((size, size)) > 0.9
    cost_surface[obstacles] = 9999
    
    # Test cases
    test_cases = [
        (0, size * size - 1, "Diagonal traverse"),
        (size // 2, size * size - size // 2, "Center paths"),
        (0, size - 1, "Top edge"),
        (size // 4 * size, 3 * size // 4 * size, "Quarter points")
    ]
    
    # Algorithms to compare
    algorithms = [
        ("Standard A*", standard_astar),
        ("Safe Optimized", safe_optimized),
        ("Dynamic Weights", optimized_with_dynamic_weights)
    ]
    
    results = []
    
    for start_idx, end_idx, test_name in test_cases:
        print(f"\nTest: {test_name}")
        print("-" * 70)
        
        # Get reference path
        ref_path = standard_astar(cost_surface, start_idx, end_idx, (size, size))
        
        if not ref_path:
            print("  No path found by standard A*")
            continue
        
        print(f"  Reference path length: {len(ref_path)} cells")
        
        # Compare other algorithms
        for algo_name, algo_func in algorithms[1:]:  # Skip standard A* (comparing with itself)
            path = algo_func(cost_surface, start_idx, end_idx, (size, size))
            
            if path:
                metrics = calculate_comprehensive_path_metrics(ref_path, path, (size, size))
                
                print(f"\n  {algo_name}:")
                print(f"    Path length: {metrics['path2_length']} cells (diff: {metrics['length_diff']})")
                print(f"    Fréchet distance: {metrics['frechet_distance']:.2f} cells")
                print(f"    Hausdorff distance: {metrics['hausdorff_distance']:.2f} cells")
                print(f"    Is identical: {'✅ YES' if metrics['is_identical'] else '❌ NO'}")
                print(f"    Is same path (Fréchet < 2): {'✅ YES' if metrics['is_same_path'] else '❌ NO'}")
                
                results.append({
                    'test': test_name,
                    'algorithm': algo_name,
                    'metrics': metrics
                })
            else:
                print(f"\n  {algo_name}: No path found")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY - Path Quality by Algorithm")
    print("="*80)
    
    for algo_name, _ in algorithms[1:]:
        algo_results = [r for r in results if r['algorithm'] == algo_name]
        if algo_results:
            avg_frechet = sum(r['metrics']['frechet_distance'] for r in algo_results) / len(algo_results)
            identical_count = sum(1 for r in algo_results if r['metrics']['is_identical'])
            same_path_count = sum(1 for r in algo_results if r['metrics']['is_same_path'])
            
            print(f"\n{algo_name}:")
            print(f"  Average Fréchet distance: {avg_frechet:.2f} cells")
            print(f"  Identical paths: {identical_count}/{len(algo_results)}")
            print(f"  Same paths (Fréchet < 2): {same_path_count}/{len(algo_results)}")
    
    print(f"\n{'='*80}")
    print("Metric Interpretation:")
    print("  - Fréchet Distance: Measures path similarity considering order")
    print("    • < 1 cell: Nearly identical paths")
    print("    • < 2 cells: Same path with minor variations")
    print("    • > 5 cells: Significantly different paths")
    print("  - Hausdorff Distance: Maximum deviation at any point")
    print("  - Is Identical: Exact same sequence of cells")
    print("="*80)


if __name__ == "__main__":
    run_frechet_analysis()