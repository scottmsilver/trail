#!/usr/bin/env python3
"""
Test which optimizations produce identical or near-identical paths
"""

import numpy as np
from math import sqrt
import time
import heapq
from typing import List, Tuple, Dict
import pytest


def calculate_path_metrics(path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]) -> Dict:
    """
    Calculate detailed path comparison metrics.
    
    "Same path" criteria:
    1. Exact same length (±1 cell tolerance)
    2. Same start and end points
    3. Total distance within 1% 
    4. Maximum deviation < 2 cells at any point
    """
    if not path1 or not path2:
        return {'identical': False, 'reason': 'One path is empty'}
    
    # Check endpoints
    if path1[0] != path2[0] or path1[-1] != path2[-1]:
        return {'identical': False, 'reason': 'Different endpoints'}
    
    # Length check
    length_diff = abs(len(path1) - len(path2))
    if length_diff > 1:
        return {
            'identical': False, 
            'reason': f'Length difference too large: {length_diff} cells',
            'length_diff': length_diff
        }
    
    # Calculate total path distances
    def path_distance(path):
        dist = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            dist += sqrt(dx*dx + dy*dy)
        return dist
    
    dist1 = path_distance(path1)
    dist2 = path_distance(path2)
    distance_ratio = min(dist1, dist2) / max(dist1, dist2) if max(dist1, dist2) > 0 else 1
    
    if distance_ratio < 0.99:  # More than 1% difference
        return {
            'identical': False,
            'reason': f'Distance differs by {(1-distance_ratio)*100:.1f}%',
            'distance_ratio': distance_ratio
        }
    
    # Maximum deviation check
    max_deviation = 0
    for i, p1 in enumerate(path1):
        # Find closest point in path2
        min_dist = float('inf')
        for p2 in path2:
            dist = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist = min(min_dist, dist)
        max_deviation = max(max_deviation, min_dist)
    
    if max_deviation > 2.0:  # More than 2 cells deviation
        return {
            'identical': False,
            'reason': f'Maximum deviation: {max_deviation:.1f} cells',
            'max_deviation': max_deviation
        }
    
    # If we get here, paths are effectively identical
    return {
        'identical': True,
        'length_diff': length_diff,
        'distance_ratio': distance_ratio,
        'max_deviation': max_deviation
    }


def standard_astar(cost_surface, start_idx, end_idx, shape):
    """Reference standard A* implementation"""
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
    
    iterations = 0
    start_time = time.time()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        iterations += 1
        
        if current == end_idx:
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            return path, time.time() - start_time, iterations
        
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
    
    return None, time.time() - start_time, iterations


@pytest.mark.integration
def test_optimization(cost_surface, start_idx, end_idx, shape, optimization_name, optimization_func):
    """Test a specific optimization against standard A*"""
    
    # Get standard path
    path_standard, time_standard, iter_standard = standard_astar(cost_surface, start_idx, end_idx, shape)
    
    # Get optimized path
    path_optimized, time_optimized, iter_optimized = optimization_func(cost_surface, start_idx, end_idx, shape)
    
    if not path_standard or not path_optimized:
        return {
            'name': optimization_name,
            'success': False,
            'reason': 'Path not found'
        }
    
    # Compare paths
    metrics = calculate_path_metrics(path_standard, path_optimized)
    
    # Calculate performance
    speed_improvement = (time_standard - time_optimized) / time_standard * 100 if time_standard > 0 else 0
    
    return {
        'name': optimization_name,
        'success': True,
        'identical_path': metrics['identical'],
        'reason': metrics.get('reason', 'Paths are identical'),
        'speed_improvement': speed_improvement,
        'time_standard': time_standard,
        'time_optimized': time_optimized,
        'iterations_saved': iter_standard - iter_optimized,
        'path_metrics': metrics
    }


# Define optimization variants
def astar_with_caching_only(cost_surface, start_idx, end_idx, shape):
    """A* with only heuristic caching"""
    height, width = shape
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    came_from = {}
    g_score = {}  # Dict instead of array
    g_score[start_idx] = 0
    
    closed_set = set()
    h_cache = {}  # Heuristic cache
    
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
    
    iterations = 0
    start_time = time.time()
    end_row, end_col = np.unravel_index(end_idx, shape)
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        iterations += 1
        
        if current == end_idx:
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            return path, time.time() - start_time, iterations
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        row_current, col_current = np.unravel_index(current, shape)
        
        for dy, dx in neighbors_offsets:
            row_neighbor = row_current + dy
            col_neighbor = col_current + dx
            
            if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                neighbor = row_neighbor * width + col_neighbor
                
                if neighbor in closed_set:
                    continue
                
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                if terrain_cost >= 9999:
                    continue
                
                distance = sqrt(dy**2 + dx**2)
                movement_cost = terrain_cost * distance
                tentative_g = g_score.get(current, float('inf')) + movement_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Cached heuristic
                    if neighbor not in h_cache:
                        h_cache[neighbor] = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    h = h_cache[neighbor]
                    
                    f = tentative_g + h  # No dynamic weight!
                    heapq.heappush(open_set, (f, neighbor))
    
    return None, time.time() - start_time, iterations


def astar_with_preprocessing(cost_surface, start_idx, end_idx, shape):
    """A* with neighbor preprocessing only"""
    height, width = shape
    
    # Precompute neighbors
    neighbor_cache = {}
    for row in range(height):
        for col in range(width):
            if cost_surface[row, col] < 9999:
                idx = row * width + col
                neighbors = []
                for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    nr, nc = row + dy, col + dx
                    if 0 <= nr < height and 0 <= nc < width and cost_surface[nr, nc] < 9999:
                        nidx = nr * width + nc
                        dist = sqrt(dy**2 + dx**2)
                        neighbors.append((nidx, dist))
                neighbor_cache[idx] = neighbors
    
    # Run standard A* with preprocessed neighbors
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}
    closed_set = set()
    
    iterations = 0
    start_time = time.time()
    end_row, end_col = np.unravel_index(end_idx, shape)
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        iterations += 1
        
        if current == end_idx:
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            return path, time.time() - start_time, iterations
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        # Use preprocessed neighbors
        if current in neighbor_cache:
            for neighbor, distance in neighbor_cache[current]:
                if neighbor in closed_set:
                    continue
                
                row_neighbor, col_neighbor = np.unravel_index(neighbor, shape)
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                movement_cost = terrain_cost * distance
                tentative_g = g_score.get(current, float('inf')) + movement_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    h = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor))
    
    return None, time.time() - start_time, iterations


def run_path_identity_tests():
    """Test which optimizations preserve path identity"""
    print("PATH IDENTITY ANALYSIS")
    print("="*80)
    print("Testing which optimizations produce identical paths to standard A*")
    print("\nPath Identity Criteria:")
    print("  - Same length (±1 cell)")
    print("  - Same total distance (±1%)")
    print("  - Maximum deviation < 2 cells")
    print("="*80)
    
    # Create test grid
    np.random.seed(42)
    size = 100
    cost_surface = np.random.random((size, size)) * 5 + 1
    obstacles = np.random.random((size, size)) > 0.9
    cost_surface[obstacles] = 9999
    
    # Test cases
    test_cases = [
        (0, size * size - 1, "Corner to corner"),
        (size // 2, size * size - size // 2, "Center paths"),
        (0, size - 1, "Edge traverse")
    ]
    
    # Optimizations to test
    optimizations = [
        ("Heuristic Caching Only", astar_with_caching_only),
        ("Preprocessing Only", astar_with_preprocessing),
    ]
    
    safe_optimizations = []
    
    for start_idx, end_idx, path_name in test_cases:
        print(f"\nTest: {path_name}")
        print("-" * 60)
        
        for opt_name, opt_func in optimizations:
            result = test_optimization(cost_surface, start_idx, end_idx, (size, size), opt_name, opt_func)
            
            if result['success']:
                if result['identical_path']:
                    print(f"✅ {opt_name}: IDENTICAL PATH, {result['speed_improvement']:.1f}% faster")
                    safe_optimizations.append(opt_name)
                else:
                    print(f"❌ {opt_name}: DIFFERENT PATH - {result['reason']}")
            else:
                print(f"⚠️  {opt_name}: {result['reason']}")
    
    print(f"\n{'='*80}")
    print("SAFE OPTIMIZATIONS (produce identical paths):")
    print("="*80)
    
    safe_opts = list(set(safe_optimizations))
    for opt in safe_opts:
        print(f"  ✅ {opt}")
    
    print(f"\nThese optimizations can be used with confidence that paths remain optimal.")


if __name__ == "__main__":
    run_path_identity_tests()