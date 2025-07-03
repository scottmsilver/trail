#!/usr/bin/env python3
"""
Test combinations of safe optimizations that preserve path quality
"""

import numpy as np
from math import sqrt
import time
import heapq


def standard_astar(cost_surface, start_idx, end_idx, shape):
    """Reference implementation"""
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


def safe_optimized_astar(cost_surface, start_idx, end_idx, shape):
    """A* with all SAFE optimizations that preserve path quality"""
    height, width = shape
    
    # 1. Preprocessing: Build neighbor cache
    preprocess_start = time.time()
    neighbor_cache = {}
    neighbor_offsets = [(-1, -1, sqrt(2)), (-1, 0, 1.0), (-1, 1, sqrt(2)),
                       (0, -1, 1.0),                    (0, 1, 1.0),
                       (1, -1, sqrt(2)),  (1, 0, 1.0),  (1, 1, sqrt(2))]
    
    for row in range(height):
        for col in range(width):
            if cost_surface[row, col] < 9999:
                idx = row * width + col
                neighbors = []
                for dy, dx, dist in neighbor_offsets:
                    nr, nc = row + dy, col + dx
                    if 0 <= nr < height and 0 <= nc < width and cost_surface[nr, nc] < 9999:
                        nidx = nr * width + nc
                        neighbors.append((nidx, nr, nc, dist))
                if neighbors:
                    neighbor_cache[idx] = neighbors
    
    preprocess_time = time.time() - preprocess_start
    
    # 2. A* with safe optimizations
    pathfind_start = time.time()
    
    # Use better data structures
    open_set = []
    tie_breaker = 0  # For consistent ordering
    heapq.heappush(open_set, (0, 0, tie_breaker, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}  # Dictionary instead of array
    closed_set = set()
    
    # Heuristic cache
    h_cache = {}
    end_row, end_col = np.unravel_index(end_idx, shape)
    
    iterations = 0
    
    while open_set:
        current_f, neg_g, _, current = heapq.heappop(open_set)
        iterations += 1
        
        if current == end_idx:
            # Reconstruct path
            path = []
            while current in came_from:
                row, col = np.unravel_index(current, shape)
                path.append((row, col))
                current = came_from[current]
            row, col = np.unravel_index(current, shape)
            path.append((row, col))
            path.reverse()
            
            total_time = (time.time() - pathfind_start) + preprocess_time
            return path, total_time, iterations
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        current_g = -neg_g
        
        # Use preprocessed neighbors
        if current in neighbor_cache:
            for neighbor, row_neighbor, col_neighbor, distance in neighbor_cache[current]:
                if neighbor in closed_set:
                    continue
                
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                movement_cost = terrain_cost * distance
                tentative_g = current_g + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Use cached heuristic
                    if neighbor not in h_cache:
                        h_cache[neighbor] = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    h = h_cache[neighbor]
                    
                    # Standard A* scoring (no dynamic weights!)
                    f = tentative_g + h
                    
                    tie_breaker += 1
                    heapq.heappush(open_set, (f, -tentative_g, tie_breaker, neighbor))
    
    total_time = (time.time() - pathfind_start) + preprocess_time
    return None, total_time, iterations


def test_safe_optimizations():
    """Compare standard vs safely optimized A*"""
    print("SAFE OPTIMIZATION PERFORMANCE TEST")
    print("="*80)
    print("Testing optimizations that guarantee identical paths")
    print("="*80)
    
    test_configs = [
        ("Small Grid (50×50)", 50),
        ("Medium Grid (100×100)", 100),
        ("Large Grid (200×200)", 200),
        ("Very Large Grid (500×500)", 500),
    ]
    
    for grid_name, size in test_configs:
        print(f"\n{grid_name} Test:")
        print("-" * 60)
        
        # Create test grid
        np.random.seed(42)
        cost_surface = np.random.random((size, size)) * 5 + 1
        obstacles = np.random.random((size, size)) > 0.9
        cost_surface[obstacles] = 9999
        
        # Test different paths
        test_paths = [
            (0, size * size - 1, "Diagonal"),
            (0, size - 1, "Top edge"),
            (size // 2 * size + size // 4, size // 2 * size + 3 * size // 4, "Center")
        ]
        
        total_speedup = 0
        path_count = 0
        
        for start_idx, end_idx, path_type in test_paths:
            # Standard A*
            path1, time1, iter1 = standard_astar(cost_surface, start_idx, end_idx, (size, size))
            
            # Safe optimized A*
            path2, time2, iter2 = safe_optimized_astar(cost_surface, start_idx, end_idx, (size, size))
            
            if path1 and path2:
                # Verify paths are identical
                identical = len(path1) == len(path2) and all(p1 == p2 for p1, p2 in zip(path1, path2))
                
                speedup = (time1 - time2) / time1 * 100 if time1 > 0 else 0
                total_speedup += speedup
                path_count += 1
                
                print(f"  {path_type}: ", end="")
                if identical:
                    print(f"✅ IDENTICAL PATH, {speedup:.1f}% faster ({time1:.3f}s → {time2:.3f}s)")
                else:
                    print(f"❌ DIFFERENT PATH!")
            else:
                print(f"  {path_type}: Path not found")
        
        if path_count > 0:
            avg_speedup = total_speedup / path_count
            print(f"  Average speedup: {avg_speedup:.1f}%")
    
    print(f"\n{'='*80}")
    print("SAFE OPTIMIZATIONS APPLIED:")
    print("  ✅ Memory-efficient storage (dictionaries vs arrays)")
    print("  ✅ Heuristic caching")
    print("  ✅ Neighbor preprocessing")
    print("  ✅ Better priority queue structure")
    print("  ❌ Dynamic weights (NOT used - affects path quality)")
    print("  ❌ Early termination (NOT used - can miss optimal path)")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_safe_optimizations()