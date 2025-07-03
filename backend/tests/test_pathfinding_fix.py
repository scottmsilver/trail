#!/usr/bin/env python3
"""
Test a fix for early termination in pathfinding.
"""

import numpy as np
import heapq
import time
from math import sqrt


def astar_with_proper_early_termination(cost_surface, indices, start_idx, end_idx, 
                                        optimization_config=None):
    """
    A* pathfinding with PROPER early termination that actually works.
    """
    
    if optimization_config is None:
        optimization_config = {}
    
    use_early_termination = optimization_config.get('early_termination', True)
    stagnation_limit = optimization_config.get('stagnation_limit', 10000)
    
    height, width = cost_surface.shape
    
    # Convert indices
    start_row, start_col = np.unravel_index(start_idx, (height, width))
    end_row, end_col = np.unravel_index(end_idx, (height, width))
    
    # A* setup
    open_set = []
    g_score = np.full(cost_surface.shape, np.inf)
    g_score[start_row, start_col] = 0
    
    # Euclidean heuristic
    def heuristic(row, col):
        return sqrt((row - end_row) ** 2 + (col - end_col) ** 2)
    
    f_score = g_score[start_row, start_col] + heuristic(start_row, start_col)
    heapq.heappush(open_set, (f_score, 0, start_idx))
    
    came_from = {}
    step_count = 0
    best_h = float('inf')
    stagnation_counter = 0
    
    # Neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    sqrt2 = sqrt(2)
    
    while open_set:
        current_f, _, current_idx = heapq.heappop(open_set)
        row_current, col_current = np.unravel_index(current_idx, (height, width))
        
        # Found goal
        if current_idx == end_idx:
            # Reconstruct path
            path = []
            current = end_idx
            while current in came_from:
                row, col = np.unravel_index(current, (height, width))
                path.append((col_current, row_current))  # (x, y) format
                current = came_from[current]
            path.append((start_col, start_row))
            return path[::-1]
        
        # Early termination check
        if use_early_termination:
            current_h = heuristic(row_current, col_current)
            if current_h < best_h:
                best_h = current_h
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
                if stagnation_counter > stagnation_limit:
                    print(f"Early termination triggered after {step_count} steps")
                    # Just return None - no path found
                    return None
        
        # Explore neighbors
        for dr, dc in neighbors:
            row_neighbor = row_current + dr
            col_neighbor = col_current + dc
            
            if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                cost = cost_surface[row_neighbor, col_neighbor]
                if cost >= 1000:  # Impassable
                    continue
                
                # Distance cost
                distance = sqrt2 if dr != 0 and dc != 0 else 1.0
                tentative_g = g_score[row_current, col_current] + cost * distance
                
                if tentative_g < g_score[row_neighbor, col_neighbor]:
                    g_score[row_neighbor, col_neighbor] = tentative_g
                    f = tentative_g + heuristic(row_neighbor, col_neighbor)
                    neighbor_idx = row_neighbor * width + col_neighbor
                    heapq.heappush(open_set, (f, step_count, neighbor_idx))
                    came_from[neighbor_idx] = current_idx
        
        step_count += 1
    
    return None  # No path found


def test_early_termination():
    """Test the impact of early termination"""
    
    # Create a simple test grid
    size = 100
    cost_surface = np.ones((size, size))
    indices = np.arange(size * size).reshape(size, size)
    
    # Start and end
    start_idx = 0  # Top-left
    end_idx = size * size - 1  # Bottom-right
    
    configs = [
        ("No early termination", {'early_termination': False}),
        ("Early termination (10k)", {'early_termination': True, 'stagnation_limit': 10000}),
        ("Early termination (1k)", {'early_termination': True, 'stagnation_limit': 1000}),
        ("Early termination (100)", {'early_termination': True, 'stagnation_limit': 100}),
    ]
    
    print("Testing Early Termination on {}x{} grid".format(size, size))
    print("="*50)
    
    for name, config in configs:
        start_time = time.time()
        path = astar_with_proper_early_termination(cost_surface, indices, start_idx, end_idx, config)
        elapsed = time.time() - start_time
        
        print(f"\n{name}:")
        print(f"  Time: {elapsed:.3f}s")
        if path:
            print(f"  Path found: {len(path)} points")
        else:
            print(f"  Path: NOT FOUND (early termination)")


if __name__ == "__main__":
    test_early_termination()