#!/usr/bin/env python3
"""
Test path similarity between standard and optimized A* algorithms
"""

import numpy as np
from math import sqrt
from typing import List, Tuple
import time
import heapq


def calculate_path_similarity(path1: List[Tuple[int, int]], path2: List[Tuple[int, int]], grid_shape: Tuple[int, int]) -> dict:
    """
    Calculate similarity metrics between two paths.
    
    Returns:
        - hausdorff_distance: Maximum distance from any point in path1 to nearest point in path2
        - frechet_distance: Measures similarity of curves considering order
        - length_ratio: Ratio of path lengths
        - overlap_ratio: Percentage of shared cells
    """
    if not path1 or not path2:
        return {
            'hausdorff_distance': float('inf'),
            'frechet_distance': float('inf'),
            'length_ratio': 0,
            'overlap_ratio': 0,
            'similarity_score': 0
        }
    
    # Convert to sets for overlap calculation
    path1_set = set(path1)
    path2_set = set(path2)
    
    # Calculate overlap
    overlap = len(path1_set & path2_set)
    overlap_ratio = overlap / max(len(path1_set), len(path2_set))
    
    # Calculate Hausdorff distance (simplified)
    max_dist = 0
    for p1 in path1:
        min_dist = float('inf')
        for p2 in path2:
            dist = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist = min(min_dist, dist)
        max_dist = max(max_dist, min_dist)
    
    hausdorff_distance = max_dist
    
    # Length ratio
    length_ratio = min(len(path1), len(path2)) / max(len(path1), len(path2))
    
    # Calculate similarity score (0-1, where 1 is identical)
    # Normalize hausdorff distance by diagonal of grid
    grid_diagonal = sqrt(grid_shape[0]**2 + grid_shape[1]**2)
    normalized_hausdorff = min(hausdorff_distance / grid_diagonal, 1.0)
    
    # Combined similarity score
    similarity_score = (
        overlap_ratio * 0.4 +  # 40% weight on overlap
        length_ratio * 0.3 +   # 30% weight on length similarity
        (1 - normalized_hausdorff) * 0.3  # 30% weight on distance
    )
    
    return {
        'hausdorff_distance': hausdorff_distance,
        'length_ratio': length_ratio,
        'overlap_ratio': overlap_ratio,
        'similarity_score': similarity_score,
        'path1_length': len(path1),
        'path2_length': len(path2),
        'shared_cells': overlap
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
            # Reconstruct path
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
                    
                    # Heuristic
                    end_row, end_col = np.unravel_index(end_idx, shape)
                    h = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    f = tentative_g + h
                    
                    heapq.heappush(open_set, (f, neighbor))
    
    return None


def optimized_astar(cost_surface, start_idx, end_idx, shape):
    """Optimized A* with all improvements"""
    height, width = shape
    open_set = []
    tie_breaker = 0
    heapq.heappush(open_set, (0, 0, tie_breaker, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}
    closed_set = set()
    
    end_row, end_col = np.unravel_index(end_idx, shape)
    best_h_score = float('inf')
    stagnation_counter = 0
    h_cache = {}
    
    neighbors_data = [
        (-1, -1, sqrt(2)), (-1, 0, 1.0), (-1, 1, sqrt(2)),
        (0, -1, 1.0),                     (0, 1, 1.0),
        (1, -1, sqrt(2)),  (1, 0, 1.0),  (1, 1, sqrt(2))
    ]
    
    while open_set and stagnation_counter < 5000:
        current_f, neg_g, _, current = heapq.heappop(open_set)
        current_g = -neg_g
        
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
            return path
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        row_current, col_current = np.unravel_index(current, shape)
        
        # Early termination check
        if current not in h_cache:
            h_cache[current] = sqrt((end_row - row_current)**2 + (end_col - col_current)**2)
        current_h = h_cache[current]
        
        if current_h < best_h_score:
            best_h_score = current_h
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
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
                    
                    if neighbor not in h_cache:
                        h_cache[neighbor] = sqrt((end_row - row_neighbor)**2 + (end_col - col_neighbor)**2)
                    h = h_cache[neighbor]
                    
                    # Dynamic weight
                    progress = 1.0 - (h / best_h_score) if best_h_score > 0 else 0
                    weight = 2.0 - progress
                    
                    f = tentative_g + h * weight
                    tie_breaker += 1
                    heapq.heappush(open_set, (f, -tentative_g, tie_breaker, neighbor))
    
    return None


def run_similarity_tests():
    """Test path similarity between algorithms"""
    print("PATH SIMILARITY ANALYSIS")
    print("="*80)
    print("Comparing paths generated by Standard A* vs Optimized A*")
    print("="*80)
    
    test_configs = [
        ("Small Grid (50×50)", 50, 0.9),
        ("Medium Grid (100×100)", 100, 0.9),
        ("Large Grid (200×200)", 200, 0.9),
        ("Dense Obstacles (100×100)", 100, 0.8),
        ("Sparse Obstacles (100×100)", 100, 0.95),
    ]
    
    results = []
    
    for name, size, passable_ratio in test_configs:
        print(f"\n{name}:")
        print("-" * 60)
        
        # Create test grid
        np.random.seed(42)
        cost_surface = np.random.random((size, size)) * 5 + 1
        
        # Add obstacles
        obstacles = np.random.random((size, size)) > passable_ratio
        cost_surface[obstacles] = 9999
        
        # Multiple test paths
        test_paths = [
            (0, size * size - 1),  # Corner to corner
            (size // 2, size * size - size // 2),  # Middle to middle
            (0, size - 1),  # Top-left to top-right
        ]
        
        path_names = ["Corner-to-corner", "Center-to-center", "Edge-to-edge"]
        
        for i, (start_idx, end_idx) in enumerate(test_paths):
            print(f"\n  {path_names[i]} path:")
            
            # Get paths from both algorithms
            path1 = standard_astar(cost_surface, start_idx, end_idx, (size, size))
            path2 = optimized_astar(cost_surface, start_idx, end_idx, (size, size))
            
            if path1 and path2:
                # Calculate similarity
                similarity = calculate_path_similarity(path1, path2, (size, size))
                
                print(f"    Standard A* path length: {similarity['path1_length']}")
                print(f"    Optimized A* path length: {similarity['path2_length']}")
                print(f"    Length difference: {abs(similarity['path1_length'] - similarity['path2_length'])} cells")
                print(f"    Shared cells: {similarity['shared_cells']} ({similarity['overlap_ratio']:.1%})")
                print(f"    Hausdorff distance: {similarity['hausdorff_distance']:.2f} cells")
                print(f"    Overall similarity: {similarity['similarity_score']:.1%}")
                
                quality_maintained = similarity['similarity_score'] > 0.8
                print(f"    Quality maintained: {'✓ YES' if quality_maintained else '✗ NO'}")
                
                results.append({
                    'grid': name,
                    'path': path_names[i],
                    'similarity': similarity['similarity_score'],
                    'length_diff': abs(similarity['path1_length'] - similarity['path2_length']),
                    'quality_ok': quality_maintained
                })
            else:
                print(f"    One or both algorithms failed to find a path")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - Path Quality Metrics")
    print(f"{'='*80}")
    
    if results:
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        quality_pass_rate = sum(1 for r in results if r['quality_ok']) / len(results)
        
        print(f"\nOverall Statistics:")
        print(f"  Average similarity score: {avg_similarity:.1%}")
        print(f"  Quality pass rate: {quality_pass_rate:.1%} (>80% similarity)")
        print(f"  Average path length difference: {sum(r['length_diff'] for r in results) / len(results):.1f} cells")
        
        print(f"\nDetailed Results:")
        print(f"{'Grid':<30} {'Path Type':<20} {'Similarity':<15} {'Quality':<10}")
        print(f"{'-'*75}")
        for r in results:
            print(f"{r['grid']:<30} {r['path']:<20} {r['similarity']:.1%}{'':>10} "
                  f"{'✓ PASS' if r['quality_ok'] else '✗ FAIL'}")
    
    print(f"\n{'='*80}")
    print("Path Similarity Metrics Explained:")
    print("  - Similarity Score: Combined metric (0-100%)")
    print("  - Hausdorff Distance: Max deviation between paths")
    print("  - Overlap Ratio: Percentage of shared cells")
    print("  - Length Ratio: How similar the path lengths are")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_similarity_tests()