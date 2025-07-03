"""
Optimized A* pathfinding implementation with performance improvements
"""

import numpy as np
import heapq
from math import sqrt
import time


def astar_pathfinding_optimized(self, cost_surface, indices, start_idx, end_idx, out_trans, transformer, dem=None,
                               optimization_config=None):
    """
    Optimized A* pathfinding with performance improvements.
    
    Optimizations included:
    1. Conservative early termination
    2. Dynamic weight adjustment
    3. Memory limiting
    4. Improved tie-breaking
    5. Corner cutting (line-of-sight)
    
    optimization_config: dict with optional parameters:
        - early_termination: bool (default True)
        - stagnation_limit: int (default 5000)
        - dynamic_weights: bool (default True)
        - memory_limit: int (default 50000 nodes)
        - corner_cutting: bool (default True)
        - max_iterations: int (default 10000000)
    """
    
    # Default optimization settings
    if optimization_config is None:
        optimization_config = {}
    
    use_early_termination = optimization_config.get('early_termination', True)
    stagnation_limit = optimization_config.get('stagnation_limit', 5000)
    use_dynamic_weights = optimization_config.get('dynamic_weights', True)
    memory_limit = optimization_config.get('memory_limit', 50000)
    use_corner_cutting = optimization_config.get('corner_cutting', True)
    max_iterations = optimization_config.get('max_iterations', 10000000)
    
    # Performance tracking
    start_time = time.time()
    
    height, width = cost_surface.shape
    open_set = []
    
    # Priority queue with tie-breaking: (f_score, -g_score, tie_breaker, node_idx)
    # Using negative g_score to prefer nodes closer to goal when f_scores are equal
    tie_breaker = 0
    heapq.heappush(open_set, (0, 0, tie_breaker, start_idx))
    
    came_from = {}
    g_score = {}  # Use dict instead of full array for memory efficiency
    g_score[start_idx] = 0
    
    closed_set = set()
    
    # Calculate initial heuristic for dynamic weighting
    start_h = heuristic_optimized(start_idx, end_idx, indices.shape, out_trans)
    
    # Early termination tracking
    best_h_score = float('inf')
    stagnation_counter = 0
    best_node = start_idx
    
    # Neighbor offsets
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),         (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
    
    # Debug info
    if self.debug_mode:
        self.debug_data = {
            'explored_nodes': [],
            'optimization_stats': {
                'early_terminations': 0,
                'memory_prunes': 0,
                'corner_cuts': 0,
                'dynamic_weight_changes': []
            }
        }
    
    step_count = 0
    
    # Calculate expected search area
    start_row, start_col = np.unravel_index(start_idx, indices.shape)
    end_row, end_col = np.unravel_index(end_idx, indices.shape)
    grid_distance = abs(end_row - start_row) + abs(end_col - start_col)
    
    if grid_distance > 1000:
        print(f"Large search area: {grid_distance} cells. Optimizations enabled.")
    
    while open_set and step_count < max_iterations:
        # Memory limiting - prune open set if too large
        if use_early_termination and len(open_set) > memory_limit:
            # Keep only the best half
            temp = []
            for _ in range(memory_limit // 2):
                if open_set:
                    temp.append(heapq.heappop(open_set))
            open_set = temp
            heapq.heapify(open_set)
            
            if self.debug_mode:
                self.debug_data['optimization_stats']['memory_prunes'] += 1
        
        current_f, neg_g, _, current = heapq.heappop(open_set)
        current_g = -neg_g
        step_count += 1
        
        # Goal check
        if current == end_idx:
            path = reconstruct_path_optimized(came_from, current, out_trans, transformer, indices.shape)
            
            if self.debug_mode:
                elapsed = time.time() - start_time
                print(f"Path found in {elapsed:.3f}s, {step_count} iterations")
                print(f"Optimizations: {self.debug_data['optimization_stats']}")
            
            return path
        
        # Skip if already processed
        if current in closed_set:
            continue
            
        closed_set.add(current)
        row_current, col_current = np.unravel_index(current, indices.shape)
        
        # Early termination check
        if use_early_termination:
            current_h = heuristic_optimized(current, end_idx, indices.shape, out_trans)
            
            if current_h < best_h_score:
                best_h_score = current_h
                best_node = current
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
                if stagnation_counter > stagnation_limit:
                    if self.debug_mode:
                        self.debug_data['optimization_stats']['early_terminations'] += 1
                        elapsed = time.time() - start_time
                        print(f"Early termination after {step_count} iterations, {elapsed:.3f}s")
                    # Try to return best path found so far
                    if best_node != current and best_node in came_from:
                        # Continue from best node instead of giving up completely
                        current = best_node
                        stagnation_counter = 0
                    else:
                        break
        
        # Corner cutting check (line of sight to goal)
        if use_corner_cutting and step_count % 100 == 0:  # Check periodically
            if has_line_of_sight(row_current, col_current, end_row, end_col, cost_surface):
                # Direct path possible!
                if self.debug_mode:
                    self.debug_data['optimization_stats']['corner_cuts'] += 1
                
                # Create direct path
                came_from[end_idx] = current
                path = reconstruct_path_optimized(came_from, end_idx, out_trans, transformer, indices.shape)
                return path
        
        # Calculate dynamic weight if enabled
        if use_dynamic_weights:
            # Progress = how close we are to goal (0 to 1)
            progress = 1.0 - (current_h / start_h) if start_h > 0 else 0
            # Start aggressive (weight=2.0), end conservative (weight=1.0)
            heuristic_weight = 2.0 - progress
            
            if self.debug_mode and step_count % 1000 == 0:
                self.debug_data['optimization_stats']['dynamic_weight_changes'].append({
                    'step': step_count,
                    'weight': heuristic_weight,
                    'progress': progress
                })
        else:
            heuristic_weight = 1.0
        
        # Explore neighbors
        for dy, dx in neighbors_offsets:
            row_neighbor = row_current + dy
            col_neighbor = col_current + dx
            
            if 0 <= row_neighbor < height and 0 <= col_neighbor < width:
                neighbor = indices[row_neighbor, col_neighbor]
                
                if neighbor in closed_set:
                    continue
                
                # Skip impassable terrain
                terrain_cost = cost_surface[row_neighbor, col_neighbor]
                if terrain_cost >= 9999:  # Obstacle
                    continue
                
                # Calculate movement cost
                distance = sqrt((dy * out_trans.e) ** 2 + (dx * out_trans.a) ** 2)
                movement_cost = terrain_cost * distance
                tentative_g = current_g + movement_cost
                
                # Skip if not an improvement
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue
                
                # Update scores
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                
                # Calculate f-score with dynamic weight
                h_score = heuristic_optimized(neighbor, end_idx, indices.shape, out_trans)
                f_score = tentative_g + h_score * heuristic_weight
                
                # Add to open set with tie-breaking
                tie_breaker += 1
                heapq.heappush(open_set, (f_score, -tentative_g, tie_breaker, neighbor))
    
    # No path found
    if self.debug_mode:
        elapsed = time.time() - start_time
        print(f"No path found after {step_count} iterations, {elapsed:.3f}s")
        if use_early_termination:
            print(f"Best progress: {best_h_score:.1f}m from goal")
    
    return None


def heuristic_optimized(node_idx, end_idx, shape, out_trans):
    """
    Optimized heuristic function with better terrain awareness.
    """
    row_node, col_node = np.unravel_index(node_idx, shape)
    row_end, col_end = np.unravel_index(end_idx, shape)
    
    dx = (col_node - col_end) * out_trans.a
    dy = (row_node - row_end) * out_trans.e
    
    # Euclidean distance
    distance = sqrt(dx**2 + dy**2)
    
    # Use a conservative multiplier to avoid overly optimistic estimates
    # This helps the algorithm make better decisions
    return distance * 0.9


def reconstruct_path_optimized(came_from, current, out_trans, transformer, shape):
    """
    Optimized path reconstruction.
    """
    path = []
    
    while current in came_from:
        row, col = np.unravel_index(current, shape)
        x = out_trans.c + col * out_trans.a + out_trans.a / 2
        y = out_trans.f + row * out_trans.e + out_trans.e / 2
        x_lon, y_lat = transformer.transform(x, y, direction='INVERSE')
        path.append((x_lon, y_lat))
        current = came_from[current]
    
    # Add starting point
    row, col = np.unravel_index(current, shape)
    x = out_trans.c + col * out_trans.a + out_trans.a / 2
    y = out_trans.f + row * out_trans.e + out_trans.e / 2
    x_lon, y_lat = transformer.transform(x, y, direction='INVERSE')
    path.append((x_lon, y_lat))
    
    path.reverse()
    return path


def has_line_of_sight(row1, col1, row2, col2, cost_surface, max_cost=1000):
    """
    Check if there's a clear line of sight between two points.
    Uses Bresenham's line algorithm.
    """
    # Get all points along the line
    points = list(bresenham_line(row1, col1, row2, col2))
    
    # Check if any point has high cost (obstacle)
    for row, col in points:
        if 0 <= row < cost_surface.shape[0] and 0 <= col < cost_surface.shape[1]:
            if cost_surface[row, col] >= max_cost:
                return False
        else:
            return False  # Out of bounds
    
    return True


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm to get all points along a line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    
    if dx > dy:
        error = dx / 2
        while x != x1:
            points.append((x, y))
            error -= dy
            if error < 0:
                y += y_inc
                error += dx
            x += x_inc
    else:
        error = dy / 2
        while y != y1:
            points.append((x, y))
            error -= dx
            if error < 0:
                x += x_inc
                error += dy
            y += y_inc
    
    points.append((x, y))
    return points