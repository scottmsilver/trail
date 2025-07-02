import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from app.services.dem_tile_cache import DEMTileCache
from pyproj import Transformer
from rasterio.transform import from_origin


def visualize_pathfinding_scenario(scenario_name, cost_surface, start_pos, end_pos, path_cells, save=True):
    """Visualize a pathfinding scenario with cost surface and path"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot cost surface
    im = ax.imshow(cost_surface, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax, label='Cost')
    
    # Plot start and end
    ax.plot(start_pos[1], start_pos[0], 'go', markersize=15, label='Start')
    ax.plot(end_pos[1], end_pos[0], 'ro', markersize=15, label='End')
    
    # Plot path
    if path_cells:
        path_rows = [p[0] for p in path_cells]
        path_cols = [p[1] for p in path_cells]
        ax.plot(path_cols, path_rows, 'w-', linewidth=3, label='Path', alpha=0.8)
        ax.plot(path_cols, path_rows, 'wo', markersize=8)
    
    ax.set_title(f'Pathfinding Test: {scenario_name}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(f'test_path_{scenario_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


def test_and_visualize_scenarios():
    """Run and visualize different pathfinding scenarios"""
    
    # Setup
    pathfinder = DEMTileCache()
    
    # Scenario 1: Simple obstacle avoidance
    print("\n1. Testing obstacle avoidance with visualization...")
    rows, cols = 20, 20
    cost_surface = np.ones((rows, cols))
    
    # Add obstacles
    cost_surface[8:12, 5:15] = 100.0  # Horizontal wall
    cost_surface[5:15, 10] = 100.0    # Vertical wall with gap
    
    # Create transform
    transform = from_origin(-111.5, 40.5, 30, 30)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
    indices = np.arange(rows * cols).reshape(rows, cols)
    
    # Find path
    start_idx = indices[5, 2]
    end_idx = indices[15, 17]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    # Convert path to grid coordinates
    path_cells = []
    if path:
        for lon, lat in path:
            x, y = transformer.transform(lon, lat)
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            if 0 <= row < rows and 0 <= col < cols:
                path_cells.append((row, col))
    
    visualize_pathfinding_scenario(
        "Obstacle Avoidance",
        cost_surface,
        (5, 2),
        (15, 17),
        path_cells
    )
    print(f"Path found with {len(path_cells)} cells")
    
    # Scenario 2: Gradient terrain
    print("\n2. Testing gradient terrain...")
    cost_surface = np.ones((rows, cols))
    
    # Create gradient from left to right
    for j in range(cols):
        cost_surface[:, j] = 1.0 + j * 0.2  # Increasing cost to the right
    
    # Add a low-cost corridor
    cost_surface[10, :] = 0.5  # Cheap path through middle
    
    start_idx = indices[5, 0]
    end_idx = indices[15, 19]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    path_cells = []
    if path:
        for lon, lat in path:
            x, y = transformer.transform(lon, lat)
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            if 0 <= row < rows and 0 <= col < cols:
                path_cells.append((row, col))
    
    visualize_pathfinding_scenario(
        "Gradient with Corridor",
        cost_surface,
        (5, 0),
        (15, 19),
        path_cells
    )
    print(f"Path found with {len(path_cells)} cells")
    
    # Scenario 3: Multiple paths
    print("\n3. Testing multiple path options...")
    cost_surface = np.ones((rows, cols)) * 2.0
    
    # Create two possible paths
    cost_surface[5, 2:18] = 1.0   # Upper path
    cost_surface[15, 2:18] = 1.0  # Lower path
    cost_surface[5:16, 10] = 1.5  # Connection in middle
    
    # Make upper path slightly cheaper
    cost_surface[5, 2:10] = 0.8
    
    start_idx = indices[10, 2]
    end_idx = indices[10, 17]
    
    path = pathfinder.astar_pathfinding(
        cost_surface, indices, start_idx, end_idx,
        transform, transformer
    )
    
    path_cells = []
    if path:
        for lon, lat in path:
            x, y = transformer.transform(lon, lat)
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            if 0 <= row < rows and 0 <= col < cols:
                path_cells.append((row, col))
    
    visualize_pathfinding_scenario(
        "Multiple Paths",
        cost_surface,
        (10, 2),
        (10, 17),
        path_cells
    )
    print(f"Path found with {len(path_cells)} cells")
    
    # Print summary of path characteristics
    if path_cells:
        rows_used = set(p[0] for p in path_cells)
        print(f"Path uses rows: {sorted(rows_used)}")
        
        # Check if it used the cheaper upper path
        upper_path_cells = sum(1 for r, c in path_cells if r == 5)
        print(f"Cells in upper (cheaper) path: {upper_path_cells}")


if __name__ == "__main__":
    print("Running pathfinding visualization tests...")
    test_and_visualize_scenarios()
    print("\nVisualization tests complete! Check the generated PNG files.")