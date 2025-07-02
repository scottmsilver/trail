import pytest
import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from pyproj import Transformer
from rasterio.transform import from_origin


class TestPathfindingAlgorithm:
    """Test the A* pathfinding algorithm with controlled scenarios"""
    
    def create_test_grid(self, rows, cols, cell_size=30):
        """Create a test grid with known terrain"""
        # Create a simple transform (30m cells)
        transform = from_origin(-111.5, 40.5, cell_size, cell_size)
        
        # Create transformer for lat/lon conversion
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
        
        # Create indices
        indices = np.arange(rows * cols).reshape(rows, cols)
        
        return transform, transformer, indices
    
    def test_straight_path_flat_terrain(self):
        """Test that algorithm finds straight path on flat terrain"""
        # Create 10x10 flat terrain (all costs = 1.0)
        rows, cols = 10, 10
        cost_surface = np.ones((rows, cols))
        transform, transformer, indices = self.create_test_grid(rows, cols)
        
        # Create pathfinder
        pathfinder = DEMTileCache()
        
        # Find path from top-left to bottom-right
        start_idx = indices[0, 0]  # Top-left
        end_idx = indices[9, 9]    # Bottom-right
        
        path = pathfinder.astar_pathfinding(
            cost_surface, indices, start_idx, end_idx, 
            transform, transformer
        )
        
        # Should find a path
        assert path is not None
        assert len(path) > 0
        
        # Path should be approximately diagonal (shortest distance)
        # In a 10x10 grid, diagonal path should have ~10-14 points
        assert 8 <= len(path) <= 15
        
        print(f"Flat terrain path length: {len(path)} points")
    
    def test_avoid_high_cost_obstacle(self):
        """Test that algorithm avoids high-cost obstacles"""
        # Create 10x10 terrain with obstacle in middle
        rows, cols = 10, 10
        cost_surface = np.ones((rows, cols))
        
        # Add high-cost obstacle (wall) in the middle
        cost_surface[4:6, 3:7] = 1000.0  # Very high cost
        
        transform, transformer, indices = self.create_test_grid(rows, cols)
        pathfinder = DEMTileCache()
        
        # Find path from top to bottom through middle
        start_idx = indices[0, 5]  # Top middle
        end_idx = indices[9, 5]    # Bottom middle
        
        path = pathfinder.astar_pathfinding(
            cost_surface, indices, start_idx, end_idx,
            transform, transformer
        )
        
        assert path is not None
        
        # Path should go around the obstacle
        # Convert path back to grid coordinates to check
        path_cells = []
        for lon, lat in path:
            x, y = transformer.transform(lon, lat)
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            path_cells.append((row, col))
        
        # Check that path doesn't go through obstacle
        for row, col in path_cells:
            if 0 <= row < rows and 0 <= col < cols:
                assert cost_surface[row, col] < 1000.0, f"Path goes through obstacle at ({row}, {col})"
        
        print(f"Obstacle avoidance path length: {len(path)} points")
    
    def test_prefer_low_cost_path(self):
        """Test that algorithm prefers lower cost paths"""
        # Create terrain with two possible paths
        rows, cols = 10, 10
        cost_surface = np.ones((rows, cols)) * 5.0  # Default cost = 5 (higher)
        
        # Create a low-cost "valley" path
        cost_surface[5, :] = 1.0  # Horizontal valley in middle (much lower)
        
        transform, transformer, indices = self.create_test_grid(rows, cols)
        pathfinder = DEMTileCache()
        
        # Find path from left to right, but closer to valley
        start_idx = indices[3, 0]   # Left side, row 3
        end_idx = indices[3, 9]     # Right side, row 3
        
        path = pathfinder.astar_pathfinding(
            cost_surface, indices, start_idx, end_idx,
            transform, transformer
        )
        
        assert path is not None
        
        # Convert path to grid coordinates
        path_cells = []
        for lon, lat in path:
            x, y = transformer.transform(lon, lat)
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            path_cells.append((row, col))
        
        # Path should use the valley (row 5) for most of its length
        valley_cells = sum(1 for row, col in path_cells if row == 5)
        print(f"Valley path uses {valley_cells} out of {len(path_cells)} cells in low-cost area")
        
        # Debug: print path
        print(f"Path cells: {path_cells}")
        
        # Should use valley for significant portion (at least 30%)
        assert valley_cells >= len(path_cells) * 0.3
    
    def test_steep_slope_avoidance(self):
        """Test that algorithm avoids steep slopes when possible"""
        rows, cols = 20, 20
        
        # Create elevation that increases steeply in middle
        dem = np.zeros((rows, cols))
        for i in range(rows):
            if 8 <= i <= 12:
                # Steep slope in middle rows
                dem[i, :] = (i - 8) * 100  # 100m per row = very steep
            else:
                dem[i, :] = 0  # Flat elsewhere
        
        # Calculate slope-based costs
        cost_surface = np.ones((rows, cols))
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Simple slope calculation
                dy = abs(dem[i+1, j] - dem[i-1, j]) / 60  # 30m cells
                dx = abs(dem[i, j+1] - dem[i, j-1]) / 60
                slope = np.sqrt(dy**2 + dx**2)
                
                # Higher cost for steeper slopes
                if slope > 0.3:  # ~17 degrees
                    cost_surface[i, j] = 10.0
                elif slope > 0.1:  # ~6 degrees
                    cost_surface[i, j] = 3.0
        
        transform, transformer, indices = self.create_test_grid(rows, cols)
        pathfinder = DEMTileCache()
        
        # Find path from left to right
        start_idx = indices[10, 0]   # Middle left
        end_idx = indices[10, 19]    # Middle right
        
        path = pathfinder.astar_pathfinding(
            cost_surface, indices, start_idx, end_idx,
            transform, transformer, dem
        )
        
        assert path is not None
        
        # Path should avoid the steep middle section
        path_cells = []
        for lon, lat in path:
            x, y = transformer.transform(lon, lat)
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            path_cells.append((row, col))
        
        # Check how many cells are in steep area
        steep_cells = sum(1 for row, col in path_cells 
                         if 8 <= row <= 12 and 0 <= col < cols)
        
        print(f"Steep avoidance: {steep_cells} out of {len(path_cells)} cells in steep area")
        
        # Should minimize time in steep area
        assert steep_cells < len(path_cells) * 0.3
    
    def test_impossible_path(self):
        """Test that algorithm returns None for impossible paths"""
        # Create terrain with complete barrier
        rows, cols = 10, 10
        cost_surface = np.ones((rows, cols))
        
        # Create impassable barrier across middle
        cost_surface[4:6, :] = np.inf  # Infinite cost = impassable
        
        transform, transformer, indices = self.create_test_grid(rows, cols)
        pathfinder = DEMTileCache()
        
        # Try to find path from top to bottom
        start_idx = indices[0, 5]  # Top
        end_idx = indices[9, 5]    # Bottom
        
        path = pathfinder.astar_pathfinding(
            cost_surface, indices, start_idx, end_idx,
            transform, transformer
        )
        
        # Should not find a path
        assert path is None or len(path) == 0
        print("Correctly identified impossible path")
    
    def test_heuristic_admissibility(self):
        """Test that heuristic never overestimates actual cost"""
        rows, cols = 10, 10
        cost_surface = np.ones((rows, cols))
        transform, transformer, indices = self.create_test_grid(rows, cols)
        
        pathfinder = DEMTileCache()
        
        # Test heuristic for various point pairs
        test_pairs = [
            (indices[0, 0], indices[0, 1]),    # Adjacent horizontal
            (indices[0, 0], indices[1, 0]),    # Adjacent vertical  
            (indices[0, 0], indices[1, 1]),    # Adjacent diagonal
            (indices[0, 0], indices[5, 5]),    # Far diagonal
        ]
        
        for start_idx, end_idx in test_pairs:
            h_cost = pathfinder.heuristic(start_idx, end_idx, indices.shape, transform)
            
            # Find actual path cost
            path = pathfinder.astar_pathfinding(
                cost_surface, indices, start_idx, end_idx,
                transform, transformer
            )
            
            if path and len(path) > 1:
                # Calculate actual path cost more accurately
                # For unit cost surface, minimum cost is distance * 1.0
                start_row, start_col = np.unravel_index(start_idx, indices.shape)
                end_row, end_col = np.unravel_index(end_idx, indices.shape)
                
                # Actual minimum distance in meters
                dx = abs(end_col - start_col) * 30.0
                dy = abs(end_row - start_row) * 30.0
                min_distance = np.sqrt(dx**2 + dy**2)
                
                # Heuristic should not overestimate the minimum possible cost
                print(f"Heuristic: {h_cost:.2f}, Min distance: {min_distance:.2f}")
                assert h_cost <= min_distance * 1.01, \
                    f"Heuristic {h_cost} overestimates minimum distance {min_distance}"
        
        print("Heuristic is admissible")


if __name__ == "__main__":
    # Run tests directly
    test = TestPathfindingAlgorithm()
    
    print("Running pathfinding algorithm tests...")
    print("\n1. Testing straight path on flat terrain:")
    test.test_straight_path_flat_terrain()
    
    print("\n2. Testing obstacle avoidance:")
    test.test_avoid_high_cost_obstacle()
    
    print("\n3. Testing low-cost path preference:")
    test.test_prefer_low_cost_path()
    
    print("\n4. Testing steep slope avoidance:")
    test.test_steep_slope_avoidance()
    
    print("\n5. Testing impossible path detection:")
    test.test_impossible_path()
    
    print("\n6. Testing heuristic admissibility:")
    test.test_heuristic_admissibility()
    
    print("\nAll tests completed!")