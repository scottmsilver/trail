"""
Test the integration of obstacle configuration with actual pathfinding
"""
import pytest
import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from pyproj import Transformer
from rasterio.transform import from_origin


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test that different configurations produce different paths"""
    
    def create_test_terrain(self):
        """Create test terrain with a steep hill and a gentle valley"""
        rows, cols = 30, 30
        dem = np.zeros((rows, cols))
        
        # Create a steep ridge in the middle (rows 10-20)
        for i in range(10, 20):
            for j in range(cols):
                # Peak at row 15
                height = 50 - abs(i - 15) * 10  # Max 50m high
                dem[i, j] = height
        
        # Create a gentle valley path at row 5
        for j in range(cols):
            dem[5, j] = 5  # Slight elevation
            
        return dem, rows, cols
    
    def test_easy_vs_experienced_hiker(self):
        """Test that easy hikers avoid steep terrain more than experienced hikers"""
        dem, rows, cols = self.create_test_terrain()
        
        # Setup pathfinding infrastructure
        transform = from_origin(-111.5, 40.5, 30, 30)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)
        indices = np.arange(rows * cols).reshape(rows, cols)
        
        # Create obstacle mask (no obstacles for this test)
        obstacle_mask = np.zeros((rows, cols), dtype=bool)
        
        # Test easy hiker configuration
        easy_config = ObstaclePresets.easy_hiker()
        easy_pathfinder = DEMTileCache(obstacle_config=easy_config)
        
        # Compute cost surface for easy hiker
        easy_cost = easy_pathfinder.compute_cost_surface(dem, transform, obstacle_mask)
        
        # Test experienced hiker configuration  
        exp_config = ObstaclePresets.experienced_hiker()
        exp_pathfinder = DEMTileCache(obstacle_config=exp_config)
        
        # Compute cost surface for experienced hiker
        exp_cost = exp_pathfinder.compute_cost_surface(dem, transform, obstacle_mask)
        
        # Find paths from left to right
        start_idx = indices[15, 0]  # Start at steep area
        end_idx = indices[15, 29]   # End at steep area
        
        easy_path = easy_pathfinder.astar_pathfinding(
            easy_cost, indices, start_idx, end_idx, transform, transformer, dem
        )
        
        exp_path = exp_pathfinder.astar_pathfinding(
            exp_cost, indices, start_idx, end_idx, transform, transformer, dem
        )
        
        # Convert paths to grid coordinates
        def path_to_rows(path):
            rows = []
            for lon, lat in path:
                x, y = transformer.transform(lon, lat)
                col = int((x - transform.c) / transform.a)
                row = int((y - transform.f) / transform.e)
                rows.append(row)
            return rows
        
        if easy_path and exp_path:
            easy_rows = path_to_rows(easy_path)
            exp_rows = path_to_rows(exp_path)
            
            # Easy hiker should deviate more from the steep ridge
            easy_avg_row = np.mean(easy_rows)
            exp_avg_row = np.mean(exp_rows)
            
            print(f"Easy hiker average row: {easy_avg_row:.1f}")
            print(f"Experienced hiker average row: {exp_avg_row:.1f}")
            
            # Easy hiker should go further from row 15 (the peak)
            assert abs(easy_avg_row - 15) > abs(exp_avg_row - 15), \
                "Easy hiker should avoid steep terrain more than experienced hiker"
    
    def test_accessibility_slope_limits(self):
        """Test that accessibility profile has strict slope limits"""
        dem, rows, cols = self.create_test_terrain()
        
        transform = from_origin(-111.5, 40.5, 30, 30)
        obstacle_mask = np.zeros((rows, cols), dtype=bool)
        
        # Test accessibility configuration
        access_config = ObstaclePresets.accessibility_focused()
        access_pathfinder = DEMTileCache(obstacle_config=access_config)
        
        # Compute cost surface
        access_cost = access_pathfinder.compute_cost_surface(dem, transform, obstacle_mask)
        
        # Check that steep areas have very high or infinite cost
        # Row 15 (peak) should be nearly impassable
        peak_costs = access_cost[14:17, :]
        
        print(f"Accessibility cost at peak: min={np.min(peak_costs):.1f}, max={np.max(peak_costs):.1f}")
        
        # Steep areas should have very high cost
        assert np.min(peak_costs) > 100, "Steep areas should have very high cost for accessibility"
        
        # Gentle valley should have low cost
        valley_costs = access_cost[5, :]
        print(f"Accessibility cost in valley: min={np.min(valley_costs):.1f}, max={np.max(valley_costs):.1f}")
        
        assert np.max(valley_costs) < 10, "Gentle areas should have low cost for accessibility"
    
    def test_cost_surface_differences(self):
        """Test that different profiles produce different cost surfaces"""
        dem, rows, cols = self.create_test_terrain()
        
        transform = from_origin(-111.5, 40.5, 30, 30)
        obstacle_mask = np.zeros((rows, cols), dtype=bool)
        
        configs = {
            'easy': ObstaclePresets.easy_hiker(),
            'experienced': ObstaclePresets.experienced_hiker(),
            'trail_runner': ObstaclePresets.trail_runner(),
            'accessibility': ObstaclePresets.accessibility_focused()
        }
        
        cost_surfaces = {}
        
        for name, config in configs.items():
            pathfinder = DEMTileCache(obstacle_config=config)
            cost_surfaces[name] = pathfinder.compute_cost_surface(dem, transform, obstacle_mask)
        
        # Compare cost surfaces
        # Sample a steep point (row 15) and a gentle point (row 5)
        steep_point = (15, 15)
        gentle_point = (5, 15)
        
        print("\nCost comparison at steep point (row 15):")
        for name, surface in cost_surfaces.items():
            cost = surface[steep_point]
            print(f"  {name}: {cost:.1f}")
        
        print("\nCost comparison at gentle point (row 5):")
        for name, surface in cost_surfaces.items():
            cost = surface[gentle_point]
            print(f"  {name}: {cost:.1f}")
        
        # Verify differences
        # Accessibility should have highest cost at steep point
        steep_costs = {name: surface[steep_point] for name, surface in cost_surfaces.items()}
        assert steep_costs['accessibility'] == max(steep_costs.values()), \
            "Accessibility should have highest cost on steep terrain"
        
        # All should have similar low cost at gentle point
        gentle_costs = {name: surface[gentle_point] for name, surface in cost_surfaces.items()}
        assert max(gentle_costs.values()) < 5, "All profiles should have low cost on gentle terrain"


if __name__ == "__main__":
    test = TestConfigurationIntegration()
    
    print("Testing configuration integration...")
    print("\n1. Testing easy vs experienced hiker:")
    test.test_easy_vs_experienced_hiker()
    
    print("\n2. Testing accessibility slope limits:")
    test.test_accessibility_slope_limits()
    
    print("\n3. Testing cost surface differences:")
    test.test_cost_surface_differences()
    
    print("\nAll configuration integration tests passed!")