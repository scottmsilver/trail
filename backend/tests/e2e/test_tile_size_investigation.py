#!/usr/bin/env python3
"""Investigation into tile size variations to fix mismatch warnings"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.services.tiled_dem_cache import TiledDEMCache
import pickle
import pytest


@pytest.mark.real_data
@pytest.mark.slow
class TestTileSizeInvestigation(unittest.TestCase):
    """Investigate why tiles have different sizes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = TiledDEMCache(tile_size_degrees=0.01)
        self.test_bounds = (40.6282, 40.6760, -111.5908, -111.5441)  # Park City area
    
    def test_analyze_existing_tiles(self):
        """Analyze size variations in existing cached tiles"""
        print("\n🔍 ANALYZING EXISTING TILE SIZES")
        print("="*60)
        
        cache_dir = os.path.abspath(self.cache.cache_dir)
        cost_dir = os.path.join(cache_dir, 'cost')
        
        if not os.path.exists(cost_dir):
            print(f"No cost tiles found at {cost_dir}")
            return
        
        tile_sizes = {}
        tile_details = []
        
        # Analyze all cached tiles
        for filename in os.listdir(cost_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(cost_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        tile_data = pickle.load(f)
                    
                    if 'cost_surface' in tile_data:
                        shape = tile_data['cost_surface'].shape
                        size_key = f"{shape[0]}x{shape[1]}"
                        
                        if size_key not in tile_sizes:
                            tile_sizes[size_key] = []
                        
                        tile_sizes[size_key].append(filename)
                        
                        # Extract tile coordinates from filename
                        parts = filename.replace('.pkl', '').split('_')
                        if len(parts) >= 3:
                            tile_x = int(parts[1])
                            tile_y = int(parts[2])
                            
                            tile_details.append({
                                'filename': filename,
                                'tile_x': tile_x,
                                'tile_y': tile_y,
                                'shape': shape,
                                'size_key': size_key
                            })
                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        # Report findings
        print("\n📊 TILE SIZE DISTRIBUTION:")
        for size, files in sorted(tile_sizes.items()):
            print(f"  {size}: {len(files)} tiles")
            if len(files) <= 5:
                for f in files:
                    print(f"    - {f}")
        
        # Analyze patterns
        if tile_details:
            # Find edge tiles
            min_x = min(t['tile_x'] for t in tile_details)
            max_x = max(t['tile_x'] for t in tile_details)
            min_y = min(t['tile_y'] for t in tile_details)
            max_y = max(t['tile_y'] for t in tile_details)
            
            print(f"\n📍 TILE GRID EXTENT:")
            print(f"  X range: {min_x} to {max_x}")
            print(f"  Y range: {min_y} to {max_y}")
            
            # Check if edge tiles are smaller
            print("\n🔲 EDGE TILE ANALYSIS:")
            edge_tiles = [t for t in tile_details if 
                         t['tile_x'] in (min_x, max_x) or 
                         t['tile_y'] in (min_y, max_y)]
            
            edge_sizes = {}
            for t in edge_tiles:
                if t['size_key'] not in edge_sizes:
                    edge_sizes[t['size_key']] = []
                edge_sizes[t['size_key']].append(f"({t['tile_x']}, {t['tile_y']})")
            
            for size, coords in edge_sizes.items():
                print(f"  {size}: {len(coords)} edge tiles")
                if len(coords) <= 5:
                    print(f"    Coordinates: {', '.join(coords)}")
    
    def test_tile_computation_determinism(self):
        """Test if tile computation is deterministic"""
        print("\n🔬 TESTING TILE COMPUTATION DETERMINISM")
        print("="*60)
        
        # Test computing the same tile multiple times
        test_tile_x = -11157  # From the error output
        test_tile_y = 4065
        
        print(f"Testing tile ({test_tile_x}, {test_tile_y})")
        
        # Mock the computation function to track calls
        original_compute = self.cache._compute_cost_for_tile
        compute_sizes = []
        
        def track_compute(tx, ty):
            result = original_compute(tx, ty)
            if result and 'cost_surface' in result:
                shape = result['cost_surface'].shape
                compute_sizes.append(shape)
                print(f"  Computed size: {shape}")
            return result
        
        self.cache._compute_cost_for_tile = track_compute
        
        # Compute the same tile 3 times
        for i in range(3):
            self.cache.memory_cache.clear()  # Clear memory cache
            result = self.cache.get_tile(test_tile_x, test_tile_y, 'cost')
            if result:
                print(f"  Attempt {i+1}: Got shape {result['cost_surface'].shape}")
        
        # Check if sizes are consistent
        if compute_sizes:
            unique_sizes = set(compute_sizes)
            if len(unique_sizes) > 1:
                print("  ❌ Non-deterministic tile sizes detected!")
            else:
                print("  ✅ Tile computation is deterministic")
        
        self.cache._compute_cost_for_tile = original_compute
    
    def test_expected_tile_size_calculation(self):
        """Test how the expected tile size should be calculated"""
        print("\n📐 CALCULATING EXPECTED TILE SIZE")
        print("="*60)
        
        # Calculate expected size based on tile_size_degrees and resolution
        tile_size_deg = self.cache.tile_size
        
        # At 3m resolution, calculate pixels per degree
        # At equator: 1 degree ≈ 111.32 km
        # At 40° latitude: 1 degree longitude ≈ 85.28 km
        # 1 degree latitude ≈ 111.32 km everywhere
        
        lat = 40.65  # Park City latitude
        
        # Calculate meters per degree
        meters_per_deg_lat = 111320  # meters
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat))  # meters
        
        resolution_m = 3  # 3-meter resolution
        
        # Expected pixels
        expected_height = int(tile_size_deg * meters_per_deg_lat / resolution_m)
        expected_width = int(tile_size_deg * meters_per_deg_lon / resolution_m)
        
        print(f"Tile size: {tile_size_deg}° x {tile_size_deg}°")
        print(f"At latitude {lat}°:")
        print(f"  Meters per degree lat: {meters_per_deg_lat:.0f}m")
        print(f"  Meters per degree lon: {meters_per_deg_lon:.0f}m")
        print(f"Resolution: {resolution_m}m")
        print(f"\nExpected tile size: {expected_height} x {expected_width} pixels")
        
        # The actual size might be slightly different due to:
        # 1. Rounding during download
        # 2. Data availability at edges
        # 3. Projection differences
        
        return expected_height, expected_width
    
    def test_tile_size_for_specific_area(self):
        """Test what determines tile size for a specific area"""
        print("\n🗺️ TESTING TILE SIZE DETERMINATION")
        print("="*60)
        
        # Get tiles needed for test area
        bounds = self.test_bounds
        tiles_needed = self.cache._get_tiles_for_bounds(bounds)
        
        print(f"Bounds: {bounds}")
        print(f"Tiles needed: {len(tiles_needed)}")
        
        # Check a few tiles
        sample_tiles = tiles_needed[:5] if len(tiles_needed) > 5 else tiles_needed
        
        for tile_x, tile_y in sample_tiles:
            # Calculate the geographic bounds for this tile
            west = tile_x * self.cache.tile_size
            east = (tile_x + 1) * self.cache.tile_size
            south = tile_y * self.cache.tile_size
            north = (tile_y + 1) * self.cache.tile_size
            
            print(f"\nTile ({tile_x}, {tile_y}):")
            print(f"  Geographic bounds: ({south:.4f}, {north:.4f}, {west:.4f}, {east:.4f})")
            print(f"  Span: {north-south:.4f}° x {east-west:.4f}°")
            
            # Check if tile exists in cache
            cache_path = os.path.join(self.cache.cache_dir, 'cost', f'cost_{tile_x}_{tile_y}.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    tile_data = pickle.load(f)
                shape = tile_data['cost_surface'].shape
                print(f"  Cached size: {shape[0]} x {shape[1]}")


def run_tile_size_investigation():
    """Run all tile size investigation tests"""
    print("\n🧪 TILE SIZE INVESTIGATION")
    print("="*80)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTileSizeInvestigation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_tile_size_investigation()
    sys.exit(0 if success else 1)