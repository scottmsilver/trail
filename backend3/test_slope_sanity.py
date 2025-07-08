#!/usr/bin/env python3
"""
Sanity tests for slope calculations - verify slopes match real-world expectations.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def test_slope_sanity():
    """Run sanity checks on slope calculations"""
    
    # Test area
    bounds = Bounds(
        south=40.650,
        north=40.655,
        west=-111.575,
        east=-111.570
    )
    
    # Initialize
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, "./slope_data")
    
    print("Slope Sanity Tests")
    print("=" * 50)
    
    # Get data
    try:
        slope_data, _ = slope_layer.get_slope_array(bounds)
        elev_array, _ = elev_lib.get_elevation_array(bounds)
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    all_tests_passed = True
    
    # Test 1: Slope range should be reasonable
    print("\nTest 1: Reasonable slope range")
    min_slope = slope_data.slope.min()
    max_slope = slope_data.slope.max()
    mean_slope = slope_data.slope.mean()
    
    print(f"  Min slope: {min_slope:.1f}°")
    print(f"  Max slope: {max_slope:.1f}°")
    print(f"  Mean slope: {mean_slope:.1f}°")
    
    # Ski terrain typically ranges from 0-60 degrees
    # Anything over 60 is essentially a cliff
    if min_slope < 0:
        print("  ❌ FAIL: Negative slopes found!")
        all_tests_passed = False
    elif max_slope > 90:
        print("  ❌ FAIL: Slopes > 90° (impossible)!")
        all_tests_passed = False
    elif max_slope > 60:
        print("  ⚠️  WARNING: Very steep slopes (>60°) - verify this is correct")
    else:
        print("  ✓ PASS: Slope range is reasonable")
    
    # Test 2: Flat areas should have near-zero slope
    print("\nTest 2: Flat area slope check")
    # Find relatively flat areas (small elevation change)
    window_size = 3
    flat_threshold = 0.5  # meters
    
    flat_areas_found = False
    for i in range(window_size, elev_array.shape[0] - window_size):
        for j in range(window_size, elev_array.shape[1] - window_size):
            window = elev_array[i-window_size:i+window_size+1, 
                              j-window_size:j+window_size+1]
            if window.max() - window.min() < flat_threshold:
                # This is a flat area
                slope_window = slope_data.slope[i-window_size:i+window_size+1,
                                               j-window_size:j+window_size+1]
                max_slope_in_flat = slope_window.max()
                
                if max_slope_in_flat > 5:  # More than 5 degrees in a flat area
                    print(f"  ❌ FAIL: Flat area at ({i},{j}) has slope {max_slope_in_flat:.1f}°")
                    all_tests_passed = False
                else:
                    flat_areas_found = True
    
    if flat_areas_found and all_tests_passed:
        print("  ✓ PASS: Flat areas have appropriate slopes")
    elif not flat_areas_found:
        print("  ⚠️  WARNING: No flat areas found to test")
    
    # Test 3: Steep areas should correspond to large elevation changes
    print("\nTest 3: Steep slope verification")
    steep_threshold = 30  # degrees
    steep_mask = slope_data.slope > steep_threshold
    
    if steep_mask.any():
        # Check some steep areas
        steep_indices = np.where(steep_mask)
        num_samples = min(10, len(steep_indices[0]))
        
        verified_steep = 0
        for idx in range(num_samples):
            i, j = steep_indices[0][idx], steep_indices[1][idx]
            
            # Check elevation change in neighborhood
            if (i > 0 and i < elev_array.shape[0]-1 and 
                j > 0 and j < elev_array.shape[1]-1):
                
                # Get elevation differences
                elev_center = elev_array[i, j]
                elev_neighbors = [
                    elev_array[i-1, j], elev_array[i+1, j],
                    elev_array[i, j-1], elev_array[i, j+1]
                ]
                
                max_diff = max(abs(elev_center - n) for n in elev_neighbors)
                
                # For 10m pixels, 30° slope means ~5.77m elevation change
                expected_diff = 10 * np.tan(np.radians(slope_data.slope[i, j]))
                
                if max_diff > expected_diff * 0.5:  # At least half the expected
                    verified_steep += 1
        
        if verified_steep > num_samples * 0.8:
            print(f"  ✓ PASS: Steep slopes correspond to elevation changes ({verified_steep}/{num_samples} verified)")
        else:
            print(f"  ❌ FAIL: Steep slopes don't match elevation changes ({verified_steep}/{num_samples} verified)")
            all_tests_passed = False
    else:
        print("  ℹ️  INFO: No steep slopes found in this area")
    
    # Test 4: Aspect should be 0-360
    print("\nTest 4: Aspect range check")
    min_aspect = slope_data.aspect.min()
    max_aspect = slope_data.aspect.max()
    
    if min_aspect < 0 or max_aspect > 360:
        print(f"  ❌ FAIL: Aspect out of range: {min_aspect:.1f}° to {max_aspect:.1f}°")
        all_tests_passed = False
    else:
        print(f"  ✓ PASS: Aspect in valid range: {min_aspect:.1f}° to {max_aspect:.1f}°")
    
    # Test 5: Slope calculation verification
    print("\nTest 5: Manual slope calculation check")
    # Pick a few random points and verify slope calculation
    num_checks = 5
    pixel_size = 10  # meters
    
    mismatches = 0
    for _ in range(num_checks):
        # Random point away from edges
        i = np.random.randint(1, elev_array.shape[0] - 1)
        j = np.random.randint(1, elev_array.shape[1] - 1)
        
        # Calculate slope manually
        dz_dx = (elev_array[i, j+1] - elev_array[i, j-1]) / (2 * pixel_size)
        dz_dy = (elev_array[i-1, j] - elev_array[i+1, j]) / (2 * pixel_size)
        
        manual_slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        computed_slope = slope_data.slope[i, j]
        
        diff = abs(manual_slope - computed_slope)
        if diff > 2:  # Allow 2 degree tolerance
            print(f"  Point ({i},{j}): manual={manual_slope:.1f}°, computed={computed_slope:.1f}°, diff={diff:.1f}°")
            mismatches += 1
    
    if mismatches == 0:
        print(f"  ✓ PASS: Manual calculations match ({num_checks} points checked)")
    else:
        print(f"  ❌ FAIL: {mismatches}/{num_checks} points don't match manual calculation")
        all_tests_passed = False
    
    # Test 6: North-facing slopes
    print("\nTest 6: North-facing slope check")
    # In northern hemisphere, north-facing slopes should exist
    north_facing = ((slope_data.aspect > 315) | (slope_data.aspect < 45)) & (slope_data.slope > 5)
    north_percent = 100 * north_facing.sum() / slope_data.slope.size
    
    if north_percent > 0:
        print(f"  ✓ PASS: {north_percent:.1f}% of terrain is north-facing")
    else:
        print(f"  ⚠️  WARNING: No north-facing slopes found")
    
    # Test 7: Slope continuity
    print("\nTest 7: Slope continuity check")
    # Large jumps in slope between adjacent pixels might indicate issues
    slope_diff_x = np.abs(np.diff(slope_data.slope, axis=1))
    slope_diff_y = np.abs(np.diff(slope_data.slope, axis=0))
    
    max_jump_x = slope_diff_x.max()
    max_jump_y = slope_diff_y.max()
    mean_jump = (slope_diff_x.mean() + slope_diff_y.mean()) / 2
    
    print(f"  Max slope jump: {max(max_jump_x, max_jump_y):.1f}°")
    print(f"  Mean slope jump: {mean_jump:.1f}°")
    
    if max(max_jump_x, max_jump_y) > 45:
        print(f"  ⚠️  WARNING: Very large slope jumps detected (might be cliffs)")
    elif mean_jump > 10:
        print(f"  ❌ FAIL: Slopes are too discontinuous")
        all_tests_passed = False
    else:
        print(f"  ✓ PASS: Slopes are reasonably continuous")
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ ALL SANITY TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - Review slope calculations")
    
    return all_tests_passed


if __name__ == "__main__":
    test_slope_sanity()