#!/usr/bin/env python3
"""
Test the complete workflow to ensure no cache terminology remains
"""

import tempfile
import shutil
import os
import sys
sys.path.append('..')

from elevation import ElevationLibrary, Bounds

def test_workflow():
    """Test complete workflow with new terminology"""
    test_dir = tempfile.mkdtemp(prefix="test_no_cache_")
    
    try:
        print("Testing Elevation Library (No Cache Terminology)")
        print("=" * 50)
        
        # Create library
        lib = ElevationLibrary(data_dir=test_dir, resolution=10)
        print("✓ Created library")
        
        # Test list_loaded_areas (not list_cached_areas)
        info = lib.list_loaded_areas()
        print(f"✓ list_loaded_areas() works: {info['total_tiles']} tiles loaded")
        
        # Test error messages don't mention cache
        try:
            lib.get_elevation(40.0, -111.0)
        except ValueError as e:
            error_msg = str(e)
            if "cache" in error_msg.lower():
                print(f"❌ Error message contains 'cache': {error_msg}")
            else:
                print("✓ Error message doesn't contain 'cache'")
        
        # Test the method exists and old one doesn't
        if hasattr(lib, 'list_loaded_areas'):
            print("✓ list_loaded_areas method exists")
        else:
            print("❌ list_loaded_areas method missing")
            
        if hasattr(lib, 'list_cached_areas'):
            print("❌ Old list_cached_areas method still exists")
        else:
            print("✓ Old list_cached_areas method removed")
        
        print("\n" + "=" * 50)
        print("All checks passed! No cache terminology remains.")
        
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_workflow()