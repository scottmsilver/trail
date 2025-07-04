#!/usr/bin/env python3
"""
Simplified test for bidirectional speedup concept
"""

import pytest
import time


@pytest.mark.unit
def test_bidirectional_concept(mock_dem_cache):
    """Test that bidirectional search concept works with mock"""
    
    # Test coordinates
    start_lat, start_lon = 40.6572, -111.5706
    end_lat, end_lon = 40.6486, -111.5639
    
    # Test with mock - it should return quickly
    start_time = time.time()
    path, stats = mock_dem_cache.find_path(start_lat, start_lon, end_lat, end_lon)
    elapsed = time.time() - start_time
    
    assert path is not None, "Should find path"
    assert len(path) > 2, "Path should have multiple points"
    assert elapsed < 1.0, f"Mock should be fast, took {elapsed}s"
    assert stats['algorithm'] == 'mock_pathfinder'
    
    # In a real implementation, we would compare:
    # - Standard A* time vs Bidirectional A* time
    # - Both should find same quality path
    # - Bidirectional should be faster for medium/long routes
    
    print(f"Mock pathfinder completed in {elapsed:.3f}s")
    print(f"Path length: {len(path)} points")
    print(f"Distance: {stats.get('distance_m', 0):.0f}m")


@pytest.mark.unit 
def test_algorithm_selection_logic():
    """Test the logic for selecting algorithms based on distance"""
    
    def should_use_bidirectional(distance_km):
        """Decide whether to use bidirectional search"""
        if distance_km < 0.1:  # Very short routes
            return False
        elif distance_km > 5.0:  # Long routes
            return True
        else:  # Medium routes
            return distance_km > 0.5
    
    # Test algorithm selection
    assert not should_use_bidirectional(0.05), "Very short routes should use standard"
    assert should_use_bidirectional(1.0), "Medium routes should use bidirectional"
    assert should_use_bidirectional(10.0), "Long routes should use bidirectional"