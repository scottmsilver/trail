"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all fixtures from cache_fixtures
from tests.fixtures.cache_fixtures import (
    test_data_dir,
    sample_dem_data,
    sample_cost_surface,
    sample_transform,
    populated_tile_cache,
    populated_terrain_cache,
    populated_cost_cache,
    mock_dem_cache,
    sample_route_request,
    expected_route_stats
)

# Re-export all fixtures
__all__ = [
    'test_data_dir',
    'sample_dem_data',
    'sample_cost_surface',
    'sample_transform',
    'populated_tile_cache',
    'populated_terrain_cache',
    'populated_cost_cache',
    'mock_dem_cache',
    'sample_route_request',
    'expected_route_stats'
]