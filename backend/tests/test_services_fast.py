import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate


class TestTrailFinderServiceFast:
    """Fast tests for TrailFinderService using mocked cache"""
    
    def test_trail_finder_service_initialization(self):
        """Test TrailFinderService can be initialized"""
        service = TrailFinderService()
        assert service is not None
        assert service.buffer == 0.05  # default buffer

    def test_validate_coordinates(self):
        """Test coordinate validation"""
        service = TrailFinderService()
        
        # Valid coordinates
        valid_start = Coordinate(lat=40.630, lon=-111.580)
        valid_end = Coordinate(lat=40.650, lon=-111.560)
        assert service.validate_route_request(valid_start, valid_end) is True
        
        # Invalid - same coordinates
        assert service.validate_route_request(valid_start, valid_start) is False
        
        # Invalid - too far apart (> 50km)
        far_end = Coordinate(lat=41.630, lon=-110.580)
        assert service.validate_route_request(valid_start, far_end) is False

    def test_calculate_bounding_box(self):
        """Test bounding box calculation"""
        service = TrailFinderService(buffer=0.02)
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        
        bbox = service.calculate_bounding_box(start, end)
        
        assert bbox["min_lat"] == pytest.approx(40.610, 0.001)
        assert bbox["max_lat"] == pytest.approx(40.670, 0.001)
        assert bbox["min_lon"] == pytest.approx(-111.600, 0.001)
        assert bbox["max_lon"] == pytest.approx(-111.540, 0.001)

    @pytest.mark.asyncio
    async def test_find_route_with_mock(self, mock_dem_cache):
        """Test route finding with mocked cache"""
        service = TrailFinderService()
        
        # Replace the DEM cache with our mock
        service.dem_cache = mock_dem_cache
        
        # Use coordinates that match our mock data
        start = Coordinate(lat=40.6560, lon=-111.5708)
        end = Coordinate(lat=40.6520, lon=-111.5688)
        
        # Mock the find_path method to return a simple path
        mock_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        mock_stats = {
            'nodes_explored': 100,
            'path_length': 4,
            'computation_time': 0.1
        }
        
        with patch.object(mock_dem_cache, 'find_path', return_value=(mock_path, mock_stats)):
            path, stats = await service.find_route(start, end, {})
            
            assert path is not None
            assert len(path) > 0
            assert stats is not None
            assert 'distance_km' in stats
            assert 'waypoints' in stats

    def test_debug_mode(self, mock_dem_cache):
        """Test debug mode functionality"""
        service = TrailFinderService(debug_mode=True)
        service.dem_cache = mock_dem_cache
        
        assert service.debug_mode is True
        assert service.dem_cache.debug_mode is True

    @pytest.mark.asyncio
    async def test_route_options(self, mock_dem_cache):
        """Test different route options"""
        service = TrailFinderService()
        service.dem_cache = mock_dem_cache
        
        start = Coordinate(lat=40.6560, lon=-111.5708)
        end = Coordinate(lat=40.6520, lon=-111.5688)
        
        # Test with different resolutions
        for resolution in ['low', 'medium', 'high']:
            mock_path = [(0, 0), (1, 1), (2, 2)]
            mock_stats = {'path_length': 3}
            
            with patch.object(mock_dem_cache, 'find_path', return_value=(mock_path, mock_stats)):
                options = {'resolution': resolution}
                path, stats = await service.find_route(start, end, options)
                
                assert path is not None
                assert len(path) > 0

    @pytest.mark.asyncio  
    async def test_error_handling(self, mock_dem_cache):
        """Test error handling in route finding"""
        service = TrailFinderService()
        service.dem_cache = mock_dem_cache
        
        start = Coordinate(lat=40.6560, lon=-111.5708)
        end = Coordinate(lat=40.6520, lon=-111.5688)
        
        # Mock find_path to return None (no path found)
        with patch.object(mock_dem_cache, 'find_path', return_value=(None, {'error': 'No path found'})):
            path, stats = await service.find_route(start, end, {})
            
            # Should handle gracefully
            assert path is None or len(path) == 0
            assert stats is not None

    def test_cache_status(self, mock_dem_cache):
        """Test cache status reporting"""
        service = TrailFinderService()
        service.dem_cache = mock_dem_cache
        
        status = service.dem_cache.get_cache_status()
        
        assert 'terrain_cache' in status
        assert 'cost_surface_cache' in status
        assert 'total_memory_mb' in status
        assert status['terrain_cache']['count'] > 0
        assert status['cost_surface_cache']['count'] > 0