import pytest
import numpy as np
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate


def test_trail_finder_service_initialization():
    """Test TrailFinderService can be initialized"""
    service = TrailFinderService()
    assert service is not None
    assert service.buffer == 0.05  # default buffer


def test_validate_coordinates():
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


def test_calculate_bounding_box():
    """Test bounding box calculation"""
    service = TrailFinderService(buffer=0.02)
    start = Coordinate(lat=40.630, lon=-111.580)
    end = Coordinate(lat=40.650, lon=-111.560)
    
    bbox = service.calculate_bounding_box(start, end)
    
    assert bbox["min_lat"] == pytest.approx(40.610, 0.001)
    assert bbox["max_lat"] == pytest.approx(40.670, 0.001)
    assert bbox["min_lon"] == pytest.approx(-111.600, 0.001)
    assert bbox["max_lon"] == pytest.approx(-111.540, 0.001)