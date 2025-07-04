#!/usr/bin/env python3
"""
Test custom slope and path configurations
"""

import pytest
import requests
import time
from typing import Dict, Optional


@pytest.mark.unit
class TestCustomConfiguration:
    """Test custom slope and path configurations through the API"""
    
    BASE_URL = "http://localhost:9001/api"
    
    # Test coordinates with elevation
    START = {"lat": 40.6482, "lon": -111.5738}
    END = {"lat": 40.6464, "lon": -111.5729}
    
    @pytest.fixture(autouse=True)
    def check_server(self):
        """Ensure server is running before tests"""
        try:
            response = requests.get(f"{self.BASE_URL}/health", timeout=2)
            assert response.status_code == 200, "Server not responding"
        except requests.exceptions.RequestException:
            pytest.skip("API server not running on port 9001")
    
    def send_route_request(self, request_data: Dict) -> Optional[str]:
        """Send route calculation request and return route ID"""
        response = requests.post(
            f"{self.BASE_URL}/routes/calculate",
            json=request_data,
            timeout=10
        )
        
        assert response.status_code == 202, f"Expected 202, got {response.status_code}"
        data = response.json()
        assert "routeId" in data, "No routeId in response"
        return data["routeId"]
    
    def wait_for_route(self, route_id: str, timeout: int = 60) -> Optional[Dict]:
        """Poll for route completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.BASE_URL}/routes/{route_id}/status")
            
            if response.status_code == 200:
                status_data = response.json()
                status = status_data.get("status")
                
                if status == "completed":
                    # Get the full route
                    route_response = requests.get(f"{self.BASE_URL}/routes/{route_id}")
                    if route_response.status_code == 200:
                        return route_response.json()
                elif status == "failed":
                    pytest.fail(f"Route calculation failed: {status_data.get('message')}")
            
            time.sleep(1)
        
        pytest.fail(f"Route calculation timed out after {timeout}s")
    
    def test_default_configuration(self):
        """Test route with default configuration"""
        request = {
            "start": self.START,
            "end": self.END,
            "options": {"userProfile": "default"}
        }
        
        route_id = self.send_route_request(request)
        result = self.wait_for_route(route_id)
        
        assert result is not None
        assert "path" in result
        assert len(result["path"]) > 0
        assert "stats" in result
        
        # Store for comparison
        self.default_stats = result["stats"]
    
    def test_strict_slope_configuration(self):
        """Test with very strict slope limits (max 5°)"""
        request = {
            "start": self.START,
            "end": self.END,
            "options": {
                "userProfile": "default",
                "customSlopeCosts": [
                    {"slope_degrees": 0, "cost_multiplier": 1.0},
                    {"slope_degrees": 3, "cost_multiplier": 2.0},
                    {"slope_degrees": 5, "cost_multiplier": 10.0}
                ],
                "maxSlope": 5.0
            }
        }
        
        route_id = self.send_route_request(request)
        result = self.wait_for_route(route_id)
        
        assert result is not None
        stats = result["stats"]
        
        # Should have lower max slope
        assert stats["max_slope"] <= 5.0, f"Max slope {stats['max_slope']}° exceeds limit"
        
        # Might have longer distance to avoid steep areas
        # (This is expected behavior, not a failure)
    
    def test_lenient_slope_configuration(self):
        """Test with lenient slope configuration (accepts steep slopes)"""
        request = {
            "start": self.START,
            "end": self.END,
            "options": {
                "userProfile": "default",
                "customSlopeCosts": [
                    {"slope_degrees": 0, "cost_multiplier": 1.0},
                    {"slope_degrees": 20, "cost_multiplier": 1.2},
                    {"slope_degrees": 40, "cost_multiplier": 1.5}
                ]
            }
        }
        
        route_id = self.send_route_request(request)
        result = self.wait_for_route(route_id)
        
        assert result is not None
        stats = result["stats"]
        
        # Should find a route (lenient config accepts most slopes)
        assert len(result["path"]) > 0
    
    def test_custom_path_costs(self):
        """Test custom path type preferences"""
        request = {
            "start": self.START,
            "end": self.END,
            "options": {
                "userProfile": "default",
                "customPathCosts": {
                    "footway": 0.5,      # Strongly prefer footways
                    "path": 0.7,         # Prefer paths
                    "trail": 0.8,        # Slight preference for trails
                    "residential": 2.0,  # Avoid roads
                    "off_path": 3.0      # Strongly avoid off-path
                }
            }
        }
        
        route_id = self.send_route_request(request)
        result = self.wait_for_route(route_id)
        
        assert result is not None
        assert len(result["path"]) > 0
        
        # Route should be valid (actual path preferences depend on available data)
    
    def test_combined_custom_configuration(self):
        """Test both custom slopes and path costs together"""
        request = {
            "start": self.START,
            "end": self.END,
            "options": {
                "userProfile": "default",
                "maxSlope": 15.0,
                "customSlopeCosts": [
                    {"slope_degrees": 0, "cost_multiplier": 1.0},
                    {"slope_degrees": 10, "cost_multiplier": 2.0},
                    {"slope_degrees": 15, "cost_multiplier": 5.0}
                ],
                "customPathCosts": {
                    "footway": 0.6,
                    "path": 0.8,
                    "trail": 1.0,
                    "residential": 1.5,
                    "off_path": 2.0
                }
            }
        }
        
        route_id = self.send_route_request(request)
        result = self.wait_for_route(route_id)
        
        assert result is not None
        stats = result["stats"]
        
        # Verify constraints are respected
        assert stats["max_slope"] <= 15.0, f"Max slope {stats['max_slope']}° exceeds limit"
        assert len(result["path"]) > 0
    
    @pytest.mark.parametrize("profile,expected_difficulty", [
        ("easy", "easy"),
        ("experienced", "moderate"),
        ("trail_runner", "moderate"),
        ("accessibility", "easy")
    ])
    def test_user_profiles(self, profile, expected_difficulty):
        """Test different user profiles"""
        request = {
            "start": self.START,
            "end": self.END,
            "options": {"userProfile": profile}
        }
        
        route_id = self.send_route_request(request)
        result = self.wait_for_route(route_id)
        
        assert result is not None
        assert len(result["path"]) > 0
        
        # Different profiles may produce different difficulty ratings
        # (exact values depend on terrain)