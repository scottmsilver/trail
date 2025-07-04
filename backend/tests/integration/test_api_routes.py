#!/usr/bin/env python3
"""
Test the API endpoints to diagnose route finding issues
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import json
from unittest import TestCase
import pytest


@pytest.mark.integration
class TestRouteAPI(TestCase):
    """Test route finding through the API"""
    
    def setUp(self):
        """Set up test client"""
        self.base_url = "http://localhost:9001"
        self.api_url = f"{self.base_url}/api"
        
        # Test coordinates (Park City area)
        self.start = {"lat": 40.6560, "lon": -111.5708}
        self.end = {"lat": 40.6482, "lon": -111.5643}
        
        # Check if server is running
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            if response.status_code != 200:
                self.skipTest("API server not running on port 9001")
        except requests.exceptions.RequestException:
            self.skipTest("API server not reachable on port 9001")
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = requests.get(f"{self.api_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        print(f"✅ Health check passed")
    
    def test_calculate_route_endpoint(self):
        """Test the main route calculation endpoint"""
        print(f"\n🧪 Testing route calculation...")
        print(f"Start: {self.start}")
        print(f"End: {self.end}")
        
        # Start route calculation
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/routes/calculate",
            json={
                "start": self.start,
                "end": self.end,
                "options": {"userProfile": "default"}
            }
        )
        
        self.assertEqual(response.status_code, 200, 
                        f"Route calculation failed: {response.text}")
        
        data = response.json()
        self.assertIn("routeId", data)
        route_id = data["routeId"]
        print(f"✅ Route ID received: {route_id}")
        
        # Poll for completion
        max_polls = 60  # 60 seconds max
        poll_interval = 1
        completed = False
        
        for i in range(max_polls):
            status_response = requests.get(f"{self.api_url}/routes/{route_id}/status")
            self.assertEqual(status_response.status_code, 200)
            
            status_data = status_response.json()
            print(f"  Poll {i+1}: Status={status_data['status']}, Progress={status_data.get('progress', 0)}%")
            
            if status_data["status"] == "completed":
                completed = True
                break
            elif status_data["status"] == "failed":
                self.fail(f"Route calculation failed: {status_data.get('message', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        elapsed = time.time() - start_time
        self.assertTrue(completed, f"Route calculation timed out after {elapsed:.1f}s")
        print(f"✅ Route completed in {elapsed:.1f}s")
        
        # Get the actual route
        route_response = requests.get(f"{self.api_url}/routes/{route_id}")
        self.assertEqual(route_response.status_code, 200)
        
        route_data = route_response.json()
        self.assertIn("path", route_data)
        self.assertIn("stats", route_data)
        
        path_length = len(route_data["path"])
        distance = route_data["stats"]["distance_km"]
        print(f"✅ Route found: {path_length} points, {distance:.2f}km")
        
        return route_data
    
    def test_debug_route_endpoint(self):
        """Test the debug route endpoint"""
        print(f"\n🧪 Testing debug route...")
        
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/routes/debug",
            json={
                "start": self.start,
                "end": self.end,
                "options": {"userProfile": "default"}
            }
        )
        
        # Debug endpoint might take longer
        self.assertEqual(response.status_code, 200, 
                        f"Debug route failed: {response.text}")
        
        elapsed = time.time() - start_time
        route_data = response.json()
        
        self.assertIn("path", route_data)
        self.assertIn("stats", route_data)
        
        print(f"✅ Debug route completed in {elapsed:.1f}s")
        
        # Check if debug data is present
        if "debug_data" in route_data.get("stats", {}):
            debug_data = route_data["stats"]["debug_data"]
            print(f"  Debug data keys: {list(debug_data.keys())}")
        else:
            print(f"  ⚠️  No debug data in response")
        
        return route_data
    
    def test_route_with_options(self):
        """Test route calculation with different options"""
        print(f"\n🧪 Testing route with custom options...")
        
        options = {
            "userProfile": "experienced",
            "maxSlope": 45,
            "customSlopeCosts": [
                {"slope_degrees": 0, "cost_multiplier": 1.0},
                {"slope_degrees": 20, "cost_multiplier": 1.2},
                {"slope_degrees": 40, "cost_multiplier": 2.0}
            ]
        }
        
        response = requests.post(
            f"{self.api_url}/routes/calculate",
            json={
                "start": self.start,
                "end": self.end,
                "options": options
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("routeId", data)
        print(f"✅ Route with options accepted: {data['routeId']}")
    
    def test_concurrent_requests(self):
        """Test multiple concurrent route requests"""
        print(f"\n🧪 Testing concurrent requests...")
        
        import concurrent.futures
        
        def make_request(offset):
            """Make a route request with slightly different endpoint"""
            end_modified = {
                "lat": self.end["lat"],
                "lon": self.end["lon"] + offset * 0.0001
            }
            
            response = requests.post(
                f"{self.api_url}/routes/calculate",
                json={
                    "start": self.start,
                    "end": end_modified,
                    "options": {"userProfile": "default"}
                }
            )
            return response.status_code, response.json()
        
        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for status, data in results:
            self.assertEqual(status, 200)
            self.assertIn("routeId", data)
        
        print(f"✅ All {len(results)} concurrent requests succeeded")
    
    def test_error_handling(self):
        """Test API error handling"""
        print(f"\n🧪 Testing error handling...")
        
        # Test invalid coordinates
        response = requests.post(
            f"{self.api_url}/routes/calculate",
            json={
                "start": {"lat": 91, "lon": -111},  # Invalid latitude
                "end": self.end,
                "options": {}
            }
        )
        
        self.assertIn(response.status_code, [400, 422])
        print(f"✅ Invalid coordinates rejected with status {response.status_code}")
        
        # Test missing fields
        response = requests.post(
            f"{self.api_url}/routes/calculate",
            json={"start": self.start}  # Missing end
        )
        
        self.assertIn(response.status_code, [400, 422])
        print(f"✅ Missing fields rejected with status {response.status_code}")


def run_api_tests():
    """Run all API tests"""
    print("🧪 API ROUTE TESTING")
    print("="*60)
    print("Testing route calculation through API endpoints...")
    print(f"Make sure the backend is running on port 9001!")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRouteAPI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_api_tests()
    sys.exit(0 if success else 1)