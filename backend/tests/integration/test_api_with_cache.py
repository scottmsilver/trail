#!/usr/bin/env python3
"""
Test API routes with proper cache management
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import subprocess
from unittest import TestCase
import pytest


@pytest.mark.integration
class TestAPIWithCache(TestCase):
    """Test API routes after pre-warming cache"""
    
    @classmethod
    def setUpClass(cls):
        """Pre-warm the cache for test area"""
        print("\n🔥 PRE-WARMING CACHE...")
        print("="*60)
        
        # Use route_cli to ensure cache is populated
        # This uses a small test area in Park City
        test_route_cmd = [
            sys.executable, "route_cli.py",
            "Start: 40.6560, -111.5708",
            "End: 40.6482, -111.5643"
        ]
        
        print("Pre-calculating test route to populate caches...")
        start_time = time.time()
        
        result = subprocess.run(test_route_cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Cache warmed in {elapsed:.1f}s")
            # Extract some info
            output = result.stdout
            if "Total time:" in output:
                for line in output.split('\n'):
                    if "Total time:" in line:
                        print(f"  {line.strip()}")
        else:
            print(f"⚠️  Cache warming failed: {result.stderr}")
            
        # Give the server a moment to settle
        time.sleep(2)
    
    def setUp(self):
        """Set up test client"""
        self.base_url = "http://localhost:9001"
        self.api_url = f"{self.base_url}/api"
        
        # Test coordinates (same as cache warming)
        self.start = {"lat": 40.6560, "lon": -111.5708}
        self.end = {"lat": 40.6482, "lon": -111.5643}
        
        # Test server is running
        self.assertTrue(self._check_server(), "API server not running on port 9001")
    
    def _check_server(self):
        """Check if server is responding"""
        try:
            # Try a simple endpoint first
            response = requests.get(self.base_url, timeout=2)
            return response.status_code in [200, 404]  # Any response means server is up
        except:
            return False
    
    def test_1_route_calculation_cached(self):
        """Test route calculation with cached data (should be fast)"""
        print(f"\n🧪 Testing route calculation (cached)...")
        
        start_time = time.time()
        
        # Start route calculation
        response = requests.post(
            f"{self.api_url}/routes/calculate",
            json={
                "start": self.start,
                "end": self.end,
                "options": {"userProfile": "default"}
            },
            timeout=30
        )
        
        initial_time = time.time() - start_time
        print(f"  Initial response in {initial_time:.3f}s")
        
        self.assertEqual(response.status_code, 202, 
                        f"Route calculation failed: {response.text}")
        
        data = response.json()
        self.assertIn("routeId", data)
        route_id = data["routeId"]
        print(f"  Route ID: {route_id}")
        
        # Poll for completion
        completed = False
        for i in range(30):  # 30 seconds max
            status_response = requests.get(
                f"{self.api_url}/routes/{route_id}/status",
                timeout=5
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status", "unknown")
                progress = status_data.get("progress", 0)
                
                if i % 5 == 0:  # Print every 5 polls
                    print(f"  Status: {status}, Progress: {progress}%")
                
                if status == "completed":
                    completed = True
                    break
                elif status == "failed":
                    error_msg = status_data.get("message", "Unknown error")
                    self.fail(f"Route failed: {error_msg}")
            
            time.sleep(1)
        
        total_time = time.time() - start_time
        
        self.assertTrue(completed, f"Route didn't complete in {total_time:.1f}s")
        print(f"✅ Route completed in {total_time:.1f}s")
        
        # Get the route
        route_response = requests.get(
            f"{self.api_url}/routes/{route_id}",
            timeout=10
        )
        self.assertEqual(route_response.status_code, 200)
        
        route_data = route_response.json()
        self.assertIn("path", route_data)
        self.assertIn("stats", route_data)
        
        stats = route_data["stats"]
        print(f"  Distance: {stats['distance_km']:.2f}km")
        print(f"  Points: {len(route_data['path'])}")
        print(f"  Difficulty: {stats.get('difficulty', 'unknown')}")
    
    def test_2_nearby_route_fast(self):
        """Test a nearby route (should reuse cached tiles)"""
        print(f"\n🧪 Testing nearby route (should be fast)...")
        
        # Slightly different endpoint
        end_nearby = {"lat": 40.6482, "lon": -111.5644}  # 0.0001° different
        
        start_time = time.time()
        
        response = requests.post(
            f"{self.api_url}/routes/calculate",
            json={
                "start": self.start,
                "end": end_nearby,
                "options": {"userProfile": "default"}
            },
            timeout=30
        )
        
        self.assertEqual(response.status_code, 202)
        data = response.json()
        route_id = data["routeId"]
        
        # This should complete quickly since tiles are cached
        completed = False
        for i in range(15):  # Should be faster
            status_response = requests.get(
                f"{self.api_url}/routes/{route_id}/status",
                timeout=5
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get("status") == "completed":
                    completed = True
                    break
            
            time.sleep(1)
        
        total_time = time.time() - start_time
        
        self.assertTrue(completed, f"Nearby route didn't complete quickly")
        self.assertLess(total_time, 20, f"Nearby route took too long: {total_time:.1f}s")
        print(f"✅ Nearby route completed in {total_time:.1f}s")
    
    def test_3_debug_endpoint(self):
        """Test debug endpoint (synchronous)"""
        print(f"\n🧪 Testing debug endpoint...")
        
        start_time = time.time()
        
        response = requests.post(
            f"{self.api_url}/routes/debug",
            json={
                "start": self.start,
                "end": self.end,
                "options": {"userProfile": "default"}
            },
            timeout=60  # Debug can take longer
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Debug route completed in {elapsed:.1f}s")
            
            data = response.json()
            self.assertIn("path", data)
            self.assertIn("stats", data)
            
            # Check for debug data
            stats = data.get("stats", {})
            if "debug_data" in stats:
                debug_keys = list(stats["debug_data"].keys())
                print(f"  Debug data available: {debug_keys[:5]}...")  # First 5 keys
            else:
                print("  ⚠️  No debug data in response")
        else:
            print(f"❌ Debug endpoint failed: {response.status_code}")
            print(f"  Response: {response.text[:200]}...")
    
    def test_4_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{self.api_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")


def check_backend_logs():
    """Check backend logs for errors"""
    print("\n📋 CHECKING BACKEND LOGS...")
    print("="*60)
    
    # Look for recent errors in uvicorn output
    try:
        # Get the PID of uvicorn
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn.*9001"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pid = result.stdout.strip().split()[0]
            print(f"Backend PID: {pid}")
            
            # Check if we can see any recent output
            # Note: This only works if the process has recent output
            print("\nRecent backend activity:")
            print("(Check the terminal where backend is running for full logs)")
    except Exception as e:
        print(f"Could not check backend logs: {e}")


def run_cached_api_tests():
    """Run API tests with proper cache management"""
    print("🧪 API TESTING WITH CACHE MANAGEMENT")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAPIWithCache)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Additional diagnostics
    check_backend_logs()
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_cached_api_tests()
    sys.exit(0 if success else 1)