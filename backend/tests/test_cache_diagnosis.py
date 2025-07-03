#!/usr/bin/env python3
"""Diagnose cache miss issues in the pathfinding system"""

import json
import requests
import time

def test_cache_behavior():
    """Test the same route multiple times to see cache behavior"""
    
    # Same route coordinates
    start = {"lat": 40.6531, "lon": -111.5679}
    end = {"lat": 40.6021, "lon": -111.5148}
    
    print("Testing cache behavior with identical route requests...")
    print(f"Route: ({start['lat']}, {start['lon']}) -> ({end['lat']}, {end['lon']})")
    print("=" * 80)
    
    # Make 3 identical requests
    for i in range(3):
        print(f"\nRequest #{i+1}:")
        print("-" * 40)
        
        # Make the request
        response = requests.post(
            "http://localhost:9001/api/routes/calculate",
            json={
                "start": start,
                "end": end,
                "options": {
                    "avoidSteep": False,
                    "userProfile": "moderate_hiker"
                }
            }
        )
        
        if response.status_code != 202:
            print(f"Error: {response.status_code} - {response.text}")
            continue
            
        route_data = response.json()
        route_id = route_data["routeId"]
        print(f"Route ID: {route_id}")
        
        # Poll for completion
        start_time = time.time()
        while True:
            status_response = requests.get(f"http://localhost:9001/api/routes/{route_id}/status")
            status = status_response.json()
            
            if status["status"] == "completed":
                elapsed = time.time() - start_time
                print(f"Completed in {elapsed:.2f}s")
                
                # Get full route data
                route_response = requests.get(f"http://localhost:9001/api/routes/{route_id}")
                route = route_response.json()
                
                stats = route["stats"]
                print(f"Distance: {stats['distance_km']:.2f} km")
                print(f"Elevation gain: {stats['elevation_gain_m']:.0f} m")
                
                break
            elif status["status"] == "failed":
                print(f"Failed: {status.get('message', 'Unknown error')}")
                break
                
            time.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("Check the backend logs to see cache hit/miss patterns!")
    print("\nExpected behavior:")
    print("- Request #1: Cache misses (initial computation)")
    print("- Request #2 & #3: Cache hits (using cached data)")
    print("\nIf you see cache misses on all requests, there's a cache key issue.")

if __name__ == "__main__":
    test_cache_behavior()