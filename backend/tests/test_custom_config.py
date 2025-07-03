#!/usr/bin/env python
"""Test custom slope and path configurations"""
import requests
import json
import time

def test_custom_configurations():
    """Test that custom slope and path costs work through the API"""
    # Test coordinates with some elevation
    start_lat, start_lon = 40.6482, -111.5738
    end_lat, end_lon = 40.6464, -111.5729
    
    print("CUSTOM CONFIGURATION TEST")
    print("="*60)
    print(f"Route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})\n")
    
    base_url = "http://localhost:9001/api/routes/calculate"
    
    # Test 1: Default configuration
    print("1. DEFAULT CONFIGURATION:")
    default_request = {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
        "options": {"userProfile": "default"}
    }
    
    route_id = send_route_request(base_url, default_request)
    if route_id:
        result = wait_for_route(route_id)
        if result and result.get('path'):
            print(f"   ✓ Success: {len(result['path'])} points")
        else:
            print("   ✗ Failed to get route")
    
    # Test 2: Custom slope configuration - very strict
    print("\n2. CUSTOM SLOPE - STRICT (max 5° slope):")
    strict_slope_request = {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
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
    
    route_id = send_route_request(base_url, strict_slope_request)
    if route_id:
        result = wait_for_route(route_id)
        if result:
            print(f"   ✓ Success: {len(result['path'])} points")
        else:
            print("   ✗ No route found (terrain too steep)")
    
    # Test 3: Custom slope configuration - relaxed
    print("\n3. CUSTOM SLOPE - RELAXED (gentle penalty):")
    relaxed_slope_request = {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
        "options": {
            "userProfile": "default",
            "customSlopeCosts": [
                {"slope_degrees": 0, "cost_multiplier": 1.0},
                {"slope_degrees": 10, "cost_multiplier": 1.2},
                {"slope_degrees": 20, "cost_multiplier": 1.5},
                {"slope_degrees": 30, "cost_multiplier": 2.0},
                {"slope_degrees": 45, "cost_multiplier": 5.0}
            ]
        }
    }
    
    route_id = send_route_request(base_url, relaxed_slope_request)
    if route_id:
        result = wait_for_route(route_id)
        if result:
            print(f"   ✓ Success: {len(result['path'])} points")
    
    # Test 4: Custom path costs
    print("\n4. CUSTOM PATH COSTS (strong trail preference):")
    trail_preference_request = {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
        "options": {
            "userProfile": "default",
            "customPathCosts": {
                "trail": 0.1,        # Very strong preference for trails
                "footway": 0.2,      # Strong preference for footways
                "residential": 0.9,   # Slight preference for streets
                "off_path": 3.0      # Strong penalty for off-path
            }
        }
    }
    
    route_id = send_route_request(base_url, trail_preference_request)
    if route_id:
        result = wait_for_route(route_id)
        if result:
            print(f"   ✓ Success: {len(result['path'])} points")
    
    # Test 5: Combined custom configuration
    print("\n5. COMBINED CUSTOM CONFIG (slope + paths):")
    combined_request = {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
        "options": {
            "userProfile": "default",
            "customSlopeCosts": [
                {"slope_degrees": 0, "cost_multiplier": 1.0},
                {"slope_degrees": 10, "cost_multiplier": 1.5},
                {"slope_degrees": 20, "cost_multiplier": 3.0}
            ],
            "maxSlope": 25.0,
            "customPathCosts": {
                "footway": 0.3,
                "residential": 0.5,
                "off_path": 2.0
            }
        }
    }
    
    route_id = send_route_request(base_url, combined_request)
    if route_id:
        result = wait_for_route(route_id)
        if result:
            print(f"   ✓ Success: {len(result['path'])} points")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("• Custom slope costs allow fine-tuning steepness penalties")
    print("• maxSlope parameter sets a hard limit on acceptable slopes")
    print("• Custom path costs let you prefer/avoid specific path types")
    print("• Configurations can be combined for precise control")


def send_route_request(url, data):
    """Send route request and return route ID"""
    try:
        response = requests.post(url, json=data)
        if response.status_code == 202:
            return response.json()['routeId']
        else:
            print(f"   ✗ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ✗ Request error: {e}")
        return None


def wait_for_route(route_id, max_attempts=10):
    """Wait for route to complete and return result"""
    for i in range(max_attempts):
        time.sleep(1)
        try:
            response = requests.get(f"http://localhost:9001/api/routes/{route_id}")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'completed':
                    return data
                elif data.get('status') == 'failed':
                    return None
        except:
            pass
    return None


if __name__ == "__main__":
    test_custom_configurations()