#!/usr/bin/env python
"""Test the specific failing route through the API"""
import requests
import time
import json
import pytest

@pytest.mark.integration
def test_route_api():
    """Test the route that reportedly fails"""
    # User's coordinates
    start_lat, start_lon = 40.6482, -111.5738
    end_lat, end_lon = 40.6464, -111.5729
    
    print(f"Testing route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
    print("Distance: ~200m\n")
    
    # Test with different profiles
    profiles = ["default", "easy", "experienced", "trail_runner", "accessibility"]
    
    results = {}
    
    for profile in profiles:
        print(f"Testing profile: {profile}")
        
        # Make request
        url = "http://localhost:9001/api/routes/calculate"
        data = {
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "options": {"userProfile": profile}
        }
        
        try:
            response = requests.post(url, json=data)
            
            if response.status_code == 202:
                result = response.json()
                route_id = result['routeId']
                
                # Poll for result
                max_attempts = 10
                for i in range(max_attempts):
                    status_url = f"http://localhost:9001/api/routes/{route_id}/status"
                    status_response = requests.get(status_url)
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        
                        if status_data['status'] == 'completed':
                            # Get full route
                            route_url = f"http://localhost:9001/api/routes/{route_id}"
                            route_response = requests.get(route_url)
                            
                            if route_response.status_code == 200:
                                route_data = route_response.json()
                                if route_data.get('route'):
                                    print(f"  ✓ SUCCESS: {len(route_data['route'])} points")
                                    results[profile] = True
                                else:
                                    print(f"  ✗ No route in response")
                                    results[profile] = False
                                break
                            else:
                                print(f"  ✗ Error getting route: {route_response.status_code}")
                                results[profile] = False
                                break
                                
                        elif status_data['status'] == 'failed':
                            print(f"  ✗ FAILED: {status_data.get('error', 'No error message')}")
                            results[profile] = False
                            break
                    
                    time.sleep(1)
                else:
                    print(f"  ✗ Timeout waiting for route")
                    results[profile] = False
            else:
                print(f"  ✗ Error starting calculation: {response.status_code}")
                results[profile] = False
                
        except Exception as e:
            print(f"  ✗ Request error: {e}")
            results[profile] = False
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    failed_profiles = [p for p, success in results.items() if not success]
    
    if not failed_profiles:
        print("✓ All profiles successfully found routes!")
        print("\nThe issue might have been:")
        print("- Temporary DEM download failure")
        print("- Different coordinates than reported")
        print("- Issue has been resolved by configuration changes")
    else:
        print(f"✗ Failed profiles: {', '.join(failed_profiles)}")
        print("\nPossible causes:")
        for profile in failed_profiles:
            if profile == "accessibility":
                print(f"- {profile}: Terrain might be too steep (>10° slope)")
            else:
                print(f"- {profile}: Check obstacle configuration")

if __name__ == "__main__":
    test_route_api()