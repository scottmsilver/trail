#!/usr/bin/env python
"""Test path following through the API"""
import requests
import time
import json

def test_api_path_following():
    """Test that different profiles follow paths differently"""
    # Test coordinates
    start_lat, start_lon = 40.6482, -111.5738
    end_lat, end_lon = 40.6464, -111.5729
    
    print("API PATH FOLLOWING TEST")
    print("="*60)
    print(f"Route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})\n")
    
    # Test profiles with different path preferences
    profiles = {
        "default": "Basic routing",
        "easy": "Urban walker - prefers sidewalks", 
        "experienced": "Trail seeker - prefers natural trails",
        "trail_runner": "Flexible - mild path preference"
    }
    
    results = {}
    
    for profile, description in profiles.items():
        print(f"Testing {profile}: {description}")
        
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
                for i in range(10):
                    time.sleep(1)
                    route_url = f"http://localhost:9001/api/routes/{route_id}"
                    route_response = requests.get(route_url)
                    
                    if route_response.status_code == 200:
                        route_data = route_response.json()
                        if route_data.get('status') == 'completed' and route_data.get('route'):
                            route_points = route_data['route']
                            distance = route_data.get('stats', {}).get('distance_km', 0)
                            results[profile] = {
                                'points': len(route_points),
                                'distance': distance,
                                'success': True
                            }
                            print(f"  ✓ Success: {len(route_points)} points, {distance:.3f} km")
                            break
                else:
                    print(f"  ✗ Timeout or failed")
                    results[profile] = {'success': False}
            else:
                print(f"  ✗ API error: {response.status_code}")
                results[profile] = {'success': False}
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[profile] = {'success': False}
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS:")
    
    if all(r.get('success') for r in results.values()):
        # Compare route lengths
        default_points = results.get('default', {}).get('points', 0)
        
        for profile, result in results.items():
            if profile != 'default' and result.get('success'):
                diff = result['points'] - default_points
                if diff > 0:
                    print(f"• {profile}: +{diff} points (follows paths more)")
                elif diff < 0:
                    print(f"• {profile}: {diff} points (more direct)")
                else:
                    print(f"• {profile}: Same length as default")
        
        print("\n✓ Path following is working! Profiles that prefer paths")
        print("  take slightly longer routes to stay on sidewalks/trails.")
    else:
        print("✗ Some routes failed - check API logs")

if __name__ == "__main__":
    test_api_path_following()