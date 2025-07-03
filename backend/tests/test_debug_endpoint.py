#!/usr/bin/env python
import requests
import json

# Test the debug endpoint directly
url = "http://localhost:8000/api/routes/debug"
data = {
    "start": {"lat": 40.6470, "lon": -111.5759},
    "end": {"lat": 40.6559, "lon": -111.5705}
}

print(f"Testing debug endpoint with: {data}")
response = requests.post(url, json=data)

print(f"Status code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Route ID: {result.get('routeId')}")
    print(f"Status: {result.get('status')}")
    print(f"Path points: {len(result.get('path', []))}")
    
    stats = result.get('stats', {})
    if 'debug_data' in stats:
        if stats['debug_data']:
            debug_data = stats['debug_data']
            print(f"Debug data present: YES")
            print(f"  - Explored nodes: {len(debug_data.get('explored_nodes', []))}")
            print(f"  - Decision points: {len(debug_data.get('decision_points', []))}")
            grid = debug_data.get('grid_exploration', {})
            if grid:
                print(f"  - Grid shape: {grid.get('shape')}")
                # Count in_path cells
                in_path = grid.get('in_path', [])
                if in_path:
                    in_path_count = sum(sum(row) if row else 0 for row in in_path)
                    print(f"  - In-path cells: {in_path_count}")
        else:
            print("Debug data present: NO (null)")
    else:
        print("Debug data field missing from stats")
    
    # Save full response for debugging
    with open('debug_endpoint_response.json', 'w') as f:
        json.dump(result, f, indent=2)
        print("\nFull response saved to debug_endpoint_response.json")
else:
    print(f"Error: {response.text}")