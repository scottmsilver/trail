#!/usr/bin/env python
import asyncio
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate
import json

async def test_debug_path():
    # Create debug-enabled service
    debug_service = TrailFinderService(debug_mode=True)
    
    # Use the problematic coordinates from the user
    start = Coordinate(lat=40.6465, lon=-111.5754)
    end = Coordinate(lat=40.6554, lon=-111.5715)
    
    print(f"Testing route from {start.lat},{start.lon} to {end.lat},{end.lon}")
    
    path, stats = await debug_service.find_route(start, end, {})
    
    if path:
        print(f"Found path with {len(path)} points")
        print(f"First few path points: {path[:3]}")
        
        if "debug_data" in stats and stats["debug_data"]:
            debug_data = stats["debug_data"]
            grid = debug_data["grid_exploration"]
            
            # Check in_path data
            in_path_count = sum(sum(row) for row in grid["in_path"])
            print(f"In-path cells: {in_path_count}")
            
            # Find in_path indices
            for i, row in enumerate(grid["in_path"]):
                for j, val in enumerate(row):
                    if val:
                        print(f"  In-path cell at ({i}, {j})")
                        if len([1 for r in grid["in_path"] for v in r if v]) >= 5:
                            break
                if len([1 for r in grid["in_path"] for v in r if v]) >= 5:
                    break
            
            # Save debug data to file for inspection
            with open('debug_output.json', 'w') as f:
                # Convert to smaller format for inspection
                small_debug = {
                    "grid_shape": grid["shape"],
                    "explored_count": sum(sum(row) for row in grid["explored"]),
                    "in_path_count": in_path_count,
                    "first_5_explored": [(i, j) for i, row in enumerate(grid["explored"]) for j, val in enumerate(row) if val][:5],
                    "first_5_in_path": [(i, j) for i, row in enumerate(grid["in_path"]) for j, val in enumerate(row) if val][:5]
                }
                json.dump(small_debug, f, indent=2)
                print("\nDebug data saved to debug_output.json")
    else:
        print("No path found")
        print(f"Error: {stats.get('error', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(test_debug_path())