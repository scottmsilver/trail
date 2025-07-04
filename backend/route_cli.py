#!/usr/bin/env python3
"""
Command-line tool for finding routes between coordinates.

Usage:
    # Using local libraries (default):
    python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
    
    # Using API service:
    python route_cli.py --api "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
    
    # With custom API URL:
    python route_cli.py --api --api-url http://localhost:8000 "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix for aiohttp compatibility issue
import aiohttp
if not hasattr(aiohttp, 'ClientConnectorDNSError'):
    aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError

import time
import re
import argparse
import requests
import json


def get_verbose_dem_cache_class():
    """Get VerboseDEMTileCache class after imports are done"""
    from app.services.dem_tile_cache import DEMTileCache
    
    class VerboseDEMTileCache(DEMTileCache):
        """DEMTileCache wrapper that provides detailed progress updates"""
        
        def __init__(self, *args, **kwargs):
            # Extract optimization config if provided
            self.optimization_config = kwargs.pop('optimization_config', None)
            super().__init__(*args, **kwargs)
        # Skip precomputed cache loading to use tiled cache system
        # self._load_precomputed_caches()
        print("   Using tiled cache system (precomputed cache disabled)")
    
    def _load_precomputed_caches(self):
        """Load any precomputed cost surfaces from disk"""
        import pickle
        import glob
        
        # Use absolute path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "precomputed_cache")
        
        if not os.path.exists(cache_dir):
            print(f"   No precomputed cache directory found at: {cache_dir}")
            return
            
        cache_files = glob.glob(f"{cache_dir}/*_cost.pkl")
        print(f"   Found {len(cache_files)} precomputed cache files in {cache_dir}")
        for cache_file in cache_files:
            try:
                with TimedStep(f"Loading precomputed cache: {os.path.basename(cache_file)}"):
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Extract cache key from filename
                    cache_key = os.path.basename(cache_file).replace('.pkl', '')
                    
                    # Store in cost surface cache
                    self.cost_surface_cache[cache_key] = cache_data
                    
                    # Also store terrain data if available
                    terrain_key = cache_key.replace('_cost', '')
                    if 'dem' in cache_data and 'out_trans' in cache_data and 'crs' in cache_data:
                        self.terrain_cache[terrain_key] = (
                            cache_data['dem'], 
                            cache_data['out_trans'], 
                            cache_data['crs']
                        )
                    
                    print(f"   Loaded cost surface shape: {cache_data['cost_surface'].shape}")
            except Exception as e:
                print(f"   Failed to load {cache_file}: {str(e)}")
    
    def find_route_verbose(self, lat1, lon1, lat2, lon2):
        """Find route with detailed progress reporting"""
        
        # Define Area of Interest
        with TimedStep("Defining area of interest"):
            min_lat, max_lat, min_lon, max_lon = self.define_area_of_interest(lat1, lon1, lat2, lon2)
            print(f"   Area: {min_lat:.4f},{max_lat:.4f} to {min_lon:.4f},{max_lon:.4f}")
        
        # Create cache key for this area
        cache_key = f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}"
        
        # Check if we have cached terrain data
        if cache_key in self.terrain_cache:
            with TimedStep("Loading cached terrain data"):
                dem, out_trans, crs = self.terrain_cache[cache_key]
                print(f"   Terrain shape: {dem.shape}")
        else:
            # Download and process terrain
            with TimedStep("Downloading DEM elevation data"):
                dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
                if not dem_file:
                    print("   ✗ Failed to download DEM data")
                    return None
                print(f"   Downloaded: {dem_file}")
            
            with TimedStep("Reading and reprojecting DEM data"):
                dem, out_trans, crs = self.read_dem(dem_file)
                if dem is None:
                    return None
                dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)
                print(f"   DEM shape: {dem.shape}")
                
            # Cache the terrain data
            self.terrain_cache[cache_key] = (dem, out_trans, crs)
        
        # Check for cached cost surface
        cost_cache_key = f"{cache_key}_cost"
        if cost_cache_key in self.cost_surface_cache:
            with TimedStep("Loading cached cost surface"):
                cached_data = self.cost_surface_cache[cost_cache_key]
                cost_surface = cached_data['cost_surface']
                indices = cached_data['indices']
                slope_degrees = cached_data['slope_degrees']
                obstacle_mask = cached_data.get('obstacle_mask')
                path_raster = cached_data.get('path_raster')
                import numpy as np
                print(f"   Cost surface shape: {cost_surface.shape}")
                print(f"   Cost stats: min={np.min(cost_surface):.2f}, max={np.max(cost_surface):.2f}, mean={np.mean(cost_surface):.2f}")
                impassable_pct = np.sum(cost_surface > 1000) / cost_surface.size * 100
                print(f"   Impassable cells (cost > 1000): {impassable_pct:.1f}%")
        else:
            # Fetch obstacles and paths
            with TimedStep("Fetching obstacle data from OpenStreetMap"):
                obstacles = self.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
                print(f"   Found {len(obstacles)} obstacles")
                
            with TimedStep("Rasterizing obstacles to grid"):
                obstacle_mask = self.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)
                import numpy as np
                obstacle_count = np.sum(obstacle_mask)
                print(f"   Obstacle cells: {obstacle_count} ({obstacle_count/obstacle_mask.size*100:.1f}% of area)")
            
            with TimedStep("Fetching preferred paths from OpenStreetMap"):
                paths = self.fetch_paths(min_lat, max_lat, min_lon, max_lon)
                print(f"   Found {len(paths)} path segments")
                
            with TimedStep("Rasterizing paths to grid"):
                path_raster, path_types = self.rasterize_paths(paths, out_trans, dem.shape, crs)
                path_count = np.sum(path_raster > 0)
                print(f"   Path cells: {path_count} ({path_count/path_raster.size*100:.1f}% of area)")
            
            # Compute cost surface
            with TimedStep("Computing cost surface"):
                cost_surface, slope_degrees = self.compute_cost_surface(dem, out_trans, obstacle_mask, path_raster, path_types)
                import numpy as np
                print(f"   Cost stats: min={np.min(cost_surface):.2f}, max={np.max(cost_surface):.2f}, mean={np.mean(cost_surface):.2f}")
                impassable_pct = np.sum(cost_surface > 1000) / cost_surface.size * 100
                print(f"   Impassable cells (cost > 1000): {impassable_pct:.1f}%")
            
            with TimedStep("Building spatial indices"):
                indices = self.build_indices(cost_surface)
                print(f"   Indices built for {len(indices.flatten())} cells")
            
            # Cache the cost surface
            self.cost_surface_cache[cost_cache_key] = {
                'cost_surface': cost_surface,
                'indices': indices,
                'slope_degrees': slope_degrees,
                'obstacle_mask': obstacle_mask,
                'path_raster': path_raster
            }
        
        # Get start and end indices
        with TimedStep("Converting coordinates to grid indices"):
            start_idx, end_idx, transformer = self.get_indices(lat1, lon1, lat2, lon2, out_trans, crs, indices)
            if start_idx is None or end_idx is None:
                print("   ✗ Coordinates out of bounds")
                return None
            print(f"   Start index: {start_idx}, End index: {end_idx}")
            # Calculate grid distance
            height, width = indices.shape
            start_row, start_col = np.unravel_index(start_idx, (height, width))
            end_row, end_col = np.unravel_index(end_idx, (height, width))
            grid_distance = abs(end_row - start_row) + abs(end_col - start_col)
            print(f"   Grid distance: {grid_distance} cells (Manhattan distance)")
        
        # Run pathfinding
        with TimedStep("Running pathfinding algorithm"):
            # Use optimization config from instance or default
            optimization_config = self.optimization_config or {
                'early_termination': True,
                'stagnation_limit': 10000,
                'dynamic_weights': False,
                'corner_cutting': False
            }
            
            # Check if we should use bidirectional A*
            algorithm = optimization_config.get('algorithm', 'standard')
            print(f"   Using algorithm: {algorithm}")
            print(f"   Using optimization config: {optimization_config}")
            
            if algorithm == 'bidirectional':
                path_coords = self.bidirectional_astar(
                    cost_surface, indices, start_idx, end_idx, 
                    out_trans, transformer, dem, optimization_config
                )
            else:
                path_coords = self.astar_pathfinding_optimized(
                    cost_surface, indices, start_idx, end_idx, 
                    out_trans, transformer, dem, optimization_config
                )
            
            if path_coords:
                print(f"   Path found with {len(path_coords)} points")
            else:
                print(f"   No path found")
                
        return path_coords
    
    return VerboseDEMTileCache


class TimedStep:
    """Context manager for timing individual steps"""
    def __init__(self, description):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        print(f"\n📍 {self.description}...")
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            print(f"   ✓ Completed in {format_time(duration)}")
        else:
            print(f"   ✗ Failed after {format_time(duration)}")


def parse_coordinate(coord_str):
    """Parse coordinate string like 'Start: 40.6572, -111.5706' or 'End: 40.6486, -111.5639'"""
    # Remove 'Start:' or 'End:' prefix and parse the numbers
    coord_str = coord_str.replace('Start:', '').replace('End:', '').strip()
    
    # Extract lat, lon using regex
    match = re.match(r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$', coord_str)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon
    else:
        raise ValueError(f"Invalid coordinate format: {coord_str}")


def format_time(seconds):
    """Format time in human-readable way"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def find_route_via_api(start_lat, start_lon, end_lat, end_lon, api_url="http://localhost:9001"):
    """Find route using the API service"""
    
    # Prepare request payload
    payload = {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon}
    }
    
    # Step 1: Start route calculation
    with TimedStep("Starting route calculation via API"):
        try:
            response = requests.post(
                f"{api_url}/api/routes/calculate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5  # 5 second timeout for initial connection
            )
            response.raise_for_status()
            result = response.json()
            route_id = result["routeId"]
            print(f"   Route ID: {route_id}")
        except requests.exceptions.ConnectionError:
            print(f"   ✗ Cannot connect to API at {api_url}")
            print(f"   Make sure the API server is running (cd frontend && npm run dev)")
            return None
        except requests.exceptions.Timeout:
            print(f"   ✗ API request timed out - server may not be running at {api_url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"   ✗ API request failed: {e}")
            return None
    
    # Step 2: Poll for status
    with TimedStep("Waiting for route calculation"):
        max_attempts = 60  # 60 seconds timeout
        for i in range(max_attempts):
            try:
                response = requests.get(f"{api_url}/api/routes/{route_id}/status")
                response.raise_for_status()
                status_data = response.json()
                
                status = status_data["status"]
                progress = status_data.get("progress", 0)
                
                # Update progress display
                print(f"\r   Progress: {progress}% - Status: {status}", end="", flush=True)
                
                if status.upper() == "COMPLETED":
                    print()  # New line after progress
                    break
                elif status.upper() == "FAILED":
                    print(f"\n   ✗ Route calculation failed: {status_data.get('message', 'Unknown error')}")
                    return None
                
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"\n   ✗ Status check failed: {e}")
                return None
        else:
            print("\n   ✗ Route calculation timed out")
            return None
    
    # Step 3: Get the final route
    with TimedStep("Retrieving route data"):
        try:
            response = requests.get(f"{api_url}/api/routes/{route_id}")
            response.raise_for_status()
            route_data = response.json()
            return route_data
        except requests.exceptions.RequestException as e:
            print(f"   ✗ Failed to retrieve route: {e}")
            return None


def main():
    """Main CLI function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Find hiking routes between coordinates using local libraries or API service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Using local libraries (default):
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
  
  # Using API service:
  python route_cli.py --api "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
  
  # With custom API URL:
  python route_cli.py --api --api-url http://myserver:8000 "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
        '''
    )
    
    parser.add_argument('start', help='Start coordinates (e.g., "Start: 40.6572, -111.5706")')
    parser.add_argument('end', help='End coordinates (e.g., "End: 40.6486, -111.5639")')
    parser.add_argument('--api', action='store_true', help='Use API service instead of local libraries')
    parser.add_argument('--api-url', default='http://localhost:9001', help='API service URL (default: http://localhost:9001)')
    
    args = parser.parse_args()
    
    overall_start = time.time()
    
    # Parse coordinates
    with TimedStep("Parsing coordinates"):
        try:
            start_lat, start_lon = parse_coordinate(args.start)
            end_lat, end_lon = parse_coordinate(args.end)
            
        except ValueError as e:
            print(f"❌ Error parsing coordinates: {e}")
            print("\nExpected format:")
            print('  "Start: 40.6572, -111.5706"')
            print('  "End: 40.6486, -111.5639"')
            sys.exit(1)
    
    # Display route info
    print("\n🏔️  TRAIL ROUTE FINDER")
    print("="*60)
    print(f"Mode: {'API Service' if args.api else 'Local Libraries'}")
    if args.api:
        print(f"API URL: {args.api_url}")
    print(f"Start: {start_lat}, {start_lon}")
    print(f"End:   {end_lat}, {end_lon}")
    print("-"*60)
    
    # Calculate straight-line distance
    with TimedStep("Calculating straight-line distance"):
        import numpy as np
        dlat = np.radians(end_lat - start_lat)
        dlon = np.radians(end_lon - start_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(start_lat)) * np.cos(np.radians(end_lat)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        straight_km = 6371 * c
        straight_miles = straight_km * 0.621371
        
        print(f"   Distance: {straight_miles:.2f} miles ({straight_km:.2f} km)")
    
    # Find route
    if args.api:
        # Use API service
        route_data = find_route_via_api(start_lat, start_lon, end_lat, end_lon, args.api_url)
        
        if route_data and route_data.get('path'):
            path = route_data['path']
            stats = route_data.get('stats', {})
            route_time = time.time() - overall_start
        else:
            path = None
            stats = {}
            route_time = time.time() - overall_start
    else:
        # Use local libraries
        try:
            # Initialize cache
            with TimedStep("Initializing terrain cache and configuration"):
                # Import libraries only when needed
                from app.services.dem_tile_cache import DEMTileCache
                from app.services.obstacle_config import ObstacleConfig
                from app.services.path_preferences import PathPreferences
                
                # Get the verbose cache class
                VerboseDEMTileCache = get_verbose_dem_cache_class()
                
                # Use the same configuration as the API for consistency
                obstacle_config = ObstacleConfig()
                path_preferences = PathPreferences()
                
                # Try dynamic weights optimization which showed 71.8x speedup in benchmark
                optimization_config = {
                    'early_termination': True,
                    'stagnation_limit': 10000,
                    'dynamic_weights': True,
                    'weight_start': 1.0,
                    'weight_end': 1.2,  # Mild dynamic weights
                    'corner_cutting': False,
                    'use_heap': True,
                    'algorithm': 'bidirectional'  # Use bidirectional A* for better performance
                }
                print(f"   Optimization config: early_termination={optimization_config['early_termination']}, stagnation_limit={optimization_config['stagnation_limit']}")
                print(f"   Buffer size: 0.02° (~2.2km) for efficient tile cache usage")
                
                cache = VerboseDEMTileCache(
                    buffer=0.02,  # Use 2km buffer for better tile cache utilization
                    obstacle_config=obstacle_config,
                    path_preferences=path_preferences,
                    debug_mode=True,  # Enable debug to see pathfinding details
                    optimization_config=optimization_config
                )
                
                # Check cache status (but handle errors)
                try:
                    cache_status = cache.get_cache_status()
                    if cache_status['terrain_cache']['count'] > 0:
                        print(f"   Using cached terrain: {cache_status['terrain_cache']['count']} tiles")
                        print(f"   Using cached cost surfaces: {cache_status['cost_surface_cache']['count']} surfaces")
                        print(f"   Total cache memory: {cache_status['total_memory_mb']:.1f} MB")
                    
                    # Also check tiled cache
                    import os
                    tile_cache_path = os.path.abspath("tile_cache/cost")
                    if os.path.exists(tile_cache_path):
                        tile_count = len([f for f in os.listdir(tile_cache_path) if f.endswith('.pkl')])
                        print(f"   Disk cache: {tile_count} tiles available at {os.path.dirname(tile_cache_path)}")
                except Exception as e:
                    # Just count the caches manually
                    print(f"   Terrain cache entries: {len(cache.terrain_cache)}")
                    print(f"   Cost surface cache entries: {len(cache.cost_surface_cache)}")
            
            # Find route
            print("\n🔍 PATHFINDING PROCESS")
            print("-"*60)
            
            # Note: The find_route method will print its own progress
            path = cache.find_route(start_lat, start_lon, end_lat, end_lon)
            
            route_time = time.time() - overall_start
            stats = {}
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Process results (common for both API and local)
    if path:
        # Calculate path distance
        path_km = 0
        for i in range(len(path) - 1):
            if isinstance(path[i], dict):
                lat1, lon1 = path[i]['lat'], path[i]['lon']
                lat2, lon2 = path[i+1]['lat'], path[i+1]['lon']
            else:
                lon1, lat1 = path[i]
                lon2, lat2 = path[i+1]
            
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            path_km += 6371 * c
        
        path_miles = path_km * 0.621371
        
        # Get stats from API or calculate for local
        if args.api:
            elevation_gain = stats.get('elevation_gain_m', 0)
            max_slope = stats.get('max_slope', 0)
        else:
            # Calculate elevation gain if available
            elevation_gain = 0
            max_slope = 0
            if isinstance(path[0], dict) and 'elevation' in path[0]:
                for i in range(1, len(path)):
                    if 'elevation' in path[i] and path[i]['elevation'] is not None:
                        # Check if previous elevation is also valid
                        if 'elevation' in path[i-1] and path[i-1]['elevation'] is not None:
                            gain = path[i]['elevation'] - path[i-1]['elevation']
                            if gain > 0:
                                elevation_gain += gain
                        if 'slope' in path[i]:
                            max_slope = max(max_slope, abs(path[i]['slope']))
        
        # Results
        print(f"\n✅ ROUTE FOUND!")
        print("="*60)
        
        with TimedStep("Processing route statistics"):
            print(f"   Path distance:    {path_miles:.2f} miles ({path_km:.2f} km)")
            print(f"   vs straight line: {(path_miles/straight_miles - 1)*100:+.1f}%")
            print(f"   Path points:      {len(path)}")
            if elevation_gain > 0:
                print(f"   Elevation gain:   {elevation_gain:.0f}m ({elevation_gain*3.28084:.0f}ft)")
                print(f"   Max slope:        {max_slope:.1f}°")
        
        # Waypoint samples
        print(f"\n📍 Sample waypoints:")
        indices = [0, len(path)//4, len(path)//2, 3*len(path)//4, len(path)-1]
        for i, idx in enumerate(indices):
            if idx < len(path):
                point = path[idx]
                if isinstance(point, dict):
                    lat, lon = point['lat'], point['lon']
                    elev = point.get('elevation', 'N/A')
                    if elev != 'N/A':
                        print(f"  {i+1}. {lat:.6f}, {lon:.6f} (elev: {elev:.0f}m)")
                    else:
                        print(f"  {i+1}. {lat:.6f}, {lon:.6f}")
                else:
                    lon, lat = point
                    print(f"  {i+1}. {lat:.6f}, {lon:.6f}")
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY")
        print("-"*60)
        print(f"  Total time:       {format_time(route_time)}")
        print(f"  Points/second:    {len(path)/route_time:.0f}")
        
        # Cache info (only for local mode)
        if not args.api:
            print(f"\n💾 Final cache status:")
            print(f"  Terrain tiles:    {len(cache.terrain_cache)}")
            print(f"  Cost surfaces:    {len(cache.cost_surface_cache)}")
            
            # Calculate approximate memory usage
            total_memory_mb = 0
            for key, (dem, _, _) in cache.terrain_cache.items():
                total_memory_mb += dem.nbytes / (1024 * 1024)
            for key, data in cache.cost_surface_cache.items():
                if isinstance(data, dict) and 'cost_surface' in data:
                    total_memory_mb += data['cost_surface'].nbytes / (1024 * 1024)
            print(f"  Memory used:      ~{total_memory_mb:.1f} MB")
        
    else:
        print(f"\n❌ No route found (total time: {format_time(route_time)})")
    
    print("\n" + "="*60)
    print(f"✓ Complete! Total time: {format_time(time.time() - overall_start)}")


if __name__ == "__main__":
    main()