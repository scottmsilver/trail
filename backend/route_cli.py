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
from app.services.gpx_generator import GPXGenerator


def get_verbose_dem_cache_class():
    """Get VerboseDEMTileCache class after imports are done"""
    from app.services.dem_tile_cache import DEMTileCache
    
    class VerboseDEMTileCache(DEMTileCache):
        """DEMTileCache wrapper that provides detailed progress updates"""
        
        def __init__(self, *args, **kwargs):
            # Extract optimization config if provided
            self.optimization_config = kwargs.pop('optimization_config', None)
            self.debug_grid = kwargs.pop('debug_grid', False)
            self.slope_viz = kwargs.pop('slope_viz', False)
            super().__init__(*args, **kwargs)
            # Skip precomputed cache loading to use tiled cache system
            # self._load_precomputed_caches()
            print("   Using tiled cache system (precomputed cache disabled)")
            if self.debug_grid:
                print("   Debug grid visualization enabled")
            if self.slope_viz:
                print("   Slope visualization enabled")
        
        def find_route(self, lat1, lon1, lat2, lon2):
            """Override find_route to use verbose version"""
            return self.find_route_verbose(lat1, lon1, lat2, lon2)
    
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
            """Find route with detailed progress reporting that uses tiled cache"""
            
            # Define Area of Interest
            with TimedStep("Defining area of interest"):
                min_lat, max_lat, min_lon, max_lon = self.define_area_of_interest(lat1, lon1, lat2, lon2)
                print(f"   Area: {min_lat:.4f},{max_lat:.4f} to {min_lon:.4f},{max_lon:.4f}")
            
            # Try to use tiled cost surface first
            with TimedStep("Checking for tiled cost surface"):
                # Get DEM data first (needed for tiled system)
                cache_key = self._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)
                
                # Check if we have cached terrain data
                if cache_key in self.terrain_cache:
                    dem, out_trans, crs = self.terrain_cache[cache_key]
                    print(f"   Using cached terrain data")
                else:
                    # Download DEM Data
                    with TimedStep("Downloading DEM elevation data"):
                        dem_file = self.download_dem(min_lat, max_lat, min_lon, max_lon)
                        if dem_file is None:
                            return None
                        print(f"   Downloaded: {dem_file}")

                    # Read DEM Data
                    with TimedStep("Reading and reprojecting DEM data"):
                        dem, out_trans, crs = self.read_dem(dem_file)
                        dem, out_trans, crs = self.reproject_dem(dem, out_trans, crs)
                        print(f"   DEM shape: {dem.shape}")
                    
                    # Cache the reprojected terrain data
                    self.terrain_cache[cache_key] = (dem, out_trans, crs)
                
                # Try tiled approach
                tiled_result = self._try_tiled_cost_surface(min_lat, max_lat, min_lon, max_lon, 
                                                            dem, out_trans, crs)
                
                if tiled_result is not None:
                    print(f"   ✓ Using precomputed tiles!")
                    cost_surface, slope_degrees, indices, dem_composed, path_raster, path_types = tiled_result
                    
                    # Use composed DEM if available
                    if dem_composed is not None:
                        dem = dem_composed
                        
                    # Update transform if composed
                    if hasattr(self, '_composed_transform'):
                        out_trans = self._composed_transform
                        crs = getattr(self, '_composed_crs', crs)
                else:
                    print(f"   Tiles not available, computing from scratch...")
                    
                    # Fallback to computing everything
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
                        path_raster, path_types, path_raw_tags = self.rasterize_paths(paths, out_trans, dem.shape, crs)
                        path_count = np.sum(path_raster > 0)
                        print(f"   Path cells: {path_count} ({path_count/path_raster.size*100:.1f}% of area)")
                    
                    # Compute cost surface
                    with TimedStep("Computing cost surface"):
                        cost_surface, slope_degrees = self.compute_cost_surface(dem, out_trans, obstacle_mask, path_raster, path_types)
                        
                    with TimedStep("Building spatial indices"):
                        indices = self.build_indices(cost_surface)
                        print(f"   Indices built for {len(indices.flatten())} cells")
            
            # Print cost surface stats
            import numpy as np
            print(f"   Cost stats: min={np.min(cost_surface):.2f}, max={np.max(cost_surface):.2f}, mean={np.mean(cost_surface):.2f}")
            impassable_pct = np.sum(cost_surface > 1000) / cost_surface.size * 100
            print(f"   Impassable cells (cost > 1000): {impassable_pct:.1f}%")
            
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
                    
                    # Show debug grid if requested
                    if self.debug_grid:
                        print("\n📊 Generating debug grid visualization...")
                        self.show_debug_grid(cost_surface, indices, start_idx, end_idx,
                                           out_trans, transformer, path_coords)
                    
                    # Show slope visualization if requested
                    if self.slope_viz:
                        print("\n🏔️ Generating slope visualization...")
                        self.show_slope_visualization(dem, cost_surface, indices, start_idx, end_idx,
                                                    out_trans, transformer, path_coords)
                else:
                    print(f"   No path found")
                    return None
            
            # Calculate slopes and path types for each segment
            with TimedStep("Calculating slopes and path types"):
                path_with_slopes = self.calculate_path_slopes(path_coords, dem, out_trans, transformer, path_raster, path_types)
                
                # Print some statistics about path types
                if path_with_slopes and len(path_with_slopes) > 0:
                    path_type_counts = {}
                    for point in path_with_slopes:
                        pt = point.get('path_type', 'unknown')
                        path_type_counts[pt] = path_type_counts.get(pt, 0) + 1
                    
                    print(f"   Path type distribution:")
                    for pt, count in sorted(path_type_counts.items()):
                        pct = (count / len(path_with_slopes)) * 100
                        print(f"     {pt}: {count} points ({pct:.1f}%)")
                    
                    # Generate elevation/slope profile graphs if visualization is requested
                    if self.debug_grid or self.slope_viz:
                        print("\n📈 Generating elevation and slope profile graphs...")
                        self.generate_profile_graphs(path_with_slopes)
                        # Generate terrain-wide slope change map
                        self.generate_terrain_slope_change_map(dem, slope_degrees, path_coords, 
                                                              out_trans, min_lat, max_lat, min_lon, max_lon)
                        
            return path_with_slopes
        
        def show_debug_grid(self, cost_surface, indices, start_idx, end_idx, 
                           out_trans, transformer, path=None):
            """Show debug grid visualization using matplotlib"""
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
            
            # Convert indices to row/col
            height, width = indices.shape
            start_row, start_col = np.unravel_index(start_idx, (height, width))
            end_row, end_col = np.unravel_index(end_idx, (height, width))
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Cost surface (linear scale)
            # Use percentile for better visualization of the range
            vmax_percentile = np.percentile(cost_surface, 99)  # 99th percentile
            im1 = ax1.imshow(cost_surface, cmap='RdYlGn_r', vmin=0, vmax=vmax_percentile)
            ax1.plot(start_col, start_row, 'gs', markersize=15, label='Start')
            ax1.plot(end_col, end_row, 'rs', markersize=15, label='End')
            
            # Add path if available
            if path:
                path_rows = []
                path_cols = []
                for point in path:
                    if isinstance(point, dict):
                        lon, lat = point['lon'], point['lat']
                    else:
                        lon, lat = point
                    # Convert to grid coordinates
                    x, y = transformer.transform(lon, lat)
                    col = int(round((x - out_trans.c) / out_trans.a))
                    row = int(round((y - out_trans.f) / out_trans.e))
                    if 0 <= row < height and 0 <= col < width:
                        path_rows.append(row)
                        path_cols.append(col)
                
                if path_rows:
                    ax1.plot(path_cols, path_rows, 'b-', linewidth=2, label='Path')
            
            ax1.set_title('Cost Surface')
            ax1.legend()
            cbar = plt.colorbar(im1, ax=ax1, label='Cost')
            # Show actual range
            cbar.ax.text(0.5, -0.1, f'Max: {vmax_percentile:.0f}', 
                        transform=cbar.ax.transAxes, ha='center')
            
            # Plot 2: Terrain features
            terrain_view = np.zeros_like(cost_surface)
            terrain_view[cost_surface > 50] = 3  # Very high cost
            terrain_view[(cost_surface > 20) & (cost_surface <= 50)] = 2  # High cost
            terrain_view[(cost_surface > 5) & (cost_surface <= 20)] = 1  # Medium cost
            
            colors = ['green', 'yellow', 'orange', 'red']
            cmap = LinearSegmentedColormap.from_list('terrain', colors, N=4)
            
            im2 = ax2.imshow(terrain_view, cmap=cmap, vmin=0, vmax=3)
            ax2.plot(start_col, start_row, 'gs', markersize=15, label='Start')
            ax2.plot(end_col, end_row, 'rs', markersize=15, label='End')
            
            if path and path_rows:
                ax2.plot(path_cols, path_rows, 'b-', linewidth=2, label='Path')
            
            ax2.set_title('Terrain Difficulty')
            ax2.legend()
            
            # Add colorbar with labels
            cbar = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(['Easy', 'Moderate', 'Hard', 'Very Hard'])
            
            plt.tight_layout()
            
            # Save the figure
            output_file = 'debug_grid.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n📊 Debug grid saved to: {output_file}")
            plt.close()
        
        def show_slope_visualization(self, dem, cost_surface, indices, start_idx, end_idx,
                                   out_trans, transformer, path=None):
            """Generate slope visualization map"""
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            
            height, width = dem.shape
            
            # Calculate slopes in degrees
            # Gradient in x and y directions
            cell_size = abs(out_trans.a)  # Cell size in meters (assuming projected coordinates)
            gy, gx = np.gradient(dem, cell_size)
            
            # Calculate slope in degrees
            slope_radians = np.arctan(np.sqrt(gx**2 + gy**2))
            slope_degrees = np.degrees(slope_radians)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Define slope categories and colors
            # Green (0-5°), Yellow (5-15°), Orange (15-30°), Red (30+°)
            colors = ['#2d7f2d', '#7fbf2d', '#ffff00', '#ff7f00', '#ff0000']
            boundaries = [0, 5, 15, 25, 35, 90]
            cmap = LinearSegmentedColormap.from_list('slope', colors, N=len(colors))
            
            # Plot slope map
            im = ax.imshow(slope_degrees, cmap=cmap, vmin=0, vmax=45, origin='upper')
            
            # Add start and end points
            start_row, start_col = np.unravel_index(start_idx, indices.shape)
            end_row, end_col = np.unravel_index(end_idx, indices.shape)
            ax.plot(start_col, start_row, 'gs', markersize=15, label='Start', 
                   markeredgecolor='black', markeredgewidth=2)
            ax.plot(end_col, end_row, 'rs', markersize=15, label='End',
                   markeredgecolor='black', markeredgewidth=2)
            
            # Add path if available
            if path:
                path_rows = []
                path_cols = []
                for point in path:
                    if isinstance(point, dict):
                        lon, lat = point['lon'], point['lat']
                    else:
                        lon, lat = point
                    # Convert to grid coordinates
                    x, y = transformer.transform(lon, lat)
                    col = int(round((x - out_trans.c) / out_trans.a))
                    row = int(round((y - out_trans.f) / out_trans.e))
                    if 0 <= row < height and 0 <= col < width:
                        path_rows.append(row)
                        path_cols.append(col)
                
                if path_rows:
                    ax.plot(path_cols, path_rows, 'b-', linewidth=3, label='Route',
                           markeredgecolor='white', markeredgewidth=1)
            
            ax.set_title('Terrain Slope Map', fontsize=16)
            ax.legend(loc='upper right')
            
            # Add colorbar with custom labels
            cbar = plt.colorbar(im, ax=ax, label='Slope (degrees)', 
                               boundaries=boundaries, ticks=[0, 5, 15, 25, 35, 45])
            cbar.ax.set_yticklabels(['0°\n(Flat)', '5°\n(Gentle)', '15°\n(Moderate)', 
                                     '25°\n(Steep)', '35°\n(Very Steep)', '45°+'])
            
            # Add grid
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            
            plt.tight_layout()
            
            # Save the figure
            output_file = 'slope_visualization.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"   ✓ Slope visualization saved to: {output_file}")
            plt.close()
        
        def generate_profile_graphs(self, path_with_slopes):
            """Generate elevation, slope, and slope change profile graphs"""
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not path_with_slopes or len(path_with_slopes) < 2:
                print("   ⚠️  Not enough path points for profile graphs")
                return
            
            # Extract data from path
            distances = []
            elevations = []
            slopes = []
            total_distance = 0
            
            for i, point in enumerate(path_with_slopes):
                if i == 0:
                    distances.append(0)
                else:
                    # Calculate distance from previous point
                    prev_point = path_with_slopes[i-1]
                    lat1, lon1 = point['lat'], point['lon']
                    lat2, lon2 = prev_point['lat'], prev_point['lon']
                    
                    # Haversine distance
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    segment_distance = 6371000 * c  # Distance in meters
                    
                    total_distance += segment_distance
                    distances.append(total_distance)
                
                # Collect elevation and slope
                elevation = point.get('elevation')
                if elevation is None or elevation == 'N/A':
                    elevation = 0
                elevations.append(float(elevation))
                
                slope = point.get('slope', 0)
                if slope is None or slope == 'N/A':
                    slope = 0
                slopes.append(abs(float(slope)))  # Use absolute slope
            
            # Convert distances to kilometers
            distances_km = [d / 1000 for d in distances]
            
            # Calculate slope change (first derivative of slope)
            if len(slopes) > 2:
                slope_changes = np.gradient(slopes, distances)
            else:
                slope_changes = [0] * len(slopes)
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot 1: Elevation Profile
            ax1.plot(distances_km, elevations, 'b-', linewidth=2)
            ax1.fill_between(distances_km, elevations, alpha=0.3)
            ax1.set_ylabel('Elevation (m)', fontsize=12)
            ax1.set_title('Route Profile Analysis', fontsize=16)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=min(elevations) - 50 if elevations else 0)
            
            # Add elevation statistics
            if elevations:
                elev_gain = sum(max(0, elevations[i] - elevations[i-1]) 
                               for i in range(1, len(elevations)))
                elev_loss = sum(max(0, elevations[i-1] - elevations[i]) 
                               for i in range(1, len(elevations)))
                ax1.text(0.02, 0.95, f'Gain: {elev_gain:.0f}m | Loss: {elev_loss:.0f}m', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 2: Slope Profile
            ax2.plot(distances_km, slopes, 'r-', linewidth=2)
            ax2.fill_between(distances_km, slopes, alpha=0.3, color='red')
            ax2.set_ylabel('Slope (degrees)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(bottom=0)
            
            # Add slope zones
            ax2.axhspan(0, 5, alpha=0.2, color='green', label='Easy (0-5°)')
            ax2.axhspan(5, 15, alpha=0.2, color='yellow', label='Moderate (5-15°)')
            ax2.axhspan(15, 25, alpha=0.2, color='orange', label='Steep (15-25°)')
            if max(slopes) > 25:
                ax2.axhspan(25, max(slopes) + 5, alpha=0.2, color='red', label='Very Steep (25°+)')
            
            # Add slope statistics
            avg_slope = np.mean(slopes) if slopes else 0
            max_slope = max(slopes) if slopes else 0
            ax2.text(0.02, 0.95, f'Avg: {avg_slope:.1f}° | Max: {max_slope:.1f}°', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 3: Net Slope Change Between Samples (color-coded)
            # Calculate net slope changes between consecutive points
            net_slope_changes = []
            for i in range(1, len(slopes)):
                net_change = slopes[i] - slopes[i-1]
                net_slope_changes.append(net_change)
            
            # Add a dummy value at the start to match array lengths
            net_slope_changes = [0] + net_slope_changes
            
            # Create color map for slope changes
            # Red for getting steeper, green for getting gentler
            colors = []
            for change in net_slope_changes:
                if change > 2:  # Getting much steeper
                    colors.append('#8B0000')  # Dark red
                elif change > 0.5:  # Getting steeper
                    colors.append('#FF0000')  # Red
                elif change > -0.5:  # Relatively constant
                    colors.append('#FFD700')  # Gold
                elif change > -2:  # Getting gentler
                    colors.append('#00FF00')  # Green
                else:  # Getting much gentler
                    colors.append('#006400')  # Dark green
            
            # Plot as colored bars
            for i in range(len(distances_km) - 1):
                ax3.fill_between(distances_km[i:i+2], 0, net_slope_changes[i:i+2], 
                               color=colors[i], alpha=0.7, edgecolor='none')
            
            ax3.set_ylabel('Net Slope Change\n(degrees)', fontsize=12)
            ax3.set_xlabel('Distance (km)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
            
            # Add color legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#8B0000', label='Much Steeper (>2°)'),
                Patch(facecolor='#FF0000', label='Steeper (0.5-2°)'),
                Patch(facecolor='#FFD700', label='Constant (±0.5°)'),
                Patch(facecolor='#00FF00', label='Gentler (-0.5 to -2°)'),
                Patch(facecolor='#006400', label='Much Gentler (<-2°)')
            ]
            ax3.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
            
            # Add statistics
            steeper_count = sum(1 for c in net_slope_changes if c > 0.5)
            gentler_count = sum(1 for c in net_slope_changes if c < -0.5)
            ax3.text(0.02, 0.95, f'Getting steeper: {steeper_count} points | Getting gentler: {gentler_count} points', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
            
            plt.tight_layout()
            
            # Save the figure
            output_file = 'route_elevation_profile.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"   ✓ Elevation/slope profile saved to: {output_file}")
            plt.close()
        
        def generate_terrain_slope_change_map(self, dem, slope_degrees, path_coords, out_trans, 
                                             min_lat, max_lat, min_lon, max_lon, 
                                             output_filename='terrain_slope_changes.png'):
            """Generate a map showing slope changes across the entire terrain"""
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            
            print("   Generating terrain-wide slope change map...")
            
            # Calculate slope changes (gradient of slope)
            # This shows how quickly the slope is changing
            slope_change_y, slope_change_x = np.gradient(slope_degrees)
            slope_change_magnitude = np.sqrt(slope_change_x**2 + slope_change_y**2)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create custom colormap
            # Red = rapid slope changes (terrain getting steeper/gentler quickly)
            # Yellow = moderate slope changes
            # Green = gentle/constant slope changes
            colors = ['#006400', '#00FF00', '#FFD700', '#FF8C00', '#FF0000', '#8B0000']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('slope_change', colors, N=n_bins)
            
            # Plot slope changes
            im = ax.imshow(slope_change_magnitude, cmap=cmap, origin='upper', 
                          extent=[0, dem.shape[1], dem.shape[0], 0],
                          vmin=0, vmax=np.percentile(slope_change_magnitude, 95))
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Slope Change Rate (°/pixel)')
            
            # Plot the route if provided
            if path_coords:
                path_rows = []
                path_cols = []
                for point in path_coords:
                    if isinstance(point, dict):
                        lon, lat = point['lon'], point['lat']
                    else:
                        lon, lat = point
                    # Convert to pixel coordinates
                    col = int((lon - min_lon) / (max_lon - min_lon) * dem.shape[1])
                    row = int((1 - (lat - min_lat) / (max_lat - min_lat)) * dem.shape[0])
                    if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                        path_rows.append(row)
                        path_cols.append(col)
                
                # Plot path
                ax.plot(path_cols, path_rows, 'b-', linewidth=3, label='Route')
                
                # Mark start and end
                if len(path_cols) > 0:
                    ax.plot(path_cols[0], path_rows[0], 'go', markersize=12, 
                           markeredgecolor='white', markeredgewidth=2, label='Start')
                    ax.plot(path_cols[-1], path_rows[-1], 'ro', markersize=12, 
                           markeredgecolor='white', markeredgewidth=2, label='End')
            
            # Set title and labels
            ax.set_title('Terrain Slope Change Rate\n(Red = Rapid changes, Green = Gentle/constant)', 
                        fontsize=16, pad=20)
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend if route is shown
            if path_coords:
                ax.legend(loc='upper right')
            
            # Add statistics
            stats_text = f'Slope change statistics:\n'
            stats_text += f'Min: {np.min(slope_change_magnitude):.1f}°/pixel\n'
            stats_text += f'Max: {np.max(slope_change_magnitude):.1f}°/pixel\n'
            stats_text += f'Mean: {np.mean(slope_change_magnitude):.1f}°/pixel\n'
            stats_text += f'Areas with rapid changes: {np.sum(slope_change_magnitude > 5) / slope_change_magnitude.size * 100:.1f}%'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            
            # Save
            plt.tight_layout()
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Terrain slope change map saved to: {output_filename}")
        
        def generate_slope_change_map(self, path_with_slopes, output_filename='slope_change_map.png'):
            """Generate a map showing slope changes color-coded along the route"""
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.collections import LineCollection
            import numpy as np
            import requests
            from PIL import Image
            from io import BytesIO
            import math
            
            if not path_with_slopes or len(path_with_slopes) < 2:
                print("   ⚠️  Not enough path points for slope change map")
                return
            
            # Extract coordinates and calculate slope changes
            lats = []
            lons = []
            slope_changes = []
            
            for i, point in enumerate(path_with_slopes):
                lats.append(point['lat'])
                lons.append(point['lon'])
                
                if i > 0:
                    # Calculate net slope change from previous point
                    prev_slope = path_with_slopes[i-1].get('slope', 0)
                    curr_slope = point.get('slope', 0)
                    if prev_slope == 'N/A' or prev_slope is None:
                        prev_slope = 0
                    if curr_slope == 'N/A' or curr_slope is None:
                        curr_slope = 0
                        
                    net_change = abs(float(curr_slope)) - abs(float(prev_slope))
                    slope_changes.append(net_change)
            
            # Add dummy value for first point
            slope_changes = [0] + slope_changes
            
            # Calculate bounds with padding
            padding = 0.002  # ~200m padding
            min_lat = min(lats) - padding
            max_lat = max(lats) + padding
            min_lon = min(lons) - padding
            max_lon = max(lons) + padding
            
            # Determine zoom level
            lat_diff = max_lat - min_lat
            lon_diff = max_lon - min_lon
            max_diff = max(lat_diff, lon_diff)
            
            if max_diff < 0.01:
                zoom = 16
            elif max_diff < 0.05:
                zoom = 14
            elif max_diff < 0.1:
                zoom = 13
            else:
                zoom = 12
            
            # Get tile coordinates
            def deg2num(lat_deg, lon_deg, zoom):
                lat_rad = math.radians(lat_deg)
                n = 2.0 ** zoom
                x = int((lon_deg + 180.0) / 360.0 * n)
                y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
                return (x, y)
            
            min_x, max_y = deg2num(min_lat, min_lon, zoom)
            max_x, min_y = deg2num(max_lat, max_lon, zoom)
            
            # Download and stitch tiles
            width = (max_x - min_x + 1) * 256
            height = (max_y - min_y + 1) * 256
            map_img = Image.new('RGB', (width, height))
            
            print(f"   Generating slope change map...")
            
            tile_server = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            headers = {'User-Agent': 'TrailFinderCLI/1.0'}
            
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    try:
                        url = tile_server.format(z=zoom, x=x, y=y)
                        response = requests.get(url, headers=headers, timeout=10)
                        response.raise_for_status()
                        tile = Image.open(BytesIO(response.content))
                        map_img.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
                    except:
                        gray_tile = Image.new('RGB', (256, 256), color=(200, 200, 200))
                        map_img.paste(gray_tile, ((x - min_x) * 256, (y - min_y) * 256))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Display map
            map_array = np.array(map_img)
            ax.imshow(map_array, extent=[min_x, max_x + 1, max_y + 1, min_y], aspect='auto')
            
            # Convert lat/lon to tile coordinates
            def latlon_to_tile_coords(lat, lon, zoom):
                lat_rad = math.radians(lat)
                n = 2.0 ** zoom
                x = (lon + 180.0) / 360.0 * n
                y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
                return x, y
            
            # Create line segments with colors based on slope change
            segments = []
            colors = []
            
            for i in range(len(lats) - 1):
                x1, y1 = latlon_to_tile_coords(lats[i], lons[i], zoom)
                x2, y2 = latlon_to_tile_coords(lats[i+1], lons[i+1], zoom)
                segments.append([(x1, y1), (x2, y2)])
                
                # Color based on slope change
                change = slope_changes[i+1] if i+1 < len(slope_changes) else 0
                if change > 2:  # Getting much steeper
                    colors.append('#8B0000')  # Dark red
                elif change > 0.5:  # Getting steeper
                    colors.append('#FF0000')  # Red
                elif change > -0.5:  # Relatively constant
                    colors.append('#FFD700')  # Gold
                elif change > -2:  # Getting gentler
                    colors.append('#00FF00')  # Green
                else:  # Getting much gentler
                    colors.append('#006400')  # Dark green
            
            # Create LineCollection
            lc = LineCollection(segments, colors=colors, linewidths=4, alpha=0.8)
            ax.add_collection(lc)
            
            # Add start and end markers
            start_x, start_y = latlon_to_tile_coords(lats[0], lons[0], zoom)
            end_x, end_y = latlon_to_tile_coords(lats[-1], lons[-1], zoom)
            
            ax.plot(start_x, start_y, 'go', markersize=15, markeredgecolor='white', 
                    markeredgewidth=2, label='Start', zorder=5)
            ax.plot(end_x, end_y, 'ro', markersize=15, markeredgecolor='white', 
                    markeredgewidth=2, label='End', zorder=5)
            
            # Set title and remove axes
            ax.set_title('Route Slope Changes\nRed=Getting Steeper, Green=Getting Gentler', 
                         fontsize=16, pad=20)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#8B0000', label='Much Steeper (>2°)'),
                Patch(facecolor='#FF0000', label='Steeper (0.5-2°)'),
                Patch(facecolor='#FFD700', label='Constant (±0.5°)'),
                Patch(facecolor='#00FF00', label='Gentler (-0.5 to -2°)'),
                Patch(facecolor='#006400', label='Much Gentler (<-2°)'),
                patches.Patch(color='none', label=''),  # Spacer
                ax.scatter([], [], s=150, c='g', marker='o', edgecolor='white', linewidth=2, label='Start'),
                ax.scatter([], [], s=150, c='r', marker='o', edgecolor='white', linewidth=2, label='End')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add attribution
            ax.text(0.99, 0.01, '© OpenStreetMap contributors', transform=ax.transAxes, 
                    fontsize=8, ha='right', va='bottom', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Add statistics
            steeper_count = sum(1 for c in slope_changes if c > 0.5)
            gentler_count = sum(1 for c in slope_changes if c < -0.5)
            stats_text = f'Slope changes:\nGetting steeper: {steeper_count} points\nGetting gentler: {gentler_count} points'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    fontsize=10, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            
            # Save
            plt.tight_layout()
            plt.savefig(output_filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✓ Slope change map saved to: {output_filename}")


        def generate_debug_grid_visualization(self, cost_surface, path_raster, path_types, path_raw_tags,
                                             indices, start_idx, end_idx, path_coords,
                                             out_trans, transformer, dem, 
                                             min_lat, max_lat, min_lon, max_lon):
            """Generate visualization of the pathfinding grid using matplotlib"""
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import ListedColormap, BoundaryNorm
            
            print("\n" + "="*80)
            print("DEBUG GRID VISUALIZATION")
            print("="*80)
            
            # Get grid dimensions
            height, width = cost_surface.shape
            
            print(f"Grid size: {height}x{width}")
            print(f"Resolution: ~{(max_lat - min_lat) / height * 111000:.1f} meters/pixel")
            
            # Convert start/end indices to row/col
            start_row, start_col = np.unravel_index(start_idx, indices.shape)
            end_row, end_col = np.unravel_index(end_idx, indices.shape)
            
            # Create grid for path
            path_grid = np.zeros_like(cost_surface, dtype=bool)
            if path_coords:
                for point in path_coords:
                    if isinstance(point, dict):
                        lon, lat = point['lon'], point['lat']
                    else:
                        lon, lat = point
                    x, y = transformer.transform(lon, lat)
                    col = int(round((x - out_trans.c) / out_trans.a))
                    row = int(round((y - out_trans.f) / out_trans.e))
                    if 0 <= row < height and 0 <= col < width:
                        path_grid[row, col] = True
            
            # Create explored grid from debug data
            explored_grid = None
            if hasattr(self, 'debug_data') and self.debug_data and 'grid_exploration' in self.debug_data:
                explored_grid = self.debug_data['grid_exploration']['explored']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Pathfinding Debug Visualization\nArea: ({min_lat:.6f}, {min_lon:.6f}) to ({max_lat:.6f}, {max_lon:.6f})', fontsize=16)
            
            # 1. Cost Surface Heatmap
            ax1 = axes[0, 0]
            # Use log scale for better visualization
            cost_display = np.log10(cost_surface + 1)  # Add 1 to avoid log(0)
            im1 = ax1.imshow(cost_display, cmap='RdYlGn_r', origin='upper')
            ax1.set_title('Cost Surface (log scale)')
            ax1.plot(start_col, start_row, 'go', markersize=10, label='Start')
            ax1.plot(end_col, end_row, 'ro', markersize=10, label='End')
            if path_coords:
                # Plot path
                path_cols = []
                path_rows = []
                for point in path_coords:
                    if isinstance(point, dict):
                        lon, lat = point['lon'], point['lat']
                    else:
                        lon, lat = point
                    x, y = transformer.transform(lon, lat)
                    col = int(round((x - out_trans.c) / out_trans.a))
                    row = int(round((y - out_trans.f) / out_trans.e))
                    if 0 <= row < height and 0 <= col < width:
                        path_cols.append(col)
                        path_rows.append(row)
                ax1.plot(path_cols, path_rows, 'b-', linewidth=2, label='Path')
            ax1.legend()
            plt.colorbar(im1, ax=ax1, label='log10(cost + 1)')
            
            # 2. Path Types
            ax2 = axes[0, 1]
            if path_raster is not None:
                # Create a custom colormap for path types
                unique_types = np.unique(path_raster[path_raster > 0])
                if len(unique_types) > 0:
                    # Create display array
                    path_display = np.zeros_like(path_raster, dtype=float)
                    path_display[path_raster == 0] = np.nan  # Set non-path areas to NaN
                    
                    # Map path types to colors
                    cmap = plt.cm.get_cmap('tab20', len(unique_types))
                    for i, path_id in enumerate(unique_types):
                        path_display[path_raster == path_id] = i
                    
                    im2 = ax2.imshow(path_display, cmap=cmap, origin='upper')
                    ax2.set_title('Path Types')
                    
                    # Add legend for path types
                    if path_types:
                        legend_elements = []
                        for i, path_id in enumerate(unique_types):
                            if path_id in path_types:
                                legend_elements.append(patches.Patch(
                                    color=cmap(i), 
                                    label=path_types[path_id][:20]  # Truncate long names
                                ))
                        if legend_elements:
                            ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax2.text(0.5, 0.5, 'No paths in area', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Path data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.plot(start_col, start_row, 'go', markersize=10)
            ax2.plot(end_col, end_row, 'ro', markersize=10)
            
            # 3. Exploration Heatmap
            ax3 = axes[1, 0]
            if explored_grid is not None:
                im3 = ax3.imshow(explored_grid.astype(float), cmap='Blues', origin='upper', alpha=0.7)
                ax3.set_title('Explored Cells During Search')
                plt.colorbar(im3, ax=ax3, label='Explored')
            else:
                # Show slope instead if no exploration data
                if hasattr(self, 'slope_degrees') or 'slope_degrees' in locals():
                    slope_data = self.cost_surface_cache.get(f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}_cost", {}).get('slope_degrees')
                    if slope_data is not None:
                        im3 = ax3.imshow(slope_data, cmap='terrain', origin='upper')
                        ax3.set_title('Terrain Slope (degrees)')
                        plt.colorbar(im3, ax=ax3, label='Slope (degrees)')
                    else:
                        ax3.text(0.5, 0.5, 'No exploration data available', ha='center', va='center', transform=ax3.transAxes)
                else:
                    ax3.text(0.5, 0.5, 'No exploration data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.plot(start_col, start_row, 'go', markersize=10)
            ax3.plot(end_col, end_row, 'ro', markersize=10)
            if path_coords:
                ax3.plot(path_cols, path_rows, 'b-', linewidth=2)
        
            # 4. Combined View with Obstacles
            ax4 = axes[1, 1]
            # Create composite view
            composite = np.zeros((height, width, 3))
        
            # Base layer: terrain (grayscale)
            normalized_cost = (cost_display - np.min(cost_display)) / (np.max(cost_display) - np.min(cost_display))
            composite[:, :, 0] = normalized_cost
            composite[:, :, 1] = normalized_cost
            composite[:, :, 2] = normalized_cost
        
            # Highlight obstacles in red
            obstacle_mask = cost_surface >= 1000
            composite[obstacle_mask, 0] = 1
            composite[obstacle_mask, 1] = 0
            composite[obstacle_mask, 2] = 0
        
            # Highlight paths in green
            if path_raster is not None:
                path_mask = path_raster > 0
                composite[path_mask, 0] = 0
                composite[path_mask, 1] = 1
                composite[path_mask, 2] = 0
        
            im4 = ax4.imshow(composite, origin='upper')
            ax4.set_title('Combined View (Red=Obstacles, Green=Paths)')
            ax4.plot(start_col, start_row, 'wo', markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax4.plot(end_col, end_row, 'yo', markersize=10, markeredgecolor='black', markeredgewidth=2)
            if path_coords:
                ax4.plot(path_cols, path_rows, 'b-', linewidth=3)
        
            # Add grid lines for all subplots
            for ax in axes.flat:
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Grid X')
                ax.set_ylabel('Grid Y')
        
            plt.tight_layout()
        
            # Save the figure
            output_file = 'route_debug_visualization.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_file}")
        
            # Also show if running interactively
            try:
                plt.show()
            except:
                pass
        
            # Print path analysis
            if path_raw_tags and path_coords:
                print("\nPATH ANALYSIS:")
                print("-"*40)
            
                # Track which paths were used
                paths_used = {}
                for point in path_coords:
                    if isinstance(point, dict) and 'path_type' in point:
                        path_type = point['path_type']
                        if path_type != 'off_path':
                            paths_used[path_type] = paths_used.get(path_type, 0) + 1
                
                if paths_used:
                    print("Paths used by route:")
                    for path_type, count in sorted(paths_used.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {path_type}: {count} points ({count/len(path_coords)*100:.1f}%)")
                        # Find the raw tags for this path type
                        if path_raw_tags:
                            for path_id, tags in path_raw_tags.items():
                                if tags.get('interpreted_type') == path_type:
                                    if 'name' in tags.get('tags', {}):
                                        print(f"    - {tags['tags']['name']}")
                                    break
        
            # Print cost statistics
            print("\nCOST STATISTICS:")
            print("-"*40)
            print(f"Min cost: {np.min(cost_surface):.2f}")
            print(f"Max cost: {np.max(cost_surface):.2f}")
            print(f"Mean cost: {np.mean(cost_surface):.2f}")
            print(f"Median cost: {np.median(cost_surface):.2f}")
            impassable_count = np.sum(cost_surface >= 1000)
            print(f"Impassable cells: {impassable_count} ({impassable_count/cost_surface.size*100:.1f}%)")
        
            if path_coords:
                # Calculate path cost statistics
                path_costs = []
                for point in path_coords:
                    if isinstance(point, dict):
                        lon, lat = point['lon'], point['lat']
                    else:
                        lon, lat = point
                    x, y = transformer.transform(lon, lat)
                    col = int(round((x - out_trans.c) / out_trans.a))
                    row = int(round((y - out_trans.f) / out_trans.e))
                    if 0 <= row < height and 0 <= col < width:
                        path_costs.append(cost_surface[row, col])
            
                if path_costs:
                    print(f"\nPath cost statistics:")
                    print(f"  Min cost along path: {min(path_costs):.2f}")
                    print(f"  Max cost along path: {max(path_costs):.2f}")
                    print(f"  Mean cost along path: {np.mean(path_costs):.2f}")
        
            print("="*80)
    
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
    """Parse coordinate string like 'Start: 40.6572, -111.5706' or 'End: 40.6486, -111.5639' or 'Point: 40.6567, -111.5706'"""
    # Remove 'Start:', 'End:', or 'Point:' prefix and parse the numbers
    coord_str = coord_str.replace('Start:', '').replace('End:', '').replace('Point:', '').strip()
    
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


def query_point(lat, lon, use_api=False, api_url='http://localhost:9001', gradient_pref=1.0, trail_pref=1.0):
    """Query cost and terrain data for a single point"""
    print(f"\n🔍 Querying point: ({lat:.6f}, {lon:.6f})")
    print("="*80)
    
    if use_api:
        # Use API service
        with TimedStep("Querying point via API"):
            try:
                response = requests.post(
                    f"{api_url}/api/terrain/cost-point",
                    json={"lat": lat, "lon": lon},
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                
                if response.status_code == 404:
                    print("   ✗ Data not precomputed for this area")
                    print("   Use the web UI to compute cost surfaces for this area first")
                    return
                
                response.raise_for_status()
                data = response.json()
                
            except requests.exceptions.ConnectionError:
                print(f"   ✗ Cannot connect to API at {api_url}")
                print(f"   Make sure the API server is running")
                return
            except requests.exceptions.RequestException as e:
                print(f"   ✗ API request failed: {e}")
                return
    else:
        # Use local libraries
        print("\n1. Loading local libraries...")
        from app.services.dem_tile_cache import DEMTileCache
        from app.services.obstacle_config import ObstaclePresets
        from app.services.path_preferences import PathPreferencePresets
        
        # Create cache with same settings as web service
        # Apply gradient preference if specified
        obstacle_config = ObstaclePresets.experienced_hiker()
        obstacle_config.gradient_preference = gradient_pref
        
        # Apply trail preference if specified
        path_preferences = PathPreferencePresets.trail_seeker()
        path_preferences.trail_preference = trail_pref
        
        cache = DEMTileCache(
            obstacle_config=obstacle_config,
            path_preferences=path_preferences
        )
        
        with TimedStep("Querying point data"):
            data = cache.get_cost_at_point(lat, lon)
            
            if data is None:
                print("   ✗ Failed to get point data")
                return
            
            if 'error' in data and not data.get('precomputed', True):
                print(f"   ✗ {data['error']}")
                print(f"   Tile: {data.get('tile', 'unknown')}")
                return
    
    # Display results
    print("\n📍 Point Information:")
    print("-"*40)
    print(f"Location: ({data['lat']:.6f}, {data['lon']:.6f})")
    
    if data.get('elevation') is not None:
        print(f"Elevation: {data['elevation']:.1f}m")
    
    print(f"Slope: {data['slope']:.1f}°")
    print(f"Path Type: {data['path_type']}")
    
    if data.get('path_id', 0) > 0:
        print(f"Path ID: {data['path_id']}")
    
    # Show raw OSM data if available
    if data.get('raw_osm_data'):
        print("\n🏷️  OSM Data (used when building tile):")
        print("-"*40)
        osm_data = data['raw_osm_data']
        print(f"OSM ID: {osm_data['osm_id']}")
        
        if osm_data.get('source_tag'):
            tag = osm_data['source_tag']
            value = osm_data['tags'].get(tag, 'N/A')
            print(f"Decision based on: {tag} = {value}")
        
        print("\nAll OSM tags:")
        for key, value in sorted(osm_data['tags'].items()):
            print(f"  {key}: {value}")
    
    # Show cost breakdown
    print("\n💰 Cost Analysis:")
    print("-"*40)
    print(f"Total Cost: {data['cost']:.2f}")
    
    factors = data['factors']
    if factors['is_obstacle']:
        print("⚠️  IMPASSABLE OBSTACLE")
    else:
        print(f"Base cost: {factors['base_cost']:.2f}")
        print(f"× Slope multiplier: {factors['slope_cost']:.2f}")
        print(f"× Path preference: {factors['path_multiplier']:.2f}")
        print(f"= {factors['base_cost'] * factors['slope_cost'] * factors['path_multiplier']:.2f}")
    
    # Show tile info if available
    if data.get('tile_info'):
        print("\n🗺️  Tile Information:")
        print("-"*40)
        tile_info = data['tile_info']
        print(f"Tile: {tile_info['tile_coords']}")
        print(f"Path types in tile: {tile_info['total_path_types']}")
        
        if tile_info['all_path_types']:
            print("\nAll path types found when building tile:")
            for path_id, path_type in sorted(tile_info['all_path_types'].items()):
                print(f"  ID {path_id}: {path_type}")


def prepopulate_area(lat1, lon1, lat2, lon2, use_api=False, api_url='http://localhost:9001'):
    """Prepopulate terrain and cost data for an area"""
    print(f"\n🗺️  Prepopulating area:")
    print(f"   From: ({lat1:.6f}, {lon1:.6f})")
    print(f"   To:   ({lat2:.6f}, {lon2:.6f})")
    print("="*80)
    
    # Calculate bounds
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    
    # Add small buffer
    buffer = 0.001  # About 100m
    min_lat -= buffer
    max_lat += buffer
    min_lon -= buffer
    max_lon += buffer
    
    # Calculate area size
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    area_km2 = lat_diff * 111 * lon_diff * 111 * 0.7
    print(f"\n📐 Area size: ~{area_km2:.2f} km²")
    
    if use_api:
        # Use API service
        with TimedStep("Prepopulating via API"):
            try:
                response = requests.post(
                    f"{api_url}/api/prepopulate",
                    json={
                        "start": {"lat": lat1, "lon": lon1},
                        "end": {"lat": lat2, "lon": lon2}
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Longer timeout for prepopulation
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"\n✅ Prepopulation complete!")
                if 'cache_growth' in result:
                    growth = result['cache_growth']
                    print(f"   Terrain entries added: {growth.get('terrain_entries_added', 0)}")
                    print(f"   Cost surfaces added: {growth.get('cost_surfaces_added', 0)}")
                    print(f"   Memory added: {growth.get('memory_added_mb', 0):.1f} MB")
                
            except requests.exceptions.RequestException as e:
                print(f"   ✗ API request failed: {e}")
                return False
    else:
        # Use local libraries
        print("\n1. Loading local libraries...")
        from app.services.dem_tile_cache import DEMTileCache
        from app.services.obstacle_config import ObstaclePresets
        from app.services.path_preferences import PathPreferencePresets
        
        # Create cache with same settings as web service
        # Apply gradient preference if specified
        obstacle_config = ObstaclePresets.experienced_hiker()
        obstacle_config.gradient_preference = args.gradient
        
        # Apply trail preference if specified
        path_preferences = PathPreferencePresets.trail_seeker()
        path_preferences.trail_preference = trail_pref
        
        cache = DEMTileCache(
            obstacle_config=obstacle_config,
            path_preferences=path_preferences
        )
        
        # Step 1: Download terrain data
        with TimedStep("Downloading terrain data"):
            result = cache.predownload_area(min_lat, max_lat, min_lon, max_lon)
            if result['status'] != 'success':
                print(f"   ✗ Failed to download terrain: {result}")
                return False
            print(f"   ✓ Downloaded terrain data")
        
        # Step 2: Preprocess the area (compute cost surfaces)
        with TimedStep("Computing cost surfaces"):
            preprocess_result = cache.preprocess_area(
                min_lat, max_lat, min_lon, max_lon, force=False
            )
            
            if preprocess_result.get('tiles_created', 0) > 0:
                print(f"   ✓ Created {preprocess_result['tiles_created']} new tiles")
            if preprocess_result.get('tiles_skipped', 0) > 0:
                print(f"   ℹ️  Skipped {preprocess_result['tiles_skipped']} existing tiles")
            
        print(f"\n✅ Prepopulation complete!")
    
    return True


def generate_natural_area_map_from_cache(output_filename, bounds, cache_data):
    """Generate a map showing natural vs urban area classification using actual cache data"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    import numpy as np
    import requests
    from PIL import Image
    from io import BytesIO
    import math
    
    min_lat = bounds['min_lat']
    max_lat = bounds['max_lat']
    min_lon = bounds['min_lon'] 
    max_lon = bounds['max_lon']
    
    print(f"   Using actual algorithm data for bounds: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
    
    # Get the actual path raster and types from cache
    path_raster = cache_data.get('path_raster')
    path_types = cache_data.get('path_types', {})
    
    if path_raster is None:
        print("   Error: No path raster data in cache")
        return
        
    # Create classification array from actual path data
    # 0 = no data, 1 = off-path natural, 2 = trails, 3 = urban, 4 = other natural features
    classification = np.zeros_like(path_raster, dtype=np.uint8)
    
    # Define categories more granularly
    trail_types = {'trail', 'path', 'track', 'bridleway', 'downhill', 'nordic', 'hike', 'skitour'}
    other_natural = {'grass', 'meadow', 'nature_reserve', 'park', 'beach', 'garden'}
    urban_paths = {'footway', 'pedestrian', 'cycleway', 'steps', 'living_street',
                  'residential', 'service', 'unclassified', 'tertiary', 
                  'secondary', 'primary'}
    
    print(f"   Path raster shape: {path_raster.shape}")
    print(f"   Path types found: {list(path_types.values())}")
    
    # Track totals for each category
    trail_pixels = 0
    other_natural_pixels = 0
    urban_pixels = 0
    unknown_pixels = 0
    
    # Classify each path type from actual data
    for path_id, path_type in path_types.items():
        if path_id > 0:  # Skip background (0)
            mask = path_raster == path_id
            pixels = np.sum(mask)
            if pixels > 0:
                if path_type in trail_types:
                    classification[mask] = 2  # Trails
                    trail_pixels += pixels
                    print(f"     {path_type}: {pixels} pixels (TRAIL)")
                elif path_type in other_natural:
                    classification[mask] = 4  # Other natural features
                    other_natural_pixels += pixels
                    print(f"     {path_type}: {pixels} pixels (natural feature)")
                elif path_type in urban_paths:
                    classification[mask] = 3  # Urban
                    urban_pixels += pixels
                    print(f"     {path_type}: {pixels} pixels (urban)")
                else:
                    classification[mask] = 5  # Unknown
                    unknown_pixels += pixels
                    print(f"     {path_type}: {pixels} pixels (unknown)")
    
    # Areas without paths are considered off-path natural
    off_path_pixels = np.sum(path_raster == 0)
    classification[path_raster == 0] = 1  # Off-path natural terrain
    print(f"     off_path: {off_path_pixels} pixels (off-path natural)")
    
    # Download OSM tiles for background
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff < 0.01:
        zoom = 16
    elif max_diff < 0.05:
        zoom = 14
    elif max_diff < 0.1:
        zoom = 13
    else:
        zoom = 12
    
    # Get tile range
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    min_x, max_y = deg2num(min_lat, min_lon, zoom)
    max_x, min_y = deg2num(max_lat, max_lon, zoom)
    
    # Download and stitch tiles
    width = (max_x - min_x + 1) * 256
    height = (max_y - min_y + 1) * 256
    map_img = Image.new('RGB', (width, height))
    
    print(f"   Downloading {(max_x - min_x + 1) * (max_y - min_y + 1)} map tiles...")
    
    tile_server = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    headers = {'User-Agent': 'TrailFinderCLI/1.0'}
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            try:
                url = tile_server.format(z=zoom, x=x, y=y)
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                tile = Image.open(BytesIO(response.content))
                map_img.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
            except:
                gray_tile = Image.new('RGB', (256, 256), color=(200, 200, 200))
                map_img.paste(gray_tile, ((x - min_x) * 256, (y - min_y) * 256))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display OSM map
    map_array = np.array(map_img)
    ax.imshow(map_array, extent=[min_x, max_x + 1, max_y + 1, min_y], aspect='auto', alpha=0.5)
    
    # Prepare classification overlay
    h, w = classification.shape
    
    # Create meshgrid for coordinate transformation
    lon_grid = np.linspace(min_lon, max_lon, w)
    lat_grid = np.linspace(max_lat, min_lat, h)  # Note: reversed for image coordinates
    
    # Convert to tile coordinates
    x_grid = np.zeros((h, w))
    y_grid = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            lat_rad = math.radians(lat_grid[i])
            n = 2.0 ** zoom
            x_grid[i, j] = (lon_grid[j] + 180.0) / 360.0 * n
            y_grid[i, j] = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    # Create colormap with more categories
    colors = [(0, 0, 0, 0),          # 0: Transparent (no data)
              (0.7, 0.9, 0.7, 0.4),  # 1: Light green (off-path natural)
              (0.6, 0.3, 0.1, 0.8),  # 2: Brown (trails)
              (0.9, 0.3, 0.1, 0.6),  # 3: Red (urban)
              (0.2, 0.6, 0.2, 0.6),  # 4: Dark green (parks/natural features)
              (0.5, 0.5, 0.5, 0.6)]  # 5: Gray (unknown)
    cmap = ListedColormap(colors)
    
    # Plot classification overlay
    im = ax.pcolormesh(x_grid, y_grid, classification, cmap=cmap, vmin=0, vmax=5, shading='auto')
    
    # Add title and labels
    ax.set_title('Trail vs Natural vs Urban Area Classification\nBrown=Trails, Light Green=Off-path, Dark Green=Parks, Red=Urban', 
                 fontsize=16, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    legend_elements = [
        patches.Patch(color=(0.6, 0.3, 0.1, 0.8), label='Trails/Paths (hiking trails, ski slopes)'),
        patches.Patch(color=(0.7, 0.9, 0.7, 0.4), label='Off-path Natural (open terrain)'),
        patches.Patch(color=(0.2, 0.6, 0.2, 0.6), label='Natural Features (parks, meadows)'),
        patches.Patch(color=(0.9, 0.3, 0.1, 0.6), label='Urban (streets, sidewalks)'),
        patches.Patch(color=(0.5, 0.5, 0.5, 0.6), label='Unknown')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add attribution
    ax.text(0.99, 0.01, '© OpenStreetMap contributors', transform=ax.transAxes, 
            fontsize=8, ha='right', va='bottom', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add statistics
    total_pixels = np.prod(classification.shape)
    
    # Calculate percentages
    off_path_pct = off_path_pixels / total_pixels * 100
    trail_pct = trail_pixels / total_pixels * 100
    urban_pct = urban_pixels / total_pixels * 100
    other_natural_pct = other_natural_pixels / total_pixels * 100
    unknown_pct = unknown_pixels / total_pixels * 100
    
    stats_text = f'Area breakdown:\n'
    stats_text += f'Trails/Paths: {trail_pct:.2f}%\n'
    stats_text += f'Off-path Natural: {off_path_pct:.1f}%\n'
    stats_text += f'Parks/Features: {other_natural_pct:.2f}%\n'
    stats_text += f'Urban: {urban_pct:.2f}%\n'
    if unknown_pct > 0:
        stats_text += f'Unknown: {unknown_pct:.2f}%\n'
    stats_text += f'\nGrid resolution: {h}x{w}'
    stats_text += f'\nTotal natural: {(off_path_pct + trail_pct + other_natural_pct):.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save as JPEG
    plt.tight_layout()
    plt.savefig(output_filename, format='jpg', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_natural_area_map(output_filename, bounds, dem_cache):
    """Generate a map showing natural vs urban area classification"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    import numpy as np
    import requests
    from PIL import Image
    from io import BytesIO
    import math
    import osmnx as ox
    
    min_lat = bounds['min_lat']
    max_lat = bounds['max_lat']
    min_lon = bounds['min_lon'] 
    max_lon = bounds['max_lon']
    
    print(f"   Generating natural area classification for bounds: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
    
    # Create a grid for classification
    grid_size = 500  # Resolution of classification grid
    classification = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Get OSM data for the area
    try:
        from app.services.path_preferences import PathPreferences
        
        # Get preferred path tags
        path_prefs = PathPreferences()
        tags = path_prefs.get_default_path_tags()
        
        # Define natural and urban path types based on path_preferences.py
        natural_paths = {'trail', 'path', 'track', 'bridleway', 'grass', 'meadow', 
                        'nature_reserve', 'park', 'beach', 'off_path', 'downhill', 
                        'nordic', 'hike', 'skitour', 'garden'}
        urban_paths = {'footway', 'pedestrian', 'cycleway', 'steps', 'living_street',
                      'residential', 'service', 'unclassified', 'tertiary', 
                      'secondary', 'primary'}
        
        # Fetch OSM data
        print("   Fetching OSM path data...")
        north, south = max_lat, min_lat
        east, west = max_lon, min_lon
        
        # Get all features with the tags we care about
        for tag_key, tag_values in tags.items():
            try:
                features = ox.geometries_from_bbox(
                    north, south, east, west,
                    tags={tag_key: tag_values}
                )
                
                if len(features) > 0:
                    # Process each feature
                    for idx, feature in features.iterrows():
                        # Get the path type
                        path_type = None
                        if tag_key == 'highway' and 'highway' in feature:
                            path_type = feature['highway']
                        elif tag_key == 'piste:type' and 'piste:type' in feature:
                            path_type = feature['piste:type'].replace(':', '_')
                        elif tag_key in feature:
                            path_type = feature[tag_key]
                        
                        # Classify
                        if path_type:
                            if path_type in natural_paths:
                                color_value = 1  # Natural
                            elif path_type in urban_paths:
                                color_value = 2  # Urban
                            else:
                                color_value = 3  # Unknown
                            
                            # Rasterize the geometry onto our grid
                            # This is simplified - just marks cells that contain the feature
                            if hasattr(feature.geometry, 'bounds'):
                                min_x, min_y, max_x, max_y = feature.geometry.bounds
                                
                                # Convert to grid coordinates
                                x1 = int((min_x - min_lon) / (max_lon - min_lon) * grid_size)
                                x2 = int((max_x - min_lon) / (max_lon - min_lon) * grid_size)
                                y1 = int((max_lat - max_y) / (max_lat - min_lat) * grid_size)
                                y2 = int((max_lat - min_y) / (max_lat - min_lat) * grid_size)
                                
                                # Clamp to grid bounds
                                x1 = max(0, min(grid_size-1, x1))
                                x2 = max(0, min(grid_size-1, x2))
                                y1 = max(0, min(grid_size-1, y1))
                                y2 = max(0, min(grid_size-1, y2))
                                
                                # Mark the area
                                classification[y1:y2+1, x1:x2+1] = color_value
                                
            except Exception as e:
                # Skip if error fetching this tag type
                pass
        
        # Fill areas without paths as natural (off_path)
        classification[classification == 0] = 1  # Natural terrain
        
    except Exception as e:
        print(f"   Error processing OSM data: {e}")
        # Create sample data if failed
        classification = np.ones((grid_size, grid_size), dtype=np.uint8)
        # Add some sample urban areas
        classification[200:300, 200:300] = 2
    
    # Download OSM tiles for background
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff < 0.01:
        zoom = 16
    elif max_diff < 0.05:
        zoom = 14
    elif max_diff < 0.1:
        zoom = 13
    else:
        zoom = 12
    
    # Get tile range
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    min_x, max_y = deg2num(min_lat, min_lon, zoom)
    max_x, min_y = deg2num(max_lat, max_lon, zoom)
    
    # Download and stitch tiles
    width = (max_x - min_x + 1) * 256
    height = (max_y - min_y + 1) * 256
    map_img = Image.new('RGB', (width, height))
    
    print(f"   Downloading {(max_x - min_x + 1) * (max_y - min_y + 1)} map tiles...")
    
    tile_server = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    headers = {'User-Agent': 'TrailFinderCLI/1.0'}
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            try:
                url = tile_server.format(z=zoom, x=x, y=y)
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                tile = Image.open(BytesIO(response.content))
                map_img.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
            except:
                gray_tile = Image.new('RGB', (256, 256), color=(200, 200, 200))
                map_img.paste(gray_tile, ((x - min_x) * 256, (y - min_y) * 256))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display OSM map
    map_array = np.array(map_img)
    ax.imshow(map_array, extent=[min_x, max_x + 1, max_y + 1, min_y], aspect='auto', alpha=0.5)
    
    # Prepare classification overlay
    # Convert classification grid coordinates to tile coordinates
    h, w = classification.shape
    extent_lon = [min_lon, max_lon]
    extent_lat = [min_lat, max_lat]
    
    # Create meshgrid for coordinate transformation
    lon_grid = np.linspace(min_lon, max_lon, w)
    lat_grid = np.linspace(max_lat, min_lat, h)  # Note: reversed for image coordinates
    
    # Convert to tile coordinates
    x_grid = np.zeros((h, w))
    y_grid = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            lat_rad = math.radians(lat_grid[i])
            n = 2.0 ** zoom
            x_grid[i, j] = (lon_grid[j] + 180.0) / 360.0 * n
            y_grid[i, j] = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    # Create colormap: transparent, green (natural), red (urban), gray (unknown)
    colors = [(0, 0, 0, 0),          # 0: Transparent (no data)
              (0.2, 0.8, 0.2, 0.6),  # 1: Green (natural)
              (0.9, 0.3, 0.1, 0.6),  # 2: Red (urban)
              (0.5, 0.5, 0.5, 0.6)]  # 3: Gray (unknown)
    cmap = ListedColormap(colors)
    
    # Plot classification overlay
    im = ax.pcolormesh(x_grid, y_grid, classification, cmap=cmap, vmin=0, vmax=3, shading='auto')
    
    # Add title and labels
    ax.set_title('Natural vs Urban Area Classification\nGreen=Natural, Red=Urban, Gray=Unknown', 
                 fontsize=16, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    legend_elements = [
        patches.Patch(color=(0.2, 0.8, 0.2, 0.6), label='Natural (trails, parks, grass)'),
        patches.Patch(color=(0.9, 0.3, 0.1, 0.6), label='Urban (streets, sidewalks)'),
        patches.Patch(color=(0.5, 0.5, 0.5, 0.6), label='Unknown')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add attribution
    ax.text(0.99, 0.01, '© OpenStreetMap contributors', transform=ax.transAxes, 
            fontsize=8, ha='right', va='bottom', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add statistics
    total_pixels = np.sum(classification > 0)
    if total_pixels > 0:
        natural_pct = np.sum(classification == 1) / total_pixels * 100
        urban_pct = np.sum(classification == 2) / total_pixels * 100
        unknown_pct = np.sum(classification == 3) / total_pixels * 100
        
        stats_text = f'Area breakdown:\nNatural: {natural_pct:.1f}%\nUrban: {urban_pct:.1f}%\nUnknown: {unknown_pct:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save as JPEG
    plt.tight_layout()
    plt.savefig(output_filename, format='jpg', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_route_map(path, output_filename, start_lat, start_lon, end_lat, end_lon):
    """Generate a map with OSM tiles and the route overlay"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    import requests
    from PIL import Image
    from io import BytesIO
    import math
    
    # Extract coordinates from path
    lats = []
    lons = []
    for point in path:
        if isinstance(point, dict):
            lats.append(point['lat'])
            lons.append(point['lon'])
        else:
            lon, lat = point
            lats.append(lat)
            lons.append(lon)
    
    # Calculate bounds with padding
    padding = 0.002  # ~200m padding
    min_lat = min(lats) - padding
    max_lat = max(lats) + padding
    min_lon = min(lons) - padding
    max_lon = max(lons) + padding
    
    # Determine appropriate zoom level based on area size
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff < 0.01:
        zoom = 16
    elif max_diff < 0.05:
        zoom = 14
    elif max_diff < 0.1:
        zoom = 13
    elif max_diff < 0.2:
        zoom = 12
    else:
        zoom = 11
    
    # Function to convert lat/lon to tile numbers
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    # Function to convert tile numbers to lat/lon
    def num2deg(xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    # Get tile range
    min_x, max_y = deg2num(min_lat, min_lon, zoom)
    max_x, min_y = deg2num(max_lat, max_lon, zoom)
    
    # Ensure we have at least a 3x3 grid
    if max_x - min_x < 2:
        center_x = (min_x + max_x) // 2
        min_x = center_x - 1
        max_x = center_x + 1
    if max_y - min_y < 2:
        center_y = (min_y + max_y) // 2
        min_y = center_y - 1
        max_y = center_y + 1
    
    # Download and stitch tiles
    width = (max_x - min_x + 1) * 256
    height = (max_y - min_y + 1) * 256
    
    # Create base image
    map_img = Image.new('RGB', (width, height))
    
    print(f"   Downloading {(max_x - min_x + 1) * (max_y - min_y + 1)} map tiles (zoom level {zoom})...")
    
    # Use OpenStreetMap tiles
    tile_server = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    headers = {
        'User-Agent': 'TrailFinderCLI/1.0 (https://github.com/yourusername/trailfinder)'
    }
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            try:
                url = tile_server.format(z=zoom, x=x, y=y)
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                tile = Image.open(BytesIO(response.content))
                map_img.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
            except Exception as e:
                # Create a gray tile if download fails
                gray_tile = Image.new('RGB', (256, 256), color=(200, 200, 200))
                map_img.paste(gray_tile, ((x - min_x) * 256, (y - min_y) * 256))
    
    # Convert to numpy array for matplotlib
    map_array = np.array(map_img)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display map
    ax.imshow(map_array, extent=[min_x, max_x + 1, max_y + 1, min_y], aspect='auto')
    
    # Convert route coordinates to fractional tile coordinates
    route_x = []
    route_y = []
    for lat, lon in zip(lats, lons):
        # Get fractional tile coordinates
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = (lon + 180.0) / 360.0 * n
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        route_x.append(x)
        route_y.append(y)
    
    # Plot route
    ax.plot(route_x, route_y, color='red', linewidth=3, alpha=0.8, label='Route')
    
    # Add start and end markers with fractional coordinates
    lat_rad = math.radians(start_lat)
    n = 2.0 ** zoom
    start_x = (start_lon + 180.0) / 360.0 * n
    start_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    lat_rad = math.radians(end_lat)
    end_x = (end_lon + 180.0) / 360.0 * n
    end_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    ax.scatter(start_x, start_y, color='green', s=200, marker='o', edgecolor='white', linewidth=2, zorder=5, label='Start')
    ax.scatter(end_x, end_y, color='red', s=200, marker='s', edgecolor='white', linewidth=2, zorder=5, label='End')
    
    # Add title and labels
    ax.set_title(f'Trail Route Map\n{len(path)} waypoints', fontsize=16, pad=20)
    
    # Remove axis labels but keep the frame
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Add attribution
    attribution = "© OpenStreetMap contributors"
    ax.text(0.99, 0.01, attribution, transform=ax.transAxes, 
            fontsize=8, ha='right', va='bottom', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Save as JPEG
    plt.tight_layout()
    plt.savefig(output_filename, format='jpg', dpi=150, bbox_inches='tight', 
                facecolor='white')
    plt.close()


def main():
    """Main CLI function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Find hiking routes between coordinates or query terrain data at a point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Find a route using local libraries (default):
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
  
  # Find a route and export to GPX:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --gpx trail.gpx
  
  # Find a route and generate a map:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --map route_map.jpg
  
  # Generate slope visualization with elevation/slope profiles:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --slope-viz
  
  # Generate debug grid visualization (for small areas):
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --debug-grid
  
  # Generate natural vs urban area classification map:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --natural-map natural_areas.jpg
  
  # Find a route with both GPX and map output:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --gpx trail.gpx --map route_map.jpg
  
  # Find a route preferring gradual slopes (gentler but possibly longer):
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --gradient 2.0
  
  # Find a route preferring natural trails over streets:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --trails 2.0
  
  # Combine preferences for gentle natural trails:
  python route_cli.py "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639" --gradient 2.0 --trails 2.0
  
  # Find a route using API service:
  python route_cli.py --api "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
  
  # Query terrain data at a point:
  python route_cli.py --point "40.6567, -111.5706"
  python route_cli.py --point "40.6567, -111.5706" --api
  
  # Prepopulate an area with terrain data:
  python route_cli.py --prepopulate "Start: 40.6472, -111.5691" "End: 40.6473, -111.5685"
  
  # With custom API URL:
  python route_cli.py --api --api-url http://myserver:8000 "Start: 40.6572, -111.5706" "End: 40.6486, -111.5639"
        '''
    )
    
    # Add mutually exclusive group for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # Route mode arguments
    mode_group.add_argument('coordinates', nargs='*', default=None,
                           help='Start and end coordinates for route finding')
    
    # Point mode argument
    mode_group.add_argument('--point', type=str, 
                           help='Query a single point (e.g., "40.6567, -111.5706")')
    
    # Prepopulate mode argument
    mode_group.add_argument('--prepopulate', nargs=2, metavar=('START', 'END'),
                           help='Prepopulate area between two points')
    
    # Common arguments
    parser.add_argument('--api', action='store_true', help='Use API service instead of local libraries')
    parser.add_argument('--api-url', default='http://localhost:9001', help='API service URL (default: http://localhost:9001)')
    parser.add_argument('--debug-grid', action='store_true', help='Generate debug grid visualization (for small areas only)')
    parser.add_argument('--slope-viz', action='store_true', help='Generate slope visualization map')
    parser.add_argument('--gpx', type=str, metavar='FILENAME', help='Export route to GPX file (e.g., route.gpx)')
    parser.add_argument('--map', type=str, metavar='FILENAME', help='Export route map to JPG file (e.g., route_map.jpg)')
    parser.add_argument('--natural-map', type=str, metavar='FILENAME', help='Export natural/urban area classification map to JPG file')
    parser.add_argument('--gradient', type=float, default=1.0, metavar='VALUE',
                       help='Gradient preference (1.0=normal, 2.0=prefer gradual slopes, 0.5=accept steep slopes)')
    parser.add_argument('--trails', type=float, default=1.0, metavar='VALUE',
                       help='Trail preference (1.0=normal, 2.0=prefer natural trails, 0.5=prefer urban paths)')
    
    args = parser.parse_args()
    
    # Handle prepopulate mode
    if args.prepopulate:
        # Parse start and end coordinates
        start_coords = parse_coordinate(args.prepopulate[0])
        end_coords = parse_coordinate(args.prepopulate[1])
        
        if not start_coords or not end_coords:
            print(f"✗ Invalid coordinate format")
            print("  Expected format: 'Start: 40.6472, -111.5691' or just '40.6472, -111.5691'")
            return
        
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords
        prepopulate_area(lat1, lon1, lat2, lon2, use_api=args.api, api_url=args.api_url)
        return
    
    # Handle point query mode
    if args.point:
        # Parse point coordinates
        coords = parse_coordinate(f"Point: {args.point}")
        if not coords:
            print(f"✗ Invalid point format: {args.point}")
            print("  Expected format: '40.6567, -111.5706' or '40.6567,-111.5706'")
            return
        
        lat, lon = coords
        query_point(lat, lon, use_api=args.api, api_url=args.api_url, 
                   gradient_pref=args.gradient, trail_pref=args.trails)
        return
    
    # Handle route finding mode
    if len(args.coordinates) != 2:
        parser.error("Route finding requires exactly 2 arguments: start and end coordinates")
    
    args.start = args.coordinates[0]
    args.end = args.coordinates[1]
    
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
                # Apply gradient preference if specified
                obstacle_config = ObstacleConfig(gradient_preference=args.gradient)
                path_preferences = PathPreferences(trail_preference=args.trails)
                
                if args.gradient != 1.0:
                    print(f"   Gradient preference: {args.gradient} ({'gentler slopes' if args.gradient > 1 else 'steeper allowed'})")
                if args.trails != 1.0:
                    print(f"   Trail preference: {args.trails} ({'natural paths' if args.trails > 1 else 'urban paths'})")
                
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
                    debug_mode=args.debug_grid,  # Enable debug mode for grid visualization
                    debug_grid=args.debug_grid,
                    slope_viz=args.slope_viz,
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
                    path_type = point.get('path_type', 'unknown')
                    slope = point.get('slope', 'N/A')
                    
                    # Build description parts
                    desc_parts = []
                    if elev is not None and elev != 'N/A':
                        desc_parts.append(f"elev: {elev:.0f}m")
                    if slope is not None and slope != 'N/A':
                        desc_parts.append(f"slope: {slope:.1f}°")
                    desc_parts.append(f"type: {path_type}")
                    
                    desc = ", ".join(desc_parts)
                    print(f"  {i+1}. {lat:.6f}, {lon:.6f} ({desc})")
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
        
        # Export to GPX if requested
        if args.gpx and path:
            print(f"\n📁 Exporting to GPX file...")
            try:
                # Calculate route statistics for GPX metadata
                stats = {
                    'distance_km': path_km,
                    'elevation_gain_m': int(elevation_gain) if elevation_gain > 0 else 0,
                    'max_slope': max_slope,
                    'difficulty': 'Moderate'  # Could be calculated based on slope/distance
                }
                
                # Generate route name from coordinates
                route_name = f"Trail Route: ({start_lat:.4f}, {start_lon:.4f}) to ({end_lat:.4f}, {end_lon:.4f})"
                route_desc = f"Generated trail route - {path_km:.2f}km with {elevation_gain:.0f}m elevation gain"
                
                # Create GPX content
                gpx_content = GPXGenerator.create_gpx(
                    path_with_slopes=path,
                    route_name=route_name,
                    route_description=route_desc,
                    stats=stats
                )
                
                # Write to file
                with open(args.gpx, 'w', encoding='utf-8') as f:
                    f.write(gpx_content)
                
                print(f"   ✓ GPX file saved to: {args.gpx}")
                print(f"   Route contains {len(path)} track points with elevation data")
                
            except Exception as e:
                print(f"   ❌ Error exporting GPX: {str(e)}")
        
        # Export to map if requested
        if args.map and path:
            print(f"\n🗺️  Generating route map...")
            try:
                generate_route_map(path, args.map, start_lat, start_lon, end_lat, end_lon)
                print(f"   ✓ Map saved to: {args.map}")
            except Exception as e:
                print(f"   ❌ Error generating map: {str(e)}")
        
        # Export natural area classification map if requested
        if args.natural_map:
            print(f"\n🌳 Generating natural area classification map...")
            try:
                # Only works with local mode since we need the cache
                if args.api:
                    print("   ❌ Natural area map only available in local mode (not with --api)")
                else:
                    # Get the actual cache data
                    cache_data = None
                    for key, data in cache.cost_surface_cache.items():
                        # Use the first (and likely only) cache entry
                        cache_data = data
                        break
                    
                    if cache_data is None:
                        print("   ❌ No cached cost surface data available")
                    else:
                        # Use the actual bounds from the cached data
                        bounds = {
                            'min_lat': min(start_lat, end_lat) - cache.buffer,
                            'max_lat': max(start_lat, end_lat) + cache.buffer,
                            'min_lon': min(start_lon, end_lon) - cache.buffer,
                            'max_lon': max(start_lon, end_lon) + cache.buffer
                        }
                        
                        generate_natural_area_map_from_cache(args.natural_map, bounds, cache_data)
                        print(f"   ✓ Natural area map saved to: {args.natural_map}")
            except Exception as e:
                print(f"   ❌ Error generating natural area map: {str(e)}")
        
    else:
        print(f"\n❌ No route found (total time: {format_time(route_time)})")
    
    print("\n" + "="*60)
    print(f"✓ Complete! Total time: {format_time(time.time() - overall_start)}")


if __name__ == "__main__":
    main()