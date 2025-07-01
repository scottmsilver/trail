#!/usr/bin/env python3

import py3dep
import rasterio
from rasterio.transform import from_bounds
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from queue import PriorityQueue
import gpxpy
import gpxpy.gpx
from scipy.ndimage import generic_filter
import seaborn as sns

# Function to calculate slope
def calculate_slope(elevation_array, resolution):
    dx, dy = np.gradient(elevation_array, resolution, resolution)
    slope = np.sqrt(dx**2 + dy**2)
    slope_degrees = np.arctan(slope) * (180 / np.pi)
    return slope_degrees

def download_slopes(start, goal, resolution, margin, file_name):
    # Define the bounding box
    min_lat = min(start[0], goal[0]) - margin
    max_lat = max(start[0], goal[0]) + margin
    min_lon = min(start[1], goal[1]) - margin
    max_lon = max(start[1], goal[1]) + margin

    # Generate coordinates for the bounding box
    xcoords = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / (resolution * 1e-5)))
    ycoords = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / (resolution * 1e-5)))

    print(f"Fetching elevation data for bbox: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
    
    # Fetch elevation data using py3dep
    elevation_data = py3dep.elevation_bygrid(
        xcoords=xcoords,
        ycoords=ycoords,
        crs="EPSG:4326",
        resolution=resolution,
        depression_filling=False
    )

    # Convert the elevation data to a 2D NumPy array
    elevation_array = np.array(elevation_data).reshape((len(ycoords), len(xcoords)))

    # Calculate the slope
    slope_data = calculate_slope(elevation_array, resolution)

    # Define the GeoTIFF transform
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, slope_data.shape[1], slope_data.shape[0])

    # Define the metadata for the GeoTIFF
    metadata = {
        'driver': 'GTiff',
        'height': slope_data.shape[0],
        'width': slope_data.shape[1],
        'count': 1,
        'dtype': slope_data.dtype,
        'crs': 'EPSG:4326',
        'transform': transform
    }

    # Save the slope data as a GeoTIFF file
    with rasterio.open(file_name, 'w', **metadata) as dst:
        dst.write(slope_data, 1)

    print(f"Slope data saved as '{file_name}'.")

def load_slope_data_from_file(filename):
    with rasterio.open(filename) as src:
        slope_data = src.read(1)  # Read the first band
        transform = src.transform
    return slope_data, transform

def plot_slopes(slope_data, transform):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the slope data
    cax = ax.imshow(slope_data, cmap='terrain', extent=(transform[2], transform[2] + transform[0] * slope_data.shape[1], transform[5] + transform[4] * slope_data.shape[0], transform[5]))
    ax.set_title('Slope Data Visualization')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.75)
    cbar.set_label('Slope (degrees)')

    # Save the plot
    plt.savefig('slope_visualization.png')
    print("Slope visualization saved as 'slope_visualization.png'")
    plt.close()

# Helper functions for A* pathfinding
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node, shape):
    neighbors = [
        (node[0] - 1, node[1]), (node[0] + 1, node[1]),
        (node[0], node[1] - 1), (node[0], node[1] + 1),
        (node[0] - 1, node[1] - 1), (node[0] + 1, node[1] + 1),
        (node[0] - 1, node[1] + 1), (node[0] + 1, node[1] - 1)
    ]
    return [(r, c) for r, c in neighbors if 0 <= r < shape[0] and 0 <= c < shape[1]]

def a_star_search(start, goal, margin, resolution, slope_data):
    min_lat = min(start[0], goal[0]) - margin
    min_lon = min(start[1], goal[1]) - margin

    start_idx = (int((start[0] - min_lat) / (resolution * 1e-5)), int((start[1] - min_lon) / (resolution * 1e-5)))
    goal_idx = (int((goal[0] - min_lat) / (resolution * 1e-5)), int((goal[1] - min_lon) / (resolution * 1e-5)))

    print(f"Starting A* search from {start_idx} to {goal_idx}")
    
    frontier = PriorityQueue()
    frontier.put((0, start_idx))
    came_from = {}
    cost_so_far = {}
    came_from[start_idx] = None
    cost_so_far[start_idx] = 0
    
    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal_idx:
            break

        for neighbor in get_neighbors(current, slope_data.shape):
            slope = slope_data[neighbor]
            
            slope_cost = slope + 1e-6  # Ensure slope cost is always positive

            # Apply a non-linear penalty if slope exceeds 10% (which is about 5.71 degrees)
            if slope > 5.71:
                base_weight = 10
                exponent = 4
                slope_cost += base_weight * ((slope - 5.71) ** exponent)

            distance_penalty = 3000
            # Encourage horizontal and diagonal movements to create switchbacks
            movement_cost = 1
            new_cost = cost_so_far[current] + slope_cost * movement_cost + distance_penalty * movement_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal_idx, neighbor)
                frontier.put((priority, neighbor))
                came_from[neighbor] = current

    if goal_idx not in came_from:
        print("Goal not reachable")
        return []

    # Reconstruct path
    path = []
    current = goal_idx
    while current != start_idx:
        path.append(current)
        current = came_from[current]
    path.append(start_idx)
    path.reverse()

    print(f"Path found with {len(path)} points")
    return path

def path_to_coords(start, goal, margin, path, resolution):
    min_lat = min(start[0], goal[0]) - margin
    min_lon = min(start[1], goal[1]) - margin

    # Convert path indices back to coordinates
    path_coords = [(min_lat + y * (resolution * 1e-5), min_lon + x * (resolution * 1e-5)) for y, x in path]
    path_coords = np.array(path_coords)
    return path_coords

def plot_path(start, goal, path_coords, slope_data, transform):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the slope data
    cax = ax.imshow(slope_data, cmap='terrain', extent=(transform[2], transform[2] + transform[0] * slope_data.shape[1], transform[5] + transform[4] * slope_data.shape[0], transform[5]))
    ax.set_title('Slope Data Visualization with Walking Path')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Plot the walking path between the two points
    ax.plot(path_coords[:, 1], path_coords[:, 0], 'r-', linewidth=2, label='Walking Path')
    ax.plot(start[1], start[0], 'bo', markersize=10, label='Start Point')
    ax.plot(goal[1], goal[0], 'go', markersize=10, label='End Point')
    ax.legend()

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.75)
    cbar.set_label('Slope (degrees)')

    # Save the plot
    plt.savefig('path_visualization.png')
    print("Path visualization saved as 'path_visualization.png'")
    plt.close()

def write_gpx_file(path_coords, filename):
    # Create GPX file
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for lat, lon in path_coords:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

    with open(filename, 'w') as f:
        f.write(gpx.to_xml())

    print(f"GPX file '{filename}' created.")

def calculate_bounding_box(start, goal, margin):
    min_lat = min(start[0], goal[0]) - margin
    max_lat = max(start[0], goal[0]) + margin
    min_lon = min(start[1], goal[1]) - margin
    max_lon = max(start[1], goal[1]) + margin
    return (min_lat, min_lon, max_lat, max_lon)

# Main execution
if __name__ == "__main__":
    # Define the coordinates for two points and desired resolution
    map_start = 40.657192271514724, -111.56876501454423  # Point 1
    map_goal = 40.69414448717707, -111.60456091449336  # murdock peak
    map_resolution = 2  # in meters

    # Define the margin around the points for the bounding box (in degrees)
    map_margin = 0.010
    map_file = 'slope_data.tif'
    map_gpx_file = 'path.gpx'

    print("Trail Finding Algorithm")
    print(f"Start: {map_start}")
    print(f"Goal: {map_goal}")
    print(f"Resolution: {map_resolution}m")
    print()

    # Download slope data
    print("Downloading elevation data and calculating slopes...")
    download_slopes(map_start, map_goal, map_resolution, map_margin, map_file)

    # Load slope data
    print("\nLoading slope data...")
    map_slope_data, map_transform = load_slope_data_from_file(map_file)
    plot_slopes(map_slope_data, map_transform)

    # Perform A* search
    print("\nPerforming A* pathfinding...")
    path = a_star_search(start=map_start, goal=map_goal, margin=map_margin,
                         resolution=map_resolution, slope_data=map_slope_data)

    if path:
        # Convert path to coordinates
        path_coords = path_to_coords(start=map_start, goal=map_goal, margin=map_margin, 
                                     path=path, resolution=map_resolution)

        # Visualize the path
        plot_path(start=map_start, goal=map_goal, 
                  path_coords=path_coords, slope_data=map_slope_data, transform=map_transform)
        
        # Write GPX file
        write_gpx_file(path_coords=path_coords, filename=map_gpx_file)
        
        print("\nExecution completed successfully!")
        print("Generated files:")
        print("- slope_data.tif: GeoTIFF file with slope data")
        print("- slope_visualization.png: Visualization of the slope data")
        print("- path_visualization.png: Visualization of the calculated path")
        print("- path.gpx: GPX file with the trail path")
    else:
        print("\nNo path found between the two points.")