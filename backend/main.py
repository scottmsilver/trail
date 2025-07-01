from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import py3dep
import rasterio
from rasterio.transform import from_bounds
from queue import PriorityQueue
import gpxpy
import gpxpy.gpx
import time
import tempfile
import os
from typing import List, Tuple, Optional
import math

app = FastAPI(title="Trail Pathfinder API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PathfindingRequest(BaseModel):
    start_lat: float
    start_lng: float
    goal_lat: float
    goal_lng: float
    margin: float = 0.003
    resolution: int = 2

class PathfindingResponse(BaseModel):
    success: bool
    message: str
    path_coords: Optional[List[List[float]]] = None
    processing_time: Optional[float] = None
    path_length_km: Optional[float] = None
    nodes_explored: Optional[int] = None
    grid_size: Optional[List[int]] = None

def calculate_slope(elevation_array, resolution):
    """Calculate slope from elevation data"""
    dx, dy = np.gradient(elevation_array, resolution, resolution)
    slope = np.sqrt(dx**2 + dy**2)
    slope_degrees = np.arctan(slope) * (180 / np.pi)
    return slope_degrees

def download_slopes(start, goal, resolution, margin):
    """Download and calculate slope data for the given area"""
    min_lat = min(start[0], goal[0]) - margin
    max_lat = max(start[0], goal[0]) + margin
    min_lon = min(start[1], goal[1]) - margin
    max_lon = max(start[1], goal[1]) + margin

    # Generate coordinates for the bounding box
    xcoords = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / (resolution * 1e-5)))
    ycoords = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / (resolution * 1e-5)))

    try:
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
        
        return slope_data, (min_lat, min_lon, max_lat, max_lon)
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Elevation data service unavailable: {str(e)}")

def heuristic(a, b):
    """A* heuristic function"""
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node, shape):
    """Get valid neighboring nodes"""
    neighbors = [
        (node[0] - 1, node[1]), (node[0] + 1, node[1]),
        (node[0], node[1] - 1), (node[0], node[1] + 1),
        (node[0] - 1, node[1] - 1), (node[0] + 1, node[1] + 1),
        (node[0] - 1, node[1] + 1), (node[0] + 1, node[1] - 1)
    ]
    return [(r, c) for r, c in neighbors if 0 <= r < shape[0] and 0 <= c < shape[1]]

def a_star_search(start, goal, margin, resolution, slope_data):
    """A* pathfinding algorithm for trail optimization"""
    min_lat = min(start[0], goal[0]) - margin
    min_lon = min(start[1], goal[1]) - margin

    start_idx = (int((start[0] - min_lat) / (resolution * 1e-5)), int((start[1] - min_lon) / (resolution * 1e-5)))
    goal_idx = (int((goal[0] - min_lat) / (resolution * 1e-5)), int((goal[1] - min_lon) / (resolution * 1e-5)))

    frontier = PriorityQueue()
    frontier.put((0, start_idx))
    came_from = {}
    cost_so_far = {}
    came_from[start_idx] = None
    cost_so_far[start_idx] = 0
    
    nodes_explored = 0
    
    while not frontier.empty():
        current = frontier.get()[1]
        nodes_explored += 1

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
            movement_cost = 1
            new_cost = cost_so_far[current] + slope_cost * movement_cost + distance_penalty * movement_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal_idx, neighbor)
                frontier.put((priority, neighbor))
                came_from[neighbor] = current

    if goal_idx not in came_from:
        return [], nodes_explored

    # Reconstruct path
    path = []
    current = goal_idx
    while current != start_idx:
        path.append(current)
        current = came_from[current]
    path.append(start_idx)
    path.reverse()

    return path, nodes_explored

def path_to_coords(start, goal, margin, path, resolution):
    """Convert path indices to geographic coordinates"""
    min_lat = min(start[0], goal[0]) - margin
    min_lon = min(start[1], goal[1]) - margin

    # Convert path indices back to coordinates
    path_coords = [(min_lat + y * (resolution * 1e-5), min_lon + x * (resolution * 1e-5)) for y, x in path]
    return path_coords

def calculate_path_length(path_coords):
    """Calculate approximate path length in kilometers"""
    total_distance = 0
    for i in range(1, len(path_coords)):
        lat1, lon1 = path_coords[i-1]
        lat2, lon2 = path_coords[i]
        # Approximate distance in km using simple projection
        dist = ((lat2-lat1)*111)**2 + ((lon2-lon1)*111*math.cos(math.radians(lat1)))**2
        total_distance += dist**0.5
    return total_distance

@app.get("/")
async def root():
    return {"message": "Trail Pathfinder API is running"}

@app.post("/find-path", response_model=PathfindingResponse)
async def find_path(request: PathfindingRequest):
    """Find optimal hiking path between two points"""
    
    start = (request.start_lat, request.start_lng)
    goal = (request.goal_lat, request.goal_lng)
    
    try:
        start_time = time.time()
        
        # Download slope data
        slope_data, bounds = download_slopes(start, goal, request.resolution, request.margin)
        
        # Calculate grid size for response
        grid_size = [slope_data.shape[0], slope_data.shape[1]]
        
        # Run A* pathfinding
        path, nodes_explored = a_star_search(start, goal, request.margin, request.resolution, slope_data)
        
        if not path:
            return PathfindingResponse(
                success=False,
                message="No path found between the specified points",
                processing_time=time.time() - start_time,
                nodes_explored=nodes_explored,
                grid_size=grid_size
            )
        
        # Convert to coordinates
        path_coords = path_to_coords(start, goal, request.margin, path, request.resolution)
        
        # Calculate path length
        path_length = calculate_path_length(path_coords)
        
        processing_time = time.time() - start_time
        
        return PathfindingResponse(
            success=True,
            message=f"Path found with {len(path_coords)} waypoints",
            path_coords=path_coords,
            processing_time=processing_time,
            path_length_km=path_length,
            nodes_explored=nodes_explored,
            grid_size=grid_size
        )
        
    except Exception as e:
        return PathfindingResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

@app.post("/download-gpx")
async def download_gpx(request: PathfindingRequest):
    """Generate and download GPX file for the calculated path"""
    
    start = (request.start_lat, request.start_lng)
    goal = (request.goal_lat, request.goal_lng)
    
    try:
        # Get the path (same logic as find_path)
        slope_data, bounds = download_slopes(start, goal, request.resolution, request.margin)
        path, _ = a_star_search(start, goal, request.margin, request.resolution, slope_data)
        
        if not path:
            raise HTTPException(status_code=404, detail="No path found")
            
        path_coords = path_to_coords(start, goal, request.margin, path, request.resolution)
        
        # Create GPX file
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        for lat, lon in path_coords:
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gpx', delete=False) as f:
            f.write(gpx.to_xml())
            temp_file = f.name

        return FileResponse(
            temp_file, 
            media_type='application/gpx+xml',
            filename='trail_path.gpx'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating GPX: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)