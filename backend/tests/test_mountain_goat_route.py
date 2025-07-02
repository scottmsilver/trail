import pytest
import asyncio
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_mountain_goat_long_route():
    """Test a route that previously failed due to search space size"""
    
    # Route that was failing
    response = client.post("/api/routes/calculate", json={
        "start": {"lat": 40.6568, "lon": -111.5713},
        "end": {"lat": 40.6428, "lon": -111.5777},
        "options": {
            "userProfile": "mountain_goat"
        }
    })
    
    assert response.status_code == 202
    route_id = response.json()["routeId"]
    
    # Poll for completion (with timeout)
    max_attempts = 60  # 60 seconds max
    for _ in range(max_attempts):
        status_response = client.get(f"/api/routes/{route_id}/status")
        status = status_response.json()["status"]
        
        if status == "completed":
            break
        elif status == "failed":
            pytest.fail("Route calculation failed")
            
        asyncio.run(asyncio.sleep(1))
    else:
        pytest.fail("Route calculation timed out")
    
    # Get the route
    route_response = client.get(f"/api/routes/{route_id}")
    assert route_response.status_code == 200
    
    route_data = route_response.json()
    assert route_data["status"] == "completed"
    assert len(route_data["path"]) > 0
    
    stats = route_data["stats"]
    print(f"Route found: {stats['distance_km']:.2f} km, "
          f"{stats['elevation_gain_m']:.0f}m gain, "
          f"{stats['waypoints']} waypoints")
    
    # Verify it's a reasonable route
    assert stats["distance_km"] > 1.0  # At least 1km
    assert stats["distance_km"] < 10.0  # Not unreasonably long
    assert stats["waypoints"] > 50  # Detailed path