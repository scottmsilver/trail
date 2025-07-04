import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.integration
def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "trail-finder-api"}


@pytest.mark.integration
def test_calculate_route_valid_request():
    """Test route calculation with valid coordinates"""
    request_data = {
        "start": {"lat": 40.630, "lon": -111.580},
        "end": {"lat": 40.650, "lon": -111.560},
        "options": {
            "avoidSteep": True,
            "buffer": 0.02
        }
    }
    
    response = client.post("/api/routes/calculate", json=request_data)
    assert response.status_code == 202  # Accepted for processing
    assert "routeId" in response.json()
    assert response.json()["status"] == "processing"


@pytest.mark.integration
def test_calculate_route_invalid_coordinates():
    """Test route calculation with invalid coordinates"""
    request_data = {
        "start": {"lat": 91, "lon": -111.580},  # Invalid latitude
        "end": {"lat": 40.650, "lon": -111.560}
    }
    
    response = client.post("/api/routes/calculate", json=request_data)
    assert response.status_code == 422  # Validation error


@pytest.mark.integration
def test_get_route_status():
    """Test getting route calculation status"""
    # First create a route
    request_data = {
        "start": {"lat": 40.630, "lon": -111.580},
        "end": {"lat": 40.650, "lon": -111.560}
    }
    
    create_response = client.post("/api/routes/calculate", json=request_data)
    route_id = create_response.json()["routeId"]
    
    # Check status
    status_response = client.get(f"/api/routes/{route_id}/status")
    assert status_response.status_code == 200
    assert "status" in status_response.json()
    assert "progress" in status_response.json()


@pytest.mark.integration
def test_get_route_not_found():
    """Test getting a non-existent route"""
    response = client.get("/api/routes/nonexistent-route-id")
    assert response.status_code == 404