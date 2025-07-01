# Trail Finder Backend

FastAPI backend for the Trail Finder application.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```

## Running the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000
API documentation at http://localhost:8000/docs

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/routes/calculate` - Start route calculation
- `GET /api/routes/{route_id}/status` - Get calculation status
- `GET /api/routes/{route_id}` - Get calculated route
- `GET /api/routes/{route_id}/gpx` - Download as GPX (TODO)