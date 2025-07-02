# Port Configuration

## Frontend
- **Port**: 9002
- **URL**: http://localhost:9002
- **Start command**: `cd frontend && npm run dev`

## Backend
- **Port**: 9001
- **URL**: http://localhost:9001
- **API Health Check**: http://localhost:9001/api/health
- **Start command**: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 9001`

## Important Notes
- Frontend expects backend API at http://localhost:9001
- Backend must be started with venv activated to use py3dep
- Both services need to be running for the application to work