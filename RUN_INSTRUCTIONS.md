# How to Run Trail Finder Application

## Prerequisites
- Python 3.8+ 
- Node.js 16+
- npm or yarn

## Backend Setup and Run

### 1. Navigate to backend directory:
```bash
cd /home/ssilver/development/trail/backend
```

### 2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you get an error about py3dep/aiohttp, you have two options:
- Option A: Comment out line 16 in `app/services/dem_tile_cache.py` (`# import py3dep`)
- Option B: Install compatible versions:
  ```bash
  pip install numpy==1.26.4
  pip install aiohttp==3.9.5
  ```

### 4. Run the backend:
```bash
# Development mode with auto-reload
uvicorn app.main:app --port 9001 --reload

# Or production mode
uvicorn app.main:app --port 9001 --workers 4
```

### 5. Verify backend is running:
Open http://localhost:9001/docs in your browser to see the API documentation.

---

## Frontend Setup and Run

### 1. Navigate to frontend directory:
```bash
cd /home/ssilver/development/trail/frontend
```

### 2. Install dependencies:
```bash
npm install
# or
yarn install
```

### 3. Create environment file:
```bash
# Copy the example env file
cp .env.example .env

# Make sure it contains:
# VITE_API_URL=http://localhost:9001
```

### 4. Run the frontend:
```bash
# Development mode
npm run dev -- --port 9002

# Or with yarn
yarn dev --port 9002

# Or default port (5173)
npm run dev
```

### 5. Open in browser:
Visit http://localhost:9002 (or http://localhost:5173 if using default port)

---

## Quick Start (Both at once)

### Terminal 1 - Backend:
```bash
cd /home/ssilver/development/trail/backend
source venv/bin/activate  # if using venv
uvicorn app.main:app --port 9001 --reload
```

### Terminal 2 - Frontend:
```bash
cd /home/ssilver/development/trail/frontend
npm run dev -- --port 9002
```

---

## Troubleshooting

### Backend Issues:

1. **Port already in use**:
   ```bash
   # Find what's using port 9001
   lsof -i :9001
   # Kill the process
   kill -9 <PID>
   ```

2. **Import errors**:
   - Make sure virtual environment is activated
   - Check all dependencies are installed: `pip list`

3. **No routes found**:
   - The DEM download requires py3dep which has dependency issues
   - Routes will fail without elevation data

### Frontend Issues:

1. **Cannot connect to backend**:
   - Check backend is running on port 9001
   - Verify .env file has correct VITE_API_URL
   - Check browser console for CORS errors

2. **Port already in use**:
   ```bash
   # Kill any running vite processes
   pkill -f vite
   ```

3. **Build errors**:
   - Delete node_modules and reinstall: 
     ```bash
     rm -rf node_modules package-lock.json
     npm install
     ```

---

## Features Available

Once both are running, you can:
1. Click on the map to set start and end points
2. Click "Find Route" to calculate a path
3. Use the layer control (top-right) to switch map types
4. Search for locations using the search box
5. View route statistics when a path is found

Note: Route finding may not work properly due to the py3dep dependency issue in the original code.