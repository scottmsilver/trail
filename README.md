# 🏔️ Trail Pathfinder

A modern web application for finding optimal hiking trails using AI-powered A* pathfinding algorithms. Built with React.js frontend and Python FastAPI backend.

![Trail Pathfinder](https://img.shields.io/badge/Trail-Pathfinder-green?style=for-the-badge&logo=hiking)
![React](https://img.shields.io/badge/React-18.2.0-blue?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)

## ✨ Features

- **🗺️ Interactive Map Interface**: Click-to-set start and goal points on a topographic map
- **🤖 AI Pathfinding**: Advanced A* algorithm optimized for hiking terrain
- **🏔️ Terrain Analysis**: Considers slope difficulty and elevation data
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **📊 Performance Metrics**: Real-time processing statistics and path analysis
- **📥 GPX Export**: Download trail routes for GPS devices and hiking apps
- **⚙️ Customizable Settings**: Adjust search parameters for speed vs accuracy

## 🛠️ Technology Stack

### Frontend
- **React.js 18.2** - Modern UI framework
- **Leaflet** - Interactive mapping library
- **React-Leaflet** - React components for Leaflet
- **Axios** - HTTP client for API communication

### Backend
- **FastAPI** - High-performance Python web framework
- **py3dep** - USGS elevation data access
- **NumPy** - Numerical computing for pathfinding algorithms
- **Rasterio** - Geospatial data processing
- **GPXPy** - GPX file generation

## 🚀 Quick Start

### Prerequisites
- **Node.js** 16+ and npm
- **Python** 3.8+
- **Git**

### 1. Clone the Repository
```bash
git clone <repository-url>
cd trail-pathfinder
```

### 2. Setup Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Frontend
```bash
cd ../frontend
npm install
```

### 4. Start Development Servers

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```
Backend will be available at `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend will be available at `http://localhost:3000`

## 📖 Usage Guide

### Setting Points
1. Select "Set Start Point" or "Set Goal Point"
2. Click anywhere on the map to place markers
3. Green marker = start, Red marker = goal

### Configuring Search
- **Search Margin**: Controls search area size
  - Smaller (0.001-0.003) = Faster processing
  - Larger (0.005-0.01) = More path options
- **Resolution**: Data precision in meters
  - Lower (1-2m) = More precise paths
  - Higher (5-10m) = Faster processing

### Finding Trails
1. Set your start and goal points
2. Adjust settings as needed
3. Click "Find Optimal Trail"
4. View the calculated path on the map
5. Download GPX file for GPS navigation

## 🧮 Algorithm Details

The pathfinding uses an optimized A* algorithm with the following features:

- **Slope Penalty**: Heavy penalties for slopes > 5.71° (10% grade)
- **Distance Optimization**: Minimizes total hiking distance
- **Terrain Awareness**: Uses USGS elevation data for accurate slope calculation
- **Switchback Friendly**: Encourages gradual ascents over steep climbs

### Performance Optimizations
- **Reduced Search Space**: 43.5% grid size reduction vs naive approach
- **Efficient Data Structures**: Priority queue for optimal node exploration
- **Smart Bounds**: Dynamic bounding box based on start/goal coordinates

## 📊 API Endpoints

### `POST /find-path`
Calculate optimal hiking path between two points.

**Request:**
```json
{
  "start_lat": 40.657192,
  "start_lng": -111.568765,
  "goal_lat": 40.694144,
  "goal_lng": -111.604561,
  "margin": 0.003,
  "resolution": 2
}
```

**Response:**
```json
{
  "success": true,
  "message": "Path found with 4180 waypoints",
  "path_coords": [[lat1, lng1], [lat2, lng2], ...],
  "processing_time": 72.4,
  "path_length_km": 10.03,
  "nodes_explored": 3872720,
  "grid_size": [2148, 2090]
}
```

### `POST /download-gpx`
Generate and download GPX file for the calculated path.

**Request:** Same as `/find-path`
**Response:** GPX file download

## 🗂️ Project Structure

```
trail-pathfinder/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html       # HTML template
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Application styles
│   │   ├── index.js         # React entry point
│   │   └── index.css        # Global styles
│   └── package.json         # Node.js dependencies
├── trail.ipynb              # Original Jupyter notebook
└── README.md               # This file
```

## 🎯 Example Use Cases

- **Trail Planning**: Find optimal routes between specific points
- **Hiking Safety**: Avoid overly steep or dangerous terrain
- **GPS Navigation**: Export routes for offline GPS devices
- **Trail Research**: Analyze elevation profiles and distances
- **Accessibility**: Find easier routes for different skill levels

## ⚡ Performance Benchmarks

- **Grid Size Reduction**: 43.5% smaller search space
- **Processing Speedup**: 1.8x faster than naive approach
- **Memory Efficiency**: ~4.5M cells vs 7.9M cells
- **Typical Processing Time**: 30-120 seconds for complex routes

## 🔧 Configuration

### Environment Variables (Backend)
```bash
# Optional: Set custom CORS origins
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Build Configuration (Frontend)
For production builds:
```bash
cd frontend
npm run build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with descriptive messages
5. Push and create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🐛 Known Issues

- **USGS Service Dependency**: Requires active internet connection for elevation data
- **Processing Time**: Complex routes may take 1-2 minutes to calculate
- **Memory Usage**: Large search areas require significant RAM

## 🚧 Roadmap

- [ ] **Offline Mode**: Cache elevation data for popular areas
- [ ] **Multi-point Routes**: Support for waypoint-based planning
- [ ] **Weather Integration**: Consider weather conditions in pathfinding
- [ ] **Social Features**: Share and save favorite routes
- [ ] **Mobile App**: Native iOS/Android applications

## 🏷️ Version History

- **v1.0.0** - Initial release with React frontend and FastAPI backend
- **v0.1.0** - Original Jupyter notebook implementation

---

Made with ❤️ for the hiking community