import React, { useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import './App.css';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Custom icons for start and goal markers
const startIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const goalIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

// Component to handle map clicks
function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => {
      onMapClick(e.latlng);
    },
  });
  return null;
}

function App() {
  // Default coordinates (Utah - Wasatch Mountains area)
  const [startPoint, setStartPoint] = useState({ lat: 40.657192, lng: -111.568765 });
  const [goalPoint, setGoalPoint] = useState({ lat: 40.694144, lng: -111.604561 });
  const [pathCoords, setPathCoords] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [clickMode, setClickMode] = useState('start'); // 'start' or 'goal'
  const [settings, setSettings] = useState({
    margin: 0.003,
    resolution: 2
  });

  const handleMapClick = useCallback((latlng) => {
    if (clickMode === 'start') {
      setStartPoint({ lat: latlng.lat, lng: latlng.lng });
      setPathCoords([]); // Clear existing path
      setResult(null);
    } else if (clickMode === 'goal') {
      setGoalPoint({ lat: latlng.lat, lng: latlng.lng });
      setPathCoords([]); // Clear existing path
      setResult(null);
    }
  }, [clickMode]);

  const findPath = async () => {
    setIsLoading(true);
    setResult(null);
    setPathCoords([]);

    try {
      const response = await axios.post('/find-path', {
        start_lat: startPoint.lat,
        start_lng: startPoint.lng,
        goal_lat: goalPoint.lat,
        goal_lng: goalPoint.lng,
        margin: settings.margin,
        resolution: settings.resolution
      });

      const data = response.data;
      setResult(data);

      if (data.success && data.path_coords) {
        setPathCoords(data.path_coords);
      }
    } catch (error) {
      console.error('Error finding path:', error);
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to connect to server'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const downloadGPX = async () => {
    try {
      const response = await axios.post('/download-gpx', {
        start_lat: startPoint.lat,
        start_lng: startPoint.lng,
        goal_lat: goalPoint.lat,
        goal_lng: goalPoint.lng,
        margin: settings.margin,
        resolution: settings.resolution
      }, {
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'trail_path.gpx');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading GPX:', error);
      alert('Failed to download GPX file');
    }
  };

  const resetPoints = () => {
    setStartPoint({ lat: 40.657192, lng: -111.568765 });
    setGoalPoint({ lat: 40.694144, lng: -111.604561 });
    setPathCoords([]);
    setResult(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏔️ Trail Pathfinder</h1>
        <p>AI-powered hiking trail optimization using A* pathfinding</p>
      </header>

      <div className="app-container">
        <div className="controls-panel">
          <div className="control-section">
            <h3>📍 Set Points</h3>
            <div className="click-mode">
              <label>
                <input
                  type="radio"
                  value="start"
                  checked={clickMode === 'start'}
                  onChange={(e) => setClickMode(e.target.value)}
                />
                Set Start Point (Green)
              </label>
              <label>
                <input
                  type="radio"
                  value="goal"
                  checked={clickMode === 'goal'}
                  onChange={(e) => setClickMode(e.target.value)}
                />
                Set Goal Point (Red)
              </label>
            </div>
            <button onClick={resetPoints} className="btn btn-secondary">
              Reset to Utah Default
            </button>
          </div>

          <div className="control-section">
            <h3>⚙️ Settings</h3>
            <div className="setting-item">
              <label>
                Search Margin (degrees):
                <input
                  type="number"
                  step="0.001"
                  min="0.001"
                  max="0.01"
                  value={settings.margin}
                  onChange={(e) => setSettings({...settings, margin: parseFloat(e.target.value)})}
                />
              </label>
              <small>Smaller = faster, larger = more path options</small>
            </div>
            <div className="setting-item">
              <label>
                Resolution (meters):
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={settings.resolution}
                  onChange={(e) => setSettings({...settings, resolution: parseInt(e.target.value)})}
                />
              </label>
              <small>Lower = more precise, higher = faster</small>
            </div>
          </div>

          <div className="control-section">
            <h3>🎯 Actions</h3>
            <button 
              onClick={findPath} 
              disabled={isLoading}
              className="btn btn-primary"
            >
              {isLoading ? '🔄 Finding Path...' : '🔍 Find Optimal Trail'}
            </button>
            
            {result && result.success && (
              <button 
                onClick={downloadGPX}
                className="btn btn-success"
              >
                📥 Download GPX
              </button>
            )}
          </div>

          {result && (
            <div className="control-section">
              <h3>📊 Results</h3>
              <div className={`result ${result.success ? 'success' : 'error'}`}>
                <p><strong>Status:</strong> {result.message}</p>
                {result.success && (
                  <>
                    <p><strong>Path Length:</strong> {result.path_length_km?.toFixed(2)} km</p>
                    <p><strong>Processing Time:</strong> {result.processing_time?.toFixed(1)}s</p>
                    <p><strong>Nodes Explored:</strong> {result.nodes_explored?.toLocaleString()}</p>
                    <p><strong>Grid Size:</strong> {result.grid_size?.[0]} × {result.grid_size?.[1]}</p>
                  </>
                )}
              </div>
            </div>
          )}

          <div className="control-section">
            <h3>ℹ️ Instructions</h3>
            <ul>
              <li>Select "Set Start Point" or "Set Goal Point"</li>
              <li>Click on the map to place markers</li>
              <li>Adjust settings for speed vs accuracy</li>
              <li>Click "Find Optimal Trail" to calculate path</li>
              <li>Download GPX file for GPS navigation</li>
            </ul>
          </div>
        </div>

        <div className="map-container">
          <MapContainer
            center={[40.67, -111.59]}
            zoom={13}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            
            {/* Topographic overlay for hiking */}
            <TileLayer
              attribution='&copy; <a href="https://www.opentopomap.org">OpenTopoMap</a>'
              url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
              opacity={0.6}
            />

            <MapClickHandler onMapClick={handleMapClick} />

            {/* Start marker */}
            <Marker position={[startPoint.lat, startPoint.lng]} icon={startIcon}>
              <Popup>
                <strong>Start Point</strong><br/>
                Lat: {startPoint.lat.toFixed(6)}<br/>
                Lng: {startPoint.lng.toFixed(6)}
              </Popup>
            </Marker>

            {/* Goal marker */}
            <Marker position={[goalPoint.lat, goalPoint.lng]} icon={goalIcon}>
              <Popup>
                <strong>Goal Point</strong><br/>
                Lat: {goalPoint.lat.toFixed(6)}<br/>
                Lng: {goalPoint.lng.toFixed(6)}
              </Popup>
            </Marker>

            {/* Path polyline */}
            {pathCoords.length > 0 && (
              <Polyline
                positions={pathCoords}
                color="blue"
                weight={4}
                opacity={0.8}
              >
                <Popup>
                  Optimal Trail Path<br/>
                  Length: {result?.path_length_km?.toFixed(2)} km
                </Popup>
              </Polyline>
            )}
          </MapContainer>
        </div>
      </div>
    </div>
  );
}

export default App;