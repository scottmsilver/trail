import React, { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import './PrepopulateArea.css';

interface Corner {
  lat: number;
  lon: number;
}

interface PrepopulateAreaProps {
  map?: L.Map;
  onClose?: () => void;
}

interface PrepopulateResponse {
  status: string;
  area: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
    area_km2: number;
  };
  cache_growth: {
    terrain_entries_added: number;
    cost_surfaces_added: number;
    memory_added_mb: number;
  };
}

const PrepopulateArea: React.FC<PrepopulateAreaProps> = ({ map, onClose }) => {
  const [corner1, setCorner1] = useState<Corner | null>(null);
  const [corner2, setCorner2] = useState<Corner | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<PrepopulateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rectangle, setRectangle] = useState<L.Rectangle | null>(null);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<L.LatLng | null>(null);
  const [tempRectangle, setTempRectangle] = useState<L.Rectangle | null>(null);

  // Calculate area in km²
  const calculateArea = (c1: Corner, c2: Corner): number => {
    const latDiff = Math.abs(c2.lat - c1.lat);
    const lonDiff = Math.abs(c2.lon - c1.lon);
    return latDiff * 111 * lonDiff * 111 * 0.7; // Rough approximation
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (rectangle && map) {
        map.removeLayer(rectangle);
      }
      if (tempRectangle && map) {
        map.removeLayer(tempRectangle);
      }
    };
  }, []);

  // Handle map interactions for click-and-drag
  useEffect(() => {
    if (!map || !isSelecting) return;

    const handleMouseDown = (e: L.LeafletMouseEvent) => {
      setIsDragging(true);
      setDragStart(e.latlng);
      setCorner1({ lat: e.latlng.lat, lon: e.latlng.lng });
      
      // Clear any existing rectangles
      if (tempRectangle) {
        map.removeLayer(tempRectangle);
        setTempRectangle(null);
      }
      if (rectangle) {
        map.removeLayer(rectangle);
        setRectangle(null);
      }
    };

    const handleMouseMove = (e: L.LeafletMouseEvent) => {
      if (!isDragging || !dragStart) return;

      // Update temporary rectangle
      if (tempRectangle) {
        map.removeLayer(tempRectangle);
      }

      const bounds = L.latLngBounds(dragStart, e.latlng);
      const rect = L.rectangle(bounds, {
        color: '#3B82F6',
        weight: 2,
        opacity: 0.5,
        fillOpacity: 0.1,
        dashArray: '5, 5'
      }).addTo(map);
      
      setTempRectangle(rect);
    };

    const handleMouseUp = (e: L.LeafletMouseEvent) => {
      if (!isDragging || !dragStart) return;

      setIsDragging(false);
      setCorner2({ lat: e.latlng.lat, lon: e.latlng.lng });
      setIsSelecting(false);

      // Remove temporary rectangle
      if (tempRectangle) {
        map.removeLayer(tempRectangle);
        setTempRectangle(null);
      }

      // Draw final rectangle
      const bounds = L.latLngBounds(dragStart, e.latlng);
      const rect = L.rectangle(bounds, {
        color: '#3B82F6',
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.2,
      }).addTo(map);
      setRectangle(rect);

      // Minimize the dialog after selection
      setIsMinimized(true);
    };

    // Add event listeners
    map.on('mousedown', handleMouseDown);
    map.on('mousemove', handleMouseMove);
    map.on('mouseup', handleMouseUp);

    // Change cursor when selecting
    map.getContainer().style.cursor = 'crosshair';

    return () => {
      map.off('mousedown', handleMouseDown);
      map.off('mousemove', handleMouseMove);
      map.off('mouseup', handleMouseUp);
      map.getContainer().style.cursor = '';
    };
  }, [map, isSelecting, isDragging, dragStart, tempRectangle, rectangle]);

  const startSelection = () => {
    // Clear previous selection
    if (rectangle && map) {
      map.removeLayer(rectangle);
      setRectangle(null);
    }
    if (tempRectangle && map) {
      map.removeLayer(tempRectangle);
      setTempRectangle(null);
    }
    
    setCorner1(null);
    setCorner2(null);
    setResult(null);
    setError(null);
    setIsSelecting(true);
    setIsMinimized(false);
  };

  const cancelSelection = () => {
    setIsSelecting(false);
    if (rectangle && map) {
      map.removeLayer(rectangle);
      setRectangle(null);
    }
    if (tempRectangle && map) {
      map.removeLayer(tempRectangle);
      setTempRectangle(null);
    }
    setCorner1(null);
    setCorner2(null);
  };

  const prepopulateArea = async () => {
    if (!corner1 || !corner2) return;

    setIsLoading(true);
    setError(null);
    setProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 1000);

      const response = await fetch('http://localhost:9001/api/cache/prepopulate-box', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          corner1,
          corner2,
        }),
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        throw new Error(`Failed to prepopulate: ${response.statusText}`);
      }

      const data: PrepopulateResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  // Minimized view
  if (isMinimized) {
    return (
      <div className="prepopulate-area-minimized">
        <button 
          className="minimize-button"
          onClick={() => setIsMinimized(false)}
          title="Expand prepopulate panel"
        >
          <span className="icon">📦</span>
          <span>Prepopulate Area</span>
          {corner1 && corner2 && !result && <span className="badge">Ready</span>}
          {result && <span className="badge success">✓</span>}
        </button>
        {onClose && (
          <button className="close-btn-mini" onClick={onClose}>
            ×
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="prepopulate-area-card">
      <div className="card-header">
        <div className="header-content">
          <h3 className="card-title">Prepopulate Area</h3>
          <p className="card-description">
            Click and drag on the map to select an area
          </p>
        </div>
        <div className="header-actions">
          <button 
            className="minimize-btn" 
            onClick={() => setIsMinimized(true)}
            title="Minimize"
          >
            −
          </button>
          {onClose && (
            <button className="close-btn" onClick={onClose}>
              ×
            </button>
          )}
        </div>
      </div>
      
      <div className="card-content">
        {/* Instructions */}
        {!isSelecting && !corner1 && !result && (
          <div className="alert alert-info">
            <span className="icon">📍</span>
            <p>Click "Select Area" then click and drag on the map to draw a rectangle around the area you want to prepopulate.</p>
          </div>
        )}

        {/* Selection status */}
        {isSelecting && (
          <div className="alert alert-primary">
            <span className="icon">⬚</span>
            <p>Click and drag on the map to select area</p>
          </div>
        )}

        {/* Area info */}
        {corner1 && corner2 && !isLoading && !result && (
          <div className="area-info">
            <h4>Selected Area</h4>
            <div className="info-grid">
              <span>Area:</span>
              <span>~{calculateArea(corner1, corner2).toFixed(1)} km²</span>
            </div>
          </div>
        )}

        {/* Progress */}
        {isLoading && (
          <div className="progress-section">
            <div className="progress-header">
              <span>Prepopulating area...</span>
              <span>{progress}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="progress-hint">
              This may take a few moments depending on the area size
            </p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="result-info">
            <h4>✓ Prepopulation Complete</h4>
            <div className="info-grid">
              <span>Area size:</span>
              <span>{result.area.area_km2.toFixed(1)} km²</span>
              <span>Terrain cached:</span>
              <span>{result.cache_growth.terrain_entries_added} entries</span>
              <span>Cost surfaces:</span>
              <span>{result.cache_growth.cost_surfaces_added} surfaces</span>
              <span>Memory used:</span>
              <span>{result.cache_growth.memory_added_mb.toFixed(1)} MB</span>
            </div>
            <p className="result-hint">
              Routes in this area will now calculate much faster!
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="alert alert-error">
            <p>{error}</p>
          </div>
        )}

        {/* Actions */}
        <div className="actions">
          {!isSelecting && !corner2 && !isLoading && (
            <button className="btn btn-primary" onClick={startSelection}>
              <span className="icon">📍</span>
              Select Area
            </button>
          )}
          
          {isSelecting && (
            <button className="btn btn-secondary" onClick={cancelSelection}>
              Cancel
            </button>
          )}
          
          {corner1 && corner2 && !isLoading && !result && (
            <>
              <button className="btn btn-secondary" onClick={startSelection}>
                Reselect
              </button>
              <button className="btn btn-primary" onClick={prepopulateArea}>
                <span className="icon">⬇</span>
                Prepopulate
              </button>
            </>
          )}
          
          {result && (
            <button className="btn btn-primary" onClick={startSelection}>
              Select Another Area
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PrepopulateArea;