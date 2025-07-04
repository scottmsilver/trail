import React, { useState, useEffect } from 'react';
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
  const [markers, setMarkers] = useState<L.Marker[]>([]);

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
      markers.forEach(marker => {
        if (map) map.removeLayer(marker);
      });
    };
  }, []);

  // Handle map clicks
  useEffect(() => {
    if (!map || !isSelecting) return;

    const handleMapClick = (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      
      if (!corner1) {
        setCorner1({ lat, lon: lng });
        
        // Add marker
        const marker = L.marker([lat, lng])
          .addTo(map)
          .bindPopup('Corner 1');
        setMarkers([marker]);
      } else if (!corner2) {
        setCorner2({ lat, lon: lng });
        setIsSelecting(false);
        
        // Add second marker
        const marker = L.marker([lat, lng])
          .addTo(map)
          .bindPopup('Corner 2');
        setMarkers(prev => [...prev, marker]);
        
        // Draw rectangle
        const bounds = L.latLngBounds(
          [corner1.lat, corner1.lon],
          [lat, lng]
        );
        const rect = L.rectangle(bounds, {
          color: '#3B82F6',
          weight: 2,
          opacity: 0.8,
          fillOpacity: 0.2,
        }).addTo(map);
        setRectangle(rect);
      }
    };

    map.on('click', handleMapClick);

    return () => {
      map.off('click', handleMapClick);
    };
  }, [map, isSelecting, corner1, corner2]);

  const startSelection = () => {
    // Clear previous selection
    if (rectangle && map) {
      map.removeLayer(rectangle);
      setRectangle(null);
    }
    markers.forEach(marker => {
      if (map) map.removeLayer(marker);
    });
    setMarkers([]);
    
    setCorner1(null);
    setCorner2(null);
    setResult(null);
    setError(null);
    setIsSelecting(true);
  };

  const cancelSelection = () => {
    setIsSelecting(false);
    if (rectangle && map) {
      map.removeLayer(rectangle);
      setRectangle(null);
    }
    markers.forEach(marker => {
      if (map) map.removeLayer(marker);
    });
    setMarkers([]);
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

  return (
    <div className="prepopulate-area-card">
      <div className="card-header">
        <div className="header-content">
          <h3 className="card-title">Prepopulate Area</h3>
          <p className="card-description">
            Click two corners on the map to define an area for prepopulation
          </p>
        </div>
        {onClose && (
          <button className="close-btn" onClick={onClose}>
            ×
          </button>
        )}
      </div>
      
      <div className="card-content">
        {/* Instructions */}
        {!isSelecting && !corner1 && !result && (
          <div className="alert alert-info">
            <span className="icon">📍</span>
            <p>Click "Select Area" then click two points on the map to define the corners of the area you want to prepopulate.</p>
          </div>
        )}

        {/* Selection status */}
        {isSelecting && (
          <div className="alert alert-primary">
            <span className="icon">⬚</span>
            <p>
              {!corner1 
                ? "Click on the map to set the first corner"
                : "Click on the map to set the second corner"}
            </p>
          </div>
        )}

        {/* Area info */}
        {corner1 && corner2 && !isLoading && !result && (
          <div className="area-info">
            <h4>Selected Area</h4>
            <div className="info-grid">
              <span>Corner 1:</span>
              <span>{corner1.lat.toFixed(4)}, {corner1.lon.toFixed(4)}</span>
              <span>Corner 2:</span>
              <span>{corner2.lat.toFixed(4)}, {corner2.lon.toFixed(4)}</span>
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