import { useState } from 'react'
import './App.css'
import Map from './components/Map/Map'
import api from './services/api'
import type { Coordinate, RouteResult } from './services/api'

function App() {
  const [start, setStart] = useState<Coordinate | null>(null)
  const [end, setEnd] = useState<Coordinate | null>(null)
  const [route, setRoute] = useState<RouteResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')

  const handleMapClick = (coord: Coordinate) => {
    if (!start) {
      setStart(coord)
      setStatus('Start point set. Click to set end point.')
    } else if (!end) {
      setEnd(coord)
      setStatus('End point set. Click "Find Route" to calculate.')
    } else {
      // Reset
      setStart(coord)
      setEnd(null)
      setRoute(null)
      setStatus('Start point set. Click to set end point.')
    }
  }

  const findRoute = async () => {
    if (!start || !end) return

    setLoading(true)
    setStatus('Calculating route...')
    
    try {
      // Use the imported api instance
      const response = await api.calculateRoute(start, end)
      
      // Poll for status
      let retries = 0
      const maxRetries = 30
      
      while (retries < maxRetries) {
        const status = await api.getRouteStatus(response.routeId)
        setStatus(`Processing: ${status.progress}%`)
        
        if (status.status === 'completed') {
          const routeData = await api.getRoute(response.routeId)
          setRoute(routeData)
          setStatus(`Route found! Distance: ${routeData.stats.distance_km}km`)
          break
        } else if (status.status === 'failed') {
          setStatus('Route calculation failed: ' + status.message)
          break
        }
        
        await new Promise(resolve => setTimeout(resolve, 1000))
        retries++
      }
    } catch (error) {
      setStatus('Error: ' + (error as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setStart(null)
    setEnd(null)
    setRoute(null)
    setStatus('Click on the map to set start point.')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Trail Finder</h1>
        <p>Find the optimal hiking route between two points</p>
      </header>
      
      <main className="app-main">
        <div className="controls">
          <div className="status">{status || 'Click on the map to set start point.'}</div>
          
          <div className="coordinates">
            {start && (
              <div>Start: {start.lat.toFixed(4)}, {start.lon.toFixed(4)}</div>
            )}
            {end && (
              <div>End: {end.lat.toFixed(4)}, {end.lon.toFixed(4)}</div>
            )}
          </div>
          
          <div className="buttons">
            <button 
              onClick={findRoute} 
              disabled={!start || !end || loading}
              className="btn-primary"
            >
              {loading ? 'Calculating...' : 'Find Route'}
            </button>
            
            <button onClick={reset} className="btn-secondary">
              Reset
            </button>
          </div>
          
          {route && (
            <div className="route-info">
              <h3>Route Information</h3>
              <ul>
                <li>Distance: {route.stats.distance_km} km</li>
                <li>Estimated time: {route.stats.estimated_time_min} min</li>
                <li>Difficulty: {route.stats.difficulty}</li>
                <li>Waypoints: {route.stats.waypoints || route.path.length}</li>
              </ul>
            </div>
          )}
        </div>
        
        <div className="map-wrapper">
          <Map 
            start={start || undefined}
            end={end || undefined}
            path={route?.path}
            onMapClick={handleMapClick}
          />
        </div>
      </main>
    </div>
  )
}

export default App