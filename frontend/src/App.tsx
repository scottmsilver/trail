import { useState, useRef } from 'react'
import './App.css'
import Map from './components/Map/Map'
import SearchBox from './components/SearchBox/SearchBox'
import AlgorithmDebug from './components/AlgorithmDebug/AlgorithmDebug'
import CalibrationToolbar from './components/CalibrationToolbar/CalibrationToolbar'
import PrepopulateArea from './components/PrepopulateArea/PrepopulateArea'
import api from './services/api'
import type { Coordinate, RouteResult, RouteOptions } from './services/api'
import L from 'leaflet'

function App() {
  const [start, setStart] = useState<Coordinate | null>(null)
  const [end, setEnd] = useState<Coordinate | null>(null)
  const [route, setRoute] = useState<RouteResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')
  const [mapCenter, setMapCenter] = useState<Coordinate | null>(null)
  const [debugData, setDebugData] = useState<any>(null)
  const [showDebug, setShowDebug] = useState(false)
  const [debugLoading, setDebugLoading] = useState(false)
  const [userProfile, setUserProfile] = useState<string>('default')
  const [routeOptions, setRouteOptions] = useState<RouteOptions>({ userProfile: 'default' })
  const [routeTime, setRouteTime] = useState<number | null>(null)
  const [showPrepopulate, setShowPrepopulate] = useState(false)
  const mapRef = useRef<L.Map | null>(null)

  const handleMapClick = (coord: Coordinate) => {
    // If a route has been found, don't accept new clicks
    if (route) {
      return
    }
    
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
    const startTime = performance.now()
    
    try {
      // Use the imported api instance with all route options
      const response = await api.calculateRoute(start, end, routeOptions)
      
      // Poll for status
      let retries = 0
      const maxRetries = 30
      
      while (retries < maxRetries) {
        const status = await api.getRouteStatus(response.routeId)
        setStatus(`Processing: ${status.progress}%`)
        
        if (status.status === 'completed') {
          try {
            const routeData = await api.getRoute(response.routeId)
            const endTime = performance.now()
            const timeTaken = (endTime - startTime) / 1000 // Convert to seconds
            setRouteTime(timeTaken)
            console.log('Route data:', routeData)
            setRoute(routeData)
            const distanceMiles = (routeData.stats.distance_km * 0.621371).toFixed(2)
            setStatus(`Route found! Distance: ${distanceMiles} miles (${timeTaken.toFixed(1)}s)`)
          } catch (err) {
            console.error('Error getting route:', err)
            setStatus('Error getting route data')
          }
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

  const debugRoute = async () => {
    if (!start || !end) return

    setDebugLoading(true)
    setStatus('Calculating route with debug info...')
    const startTime = performance.now()
    
    try {
      const debugResult: RouteResult = await api.calculateDebugRoute(start, end, routeOptions)
      const endTime = performance.now()
      const timeTaken = (endTime - startTime) / 1000 // Convert to seconds
      setRouteTime(timeTaken)
      console.log('Debug result:', debugResult)
      console.log('Debug result stats:', debugResult.stats)
      console.log('Debug data present:', !!debugResult.stats?.debug_data)
      
      // Set the route data
      setRoute(debugResult)
      
      // Set debug data and show debug panel
      if (debugResult.stats && debugResult.stats.debug_data) {
        setDebugData(debugResult.stats.debug_data)
        setShowDebug(true)
        const distanceMiles = (debugResult.stats.distance_km * 0.621371).toFixed(2)
        setStatus(`Debug route found! Distance: ${distanceMiles} miles (${timeTaken.toFixed(1)}s) - Check debug panel`)
      } else {
        const distanceMiles = ((debugResult.stats?.distance_km || 0) * 0.621371).toFixed(2)
        setStatus(`Debug route found! Distance: ${distanceMiles} miles (${timeTaken.toFixed(1)}s) (no debug data available)`)
        console.warn('Debug route completed but no debug data returned:', debugResult)
        console.warn('Stats object:', debugResult.stats)
      }
    } catch (error) {
      setStatus('Debug route calculation failed: ' + (error as Error).message)
      console.error('Debug route error:', error)
    } finally {
      setDebugLoading(false)
    }
  }

  const reset = () => {
    setStart(null)
    setEnd(null)
    setRoute(null)
    setDebugData(null)
    setShowDebug(false)
    setRouteTime(null)
    setStatus('Click on the map to set start point.')
  }

  const exportGPX = async () => {
    if (!route) return
    
    try {
      setStatus('Exporting route as GPX...')
      
      // Use the route ID if it exists, otherwise use the export endpoint
      if (route.routeId && route.routeId !== 'debug') {
        // Download from existing route
        const response = await fetch(`http://localhost:9001/api/routes/${route.routeId}/gpx`)
        if (!response.ok) {
          throw new Error('Failed to export GPX')
        }
        
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `trail_route_${route.routeId.substring(0, 8)}.gpx`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
        
        setStatus('GPX file downloaded successfully!')
      } else {
        // Export directly from current route data
        if (!start || !end) {
          setStatus('No route data to export')
          return
        }
        
        const response = await api.exportRouteAsGPX(start, end, routeOptions)
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `trail_route_${new Date().toISOString().replace(/[:.]/g, '-')}.gpx`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
        
        setStatus('GPX file exported successfully!')
      }
    } catch (error) {
      console.error('Error exporting GPX:', error)
      setStatus('Error exporting GPX: ' + (error as Error).message)
    }
  }

  const handleLocationSelect = (lat: number, lon: number, name?: string) => {
    setMapCenter({ lat, lon })
    if (name) {
      setStatus(`Centered on: ${name}`)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Trail Finder</h1>
        <p>Find the optimal hiking route between two points</p>
      </header>
      
      <main className="app-main">
        <div className="controls">
          <div className="profile-selector">
            <label htmlFor="user-profile">User Profile: </label>
            <select 
              id="user-profile"
              value={userProfile} 
              onChange={(e) => {
                setUserProfile(e.target.value)
                setRouteOptions({...routeOptions, userProfile: e.target.value})
              }}
              className="profile-select"
            >
              <option value="default">Default</option>
              <option value="easy">Easy Hiker</option>
              <option value="experienced">Experienced Hiker</option>
              <option value="trail_runner">Trail Runner</option>
              <option value="accessibility">Accessibility Focused</option>
            </select>
          </div>

          <CalibrationToolbar 
            options={routeOptions} 
            onChange={setRouteOptions} 
            onPrepopulateClick={() => setShowPrepopulate(true)}
          />
          
          <div className="status">{status || 'Click on the map to set start point.'}</div>
          
          <div className="coordinates">
            {(start || end) && (
              <>
                {start && end && (
                  <button 
                    className="copy-coords-btn"
                    onClick={() => {
                      const coordsText = `"Start: ${start.lat.toFixed(4)}, ${start.lon.toFixed(4)}" "End: ${end.lat.toFixed(4)}, ${end.lon.toFixed(4)}"`
                      navigator.clipboard.writeText(coordsText)
                      setStatus('Coordinates copied to clipboard in CLI format!')
                    }}
                    title="Copy coordinates for route_cli.py"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                  </button>
                )}
                <div className="coord-list">
                  {start && (
                    <div className="coord-item">
                      <span className="coord-label">Start</span>
                      <span className="coord-value">{start.lat.toFixed(4)}, {start.lon.toFixed(4)}</span>
                    </div>
                  )}
                  {end && (
                    <div className="coord-item">
                      <span className="coord-label">End</span>
                      <span className="coord-value">{end.lat.toFixed(4)}, {end.lon.toFixed(4)}</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
          
          <div className="buttons">
            <button 
              onClick={findRoute} 
              disabled={!start || !end || loading || debugLoading}
              className="btn-primary"
            >
              {loading ? 'Calculating...' : 'Find Route'}
            </button>
            
            <button 
              onClick={debugRoute} 
              disabled={!start || !end || loading || debugLoading}
              className="btn-debug"
            >
              {debugLoading ? 'Debug Calculating...' : 'Debug Route'}
            </button>
            
            <button onClick={reset} className="btn-secondary">
              Reset
            </button>
          </div>
          
          {route && (
            <div className="route-info">
              <h3>Route Information</h3>
              <ul>
                <li>Distance: {(route.stats.distance_km * 0.621371).toFixed(2)} miles ({route.stats.distance_km} km)</li>
                <li>Estimated time: {route.stats.estimated_time_min} min</li>
                <li>Difficulty: {route.stats.difficulty}</li>
                <li>Waypoints: {route.path.length}</li>
                {routeTime && <li>Calculation time: {routeTime.toFixed(1)}s</li>}
              </ul>
              <button 
                onClick={exportGPX} 
                className="btn-export"
                title="Download route as GPX file"
              >
                Export GPX
              </button>
            </div>
          )}
        </div>
        
        <div className="map-wrapper">
          <Map 
            start={start || undefined}
            end={end || undefined}
            path={route?.path}
            pathWithSlopes={route?.stats?.path_with_slopes}
            center={mapCenter || undefined}
            onMapClick={handleMapClick}
            onMapReady={(map) => { mapRef.current = map }}
          />
          <SearchBox onLocationSelect={handleLocationSelect} />
          {showPrepopulate && mapRef.current && (
            <PrepopulateArea 
              map={mapRef.current} 
              onClose={() => setShowPrepopulate(false)} 
            />
          )}
        </div>
      </main>
      
      {showDebug && (
        <AlgorithmDebug 
          debugData={debugData} 
          onClose={() => setShowDebug(false)} 
        />
      )}
    </div>
  )
}

export default App