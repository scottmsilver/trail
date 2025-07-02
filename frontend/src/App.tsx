import { useState } from 'react'
import './App.css'
import Map from './components/Map/Map'
import SearchBox from './components/SearchBox/SearchBox'
import AlgorithmDebug from './components/AlgorithmDebug/AlgorithmDebug'
import AdvancedSettings from './components/AdvancedSettings/AdvancedSettings'
import api from './services/api'
import type { Coordinate, RouteResult, RouteOptions } from './services/api'

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
            console.log('Route data:', routeData)
            setRoute(routeData)
            setStatus(`Route found! Distance: ${routeData.stats.distance_km}km`)
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
    
    try {
      const debugResult: RouteResult = await api.calculateDebugRoute(start, end, routeOptions)
      console.log('Debug result:', debugResult)
      console.log('Debug result stats:', debugResult.stats)
      console.log('Debug data present:', !!debugResult.stats?.debug_data)
      
      // Set the route data
      setRoute(debugResult)
      
      // Set debug data and show debug panel
      if (debugResult.stats && debugResult.stats.debug_data) {
        setDebugData(debugResult.stats.debug_data)
        setShowDebug(true)
        setStatus(`Debug route found! Distance: ${debugResult.stats.distance_km}km - Check debug panel`)
      } else {
        setStatus(`Debug route found! Distance: ${debugResult.stats?.distance_km || 'N/A'}km (no debug data available)`)
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
    setStatus('Click on the map to set start point.')
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
          <div className="status">{status || 'Click on the map to set start point.'}</div>
          
          <div className="coordinates">
            {start && (
              <div>Start: {start.lat.toFixed(4)}, {start.lon.toFixed(4)}</div>
            )}
            {end && (
              <div>End: {end.lat.toFixed(4)}, {end.lon.toFixed(4)}</div>
            )}
          </div>
          
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
          
          <AdvancedSettings 
            options={routeOptions} 
            onChange={setRouteOptions} 
          />
          
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
                <li>Distance: {route.stats.distance_km} km</li>
                <li>Estimated time: {route.stats.estimated_time_min} min</li>
                <li>Difficulty: {route.stats.difficulty}</li>
                <li>Waypoints: {route.path.length}</li>
              </ul>
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
          />
          <SearchBox onLocationSelect={handleLocationSelect} />
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