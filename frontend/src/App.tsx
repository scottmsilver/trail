import { useState, useRef } from 'react'
import './App.css'
import Map from './components/Map/Map'
import SearchBox from './components/SearchBox/SearchBox'
import AlgorithmDebug from './components/AlgorithmDebug/AlgorithmDebug'
import CalibrationToolbar from './components/CalibrationToolbar/CalibrationToolbar'
import PrepopulateArea from './components/PrepopulateArea/PrepopulateArea'
import SavedLocations from './components/SavedLocations/SavedLocations'
import { useSavedLocations, type SavedLocation } from './hooks/useSavedLocations'
import api, { API_BASE } from './services/api'
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
  const [showCostSurface, setShowCostSurface] = useState(false)
  const [costSurfaceBounds, setCostSurfaceBounds] = useState<{north: number, south: number, east: number, west: number} | null>(null)
  const [costPointMode, setCostPointMode] = useState(false)
  const [cacheProgress, setCacheProgress] = useState<{active: boolean, message: string} | null>(null)
  // Overlay the engine's trail/path network for the current viewport.
  const [showTrails, setShowTrails] = useState(false)
  const [trailCount, setTrailCount] = useState<number | null>(null)
  const mapRef = useRef<L.Map | null>(null)

  const {
    presets,
    recents,
    addPreset,
    updatePreset,
    deletePreset,
    recordRecent,
    promoteRecentToPreset,
  } = useSavedLocations()

  // The point offered to "Save current point": prefer end, then start, then center.
  const currentPoint = end || start || mapCenter

  const useAsStart = (loc: SavedLocation) => {
    const coord = { lat: loc.lat, lon: loc.lon }
    setStart(coord)
    setMapCenter(coord)
    recordRecent(loc.lat, loc.lon, loc.name)
    setStatus(`Start set to "${loc.name}".`)
  }

  const useAsEnd = (loc: SavedLocation) => {
    const coord = { lat: loc.lat, lon: loc.lon }
    setEnd(coord)
    setMapCenter(coord)
    recordRecent(loc.lat, loc.lon, loc.name)
    setStatus(`End set to "${loc.name}".`)
  }

  const handleMapClick = (coord: Coordinate) => {
    // If a route has been found, don't accept new clicks
    if (route) {
      return
    }

    if (!start) {
      setStart(coord)
      recordRecent(coord.lat, coord.lon)
      setStatus('Start point set. Click to set end point.')
    } else if (!end) {
      setEnd(coord)
      recordRecent(coord.lat, coord.lon)
      setStatus('End point set. Click "Find Route" to calculate.')
    } else {
      // Reset
      setStart(coord)
      setEnd(null)
      setRoute(null)
      recordRecent(coord.lat, coord.lon)
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
      const response = await api.calculateRoute([start, end], routeOptions)

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
        const response = await fetch(`${API_BASE}/api/routes/${route.routeId}/gpx`)
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

        const blob = await api.exportRouteAsGPX([start, end], routeOptions)
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
    recordRecent(lat, lon, name)
    if (name) {
      setStatus(`Centered on: ${name}`)
    }
  }

  const handleQuickCache = async () => {
    if (!mapRef.current) return

    const bounds = mapRef.current.getBounds()
    // Add 20% buffer to each side
    const latBuffer = (bounds.getNorth() - bounds.getSouth()) * 0.2
    const lonBuffer = (bounds.getEast() - bounds.getWest()) * 0.2

    const expandedBounds = {
      north: bounds.getNorth() + latBuffer,
      south: bounds.getSouth() - latBuffer,
      east: bounds.getEast() + lonBuffer,
      west: bounds.getWest() - lonBuffer
    }

    // Calculate approximate area
    const approxAreaKm2 = (expandedBounds.north - expandedBounds.south) * 111 *
                         (expandedBounds.east - expandedBounds.west) * 111 * 0.7

    setCacheProgress({ active: true, message: `Caching ~${approxAreaKm2.toFixed(1)} km² for current view...` })

    try {
      const response = await fetch(`${API_BASE}/api/cache/prepopulate-box`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          corner1: { lat: expandedBounds.south, lon: expandedBounds.west },
          corner2: { lat: expandedBounds.north, lon: expandedBounds.east },
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Cache error response:', errorText)
        throw new Error(`Failed to cache: ${response.statusText} - ${errorText}`)
      }

      const data = await response.json()
      setCacheProgress({
        active: false,
        message: `✓ Cached ${data.area.area_km2.toFixed(1)} km² (${data.cache_growth.terrain_entries_added} tiles)`
      })

      // Clear the success message after 5 seconds
      setTimeout(() => {
        setCacheProgress(null)
      }, 5000)

    } catch (err) {
      setCacheProgress({
        active: false,
        message: `Error: ${err instanceof Error ? err.message : 'Failed to cache area'}`
      })
      setTimeout(() => {
        setCacheProgress(null)
      }, 5000)
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

          <div className="engine-selector">
            <label htmlFor="routing-engine">Routing Engine: </label>
            <select
              id="routing-engine"
              value={routeOptions.engine ?? 'v1'}
              onChange={(e) => {
                const engine = e.target.value as 'v1' | 'v2'
                setRouteOptions({
                  ...routeOptions,
                  engine,
                  // Drop v2-only params when switching back to v1
                  heuristicWeight: engine === 'v2' ? routeOptions.heuristicWeight : undefined,
                })
              }}
              className="profile-select"
            >
              <option value="v1">v1 (legacy DEM)</option>
              <option value="v2">v2 (terrain A*)</option>
            </select>
            {routeOptions.engine === 'v2' && (
              <label htmlFor="heuristic-weight" className="heuristic-weight-label">
                Heuristic weight:{' '}
                <input
                  id="heuristic-weight"
                  type="number"
                  min={1}
                  max={3}
                  step={0.1}
                  value={routeOptions.heuristicWeight ?? 1}
                  onChange={(e) => {
                    const raw = parseFloat(e.target.value)
                    setRouteOptions({
                      ...routeOptions,
                      heuristicWeight: Number.isNaN(raw) ? undefined : raw,
                    })
                  }}
                  className="heuristic-weight-input"
                  title="1.0 = optimal (slowest), higher = faster/greedier"
                />
              </label>
            )}
          </div>

          <SavedLocations
            presets={presets}
            recents={recents}
            currentPoint={currentPoint}
            onUseAsStart={useAsStart}
            onUseAsEnd={useAsEnd}
            onAddPreset={addPreset}
            onUpdatePreset={updatePreset}
            onDeletePreset={deletePreset}
            onPromoteRecent={promoteRecentToPreset}
          />

          <CalibrationToolbar
            options={routeOptions}
            onChange={setRouteOptions}
            onPrepopulateClick={() => setShowPrepopulate(true)}
          />

          <div className="cost-explorer-controls">
            <button
              className={`btn-cost-mode ${costPointMode ? 'active' : ''}`}
              onClick={() => {
                setCostPointMode(!costPointMode)
                setShowCostSurface(false)
                setCostSurfaceBounds(null)
              }}
              title="Click on map to see terrain cost at any point"
            >
              {costPointMode ? 'Exit Cost Explorer' : 'Quick Cost Explorer'}
            </button>

            <button
              className="btn-explore-cost"
              onClick={() => {
                if (mapRef.current) {
                  const bounds = mapRef.current.getBounds()
                  setCostSurfaceBounds({
                    north: bounds.getNorth(),
                    south: bounds.getSouth(),
                    east: bounds.getEast(),
                    west: bounds.getWest()
                  })
                  setShowCostSurface(true)
                  setCostPointMode(false)
                }
              }}
              title="Load full cost surface overlay for current view"
              disabled={costPointMode}
            >
              Full Cost Overlay
            </button>

            <button
              className="btn-quick-cache"
              onClick={() => handleQuickCache()}
              title="Cache terrain data for current view area"
            >
              Quick Cache View
            </button>
          </div>

          <label className="trails-toggle">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={(e) => setShowTrails(e.target.checked)}
            />
            Show trails the engine uses
            {showTrails && trailCount != null && (
              <span className="eval-trail-count"> ({trailCount} in view)</span>
            )}
          </label>

          <div className="status">{status || 'Click on the map to set start point.'}</div>

          {cacheProgress && (
            <div className={`cache-progress ${cacheProgress.active ? 'active' : 'complete'}`}>
              {cacheProgress.active && <span className="spinner">⏳</span>}
              <span>{cacheProgress.message}</span>
            </div>
          )}

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
              <button
                onClick={() => setShowCostSurface(!showCostSurface)}
                className="btn-cost-surface"
                title="Explore cost surface visualization"
              >
                {showCostSurface ? 'Hide' : 'Show'} Cost Surface
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
            showCostSurface={showCostSurface}
            onCloseCostSurface={() => {
              setShowCostSurface(false)
              setCostSurfaceBounds(null)
            }}
            costSurfaceBounds={costSurfaceBounds || undefined}
            costPointMode={costPointMode}
            showTrails={showTrails}
            onTrailCount={setTrailCount}
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
