import { useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import type { LatLngExpression } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import './EvalPage.css'
import api from '../../services/api'
import type { Coordinate, RouteOptions } from '../../services/api'
import { scorePath } from '../../services/evalApi'
import type { ScoredPath } from '../../services/evalApi'
import DrawLayer from './DrawLayer'

// Fix for default markers not showing (mirrors Map.tsx). Idempotent.
import icon from 'leaflet/dist/images/marker-icon.png'
import iconShadow from 'leaflet/dist/images/marker-shadow.png'

L.Marker.prototype.options.icon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
})

/** Forwards map clicks to EvalPage for start/end placement. Disabled while
 *  drawing so DrawLayer owns clicks instead. */
function EvalClickHandler({
  disabled,
  onClick,
}: {
  disabled: boolean
  onClick: (coord: Coordinate) => void
}) {
  useMapEvents({
    click: (e) => {
      if (disabled) return
      onClick({ lat: e.latlng.lat, lon: e.latlng.lng })
    },
  })
  return null
}

/** The Eval workbench: place start/end, run the engine's optimal path and
 *  score it, and draw a candidate path that also gets scored. Comparison and
 *  attribution panels mount in the marked slot in a later task (T5); the
 *  calibration toolbar is T6. */
export default function EvalPage() {
  const [start, setStart] = useState<Coordinate | null>(null)
  const [end, setEnd] = useState<Coordinate | null>(null)
  const [optimal, setOptimal] = useState<ScoredPath | null>(null)
  const [drawn, setDrawn] = useState<ScoredPath | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [options] = useState<RouteOptions>({ userProfile: 'default' })
  const [status, setStatus] = useState('')
  const [running, setRunning] = useState(false)

  const defaultCenter: LatLngExpression = [40.64, -111.57]

  const handleMapClick = (coord: Coordinate) => {
    if (!start) {
      setStart(coord)
      setStatus('Start set. Click to set end.')
    } else if (!end) {
      setEnd(coord)
      setStatus('End set. Click "Run" to score the optimal path.')
    } else {
      // Third click restarts the selection.
      setStart(coord)
      setEnd(null)
      setStatus('Start set. Click to set end.')
    }
  }

  const run = async () => {
    if (!start || !end) return
    setRunning(true)
    setStatus('Running…')
    try {
      const { routeId } = await api.calculateRoute(start, end, options)

      // Poll until the engine finishes, mirroring App.tsx.
      let path: Coordinate[] | null = null
      for (let retries = 0; retries < 30; retries++) {
        const s = await api.getRouteStatus(routeId)
        setStatus(`Running… ${s.progress}%`)
        if (s.status === 'completed') {
          const result = await api.getRoute(routeId)
          path = result.path
          break
        }
        if (s.status === 'failed') {
          throw new Error(s.message || 'route calculation failed')
        }
        await new Promise((r) => setTimeout(r, 1000))
      }

      if (!path) throw new Error('route did not complete in time')

      setStatus('Scoring optimal path…')
      const scored = await scorePath(path, options)
      setOptimal(scored)
      setStatus(`Optimal path scored (cost ${Math.round(scored.totalCost)}).`)
    } catch (err) {
      setStatus('Error: ' + (err as Error).message)
    } finally {
      setRunning(false)
    }
  }

  const handleDrawFinish = async (points: Coordinate[]) => {
    setDrawing(false)
    if (points.length < 2) {
      setStatus('Drawn path needs at least two points.')
      return
    }
    setStatus('Scoring drawn path…')
    try {
      const scored = await scorePath(points, options)
      setDrawn(scored)
      setStatus(`Drawn path scored (cost ${Math.round(scored.totalCost)}).`)
    } catch (err) {
      setStatus('Error scoring drawn path: ' + (err as Error).message)
    }
  }

  const clear = () => {
    setStart(null)
    setEnd(null)
    setOptimal(null)
    setDrawn(null)
    setDrawing(false)
    setStatus('')
  }

  return (
    <div className="eval-page" data-testid="eval-page">
      <div className="eval-map-wrapper">
        <MapContainer center={defaultCenter} zoom={13} className="eval-map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <EvalClickHandler disabled={drawing} onClick={handleMapClick} />
          <DrawLayer active={drawing} onFinish={handleDrawFinish} />

          {start && (
            <Marker position={[start.lat, start.lon]}>
              <Popup>Start</Popup>
            </Marker>
          )}
          {end && (
            <Marker position={[end.lat, end.lon]}>
              <Popup>End</Popup>
            </Marker>
          )}

          {/* Engine's optimal path in blue. */}
          {optimal && optimal.path.length > 0 && (
            <Polyline
              positions={optimal.path.map((c) => [c.lat, c.lon] as [number, number])}
              pathOptions={{ color: '#2563eb', weight: 4 }}
            />
          )}

          {/* User-drawn candidate path in orange. */}
          {drawn && drawn.path.length > 0 && (
            <Polyline
              positions={drawn.path.map((c) => [c.lat, c.lon] as [number, number])}
              pathOptions={{ color: '#f97316', weight: 4 }}
            />
          )}
        </MapContainer>
      </div>

      <aside className="eval-sidebar">
        <h2>Eval</h2>

        <section className="eval-readout">
          <div className="eval-coord">
            <span className="eval-coord-label">Start</span>
            <span className="eval-coord-value">
              {start ? `${start.lat.toFixed(4)}, ${start.lon.toFixed(4)}` : '—'}
            </span>
          </div>
          <div className="eval-coord">
            <span className="eval-coord-label">End</span>
            <span className="eval-coord-value">
              {end ? `${end.lat.toFixed(4)}, ${end.lon.toFixed(4)}` : '—'}
            </span>
          </div>
        </section>

        <section className="eval-actions">
          <button
            className="eval-btn eval-btn-primary"
            onClick={run}
            disabled={!start || !end || running}
          >
            {running ? 'Running…' : 'Run'}
          </button>
          <button
            className={`eval-btn ${drawing ? 'eval-btn-active' : ''}`}
            onClick={() => {
              setDrawing((d) => !d)
              setStatus(
                drawing
                  ? ''
                  : 'Drawing: click to add points, double-click or Enter to finish.',
              )
            }}
          >
            {drawing ? 'Drawing… (finish)' : 'Draw path'}
          </button>
          <button className="eval-btn" onClick={clear}>
            Clear
          </button>
        </section>

        <div className="eval-status">{status || 'Click the map to set start, then end.'}</div>

        {/* Comparison + attribution panels mount here in a later task (T5). */}
        <div data-testid="eval-panels-slot" className="eval-panels-slot" />
      </aside>
    </div>
  )
}
