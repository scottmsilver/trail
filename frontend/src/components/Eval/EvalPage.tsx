import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import type { LatLngExpression } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import './EvalPage.css'
import api from '../../services/api'
import type { Coordinate, RouteOptions } from '../../services/api'
import { scorePath } from '../../services/evalApi'
import type { ScoredPath, EvalCase } from '../../services/evalApi'
import DrawLayer from './DrawLayer'
import CalibrationToolbar from '../CalibrationToolbar/CalibrationToolbar'
import ComparePanel from './ComparePanel'
import AttributionPanel from './AttributionPanel'
import CasesPanel from './CasesPanel'

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

/** The Eval workbench: place start/end, run the engine's optimal path and score
 *  it, draw a candidate path that also gets scored, compare + attribute the
 *  cost difference, and tune the cost weights (calibration) to see whether the
 *  engine would then prefer the drawn path. */
export default function EvalPage() {
  const [start, setStart] = useState<Coordinate | null>(null)
  const [end, setEnd] = useState<Coordinate | null>(null)
  const [optimal, setOptimal] = useState<ScoredPath | null>(null)
  const [drawn, setDrawn] = useState<ScoredPath | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [options, setOptions] = useState<RouteOptions>({ userProfile: 'default' })
  const [status, setStatus] = useState('')
  const [running, setRunning] = useState(false)
  const [hoveredSegment, setHoveredSegment] = useState<number | null>(null)

  // Latest values for the debounced calibration effect to read without
  // re-triggering itself (it must fire on `options` change only).
  const startRef = useRef(start)
  const endRef = useRef(end)
  const drawnRef = useRef(drawn)
  const ranOnceRef = useRef(false)
  startRef.current = start
  endRef.current = end
  drawnRef.current = drawn

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

  /** Compute the engine's optimal route for the given options and score it. */
  const runOptimal = async (opts: RouteOptions) => {
    const s = startRef.current
    const e = endRef.current
    if (!s || !e) return
    setRunning(true)
    setStatus('Running…')
    try {
      const { routeId } = await api.calculateRoute(s, e, opts)
      let path: Coordinate[] | null = null
      for (let retries = 0; retries < 30; retries++) {
        const st = await api.getRouteStatus(routeId)
        setStatus(`Running… ${st.progress}%`)
        if (st.status === 'completed') {
          path = (await api.getRoute(routeId)).path
          break
        }
        if (st.status === 'failed') throw new Error(st.message || 'route calculation failed')
        await new Promise((r) => setTimeout(r, 1000))
      }
      if (!path) throw new Error('route did not complete in time')
      setStatus('Scoring optimal path…')
      const scored = await scorePath(path, opts)
      setOptimal(scored)
      ranOnceRef.current = true
      setStatus(`Optimal scored (cost ${Math.round(scored.totalCost)}).`)
    } catch (err) {
      setStatus('Error: ' + (err as Error).message)
    } finally {
      setRunning(false)
    }
  }

  /** Re-score the already-drawn candidate under the given options (cheap). */
  const rescoreDrawn = async (opts: RouteOptions) => {
    const d = drawnRef.current
    if (!d || d.path.length < 2) return
    try {
      setDrawn(await scorePath(d.path, opts))
    } catch (err) {
      setStatus('Error scoring drawn path: ' + (err as Error).message)
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

  // Calibration loop: when weights change, re-run the optimal (if we've run at
  // least once) and re-score the drawn path — both under the new options — so
  // the comparison and verdict stay apples-to-apples. Debounced.
  useEffect(() => {
    if (!ranOnceRef.current && !drawnRef.current) return
    const t = setTimeout(() => {
      if (ranOnceRef.current) runOptimal(options)
      rescoreDrawn(options)
    }, 600)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options])

  /** Load a saved case: restore endpoints/weights, re-score its reference path
   *  as the drawn candidate, and re-run the optimal so both are fresh. */
  const loadCase = async (c: EvalCase) => {
    setStart(c.start)
    setEnd(c.end)
    setOptions(c.options)
    startRef.current = c.start
    endRef.current = c.end
    setHoveredSegment(null)
    setStatus(`Loaded "${c.name}".`)
    if (c.referencePath.length >= 2) {
      try {
        const scored = await scorePath(c.referencePath, c.options)
        setDrawn(scored)
        drawnRef.current = scored
      } catch (err) {
        setStatus('Error scoring case path: ' + (err as Error).message)
      }
    } else {
      setDrawn(null)
      drawnRef.current = null
    }
    await runOptimal(c.options)
  }

  const clear = () => {
    setStart(null)
    setEnd(null)
    setOptimal(null)
    setDrawn(null)
    setDrawing(false)
    setHoveredSegment(null)
    ranOnceRef.current = false
    setStatus('')
  }

  // Verdict for the what-if loop.
  let verdict: { text: string; win: boolean } | null = null
  if (optimal && drawn) {
    const win = drawn.totalCost <= optimal.totalCost
    verdict = {
      win,
      text: win
        ? `Engine would prefer yours (${Math.round(drawn.totalCost)} ≤ ${Math.round(optimal.totalCost)})`
        : `Engine still prefers optimal (${Math.round(optimal.totalCost)} < ${Math.round(drawn.totalCost)})`,
    }
  }

  const hovered = hoveredSegment != null && drawn ? drawn.segments[hoveredSegment] : null

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

          {/* Highlight the segment currently hovered in the attribution panel. */}
          {hovered && (
            <Polyline
              positions={[
                [hovered.from.lat, hovered.from.lon],
                [hovered.to.lat, hovered.to.lon],
              ]}
              pathOptions={{ color: '#dc2626', weight: 8, opacity: 0.9 }}
            />
          )}
        </MapContainer>
      </div>

      <aside className="eval-sidebar">
        <h2>Eval</h2>

        <CalibrationToolbar options={options} onChange={setOptions} />

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
            onClick={() => runOptimal(options)}
            disabled={!start || !end || running}
          >
            {running ? 'Running…' : 'Run'}
          </button>
          <button
            className={`eval-btn ${drawing ? 'eval-btn-active' : ''}`}
            onClick={() => {
              setDrawing((d) => !d)
              setStatus(
                drawing ? '' : 'Drawing: click to add points, double-click or Enter to finish.',
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

        {verdict && (
          <div className={`eval-verdict ${verdict.win ? 'eval-verdict-win' : 'eval-verdict-lose'}`}>
            {verdict.text}
          </div>
        )}

        <div data-testid="eval-panels-slot" className="eval-panels-slot">
          <ComparePanel optimal={optimal} drawn={drawn} />
          {drawn && <AttributionPanel scored={drawn} onHoverSegment={setHoveredSegment} />}
        </div>

        <CasesPanel
          current={{ start, end, options, referencePath: drawn?.path ?? [] }}
          onLoad={loadCase}
        />
      </aside>
    </div>
  )
}
