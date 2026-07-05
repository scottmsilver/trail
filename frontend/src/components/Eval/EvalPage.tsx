import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import type { LatLngExpression } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import './EvalPage.css'
import api from '../../services/api'
import type { Coordinate, RouteOptions } from '../../services/api'
import { scorePath, formatCost, isImpassable, getRouteVariants } from '../../services/evalApi'
import { parseGpx, downsamplePath, chooseGpxComparison } from '../../services/gpx'
import type { ScoredPath, EvalCase, RouteVariant } from '../../services/evalApi'
import DrawLayer from './DrawLayer'
import TrailsLayer from '../TrailsLayer'
import CalibrationToolbar from '../CalibrationToolbar/CalibrationToolbar'
import ComparePanel from './ComparePanel'
import AttributionPanel from './AttributionPanel'
import CasesPanel from './CasesPanel'
import ExpertisePanel from './ExpertisePanel'
import { variantsForDisplay } from './expertise'

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
  // "Go via this trail": snap the drawn candidate onto nearby trails before
  // scoring. Default on (per design); degrades to exact-drawn where no trail.
  const [snap, setSnap] = useState<'none' | 'trail'>('trail')
  // Overlay the engine's trail/path network for the current viewport.
  const [showTrails, setShowTrails] = useState(false)
  const [trailCount, setTrailCount] = useState<number | null>(null)
  // Expertise route family: one line per hiker level (casual…alpinist) for the
  // same start/end, each toggleable on the map.
  const [variants, setVariants] = useState<RouteVariant[] | null>(null)
  const [visibleLevels, setVisibleLevels] = useState<Record<string, boolean>>({})
  const [familyRunning, setFamilyRunning] = useState(false)

  // Latest values for the debounced calibration effect to read without
  // re-triggering itself (it must fire on `options` change only).
  const startRef = useRef(start)
  const endRef = useRef(end)
  const drawnRef = useRef(drawn)
  const snapRef = useRef(snap)
  // The snap mode the CURRENT drawn/reference path was scored with. A GPX is a
  // real recording, scored as-walked ('none'), and must stay that way when
  // calibration re-scores it — not flip to the UI snap toggle.
  const drawnSnapRef = useRef<'none' | 'trail'>(snap)
  const ranOnceRef = useRef(false)
  startRef.current = start
  endRef.current = end
  drawnRef.current = drawn
  snapRef.current = snap

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
      setStatus(`Optimal scored (cost ${formatCost(scored.totalCost)}).`)
    } catch (err) {
      setStatus('Error: ' + (err as Error).message)
    } finally {
      setRunning(false)
    }
  }

  /** Route the same start→end at every expertise level in one call and show the
   *  family. Identical lines are drawn once (backend marks them `duplicateOf`). */
  const runFamily = async () => {
    const s = startRef.current
    const e = endRef.current
    if (!s || !e) return
    setFamilyRunning(true)
    setStatus('Routing expertise family…')
    try {
      const vs = await getRouteVariants(s, e, options)
      setVariants(vs)
      // Show every level by default; the legend toggles remove them.
      setVisibleLevels(Object.fromEntries(vs.map((v) => [v.level, true])))
      const distinct = vs.filter((v) => v.path.length > 0 && !v.duplicateOf).length
      setStatus(`Family routed: ${distinct} distinct route(s) across ${vs.length} levels.`)
    } catch (err) {
      setStatus('Error routing family: ' + (err as Error).message)
    } finally {
      setFamilyRunning(false)
    }
  }

  const toggleLevel = (level: string) =>
    setVisibleLevels((v) => ({ ...v, [level]: v[level] === false }))

  /** Re-score the already-drawn candidate under the given options (cheap). */
  const rescoreDrawn = async (opts: RouteOptions) => {
    const d = drawnRef.current
    if (!d || d.path.length < 2) return
    try {
      // Re-score with the SAME snap mode this path was imported/drawn with, so a
      // GPX stays as-walked across calibration changes.
      setDrawn(await scorePath(d.path, opts, drawnSnapRef.current))
    } catch (err) {
      setStatus('Error scoring drawn path: ' + (err as Error).message)
    }
  }

  const gpxInputRef = useRef<HTMLInputElement>(null)

  /** Load a GPX the user walked as the reference track: its endpoints become
   *  start/end, the track becomes the candidate (scored AS-WALKED — no snapping,
   *  it's a real recording), then run the engine's optimal for the same
   *  endpoints so both show side-by-side. */
  const handleGpxFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = '' // let the same file be re-loaded later
    if (!file) return
    const MAX_GPX_BYTES = 20 * 1024 * 1024
    if (file.size > MAX_GPX_BYTES) {
      setStatus(`GPX too large (${(file.size / 1e6).toFixed(1)} MB, max 20 MB)`)
      return
    }
    setStatus(`Loading ${file.name}…`)
    try {
      // Loops have no meaningful start→end optimal, so compare the outbound leg
      // to its high point (see chooseGpxComparison).
      const { start: s, end: en, track, note } = chooseGpxComparison(parseGpx(await file.text()))
      const pts = downsamplePath(track, 1500)
      setStart(s)
      setEnd(en)
      startRef.current = s
      endRef.current = en
      setStatus(note ? `${note} Scoring your track…` : `Scoring your GPX track (${pts.length} pts)…`)
      // A GPX is a real recording: score it exactly as walked, no snapping.
      drawnSnapRef.current = 'none'
      const scored = await scorePath(pts, options, 'none')
      setDrawn(scored)
      drawnRef.current = scored
      await runOptimal(options)
      setStatus(`Loaded "${file.name}"${note ? ` — ${note}` : ''} Your track vs the engine's optimal.`)
    } catch (err) {
      setStatus('GPX load failed: ' + (err as Error).message)
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
      drawnSnapRef.current = snapRef.current
      const scored = await scorePath(points, options, snapRef.current)
      setDrawn(scored)
      setStatus(`Drawn path scored (cost ${formatCost(scored.totalCost)}).`)
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
        drawnSnapRef.current = snapRef.current
        const scored = await scorePath(c.referencePath, c.options, snapRef.current)
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
    setVariants(null)
    setVisibleLevels({})
    ranOnceRef.current = false
    setStatus('')
  }

  // Verdict for the what-if loop.
  let verdict: { text: string; win: boolean } | null = null
  if (optimal && drawn) {
    if (isImpassable(drawn.totalCost)) {
      verdict = { win: false, text: 'Your path is impassable (crosses terrain over the max slope)' }
    } else {
      const win = drawn.totalCost <= optimal.totalCost
      verdict = {
        win,
        text: win
          ? `Engine would prefer yours (${formatCost(drawn.totalCost)} ≤ ${formatCost(optimal.totalCost)})`
          : `Engine still prefers optimal (${formatCost(optimal.totalCost)} < ${formatCost(drawn.totalCost)})`,
      }
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

          <TrailsLayer active={showTrails} onCount={setTrailCount} />
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

          {/* Expertise route family — one line per visible, distinct level. */}
          {variants &&
            variantsForDisplay(variants, (lvl) => visibleLevels[lvl] !== false)
              .filter((d) => d.draw)
              .map((d) => (
                <Polyline
                  key={d.variant.level}
                  positions={d.variant.path.map((c) => [c.lat, c.lon] as [number, number])}
                  pathOptions={{ color: d.color, weight: 4, opacity: 0.85 }}
                />
              ))}

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
          <button className="eval-btn" onClick={() => gpxInputRef.current?.click()}>
            Load GPX
          </button>
          <input
            ref={gpxInputRef}
            type="file"
            accept=".gpx,application/gpx+xml,application/xml"
            style={{ display: 'none' }}
            onChange={handleGpxFile}
          />
          <button className="eval-btn" onClick={clear}>
            Clear
          </button>
        </section>

        <label className="eval-snap-toggle">
          <input
            type="checkbox"
            checked={snap === 'trail'}
            onChange={(e) => setSnap(e.target.checked ? 'trail' : 'none')}
          />
          Snap drawn path to trails
        </label>

        <label className="eval-snap-toggle">
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

        <ExpertisePanel
          variants={variants}
          visible={visibleLevels}
          onToggle={toggleLevel}
          onRun={runFamily}
          disabled={!start || !end}
          running={familyRunning}
        />

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
