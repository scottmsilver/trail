import React, { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { useMap } from 'react-leaflet'
import api from '../../services/api'
import type { CostPointResponse } from '../../services/api'
import './CostPointExplorer.css'

interface CostPointExplorerProps {
  enabled: boolean
}

// Escape strings before putting them in a Leaflet popup: path_type and other
// fields derive from OSM tags, which anyone can edit, so treat them as untrusted.
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/**
 * When enabled, a map click queries /api/terrain/cost-point and shows the
 * terrain-cost breakdown for that spot (cost, slope, elevation, path type and
 * the multiplier factors). Reconstructed from the Map.tsx prop contract and the
 * cost-point response shape. Backend returns 404 for un-cached areas.
 */
const CostPointExplorer: React.FC<CostPointExplorerProps> = ({ enabled }) => {
  const map = useMap()
  const markerRef = useRef<L.Marker | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<CostPointResponse | null>(null)

  const clearMarker = () => {
    if (markerRef.current) {
      map.removeLayer(markerRef.current)
      markerRef.current = null
    }
  }

  useEffect(() => {
    if (!enabled) {
      clearMarker()
      setInfo(null)
      setError(null)
      return
    }

    // Show a crosshair to signal "click to inspect".
    const container = map.getContainer()
    const prevCursor = container.style.cursor
    container.style.cursor = 'crosshair'

    // Guard late/out-of-order responses: only the most recent click may apply
    // its result, and nothing applies after this effect is torn down.
    let active = true
    let clickSeq = 0

    const handleClick = async (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng
      const myClick = ++clickSeq
      const isStale = () => !active || myClick !== clickSeq

      setLoading(true)
      setError(null)
      setInfo(null)

      clearMarker()
      const marker = L.marker([lat, lng]).addTo(map)
      markerRef.current = marker

      try {
        const data = await api.getCostAtPoint(lat, lng)
        if (isStale()) return
        setInfo(data)
        marker
          .bindPopup(
            `<b>Cost:</b> ${data.cost.toFixed(2)}<br/>` +
              `<b>Slope:</b> ${data.slope.toFixed(1)}°<br/>` +
              `<b>Path:</b> ${escapeHtml(String(data.path_type))}`
          )
          .openPopup()
      } catch (err: any) {
        if (isStale()) return
        const status = err?.response?.status
        if (status === 404) {
          setError('This area is not cached yet. Use "Quick Cache View" to cache it, then click again.')
        } else {
          setError(err?.response?.data?.detail || err?.message || 'Failed to get cost at point.')
        }
        marker.bindPopup('No cost data here').openPopup()
      } finally {
        if (!isStale()) setLoading(false)
      }
    }

    map.on('click', handleClick)
    return () => {
      active = false
      map.off('click', handleClick)
      container.style.cursor = prevCursor
    }
  }, [enabled, map])

  useEffect(() => () => clearMarker(), [])

  if (!enabled) return null

  return (
    <div className="cost-point-panel">
      <div className="cost-point-header">
        <h4>Cost Explorer</h4>
      </div>

      {!info && !loading && !error && (
        <div className="cost-point-hint">Click anywhere on the map to inspect terrain cost.</div>
      )}
      {loading && <div className="cost-point-hint">Reading cost…</div>}
      {error && <div className="cost-point-error">{error}</div>}

      {info && (
        <div className="cost-point-details">
          <div className="cost-point-row">
            <span>Final cost</span>
            <b>{info.cost.toFixed(2)}</b>
          </div>
          <div className="cost-point-row">
            <span>Slope</span>
            <b>{info.slope.toFixed(1)}°</b>
          </div>
          {info.elevation != null && (
            <div className="cost-point-row">
              <span>Elevation</span>
              <b>{info.elevation.toFixed(0)} m</b>
            </div>
          )}
          <div className="cost-point-row">
            <span>Path type</span>
            <b>{info.path_type}</b>
          </div>
          {info.factors && (
            <>
              <div className="cost-point-row">
                <span>Slope factor</span>
                <b>{info.factors.slope_cost.toFixed(2)}×</b>
              </div>
              <div className="cost-point-row">
                <span>Path factor</span>
                <b>{info.factors.path_multiplier.toFixed(2)}×</b>
              </div>
              {info.factors.is_obstacle && <div className="cost-point-obstacle">⚠ Obstacle / impassable</div>}
            </>
          )}
          {info.cost_breakdown?.calculation_steps && (
            <div className="cost-point-formula">
              <div>{info.cost_breakdown.calculation_steps.step1}</div>
              <div>{info.cost_breakdown.calculation_steps.step3}</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default CostPointExplorer
