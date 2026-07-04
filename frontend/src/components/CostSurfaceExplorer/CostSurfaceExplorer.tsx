import React, { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { useMap } from 'react-leaflet'
import api from '../../services/api'
import type { CostSurfaceResponse } from '../../services/api'
import './CostSurfaceExplorer.css'

interface CostSurfaceExplorerProps {
  startCoord: [number, number] | null
  endCoord: [number, number] | null
  visible: boolean
  onClose: () => void
  bounds?: { north: number; south: number; east: number; west: number }
}

// Obstacle / impassable cells come back from the backend as very large costs.
const OBSTACLE_COST = 1000

// Guard against a malformed or oversized response: cap the raster we will
// allocate a canvas for (~16M px). The backend downsamples, so real grids are
// far smaller; this only trips on bad/hostile data.
const MAX_GRID_PIXELS = 16_000_000

// Map a normalized cost (0..1) to a green -> yellow -> red RGB triple.
function costColor(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.5) {
    // green -> yellow
    const k = c / 0.5
    return [Math.round(46 + k * (241 - 46)), Math.round(204 + k * (196 - 204)), Math.round(113 + k * (15 - 113))]
  }
  // yellow -> red
  const k = (c - 0.5) / 0.5
  return [Math.round(241 + k * (231 - 241)), Math.round(196 + k * (76 - 196)), Math.round(15 + k * (60 - 15))]
}

/**
 * Renders the backend cost surface for a bounding box as a semi-transparent
 * raster overlay. Reconstructed from the Map.tsx prop contract + the
 * /api/terrain/cost-surface response shape (cost_surface[height][width]).
 */
const CostSurfaceExplorer: React.FC<CostSurfaceExplorerProps> = ({ visible, onClose, bounds }) => {
  const map = useMap()
  const overlayRef = useRef<L.ImageOverlay | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [range, setRange] = useState<{ min: number; max: number } | null>(null)

  const removeOverlay = () => {
    if (overlayRef.current) {
      map.removeLayer(overlayRef.current)
      overlayRef.current = null
    }
  }

  useEffect(() => {
    if (!visible || !bounds) {
      removeOverlay()
      setError(null)
      setRange(null)
      return
    }

    let cancelled = false

    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const data: CostSurfaceResponse = await api.getCostSurface(bounds)
        if (cancelled) return

        const grid = data.cost_surface
        // Derive dims from the grid itself, not the (untrusted) shape field, so
        // indexing never runs past the actual arrays.
        if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0])) {
          setError('Cost surface came back empty.')
          return
        }
        const height = grid.length
        const width = grid[0].length
        if (!width) {
          setError('Cost surface came back empty.')
          return
        }
        if (!Number.isFinite(height * width) || height * width > MAX_GRID_PIXELS) {
          setError('Cost surface is too large to render for this view. Zoom in and try again.')
          return
        }

        // Determine a color-scale range from the passable cells only.
        let min = Infinity
        let max = -Infinity
        for (let r = 0; r < height; r++) {
          for (let c = 0; c < width; c++) {
            const v = grid[r][c]
            if (v >= OBSTACLE_COST || !isFinite(v)) continue
            if (v < min) min = v
            if (v > max) max = v
          }
        }
        if (!isFinite(min)) {
          min = 0
          max = 1
        }
        if (max <= min) max = min + 1
        setRange({ min, max })

        // Paint the grid into a canvas, then hand it to Leaflet as an image overlay.
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          setError('Canvas not available.')
          return
        }
        const img = ctx.createImageData(width, height)
        for (let r = 0; r < height; r++) {
          for (let c = 0; c < width; c++) {
            const v = grid[r][c]
            const idx = (r * width + c) * 4
            if (v >= OBSTACLE_COST || !isFinite(v)) {
              // Impassable: opaque dark plum.
              img.data[idx] = 60
              img.data[idx + 1] = 20
              img.data[idx + 2] = 60
              img.data[idx + 3] = 200
              continue
            }
            const [rr, gg, bb] = costColor((v - min) / (max - min))
            img.data[idx] = rr
            img.data[idx + 1] = gg
            img.data[idx + 2] = bb
            img.data[idx + 3] = 150
          }
        }
        ctx.putImageData(img, 0, 0)

        const b = data.bounds ?? bounds
        const latLngBounds = L.latLngBounds([b.south, b.west], [b.north, b.east])

        removeOverlay()
        const overlay = L.imageOverlay(canvas.toDataURL(), latLngBounds, {
          opacity: 0.65,
          interactive: false,
        })
        overlay.addTo(map)
        overlayRef.current = overlay
      } catch (err: any) {
        if (cancelled) return
        // Don't leave a stale overlay for the previous bounds on screen while
        // the panel reports an error for the new bounds.
        removeOverlay()
        const status = err?.response?.status
        if (status === 404) {
          setError('This area is not cached yet. Use "Quick Cache View" first, then reload the cost surface.')
        } else {
          setError(err?.response?.data?.detail || err?.message || 'Failed to load cost surface.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [visible, bounds, map])

  // Clean up the overlay if the component unmounts.
  useEffect(() => () => removeOverlay(), [])

  if (!visible) return null

  return (
    <div className="cost-surface-panel">
      <div className="cost-surface-header">
        <h4>Cost Surface</h4>
        <button className="cost-surface-close" onClick={onClose} title="Hide cost surface">
          ×
        </button>
      </div>

      {loading && <div className="cost-surface-status">Loading cost surface…</div>}
      {error && <div className="cost-surface-error">{error}</div>}

      {!loading && !error && range && (
        <>
          <div className="cost-surface-gradient" />
          <div className="cost-surface-scale">
            <span>{range.min.toFixed(1)}</span>
            <span>higher cost →</span>
            <span>{range.max.toFixed(1)}</span>
          </div>
          <div className="cost-surface-legend-item">
            <span className="cost-surface-swatch obstacle" />
            <span>Impassable / obstacle</span>
          </div>
        </>
      )}
    </div>
  )
}

export default CostSurfaceExplorer
