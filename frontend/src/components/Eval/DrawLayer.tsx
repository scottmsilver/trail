import { useEffect, useState } from 'react'
import { Polyline, CircleMarker, useMap, useMapEvents } from 'react-leaflet'
import type { Coordinate } from '../../services/api'

/** Append a Leaflet latlng ({lat, lng}) to the vertex list as a {lat, lon}
 *  Coordinate. Pure and exported so the accumulation logic can be unit-tested
 *  without rendering a live Leaflet map — react-leaflet hooks throw outside a
 *  MapContainer under jsdom. Returns a new array (never mutates prev). */
export function nextVertices(
  prev: Coordinate[],
  latlng: { lat: number; lng: number },
): Coordinate[] {
  return [...prev, { lat: latlng.lat, lon: latlng.lng }]
}

interface DrawLayerProps {
  active: boolean
  onFinish: (points: Coordinate[]) => void
}

/** While `active`, accumulates map clicks into an in-progress polyline. A
 *  double-click or Enter keypress finishes the path via onFinish and resets. */
export default function DrawLayer({ active, onFinish }: DrawLayerProps) {
  const [vertices, setVertices] = useState<Coordinate[]>([])
  const map = useMap()

  // Drop any half-drawn path whenever drawing is toggled off.
  useEffect(() => {
    if (!active) setVertices([])
  }, [active])

  // Suppress double-click zoom while drawing so dblclick can end the path.
  useEffect(() => {
    if (active) map.doubleClickZoom.disable()
    else map.doubleClickZoom.enable()
    return () => {
      map.doubleClickZoom.enable()
    }
  }, [active, map])

  const finish = (pts: Coordinate[]) => {
    if (pts.length > 0) onFinish(pts)
    setVertices([])
  }

  // react-leaflet re-registers these handlers each render, so the `vertices`
  // captured in the dblclick closure is always current.
  useMapEvents({
    click: (e) => {
      if (!active) return
      setVertices((prev) => nextVertices(prev, e.latlng))
    },
    dblclick: () => {
      if (!active) return
      finish(vertices)
    },
  })

  // Enter finishes the path. Leaflet's map doesn't emit keydown reliably, so
  // listen on the window while active.
  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter') finish(vertices)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, vertices])

  if (!active || vertices.length === 0) return null

  const positions = vertices.map((v) => [v.lat, v.lon] as [number, number])
  return (
    <>
      <Polyline
        positions={positions}
        pathOptions={{ color: '#f97316', weight: 3, dashArray: '6 6' }}
      />
      {vertices.map((v, i) => (
        <CircleMarker
          key={i}
          center={[v.lat, v.lon]}
          radius={4}
          pathOptions={{ color: '#f97316', fillColor: '#f97316', fillOpacity: 1 }}
        />
      ))}
    </>
  )
}
