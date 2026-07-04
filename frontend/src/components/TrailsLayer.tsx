import { useEffect, useRef, useState } from 'react'
import { Polyline, useMap } from 'react-leaflet'
import { getTrails } from '../services/evalApi'

/** Overlay of the trail/path network the engine routes on (cached OSM ways),
 *  refetched for the visible viewport whenever the map moves. Non-interactive so
 *  the lines never intercept map clicks used to place start/end/draw. */
export default function TrailsLayer({
  active,
  onCount,
}: {
  active: boolean
  onCount: (n: number | null) => void
}) {
  const map = useMap()
  const [lines, setLines] = useState<[number, number][][]>([])
  const reqId = useRef(0)
  useEffect(() => {
    if (!active) {
      setLines([])
      onCount(null)
      return
    }
    let alive = true
    const load = async () => {
      // Monotonic id: a slow response from an earlier viewport must never
      // overwrite a newer one after a rapid pan/zoom.
      const id = ++reqId.current
      const b = map.getBounds()
      try {
        const res = await getTrails({
          south: b.getSouth(),
          west: b.getWest(),
          north: b.getNorth(),
          east: b.getEast(),
        })
        if (alive && id === reqId.current) {
          setLines(res.lines)
          onCount(res.count)
        }
      } catch {
        if (alive && id === reqId.current) {
          setLines([])
          onCount(0)
        }
      }
    }
    load()
    map.on('moveend', load)
    return () => {
      alive = false
      map.off('moveend', load)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, map])
  return (
    <>
      {lines.map((ln, i) => (
        <Polyline
          key={i}
          positions={ln}
          pathOptions={{ color: '#15803d', weight: 2, opacity: 0.5 }}
          interactive={false}
        />
      ))}
    </>
  )
}
