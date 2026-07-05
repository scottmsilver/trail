import { useEffect, useRef, useState } from 'react'
import { Polygon, useMap } from 'react-leaflet'
import { getTerrain, type TerrainFeature } from '../services/evalApi'
import { kindColor } from './terrainKinds'

/** Overlay marking notable terrain (glaciers, water, cliffs, scree, ...) for the
 *  visible viewport, refetched on map move. Non-interactive so the polygons never
 *  intercept clicks used to place start/end/draw. Reports the kinds in view so a
 *  legend can show only what's on screen. */
export default function TerrainLayer({
  active,
  onKinds,
}: {
  active: boolean
  onKinds: (kinds: string[]) => void
}) {
  const map = useMap()
  const [feats, setFeats] = useState<TerrainFeature[]>([])
  const reqId = useRef(0)
  useEffect(() => {
    if (!active) {
      setFeats([])
      onKinds([])
      return
    }
    let alive = true
    const load = async () => {
      const id = ++reqId.current
      const b = map.getBounds()
      try {
        const res = await getTerrain({
          south: b.getSouth(),
          west: b.getWest(),
          north: b.getNorth(),
          east: b.getEast(),
        })
        if (alive && id === reqId.current) {
          setFeats(res.features)
          onKinds(res.features.map((f) => f.kind))
        }
      } catch {
        if (alive && id === reqId.current) {
          setFeats([])
          onKinds([])
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
      {feats.map((f, i) => {
        const c = kindColor(f.kind)
        return (
          <Polygon
            key={i}
            positions={f.polygon}
            interactive={false}
            pathOptions={{ color: c, weight: 1, opacity: 0.7, fillColor: c, fillOpacity: 0.35 }}
          />
        )
      })}
    </>
  )
}
