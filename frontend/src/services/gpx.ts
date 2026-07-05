import type { Coordinate } from './api'

/** A GPX track point: lat/lon plus optional elevation (meters) from `<ele>`. */
export interface GpxPoint extends Coordinate {
  ele?: number
}

/**
 * Parse a GPX document into an ordered list of track points.
 *
 * Reads `<trkpt>` (a recorded track); falls back to `<rtept>` (a planned route)
 * when there is no track. Elevation is read from a child `<ele>` when present.
 * Points missing/!finite lat|lon are skipped. Throws on invalid XML or fewer
 * than 2 usable points.
 */
export function parseGpx(text: string): GpxPoint[] {
  const doc = new DOMParser().parseFromString(text, 'application/xml')
  if (doc.querySelector('parsererror')) throw new Error('not valid GPX/XML')
  let nodes = Array.from(doc.getElementsByTagName('trkpt'))
  if (nodes.length === 0) nodes = Array.from(doc.getElementsByTagName('rtept'))
  const coords: GpxPoint[] = []
  for (const n of nodes) {
    const lat = Number(n.getAttribute('lat'))
    const lon = Number(n.getAttribute('lon'))
    // Number() (not parseFloat) rejects partial junk like "47<x>"; range-check
    // so a malformed file can't feed nonsense coordinates downstream.
    if (
      Number.isFinite(lat) &&
      Number.isFinite(lon) &&
      lat >= -90 &&
      lat <= 90 &&
      lon >= -180 &&
      lon <= 180
    ) {
      const eleEl = n.getElementsByTagName('ele')[0]
      const ele = eleEl ? Number(eleEl.textContent) : NaN
      coords.push(Number.isFinite(ele) ? { lat, lon, ele } : { lat, lon })
    }
  }
  if (coords.length < 2) throw new Error('GPX has fewer than 2 valid track points')
  return coords
}

function haversineM(a: Coordinate, b: Coordinate): number {
  const R = 6371000
  const dLat = ((b.lat - a.lat) * Math.PI) / 180
  const dLon = ((b.lon - a.lon) * Math.PI) / 180
  const la1 = (a.lat * Math.PI) / 180
  const la2 = (b.lat * Math.PI) / 180
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

export interface GpxComparison {
  start: Coordinate
  end: Coordinate
  track: Coordinate[]
  note: string
}

/**
 * Decide what to compare from a GPX. The engine routes point-to-point, so a
 * LOOP (start ≈ end) has no meaningful start→end optimal. For a loop we compare
 * the OUTBOUND leg to its high point (the summit) — the farthest point if the
 * GPX has no elevation. A point-to-point track is used whole, end to end.
 */
export function chooseGpxComparison(points: GpxPoint[], loopThresholdM = 500): GpxComparison {
  const first = points[0]
  const last = points[points.length - 1]
  const strip = (p: GpxPoint): Coordinate => ({ lat: p.lat, lon: p.lon })
  if (haversineM(first, last) >= loopThresholdM) {
    return { start: strip(first), end: strip(last), track: points.map(strip), note: '' }
  }
  const hasEle = points.some((p) => p.ele != null)
  let apex = 0
  for (let i = 1; i < points.length; i++) {
    const better = hasEle
      ? (points[i].ele ?? -Infinity) > (points[apex].ele ?? -Infinity)
      : haversineM(first, points[i]) > haversineM(first, points[apex])
    if (better) apex = i
  }
  const kind = hasEle ? `highest point (${Math.round(points[apex].ele as number)} m)` : 'farthest point'
  return {
    start: strip(first),
    end: strip(points[apex]),
    track: points.slice(0, apex + 1).map(strip),
    note: `Loop detected — comparing trailhead → ${kind}.`,
  }
}

/**
 * Reduce a path to at most `max` points with a uniform stride, always keeping
 * the first and last point (a recorded GPX can have thousands of points; the
 * scorer caps at MAX_PATH_POINTS server-side).
 */
export function downsamplePath(path: Coordinate[], max: number): Coordinate[] {
  if (max < 2) throw new Error('max must be >= 2')
  if (path.length <= max) return path
  const stride = Math.ceil(path.length / max)
  const out: Coordinate[] = []
  for (let i = 0; i < path.length; i += stride) out.push(path[i])
  // Guarantee the true endpoint is present WITHOUT exceeding `max`: replace the
  // last kept sample if we're already at capacity, else append it.
  const last = path[path.length - 1]
  const kept = out[out.length - 1]
  if (kept.lat !== last.lat || kept.lon !== last.lon) {
    if (out.length >= max) out[out.length - 1] = last
    else out.push(last)
  }
  return out
}
