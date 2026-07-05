import type { Coordinate, RouteOptions } from '../../services/api'
import type { EvalCase } from '../../services/evalApi'

/**
 * A compact, copy/paste-able snapshot of an eval setup: the endpoints, the cost
 * weights, and the (already downsampled) drawn path. Small enough to live on the
 * clipboard and reconstruct a case on another machine/tab.
 */
export interface RouteReference {
  start: Coordinate
  end: Coordinate
  options: RouteOptions
  path: Coordinate[]
}

/**
 * Serialize a reference to compact JSON. Throws if the endpoints are missing —
 * a reference with no start/end can't reconstruct anything.
 */
export function encodeReference(ref: {
  start: Coordinate | null
  end: Coordinate | null
  options: RouteOptions
  path: Coordinate[]
}): string {
  if (!ref.start || !ref.end) throw new Error('reference needs both a start and an end')
  const out: RouteReference = {
    start: ref.start,
    end: ref.end,
    options: ref.options,
    path: ref.path,
  }
  return JSON.stringify(out)
}

/** A finite lat/lon within valid geographic range. */
function isCoord(v: unknown): v is Coordinate {
  if (typeof v !== 'object' || v === null) return false
  const c = v as Record<string, unknown>
  return (
    typeof c.lat === 'number' &&
    typeof c.lon === 'number' &&
    Number.isFinite(c.lat) &&
    Number.isFinite(c.lon) &&
    c.lat >= -90 &&
    c.lat <= 90 &&
    c.lon >= -180 &&
    c.lon <= 180
  )
}

// Bounds on untrusted pasted input, so a giant blob can't freeze the tab during
// validation or push an oversized re-score at the backend. The path cap matches
// the server scorer's own limit; a reference is normally a downsampled ~300 pts.
export const MAX_REFERENCE_BYTES = 1_000_000
export const MAX_REFERENCE_POINTS = 5000

/**
 * Parse and validate pasted reference text. This is the trust boundary for
 * untrusted clipboard input: reject anything that isn't shaped like a
 * RouteReference (too large, bad JSON, missing/invalid endpoints, non-array or
 * non-coord/oversized path, non-object options) with a clear Error rather than
 * passing junk on.
 */
export function decodeReference(text: string): RouteReference {
  if (text.length > MAX_REFERENCE_BYTES) {
    throw new Error('reference is too large')
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('reference is not valid JSON')
  }
  if (typeof parsed !== 'object' || parsed === null) {
    throw new Error('reference must be a JSON object')
  }
  const r = parsed as Record<string, unknown>
  if (!isCoord(r.start)) throw new Error('reference has an invalid or missing "start"')
  if (!isCoord(r.end)) throw new Error('reference has an invalid or missing "end"')
  if (typeof r.options !== 'object' || r.options === null || Array.isArray(r.options)) {
    throw new Error('reference has an invalid or missing "options"')
  }
  if (!Array.isArray(r.path) || r.path.length > MAX_REFERENCE_POINTS || !r.path.every(isCoord)) {
    throw new Error(`reference "path" must be an array of at most ${MAX_REFERENCE_POINTS} coordinates`)
  }
  return {
    start: r.start,
    end: r.end,
    options: r.options as RouteOptions,
    path: r.path,
  }
}

/**
 * Adapt a decoded reference into an EvalCase-shaped object so it can flow through
 * the existing `loadCase` restore path (its `path` becomes `referencePath`).
 */
export function referenceToCase(ref: RouteReference): EvalCase {
  return {
    id: 'pasted',
    name: 'pasted reference',
    start: ref.start,
    end: ref.end,
    options: ref.options,
    referencePath: ref.path,
    labels: [],
  }
}
