import { API_BASE } from './api'
import type { Coordinate, RouteOptions } from './api'

export interface ScoredSegment {
  from: Coordinate
  to: Coordinate
  cost: number
  factors: Record<string, number>
  dominantFactor: string
}

// The backend maps an impassable move (slope over the max) to a large finite
// sentinel so the JSON stays valid (scoring.py `_IMPASSABLE`). We detect it at
// half the sentinel: comfortably above any real passable cost — which, even with
// unbounded customPathCosts or a heavy deviation penalty, stays many orders of
// magnitude below 1e18 — yet below the sentinel (a total with any impassable
// segment is >= 1e18). Tied to the sentinel value, not an arbitrary cutoff.
export const IMPASSABLE_SENTINEL = 1e18
export function isImpassable(cost: number): boolean {
  return cost >= IMPASSABLE_SENTINEL / 2
}

/** Human-readable cost: "impassable" for the sentinel, else a rounded, grouped number. */
export function formatCost(cost: number): string {
  return isImpassable(cost) ? 'impassable' : Math.round(cost).toLocaleString('en-US')
}

export interface ScoredPath {
  path: Coordinate[]
  snapped: boolean
  totalCost: number
  distanceM: number
  elevationGainM: number
  segments: ScoredSegment[]
}

export interface EvalLabel {
  ts: string
  verdict: 'pass' | 'fail' | 'unsure'
  note?: string
}

export interface EvalCase {
  id: string
  name: string
  notes?: string
  start: Coordinate
  end: Coordinate
  options: RouteOptions
  referencePath: Coordinate[]
  labels: EvalLabel[]
}

export async function scorePath(
  path: Coordinate[],
  options: RouteOptions,
  snap: 'none' | 'trail' = 'none',
): Promise<ScoredPath> {
  const r = await fetch(`${API_BASE}/api/eval/score-path`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, options, snap }),
  })
  if (!r.ok) throw new Error(`score-path ${r.status}`)
  return r.json()
}

export interface TrailLines {
  lines: [number, number][][]
  count: number
}

/** Fetch the trail/path geometry the engine routes on (cached OSM highway=*
 *  ways) within a map viewport, for a display overlay. */
export async function getTrails(b: {
  south: number
  west: number
  north: number
  east: number
}): Promise<TrailLines> {
  const q = new URLSearchParams({
    south: String(b.south),
    west: String(b.west),
    north: String(b.north),
    east: String(b.east),
  })
  const r = await fetch(`${API_BASE}/api/eval/trails?${q}`)
  if (!r.ok) throw new Error(`trails ${r.status}`)
  return r.json()
}

export async function listCases(): Promise<EvalCase[]> {
  const r = await fetch(`${API_BASE}/api/eval/cases`)
  if (!r.ok) throw new Error(`list cases ${r.status}`)
  return r.json()
}

export async function saveCase(c: EvalCase): Promise<EvalCase> {
  const r = await fetch(`${API_BASE}/api/eval/cases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(c),
  })
  if (!r.ok) throw new Error(`save case ${r.status}`)
  return r.json()
}

export async function deleteCase(id: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/eval/cases/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`delete case ${r.status}`)
}
