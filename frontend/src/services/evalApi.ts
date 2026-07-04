import { API_BASE } from './api'
import type { Coordinate, RouteOptions } from './api'

export interface ScoredSegment {
  from: Coordinate
  to: Coordinate
  cost: number
  factors: Record<string, number>
  dominantFactor: string
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
