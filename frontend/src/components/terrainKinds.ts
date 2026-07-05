// Colors + labels for the terrain-marking overlay. Kept pure (no react-leaflet)
// so it can be unit-tested; TerrainLayer and the legend both import from here.

export interface KindMeta {
  label: string
  color: string
}

// Ordered for the legend; glacier first since it's the one that reads as a pond.
export const TERRAIN_KINDS: Record<string, KindMeta> = {
  glacier: { label: 'Glacier / snowfield', color: '#67e8f9' },
  water: { label: 'Water', color: '#0ea5e9' },
  wetland: { label: 'Wetland', color: '#14b8a6' },
  reservoir: { label: 'Reservoir', color: '#0284c7' },
  basin: { label: 'Basin', color: '#0284c7' },
  bay: { label: 'Bay', color: '#0ea5e9' },
  cliff: { label: 'Cliff', color: '#78350f' },
  rock: { label: 'Rock', color: '#9ca3af' },
  bare_rock: { label: 'Bare rock', color: '#9ca3af' },
  scree: { label: 'Scree', color: '#d6d3d1' },
}

const FALLBACK = '#a3a3a3'

export function kindColor(kind: string): string {
  return TERRAIN_KINDS[kind]?.color ?? FALLBACK
}

export function kindLabel(kind: string): string {
  return TERRAIN_KINDS[kind]?.label ?? kind
}

/** Distinct kinds present in a feature list, in TERRAIN_KINDS order (unknown
 *  kinds appended), for building a legend of only what's on screen. */
export function kindsInView(kinds: string[]): string[] {
  const present = new Set(kinds)
  const ordered = Object.keys(TERRAIN_KINDS).filter((k) => present.has(k))
  const extra = [...present].filter((k) => !(k in TERRAIN_KINDS))
  return [...ordered, ...extra]
}
