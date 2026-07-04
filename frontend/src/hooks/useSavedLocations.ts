import { useCallback, useState } from 'react'
import { seedPresets } from '../config/seedPresets'

export type LocationKind = 'preset' | 'recent'

export interface SavedLocation {
  id: string
  name: string
  lat: number
  lon: number
  kind: LocationKind
}

/** Shape of entries in the git-ignored seed file: name + coords only. */
export interface SeedLocation {
  name: string
  lat: number
  lon: number
}

const PRESETS_KEY = 'trail.presets'
const RECENTS_KEY = 'trail.recents'
const RECENTS_CAP = 8
const DEDUP_METERS = 50

function genId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `loc-${Date.now()}-${Math.floor(Math.random() * 1e9)}`
}

/** Equirectangular approximation — plenty accurate at the ~50m dedup scale. */
function distanceMeters(
  a: { lat: number; lon: number },
  b: { lat: number; lon: number }
): number {
  const R = 6371000
  const toRad = Math.PI / 180
  const lat1 = a.lat * toRad
  const lat2 = b.lat * toRad
  const x = (b.lon - a.lon) * toRad * Math.cos((lat1 + lat2) / 2)
  const y = (b.lat - a.lat) * toRad
  return Math.sqrt(x * x + y * y) * R
}

function formatCoords(lat: number, lon: number): string {
  return `${lat.toFixed(4)}, ${lon.toFixed(4)}`
}

/**
 * Coerce one raw parsed entry into a valid SavedLocation, or null if it can't be
 * trusted. localStorage is same-origin but can be corrupted (bad writes, other
 * tools, a malicious extension); a bad `lat` would otherwise crash render at
 * `loc.lat.toFixed(...)`. Numbers must be finite; missing ids/kinds are repaired.
 */
function sanitizeEntry(raw: unknown, fallbackKind: LocationKind): SavedLocation | null {
  if (typeof raw !== 'object' || raw === null) return null
  const e = raw as Record<string, unknown>
  const lat = Number(e.lat)
  const lon = Number(e.lon)
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null
  const kind: LocationKind = e.kind === 'preset' || e.kind === 'recent' ? e.kind : fallbackKind
  return {
    id: typeof e.id === 'string' && e.id ? e.id : genId(),
    name: typeof e.name === 'string' && e.name ? e.name : formatCoords(lat, lon),
    lat,
    lon,
    kind,
  }
}

function readStore(key: string, fallbackKind: LocationKind): SavedLocation[] | null {
  const raw = localStorage.getItem(key)
  if (raw === null) return null
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .map((entry) => sanitizeEntry(entry, fallbackKind))
      .filter((entry): entry is SavedLocation => entry !== null)
  } catch {
    return []
  }
}

function writeStore(key: string, value: SavedLocation[]): void {
  localStorage.setItem(key, JSON.stringify(value))
}

/**
 * Saved locations: user-managed presets + auto-captured recents.
 *
 * Presets seed once from the (git-ignored) seed file when storage has never been
 * populated. After that, localStorage is authoritative so user edits — including
 * deleting every preset — are never overwritten by the seed.
 */
export function useSavedLocations(seed: SeedLocation[] = seedPresets) {
  const [presets, setPresets] = useState<SavedLocation[]>(() => {
    const stored = readStore(PRESETS_KEY, 'preset')
    if (stored !== null) return stored
    const seeded: SavedLocation[] = seed.map((s) => ({
      id: genId(),
      name: s.name,
      lat: s.lat,
      lon: s.lon,
      kind: 'preset' as const,
    }))
    writeStore(PRESETS_KEY, seeded)
    return seeded
  })

  const [recents, setRecents] = useState<SavedLocation[]>(
    () => readStore(RECENTS_KEY, 'recent') ?? []
  )

  const persistPresets = useCallback((next: SavedLocation[]) => {
    setPresets(next)
    writeStore(PRESETS_KEY, next)
  }, [])

  const persistRecents = useCallback((next: SavedLocation[]) => {
    setRecents(next)
    writeStore(RECENTS_KEY, next)
  }, [])

  const addPreset = useCallback(
    (name: string, lat: number, lon: number) => {
      setPresets((prev) => {
        const next = [
          ...prev,
          { id: genId(), name: name.trim() || formatCoords(lat, lon), lat, lon, kind: 'preset' as const },
        ]
        writeStore(PRESETS_KEY, next)
        return next
      })
    },
    []
  )

  const updatePreset = useCallback((id: string, name: string) => {
    setPresets((prev) => {
      const next = prev.map((p) => (p.id === id ? { ...p, name: name.trim() || p.name } : p))
      writeStore(PRESETS_KEY, next)
      return next
    })
  }, [])

  const deletePreset = useCallback((id: string) => {
    setPresets((prev) => {
      const next = prev.filter((p) => p.id !== id)
      writeStore(PRESETS_KEY, next)
      return next
    })
  }, [])

  const recordRecent = useCallback((lat: number, lon: number, name?: string) => {
    setRecents((prev) => {
      const entry: SavedLocation = {
        id: genId(),
        name: name?.trim() || formatCoords(lat, lon),
        lat,
        lon,
        kind: 'recent',
      }
      const deduped = prev.filter((r) => distanceMeters(r, entry) > DEDUP_METERS)
      const next = [entry, ...deduped].slice(0, RECENTS_CAP)
      writeStore(RECENTS_KEY, next)
      return next
    })
  }, [])

  const promoteRecentToPreset = useCallback(
    (recent: SavedLocation, name?: string) => {
      addPreset(name ?? recent.name, recent.lat, recent.lon)
    },
    [addPreset]
  )

  return {
    presets,
    recents,
    addPreset,
    updatePreset,
    deletePreset,
    recordRecent,
    promoteRecentToPreset,
    persistPresets,
    persistRecents,
  }
}
