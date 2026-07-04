import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useSavedLocations, type SeedLocation } from './useSavedLocations'

const SEED: SeedLocation[] = [
  { name: 'Home', lat: 40.6461, lon: -111.498 },
  { name: 'Trailhead', lat: 40.6, lon: -111.54 },
]

beforeEach(() => {
  localStorage.clear()
})

describe('useSavedLocations — seeding', () => {
  it('seeds presets from seed data when storage is empty', () => {
    const { result } = renderHook(() => useSavedLocations(SEED))
    expect(result.current.presets.map((p) => p.name)).toEqual(['Home', 'Trailhead'])
    expect(result.current.presets[0].kind).toBe('preset')
    expect(result.current.presets[0].id).toBeTruthy()
  })

  it('does not re-seed once presets exist (user edits win)', () => {
    // First mount seeds + user deletes everything.
    const first = renderHook(() => useSavedLocations(SEED))
    act(() => first.result.current.deletePreset(first.result.current.presets[0].id))
    act(() => first.result.current.deletePreset(first.result.current.presets[0].id))
    expect(first.result.current.presets).toHaveLength(0)

    // Remount: must respect the now-empty user state, not resurrect the seed.
    const second = renderHook(() => useSavedLocations(SEED))
    expect(second.result.current.presets).toHaveLength(0)
  })
})

describe('useSavedLocations — preset CRUD', () => {
  it('addPreset appends and persists', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.addPreset('Cabin', 40.7, -111.6))
    expect(result.current.presets.map((p) => p.name)).toEqual(['Cabin'])
    expect(JSON.parse(localStorage.getItem('trail.presets')!)).toHaveLength(1)
  })

  it('updatePreset renames', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.addPreset('Cabin', 40.7, -111.6))
    const id = result.current.presets[0].id
    act(() => result.current.updatePreset(id, 'Ski Cabin'))
    expect(result.current.presets[0].name).toBe('Ski Cabin')
  })

  it('deletePreset removes', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.addPreset('A', 1, 1))
    act(() => result.current.addPreset('B', 2, 2))
    const id = result.current.presets[0].id
    act(() => result.current.deletePreset(id))
    expect(result.current.presets.map((p) => p.name)).toEqual(['B'])
  })
})

describe('useSavedLocations — recents', () => {
  it('records newest-first', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.recordRecent(40.1, -111.1, 'First'))
    act(() => result.current.recordRecent(41.2, -112.2, 'Second'))
    expect(result.current.recents.map((r) => r.name)).toEqual(['Second', 'First'])
    expect(result.current.recents[0].kind).toBe('recent')
  })

  it('caps at 8', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => {
      for (let i = 0; i < 12; i++) result.current.recordRecent(40 + i, -111 - i, `p${i}`)
    })
    expect(result.current.recents).toHaveLength(8)
    // newest (p11) first, oldest kept is p4
    expect(result.current.recents[0].name).toBe('p11')
    expect(result.current.recents[7].name).toBe('p4')
  })

  it('dedups within ~50m — a near-duplicate promotes instead of appending', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.recordRecent(40.6461, -111.498, 'Home'))
    act(() => result.current.recordRecent(41.0, -112.0, 'Far'))
    // ~30m north of Home (0.0003 deg lat ≈ 33m)
    act(() => result.current.recordRecent(40.6464, -111.498, 'Home again'))
    expect(result.current.recents).toHaveLength(2)
    expect(result.current.recents[0].name).toBe('Home again')
    expect(result.current.recents.map((r) => r.name)).not.toContain('Home')
  })

  it('does not dedup points clearly farther than 50m apart', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.recordRecent(40.6461, -111.498, 'Home'))
    // ~0.001 deg lat ≈ 111m away
    act(() => result.current.recordRecent(40.6471, -111.498, 'Nearby'))
    expect(result.current.recents).toHaveLength(2)
  })
})

describe('useSavedLocations — promote', () => {
  it('promoteRecentToPreset creates a preset', () => {
    const { result } = renderHook(() => useSavedLocations([]))
    act(() => result.current.recordRecent(40.5, -111.5, 'Spot'))
    const recent = result.current.recents[0]
    act(() => result.current.promoteRecentToPreset(recent, 'Saved Spot'))
    expect(result.current.presets.map((p) => p.name)).toEqual(['Saved Spot'])
    expect(result.current.presets[0].lat).toBe(40.5)
  })
})
