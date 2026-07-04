import { test, expect } from 'vitest'
import { nextVertices } from './DrawLayer'

// These tests target the pure vertex-accumulation helper only. react-leaflet
// hooks (useMapEvents/useMap) throw outside a live MapContainer under jsdom,
// so the draw logic lives in nextVertices to stay unit-testable.

test('accumulates clicked vertices in order', () => {
  let v: { lat: number; lon: number }[] = []
  v = nextVertices(v, { lat: 1, lng: 2 })
  v = nextVertices(v, { lat: 3, lng: 4 })
  expect(v).toEqual([
    { lat: 1, lon: 2 },
    { lat: 3, lon: 4 },
  ])
})

test('converts Leaflet lng to lon and does not mutate the previous array', () => {
  const prev = [{ lat: 10, lon: 20 }]
  const next = nextVertices(prev, { lat: 30, lng: 40 })
  expect(next).toEqual([
    { lat: 10, lon: 20 },
    { lat: 30, lon: 40 },
  ])
  // Original array is left untouched (React state immutability).
  expect(prev).toEqual([{ lat: 10, lon: 20 }])
  expect(next).not.toBe(prev)
})

test('starts from an empty list', () => {
  expect(nextVertices([], { lat: 5, lng: 6 })).toEqual([{ lat: 5, lon: 6 }])
})
