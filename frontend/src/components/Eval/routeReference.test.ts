import { test, expect } from 'vitest'
import { encodeReference, decodeReference, referenceToCase } from './routeReference'

const REF = {
  start: { lat: 40.62, lon: -111.57 },
  end: { lat: 40.64, lon: -111.55 },
  options: { userProfile: 'default', maxSlope: 35 },
  path: [
    { lat: 40.62, lon: -111.57 },
    { lat: 40.63, lon: -111.56 },
    { lat: 40.64, lon: -111.55 },
  ],
}

test('encode/decode round-trips a reference', () => {
  const decoded = decodeReference(encodeReference(REF))
  expect(decoded).toEqual(REF)
})

test('encodeReference emits compact JSON (no pretty whitespace)', () => {
  const s = encodeReference(REF)
  expect(s).not.toContain('\n')
  expect(JSON.parse(s).start).toEqual(REF.start)
})

test('encodeReference throws when an endpoint is missing', () => {
  expect(() => encodeReference({ ...REF, start: null })).toThrow(/start and an end/)
  expect(() => encodeReference({ ...REF, end: null })).toThrow(/start and an end/)
})

test('decodeReference rejects malformed JSON', () => {
  expect(() => decodeReference('not json {')).toThrow(/valid JSON/)
})

test('decodeReference rejects a non-object top level', () => {
  expect(() => decodeReference('42')).toThrow(/JSON object/)
})

test('decodeReference rejects missing or out-of-range endpoints', () => {
  expect(() => decodeReference(JSON.stringify({ ...REF, start: undefined }))).toThrow(/"start"/)
  expect(() =>
    decodeReference(JSON.stringify({ ...REF, end: { lat: 999, lon: 0 } })),
  ).toThrow(/"end"/)
})

test('decodeReference rejects a bad options block', () => {
  expect(() => decodeReference(JSON.stringify({ ...REF, options: [1, 2] }))).toThrow(/"options"/)
})

test('decodeReference rejects a path that is not an array of coordinates', () => {
  expect(() => decodeReference(JSON.stringify({ ...REF, path: 'nope' }))).toThrow(/"path"/)
  expect(() =>
    decodeReference(JSON.stringify({ ...REF, path: [{ lat: 1 }] })),
  ).toThrow(/"path"/)
})

test('decodeReference accepts an empty path', () => {
  const decoded = decodeReference(JSON.stringify({ ...REF, path: [] }))
  expect(decoded.path).toEqual([])
})

test('referenceToCase maps path to referencePath and fills case fields', () => {
  const c = referenceToCase(REF)
  expect(c.id).toBe('pasted')
  expect(c.name).toBe('pasted reference')
  expect(c.start).toEqual(REF.start)
  expect(c.end).toEqual(REF.end)
  expect(c.options).toEqual(REF.options)
  expect(c.referencePath).toEqual(REF.path)
  expect(c.labels).toEqual([])
})

import { MAX_REFERENCE_POINTS } from './routeReference'

test('decodeReference rejects an oversized path', () => {
  const pt = { lat: 40, lon: -111 }
  const big = {
    start: pt,
    end: pt,
    options: {},
    path: new Array(MAX_REFERENCE_POINTS + 1).fill(pt),
  }
  expect(() => decodeReference(JSON.stringify(big))).toThrow(/at most/)
})

test('decodeReference rejects text over the byte cap', () => {
  const huge = 'x'.repeat(1_000_001)
  expect(() => decodeReference(huge)).toThrow(/too large/)
})
