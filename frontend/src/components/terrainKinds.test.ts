import { test, expect } from 'vitest'
import { TERRAIN_KINDS, kindColor, kindLabel, kindsInView } from './terrainKinds'

test('glacier and water have distinct colors and helpful labels', () => {
  expect(TERRAIN_KINDS.glacier).toBeTruthy()
  expect(kindColor('glacier')).not.toBe(kindColor('water'))
  expect(kindLabel('glacier')).toMatch(/glacier|snowfield/i)
})

test('kindColor / kindLabel fall back for unknown kinds', () => {
  expect(kindColor('bogus')).toBe('#a3a3a3')
  expect(kindLabel('bogus')).toBe('bogus')
})

test('kindsInView keeps known-kind order and appends unknowns, de-duped', () => {
  const out = kindsInView(['scree', 'glacier', 'scree', 'mystery'])
  expect(out).toEqual(['glacier', 'scree', 'mystery'])
})
