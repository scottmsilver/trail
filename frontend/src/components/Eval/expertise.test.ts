import { test, expect } from 'vitest'
import {
  EXPERTISE_ORDER,
  LEVEL_META,
  levelColor,
  levelLabel,
  variantsForDisplay,
} from './expertise'
import type { RouteVariant } from '../../services/evalApi'

const v = (level: string, extra: Partial<RouteVariant> = {}): RouteVariant => ({
  level,
  scrambleBudgetM: 4,
  path: [
    { lat: 40, lon: -111 },
    { lat: 40.1, lon: -111.1 },
  ],
  stats: {},
  ...extra,
})

test('EXPERTISE_ORDER is easiest → hardest and every level has metadata', () => {
  expect(EXPERTISE_ORDER).toEqual(['casual', 'hiker', 'scrambler', 'alpinist'])
  for (const level of EXPERTISE_ORDER) {
    expect(LEVEL_META[level]).toBeTruthy()
    expect(LEVEL_META[level].color).toMatch(/^#[0-9a-f]{6}$/i)
  }
})

test('level colors are all distinct and avoid the optimal/drawn/hover hues', () => {
  const colors = EXPERTISE_ORDER.map(levelColor)
  expect(new Set(colors).size).toBe(colors.length) // all distinct
  const reserved = ['#2563eb', '#f97316', '#dc2626'] // optimal, drawn, hover
  for (const c of colors) expect(reserved).not.toContain(c.toLowerCase())
})

test('levelColor / levelLabel fall back for unknown levels', () => {
  expect(levelColor('casual')).toBe(LEVEL_META.casual.color)
  expect(levelColor('bogus')).toBe('#64748b')
  expect(levelLabel('hiker')).toBe('Hiker')
  expect(levelLabel('bogus')).toBe('bogus')
})

test('variantsForDisplay draws visible, distinct, routed variants only', () => {
  const variants = [
    v('casual'),
    v('hiker', { duplicateOf: 'casual' }), // identical line → do not double-draw
    v('scrambler', { path: [] }), // no route
    v('alpinist'),
  ]
  const out = variantsForDisplay(variants, () => true)

  expect(out.map((d) => d.variant.level)).toEqual(['casual', 'hiker', 'scrambler', 'alpinist'])
  expect(out[0].draw).toBe(true) // casual: routed, distinct, visible
  expect(out[1].draw).toBe(false) // hiker: duplicate of casual
  expect(out[1].duplicateOf).toBe('casual')
  expect(out[2].draw).toBe(false) // scrambler: no route
  expect(out[2].hasRoute).toBe(false)
  expect(out[3].draw).toBe(true) // alpinist: routed, distinct, visible
  expect(out[0].color).toBe(LEVEL_META.casual.color)
})

test('variantsForDisplay respects the visibility predicate', () => {
  const variants = [v('casual'), v('alpinist')]
  const out = variantsForDisplay(variants, (lvl) => lvl === 'alpinist')
  expect(out[0].draw).toBe(false) // casual hidden
  expect(out[1].draw).toBe(true) // alpinist visible
})
