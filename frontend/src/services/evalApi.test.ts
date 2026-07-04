import { test, expect } from 'vitest'
import { isImpassable, formatCost, IMPASSABLE_SENTINEL } from './evalApi'

// Regression for the codex audit finding: the impassable threshold must sit
// close to the sentinel, not at a low cutoff that a legitimately expensive but
// passable path could exceed (unbounded customPathCosts / deviation penalty).
test('the impassable sentinel is detected', () => {
  expect(isImpassable(IMPASSABLE_SENTINEL)).toBe(true)
  // A total with several impassable segments overshoots the sentinel slightly.
  expect(isImpassable(IMPASSABLE_SENTINEL * 3)).toBe(true)
})

test('expensive-but-passable costs are NOT labeled impassable', () => {
  expect(isImpassable(0)).toBe(false)
  expect(isImpassable(19_757)).toBe(false)
  // Pathologically expensive yet passable — still far below half the sentinel.
  expect(isImpassable(1e14)).toBe(false)
  expect(isImpassable(1e17)).toBe(false)
})

test('formatCost prints "impassable" only for the sentinel, else a grouped number', () => {
  expect(formatCost(IMPASSABLE_SENTINEL)).toBe('impassable')
  expect(formatCost(4512)).toBe('4,512')
  expect(formatCost(1_234_567)).toBe('1,234,567')
})
