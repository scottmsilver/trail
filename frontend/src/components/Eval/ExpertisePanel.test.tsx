import { test, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ExpertisePanel from './ExpertisePanel'
import type { RouteVariant } from '../../services/evalApi'

const line = [
  { lat: 40, lon: -111 },
  { lat: 40.1, lon: -111.1 },
]

const variants: RouteVariant[] = [
  { level: 'casual', scrambleBudgetM: 1.5, path: line, stats: {} },
  { level: 'hiker', scrambleBudgetM: 4, path: line, stats: {}, duplicateOf: 'casual' },
  { level: 'scrambler', scrambleBudgetM: 8, path: [], stats: { error: 'No route found' } },
  { level: 'alpinist', scrambleBudgetM: 15, path: line, stats: {} },
]

const noop = () => {}

test('Run family button calls onRun', () => {
  const onRun = vi.fn()
  render(
    <ExpertisePanel
      variants={null}
      visible={{}}
      onToggle={noop}
      onRun={onRun}
      disabled={false}
      running={false}
    />,
  )
  fireEvent.click(screen.getByRole('button', { name: /run family/i }))
  expect(onRun).toHaveBeenCalledOnce()
})

test('Run button is disabled without endpoints and while running', () => {
  const { rerender } = render(
    <ExpertisePanel
      variants={null}
      visible={{}}
      onToggle={noop}
      onRun={noop}
      disabled={true}
      running={false}
    />,
  )
  expect(screen.getByRole('button', { name: /run family/i })).toBeDisabled()
  rerender(
    <ExpertisePanel
      variants={null}
      visible={{}}
      onToggle={noop}
      onRun={noop}
      disabled={false}
      running={true}
    />,
  )
  expect(screen.getByRole('button', { name: /running/i })).toBeDisabled()
})

test('renders a legend row per level with duplicate and no-route notes', () => {
  render(
    <ExpertisePanel
      variants={variants}
      visible={{}}
      onToggle={noop}
      onRun={noop}
      disabled={false}
      running={false}
    />,
  )
  expect(screen.getByText('Casual')).toBeInTheDocument()
  expect(screen.getByText('Alpinist')).toBeInTheDocument()
  expect(screen.getByText(/same as casual/i)).toBeInTheDocument() // hiker duplicates casual
  expect(screen.getByText(/no route/i)).toBeInTheDocument() // scrambler found none
})

test('toggling a drawable level calls onToggle; non-drawable checkboxes are disabled', () => {
  const onToggle = vi.fn()
  render(
    <ExpertisePanel
      variants={variants}
      visible={{ casual: true }}
      onToggle={onToggle}
      onRun={noop}
      disabled={false}
      running={false}
    />,
  )
  fireEvent.click(screen.getByRole('checkbox', { name: /show casual/i }))
  expect(onToggle).toHaveBeenCalledWith('casual')
  // scrambler (no route) and hiker (duplicate) draw nothing → their toggles are off-limits
  expect(screen.getByRole('checkbox', { name: /show scrambler/i })).toBeDisabled()
  expect(screen.getByRole('checkbox', { name: /show hiker/i })).toBeDisabled()
})
