import { test, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const listCases = vi.fn()
const saveCase = vi.fn()
const deleteCase = vi.fn()
vi.mock('../../services/evalApi', () => ({
  listCases: () => listCases(),
  saveCase: (c: unknown) => saveCase(c),
  deleteCase: (id: string) => deleteCase(id),
}))

import CasesPanel, { slug } from './CasesPanel'

beforeEach(() => {
  listCases.mockReset().mockResolvedValue([])
  saveCase.mockReset().mockResolvedValue({})
  deleteCase.mockReset().mockResolvedValue(undefined)
})

test('slug sanitizes names to id-safe form', () => {
  expect(slug('Half Dome via Panorama!')).toBe('half-dome-via-panorama')
  expect(slug('  Mist   Trail  ')).toBe('mist-trail')
})

const current = {
  start: { lat: 1, lon: 2 },
  end: { lat: 3, lon: 4 },
  options: { userProfile: 'default' },
  referencePath: [{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }],
}

test('saves current setup as a case with a slugged id', async () => {
  render(<CasesPanel current={current} onLoad={() => {}} />)
  fireEvent.change(screen.getByLabelText('Case name'), { target: { value: 'My Route' } })
  fireEvent.click(screen.getByRole('button', { name: /save current/i }))
  await waitFor(() => expect(saveCase).toHaveBeenCalledTimes(1))
  const payload = saveCase.mock.calls[0][0]
  expect(payload.id).toBe('my-route')
  expect(payload.name).toBe('My Route')
  expect(payload.referencePath).toHaveLength(2)
  expect(payload.labels).toEqual([])
})

test('lists cases and appends a label on verdict click', async () => {
  const existing = {
    id: 'c1', name: 'Case 1', notes: '', start: current.start, end: current.end,
    options: current.options, referencePath: [], labels: [],
  }
  listCases.mockResolvedValue([existing])
  render(<CasesPanel current={current} onLoad={() => {}} />)
  await screen.findByText('Case 1')
  fireEvent.click(screen.getByTitle('Mark good'))
  await waitFor(() => expect(saveCase).toHaveBeenCalled())
  const saved = saveCase.mock.calls[0][0]
  expect(saved.labels).toHaveLength(1)
  expect(saved.labels[0].verdict).toBe('pass')
})
