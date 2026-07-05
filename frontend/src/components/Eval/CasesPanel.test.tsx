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

const buildGpx = vi.fn()
const downloadText = vi.fn()
vi.mock('../../services/gpx', () => ({
  buildGpx: (...args: unknown[]) => buildGpx(...args),
}))
vi.mock('../../services/download', () => ({
  downloadText: (...args: unknown[]) => downloadText(...args),
}))

import CasesPanel, { slug } from './CasesPanel'

beforeEach(() => {
  listCases.mockReset().mockResolvedValue([])
  saveCase.mockReset().mockResolvedValue({})
  deleteCase.mockReset().mockResolvedValue(undefined)
  buildGpx.mockReset().mockReturnValue('<gpx/>')
  downloadText.mockReset()
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

test('warns when saving without a drawn path, confirms when one is present', () => {
  const { rerender } = render(
    <CasesPanel current={{ ...current, referencePath: [] }} onLoad={() => {}} />,
  )
  expect(screen.getByText(/No drawn path yet/i)).toBeTruthy()
  rerender(<CasesPanel current={current} onLoad={() => {}} />)
  expect(screen.getByText(/Will include your drawn path \(2 points\)/i)).toBeTruthy()
})

test('flags saved cases that have no reference path', async () => {
  listCases.mockResolvedValue([
    { id: 'p', name: 'HasPath', notes: '', start: current.start, end: current.end, options: current.options, referencePath: [{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }], labels: [] },
    { id: 'e', name: 'Empty', notes: '', start: current.start, end: current.end, options: current.options, referencePath: [], labels: [] },
  ])
  render(<CasesPanel current={current} onLoad={() => {}} />)
  await screen.findByText('HasPath')
  expect(screen.getByText('2 pts')).toBeTruthy()
  expect(screen.getByText('no path')).toBeTruthy()
})

test('offers an enabled GPX download for a case with a reference path and downloads on click', async () => {
  listCases.mockResolvedValue([
    { id: 'p', name: 'HasPath', notes: '', start: current.start, end: current.end, options: current.options, referencePath: [{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }], labels: [] },
  ])
  render(<CasesPanel current={current} onLoad={() => {}} />)
  await screen.findByText('HasPath')
  const btn = screen.getByRole('button', { name: /gpx/i })
  expect((btn as HTMLButtonElement).disabled).toBe(false)
  fireEvent.click(btn)
  expect(buildGpx).toHaveBeenCalledWith([{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }], 'HasPath')
  expect(downloadText).toHaveBeenCalledTimes(1)
  expect(downloadText.mock.calls[0][0]).toBe('HasPath.gpx')
  expect(downloadText.mock.calls[0][2]).toBe('application/gpx+xml')
})

test('disables the GPX download for a case with too few reference points', async () => {
  listCases.mockResolvedValue([
    { id: 'e', name: 'Empty', notes: '', start: current.start, end: current.end, options: current.options, referencePath: [], labels: [] },
  ])
  render(<CasesPanel current={current} onLoad={() => {}} />)
  await screen.findByText('Empty')
  const btn = screen.getByRole('button', { name: /gpx/i })
  expect((btn as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(btn)
  expect(downloadText).not.toHaveBeenCalled()
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
