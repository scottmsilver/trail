import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SavedLocations from './SavedLocations'
import type { SavedLocation } from '../../hooks/useSavedLocations'

const presets: SavedLocation[] = [
  { id: 'p1', name: 'Home', lat: 40.6461, lon: -111.498, kind: 'preset' },
]
const recents: SavedLocation[] = [
  { id: 'r1', name: 'Trailhead', lat: 40.6, lon: -111.54, kind: 'recent' },
]

function setup(overrides: Partial<React.ComponentProps<typeof SavedLocations>> = {}) {
  const props = {
    presets,
    recents,
    currentPoint: { lat: 40.7, lon: -111.6 } as { lat: number; lon: number } | null,
    onUseAsStart: vi.fn(),
    onUseAsEnd: vi.fn(),
    onAddPreset: vi.fn(),
    onUpdatePreset: vi.fn(),
    onDeletePreset: vi.fn(),
    onPromoteRecent: vi.fn(),
    ...overrides,
  }
  render(<SavedLocations {...props} />)
  return props
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('SavedLocations', () => {
  it('renders preset and recent names', () => {
    setup()
    expect(screen.getByText('Home')).toBeInTheDocument()
    expect(screen.getByText('Trailhead')).toBeInTheDocument()
  })

  it('Start button fires onUseAsStart with the location', () => {
    const props = setup()
    fireEvent.click(screen.getAllByRole('button', { name: /use as start/i })[0])
    expect(props.onUseAsStart).toHaveBeenCalledWith(presets[0])
  })

  it('End button fires onUseAsEnd with the location', () => {
    const props = setup()
    fireEvent.click(screen.getAllByRole('button', { name: /use as end/i })[0])
    expect(props.onUseAsEnd).toHaveBeenCalledWith(presets[0])
  })

  it('Save current point prompts for a name and calls onAddPreset', () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Cabin')
    const props = setup()
    fireEvent.click(screen.getByRole('button', { name: /save current point/i }))
    expect(props.onAddPreset).toHaveBeenCalledWith('Cabin', 40.7, -111.6)
  })

  it('Save current point is disabled when there is no current point', () => {
    setup({ currentPoint: null })
    expect(screen.getByRole('button', { name: /save current point/i })).toBeDisabled()
  })

  it('rename calls onUpdatePreset', () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Ski Cabin')
    const props = setup()
    fireEvent.click(screen.getByRole('button', { name: /rename home/i }))
    expect(props.onUpdatePreset).toHaveBeenCalledWith('p1', 'Ski Cabin')
  })

  it('delete calls onDeletePreset', () => {
    const props = setup()
    fireEvent.click(screen.getByRole('button', { name: /delete home/i }))
    expect(props.onDeletePreset).toHaveBeenCalledWith('p1')
  })

  it('star on a recent calls onPromoteRecent', () => {
    const props = setup()
    fireEvent.click(screen.getByRole('button', { name: /save trailhead as preset/i }))
    expect(props.onPromoteRecent).toHaveBeenCalledWith(recents[0])
  })

  it('shows empty hints when there are no presets or recents', () => {
    setup({ presets: [], recents: [] })
    expect(screen.getByText(/no presets yet/i)).toBeInTheDocument()
    expect(screen.getByText(/no recent locations/i)).toBeInTheDocument()
  })
})
