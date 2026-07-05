import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import TrailsLayer from './TrailsLayer'

vi.mock('react-leaflet', () => {
  const mockMap = {
    on: () => {},
    off: () => {},
    getBounds: () => ({ getSouth: () => 0, getWest: () => 0, getNorth: () => 1, getEast: () => 1 }),
  }
  return {
    Polyline: ({ positions }: any) => <div data-testid="polyline" data-len={positions.length} />,
    useMap: () => mockMap,
  }
})

const getTrails = vi.fn()
vi.mock('../services/evalApi', () => ({
  getTrails: (b: any) => getTrails(b),
}))

describe('TrailsLayer', () => {
  beforeEach(() => {
    getTrails.mockReset()
  })

  it('renders nothing and reports null count when inactive', () => {
    const onCount = vi.fn()
    render(<TrailsLayer active={false} onCount={onCount} />)
    expect(screen.queryByTestId('polyline')).not.toBeInTheDocument()
    expect(onCount).toHaveBeenCalledWith(null)
    expect(getTrails).not.toHaveBeenCalled()
  })

  it('fetches and renders a polyline per line when active, reporting the count', async () => {
    getTrails.mockResolvedValue({
      lines: [
        [
          [0, 0],
          [1, 1],
        ],
        [
          [1, 1],
          [2, 2],
        ],
      ],
      count: 2,
    })
    const onCount = vi.fn()
    render(<TrailsLayer active={true} onCount={onCount} />)
    await waitFor(() => expect(screen.getAllByTestId('polyline')).toHaveLength(2))
    expect(onCount).toHaveBeenCalledWith(2)
  })
})
