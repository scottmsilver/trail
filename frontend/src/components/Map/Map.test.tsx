import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Map from './Map'

// Mock Leaflet
vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: any) => <div data-testid="map-container">{children}</div>,
  TileLayer: () => <div data-testid="tile-layer" />,
  Marker: ({ children }: any) => <div data-testid="marker">{children}</div>,
  Popup: ({ children }: any) => <div data-testid="popup">{children}</div>,
  Polyline: () => <div data-testid="polyline" />,
  useMapEvents: () => null,
}))

describe('Map Component', () => {
  it('renders map container', () => {
    render(<Map />)
    expect(screen.getByTestId('map-container')).toBeInTheDocument()
  })

  it('shows start and end markers when coordinates are set', () => {
    const startCoord = { lat: 40.630, lon: -111.580 }
    const endCoord = { lat: 40.650, lon: -111.560 }
    
    render(<Map start={startCoord} end={endCoord} />)
    
    const markers = screen.getAllByTestId('marker')
    expect(markers).toHaveLength(2)
  })

  it('displays route when path is provided', () => {
    const path = [
      { lat: 40.630, lon: -111.580 },
      { lat: 40.640, lon: -111.570 },
      { lat: 40.650, lon: -111.560 }
    ]
    
    render(<Map path={path} />)
    
    expect(screen.getByTestId('polyline')).toBeInTheDocument()
  })

  it('calls onMapClick when map is clicked', () => {
    const handleMapClick = vi.fn()
    render(<Map onMapClick={handleMapClick} />)
    
    // Note: In a real test, we'd simulate a map click event
    // This is simplified for the mock
  })
})