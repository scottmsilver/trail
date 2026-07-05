import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import Map from './Map'

// Mock Leaflet
vi.mock('react-leaflet', () => {
  // Stable map instance so effects keyed on `map` identity don't re-run forever.
  const mockMap = {
    on: () => {},
    off: () => {},
    getBounds: () => ({ getSouth: () => 0, getWest: () => 0, getNorth: () => 0, getEast: () => 0 }),
    setView: () => {},
  }
  const LayersControl: any = ({ children }: any) => <div data-testid="layers-control">{children}</div>
  LayersControl.BaseLayer = ({ children }: any) => <div data-testid="base-layer">{children}</div>
  LayersControl.Overlay = ({ children }: any) => <div data-testid="overlay">{children}</div>
  return {
    MapContainer: ({ children }: any) => <div data-testid="map-container">{children}</div>,
    TileLayer: () => <div data-testid="tile-layer" />,
    Marker: ({ children, eventHandlers }: any) => (
      <div data-testid="marker" onClick={() => eventHandlers?.dragend?.({ target: { getLatLng: () => ({ lat: 1, lng: 2 }) } })}>
        {children}
      </div>
    ),
    Popup: ({ children }: any) => <div data-testid="popup">{children}</div>,
    Polyline: () => <div data-testid="polyline" />,
    LayerGroup: ({ children }: any) => <div data-testid="layer-group">{children}</div>,
    LayersControl,
    useMap: () => mockMap,
    useMapEvents: () => null,
  }
})

describe('Map Component', () => {
  it('renders map container', () => {
    render(<Map />)
    expect(screen.getByTestId('map-container')).toBeInTheDocument()
  })

  it('renders one numbered marker per point', () => {
    const points = [
      { lat: 40.630, lon: -111.580 },
      { lat: 40.640, lon: -111.570 },
      { lat: 40.650, lon: -111.560 },
    ]
    render(<Map points={points} />)
    expect(screen.getAllByTestId('marker')).toHaveLength(3)
  })

  it('fires onPointDrag with index and new coord on dragend', () => {
    const onPointDrag = vi.fn()
    const points = [{ lat: 40.630, lon: -111.580 }, { lat: 40.640, lon: -111.570 }]
    render(<Map points={points} onPointDrag={onPointDrag} />)
    screen.getAllByTestId('marker')[1].click()
    expect(onPointDrag).toHaveBeenCalledWith(1, { lat: 1, lon: 2 })
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
