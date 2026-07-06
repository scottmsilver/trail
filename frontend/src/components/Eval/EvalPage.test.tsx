import { test, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MOCK_SCORED_PATH } from './mockScoredPath'

// react-leaflet hooks throw outside a live MapContainer under jsdom, so stub the
// whole module (mirrors Map.test.tsx). EvalClickHandler/DrawLayer become inert.
vi.mock('react-leaflet', () => {
  const mockMap = {
    on: () => {},
    off: () => {},
    getBounds: () => ({ getSouth: () => 0, getWest: () => 0, getNorth: () => 0, getEast: () => 0 }),
    setView: () => {},
    doubleClickZoom: { enable: () => {}, disable: () => {} },
  }
  return {
    MapContainer: ({ children }: any) => <div data-testid="map-container">{children}</div>,
    TileLayer: () => <div data-testid="tile-layer" />,
    Marker: ({ children }: any) => <div data-testid="marker">{children}</div>,
    Popup: ({ children }: any) => <div data-testid="popup">{children}</div>,
    Polyline: () => <div data-testid="polyline" />,
    useMap: () => mockMap,
    useMapEvents: () => null,
  }
})

const calculateRoute = vi.fn()
const getRouteStatus = vi.fn()
const getRoute = vi.fn()
vi.mock('../../services/api', () => ({
  default: {
    calculateRoute: (...args: unknown[]) => calculateRoute(...args),
    getRouteStatus: (...args: unknown[]) => getRouteStatus(...args),
    getRoute: (...args: unknown[]) => getRoute(...args),
  },
}))

const scorePath = vi.fn()
const getRouteVariants = vi.fn()
const listCases = vi.fn()
vi.mock('../../services/evalApi', async () => {
  const actual = await vi.importActual<typeof import('../../services/evalApi')>(
    '../../services/evalApi',
  )
  return {
    // Keep the real pure helpers so verdict/format logic behaves normally.
    isImpassable: actual.isImpassable,
    formatCost: actual.formatCost,
    IMPASSABLE_SENTINEL: actual.IMPASSABLE_SENTINEL,
    // Stub the network calls.
    scorePath: (...args: unknown[]) => scorePath(...args),
    getRouteVariants: (...args: unknown[]) => getRouteVariants(...args),
    getTrails: vi.fn(),
    getTerrain: vi.fn(),
    listCases: () => listCases(),
    saveCase: vi.fn(),
    deleteCase: vi.fn(),
  }
})

import EvalPage from './EvalPage'

const reference = JSON.stringify({
  start: { lat: 40.63, lon: -111.58 },
  end: { lat: 40.65, lon: -111.56 },
  options: { userProfile: 'default' },
  path: [],
})

/** Seed start/end and trigger runOptimal via the paste-reference flow (the map
 *  is mocked, so clicks can't set endpoints). */
function pasteReferenceAndRun() {
  fireEvent.click(screen.getByRole('button', { name: 'Paste reference' }))
  fireEvent.change(screen.getByLabelText('Route reference JSON'), {
    target: { value: reference },
  })
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
}

beforeEach(() => {
  calculateRoute.mockReset().mockResolvedValue({ routeId: 'r1' })
  getRouteStatus.mockReset().mockResolvedValue({ status: 'completed', progress: 100 })
  getRoute.mockReset()
  scorePath.mockReset().mockResolvedValue(MOCK_SCORED_PATH)
  getRouteVariants.mockReset().mockResolvedValue([])
  listCases.mockReset().mockResolvedValue([])
})

test('shows the OSM-missing warning + reload button when the route stats flag missing data', async () => {
  getRoute.mockResolvedValue({
    routeId: 'r1',
    status: 'completed',
    path: MOCK_SCORED_PATH.path,
    stats: { osmDataMissing: true, osmMissingTiles: 3 },
    createdAt: '',
  })

  render(<EvalPage />)
  pasteReferenceAndRun()

  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'Reload OSM data' })).toBeInTheDocument(),
  )
  expect(screen.getByRole('alert')).toHaveTextContent(/OSM data was missing/i)
})

test('does not show the warning when OSM data is complete', async () => {
  getRoute.mockResolvedValue({
    routeId: 'r1',
    status: 'completed',
    path: MOCK_SCORED_PATH.path,
    stats: {},
    createdAt: '',
  })

  render(<EvalPage />)
  pasteReferenceAndRun()

  // Wait for a run to complete (optimal path scored), then assert no warning.
  await waitFor(() => expect(scorePath).toHaveBeenCalled())
  expect(screen.queryByRole('button', { name: 'Reload OSM data' })).not.toBeInTheDocument()
})

test('clicking "Reload OSM data" re-runs the route with refreshOsm: true', async () => {
  getRoute.mockResolvedValue({
    routeId: 'r1',
    status: 'completed',
    path: MOCK_SCORED_PATH.path,
    stats: { osmDataMissing: true },
    createdAt: '',
  })

  render(<EvalPage />)
  pasteReferenceAndRun()

  const reloadBtn = await screen.findByRole('button', { name: 'Reload OSM data' })
  fireEvent.click(reloadBtn)

  await waitFor(() =>
    expect(
      calculateRoute.mock.calls.some(
        (call) => (call[2] as { refreshOsm?: boolean } | undefined)?.refreshOsm === true,
      ),
    ).toBe(true),
  )
})
