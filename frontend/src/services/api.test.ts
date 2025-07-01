import { describe, it, expect, vi, beforeEach } from 'vitest'
import axios from 'axios'
import { TrailAPI } from './api'

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      post: vi.fn(),
      get: vi.fn(),
    })),
  },
}))

describe('TrailAPI', () => {
  let api: TrailAPI
  let mockClient: any

  beforeEach(() => {
    vi.clearAllMocks()
    mockClient = {
      post: vi.fn(),
      get: vi.fn(),
    }
    ;(axios.create as any).mockReturnValue(mockClient)
    api = new TrailAPI('http://localhost:8000')
  })

  describe('calculateRoute', () => {
    it('sends route calculation request', async () => {
      const mockResponse = {
        data: { routeId: 'test-123', status: 'processing' }
      }
      mockClient.post.mockResolvedValue(mockResponse)

      const result = await api.calculateRoute(
        { lat: 40.630, lon: -111.580 },
        { lat: 40.650, lon: -111.560 }
      )

      expect(mockClient.post).toHaveBeenCalledWith(
        '/api/routes/calculate',
        {
          start: { lat: 40.630, lon: -111.580 },
          end: { lat: 40.650, lon: -111.560 },
          options: {}
        }
      )
      expect(result).toEqual(mockResponse.data)
    })

    it('includes options when provided', async () => {
      const mockResponse = {
        data: { routeId: 'test-456', status: 'processing' }
      }
      mockClient.post.mockResolvedValue(mockResponse)

      const options = { avoidSteep: true, buffer: 0.02 }
      await api.calculateRoute(
        { lat: 40.630, lon: -111.580 },
        { lat: 40.650, lon: -111.560 },
        options
      )

      expect(mockClient.post).toHaveBeenCalledWith(
        '/api/routes/calculate',
        expect.objectContaining({ options })
      )
    })
  })

  describe('getRouteStatus', () => {
    it('fetches route status', async () => {
      const mockResponse = {
        data: { status: 'completed', progress: 100 }
      }
      mockClient.get.mockResolvedValue(mockResponse)

      const result = await api.getRouteStatus('test-123')

      expect(mockClient.get).toHaveBeenCalledWith(
        '/api/routes/test-123/status'
      )
      expect(result).toEqual(mockResponse.data)
    })
  })
})