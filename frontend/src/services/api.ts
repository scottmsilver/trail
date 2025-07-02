import axios from 'axios'
import type { AxiosInstance } from 'axios'

export interface Coordinate {
  lat: number
  lon: number
}

export interface SlopeConfig {
  slope_degrees: number
  cost_multiplier: number
}

export interface CustomPathCosts {
  footway?: number
  path?: number
  trail?: number
  residential?: number
  off_path?: number
}

export interface RouteOptions {
  avoidSteep?: boolean
  buffer?: number
  slopeThreshold?: number
  userProfile?: string
  customSlopeCosts?: SlopeConfig[]
  customPathCosts?: CustomPathCosts
  maxSlope?: number
}

export interface RouteResponse {
  routeId: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
}

export interface RouteStatusResponse {
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  message?: string
}

export interface RouteResult {
  routeId: string
  status: string
  path: Coordinate[]
  stats: {
    distance_km: number
    elevation_gain_m: number
    estimated_time_min: number
    difficulty: string
    debug_data?: any
  }
  createdAt: string
}

export interface DebugRouteResult {
  path: Coordinate[]
  stats: {
    distance_km: number
    elevation_gain_m: number
    estimated_time_min: number
    difficulty: string
    debug_data: any
  }
  debug_info: string
}

export class TrailAPI {
  private client: AxiosInstance

  constructor(baseURL: string = import.meta.env.VITE_API_URL || 'http://localhost:9001') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }

  async calculateRoute(
    start: Coordinate,
    end: Coordinate,
    options: RouteOptions = {}
  ): Promise<RouteResponse> {
    const response = await this.client.post<RouteResponse>('/api/routes/calculate', {
      start,
      end,
      options,
    })
    return response.data
  }

  async getRouteStatus(routeId: string): Promise<RouteStatusResponse> {
    const response = await this.client.get<RouteStatusResponse>(
      `/api/routes/${routeId}/status`
    )
    return response.data
  }

  async getRoute(routeId: string): Promise<RouteResult> {
    const response = await this.client.get<RouteResult>(`/api/routes/${routeId}`)
    return response.data
  }

  async downloadGPX(routeId: string): Promise<Blob> {
    const response = await this.client.get(`/api/routes/${routeId}/gpx`, {
      responseType: 'blob',
    })
    return response.data
  }

  async calculateDebugRoute(
    start: Coordinate,
    end: Coordinate,
    options: RouteOptions = {}
  ): Promise<RouteResult> {
    const response = await this.client.post<RouteResult>('/api/routes/debug', {
      start,
      end,
      options,
    })
    return response.data
  }

  async getTerrainSlopes(bounds: {
    minLat: number
    maxLat: number
    minLon: number
    maxLon: number
  }): Promise<{
    lats: number[]
    lons: number[]
    slopes: number[]
    bounds: typeof bounds
  }> {
    const response = await this.client.post('/api/terrain/slopes', bounds)
    return response.data
  }

  async exportRouteAsGPX(
    start: Coordinate,
    end: Coordinate,
    options: RouteOptions = {}
  ): Promise<Blob> {
    const response = await this.client.post('/api/routes/export/gpx', {
      start,
      end,
      options,
    }, {
      responseType: 'blob',
    })
    return response.data
  }
}

const api = new TrailAPI()
export default api