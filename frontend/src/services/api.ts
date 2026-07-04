import axios from 'axios'
import type { AxiosInstance } from 'axios'

// Single source for the API base. Prefer the build-time VITE_API_URL; an
// explicit empty string means "same origin" (relative), for single-origin
// serving. Falls back to the dev backend only when the var is unset.
export const API_BASE: string = import.meta.env.VITE_API_URL ?? 'http://localhost:9001'

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
  gradientPreference?: number
  trailPreference?: number
  engine?: 'v1' | 'v2'
  heuristicWeight?: number
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
    path_with_slopes?: any[]
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

export interface CostSurfaceResponse {
  cost_surface: number[][]
  slope_degrees?: number[][]
  elevation?: number[][] | null
  path_raster?: number[][] | null
  bounds: { north: number; south: number; east: number; west: number }
  shape: { height: number; width: number }
  downsampling_factor?: number
  start?: Coordinate
  end?: Coordinate
}

export interface CostPointResponse {
  lat: number
  lon: number
  cost: number
  slope: number
  elevation: number | null
  path_type: string
  path_id: number
  raw_osm_data?: Record<string, unknown> | null
  factors: {
    base_cost: number
    slope_cost: number
    path_multiplier: number
    is_obstacle: boolean
  }
  cost_breakdown?: {
    formula: string
    calculation_steps: { step1: string; step2: string; step3: string }
  }
}

export class TrailAPI {
  private client: AxiosInstance

  constructor(baseURL: string = API_BASE) {
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

  // Cost surface for a bounding box (used by CostSurfaceExplorer overlay)
  async getCostSurface(bounds: {
    north: number
    south: number
    east: number
    west: number
  }): Promise<CostSurfaceResponse> {
    const response = await this.client.post<CostSurfaceResponse>('/api/terrain/cost-surface', {
      bounds,
    })
    return response.data
  }

  // Cost breakdown at a single point (used by CostPointExplorer).
  // Backend returns 404 when the area has not been precomputed/cached.
  async getCostAtPoint(lat: number, lon: number): Promise<CostPointResponse> {
    const response = await this.client.post<CostPointResponse>('/api/terrain/cost-point', {
      lat,
      lon,
    })
    return response.data
  }
}

const api = new TrailAPI()
export default api
