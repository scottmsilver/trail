import axios, { AxiosInstance } from 'axios'

export interface Coordinate {
  lat: number
  lon: number
}

export interface RouteOptions {
  avoidSteep?: boolean
  buffer?: number
  slopeThreshold?: number
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
  }
  createdAt: string
}

export class TrailAPI {
  private client: AxiosInstance

  constructor(baseURL: string = import.meta.env.VITE_API_URL || 'http://localhost:8000') {
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
}

export default new TrailAPI()