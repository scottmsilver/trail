import React, { useEffect, useState } from 'react'
import L from 'leaflet'
import { useMap } from 'react-leaflet'
import api from '../../services/api'
import './SlopeOverlay.css'

interface SlopeOverlayProps {
  enabled: boolean
}

const SlopeOverlay: React.FC<SlopeOverlayProps> = ({ enabled }) => {
  const map = useMap()
  const [heatmapLayer, setHeatmapLayer] = useState<L.Layer | null>(null)
  const [loading, setLoading] = useState(false)
  const [legendVisible, setLegendVisible] = useState(true)
  const [mapBounds, setMapBounds] = useState<L.LatLngBounds | null>(null)
  const [pluginLoaded, setPluginLoaded] = useState(false)
  
  console.log('SlopeOverlay enabled:', enabled, 'plugin loaded:', pluginLoaded)

  // Load leaflet.heat plugin dynamically
  useEffect(() => {
    if (!window.L.heatLayer) {
      const script = document.createElement('script')
      script.src = 'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js'
      script.onload = () => {
        console.log('Leaflet.heat plugin loaded')
        setPluginLoaded(true)
      }
      script.onerror = () => {
        console.error('Failed to load leaflet.heat plugin')
      }
      document.head.appendChild(script)
    } else {
      setPluginLoaded(true)
    }
  }, [])

  useEffect(() => {
    if (!enabled || !pluginLoaded) {
      if (heatmapLayer) {
        console.log('Removing heatmap layer')
        map.removeLayer(heatmapLayer)
        setHeatmapLayer(null)
      }
      return
    }
    
    // Trigger initial load
    if (!mapBounds) {
      setMapBounds(map.getBounds())
    }

    const loadSlopeData = async () => {
      setLoading(true)
      console.log('Loading slope data...')
      
      try {
        const bounds = map.getBounds()
        console.log('Map bounds:', {
          minLat: bounds.getSouth(),
          maxLat: bounds.getNorth(),
          minLon: bounds.getWest(),
          maxLon: bounds.getEast()
        })
        
        const slopeData = await api.getTerrainSlopes({
          minLat: bounds.getSouth(),
          maxLat: bounds.getNorth(),
          minLon: bounds.getWest(),
          maxLon: bounds.getEast()
        })
        console.log('Slope data received:', slopeData)

        // Remove existing layer
        if (heatmapLayer) {
          map.removeLayer(heatmapLayer)
        }

        // Create heatmap data points
        const heatData = slopeData.lats.map((lat, i) => {
          const slope = slopeData.slopes[i]
          // Use slope value directly as intensity - leaflet.heat expects higher values for "hotter" spots
          // We'll use the actual slope in degrees
          return [lat, slopeData.lons[i], slope] as [number, number, number]
        })
        
        console.log(`Received ${heatData.length} slope data points`)
        const slopeValues = slopeData.slopes
        console.log(`Slope range: ${Math.min(...slopeValues)}° to ${Math.max(...slopeValues)}°`)

        // Create custom gradient for slopes
        // Values are normalized 0-1 by leaflet.heat based on the 'max' parameter
        const gradient = {
          0.0: 'rgba(46, 204, 113, 0)',     // Transparent - very flat (0°)
          0.05: 'rgba(46, 204, 113, 0.4)',  // Green - flat (0-2°)
          0.125: 'rgba(39, 174, 96, 0.5)',  // Dark green - gentle (2-5°)
          0.25: 'rgba(241, 196, 15, 0.5)',  // Yellow - moderate (5-10°)
          0.375: 'rgba(230, 126, 34, 0.6)', // Orange - steep (10-15°)
          0.5: 'rgba(231, 76, 60, 0.6)',    // Red - very steep (15-20°)
          0.625: 'rgba(192, 57, 43, 0.7)',  // Dark red - extreme (20-25°)
          0.75: 'rgba(142, 68, 173, 0.7)',  // Purple - nearly vertical (25-30°)
          1.0: 'rgba(44, 62, 80, 0.8)'      // Dark - vertical (30°+)
        }

        if (window.L.heatLayer) {
          console.log('Creating heatmap layer...')
          const heat = window.L.heatLayer(heatData, {
            radius: 15,
            blur: 20,
            max: 35,  // Maximum slope value for normalization (35 degrees)
            maxZoom: 17,
            gradient: gradient,
            minOpacity: 0.05,
            maxOpacity: 0.4
          })

          heat.addTo(map)
          
          // Ensure the heatmap doesn't block map clicks
          const heatmapPane = map.getPane('overlayPane')
          if (heatmapPane) {
            heatmapPane.style.pointerEvents = 'none'
          }
          
          setHeatmapLayer(heat)
          console.log('Heatmap layer added to map')
        } else {
          console.error('window.L.heatLayer is not available')
        }
      } catch (error) {
        console.error('Error loading slope data:', error)
      } finally {
        setLoading(false)
      }
    }

    // Debounce to avoid too many requests
    const timeoutId = setTimeout(loadSlopeData, 500)
    return () => clearTimeout(timeoutId)
  }, [enabled, map, mapBounds, pluginLoaded])

  // Reload on map move
  useEffect(() => {
    if (!enabled) return

    const handleMoveEnd = () => {
      // Trigger reload by changing a dependency
      setMapBounds(map.getBounds())
    }

    map.on('moveend', handleMoveEnd)
    return () => {
      map.off('moveend', handleMoveEnd)
    }
  }, [enabled, map])

  if (!enabled) return null

  return (
    <>
      {loading && (
        <div className="slope-loading">
          Loading terrain data...
        </div>
      )}
      
      {legendVisible && (
        <div className="slope-legend">
          <div className="legend-header">
            <h4>Terrain Slope</h4>
            <button 
              className="legend-close"
              onClick={() => setLegendVisible(false)}
            >
              ×
            </button>
          </div>
          <div className="legend-items">
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(46, 204, 113, 0.6)' }}></span>
              <span className="legend-label">0-2° Flat</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(39, 174, 96, 0.6)' }}></span>
              <span className="legend-label">2-5° Gentle</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(241, 196, 15, 0.6)' }}></span>
              <span className="legend-label">5-10° Moderate</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(230, 126, 34, 0.7)' }}></span>
              <span className="legend-label">10-15° Steep</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(231, 76, 60, 0.7)' }}></span>
              <span className="legend-label">15-20° Very Steep</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(192, 57, 43, 0.8)' }}></span>
              <span className="legend-label">20-25° Extreme</span>
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: 'rgba(142, 68, 173, 0.8)' }}></span>
              <span className="legend-label">25°+ Nearly Vertical</span>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

// Extend window type for leaflet.heat
declare global {
  interface Window {
    L: typeof L & {
      heatLayer?: (
        latlngs: [number, number, number][],
        options: {
          radius?: number
          blur?: number
          maxZoom?: number
          gradient?: Record<number, string>
          minOpacity?: number
          maxOpacity?: number
        }
      ) => L.Layer
    }
  }
}

export default SlopeOverlay