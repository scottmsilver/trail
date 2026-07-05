import { MapContainer, TileLayer, Marker, Popup, useMapEvents, LayersControl, useMap, LayerGroup } from 'react-leaflet'
import { useEffect, useState } from 'react'
import L from 'leaflet'
import type { LatLngExpression } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import './Map.css'
import SlopeOverlay from '../SlopeOverlay/SlopeOverlay'
import TrailsLayer from '../TrailsLayer'
import PathWithSlopes from '../PathWithSlopes/PathWithSlopes'
import CostSurfaceExplorer from '../CostSurfaceExplorer/CostSurfaceExplorer'
import CostPointExplorer from '../CostPointExplorer/CostPointExplorer'

// Fix for default markers not showing
import icon from 'leaflet/dist/images/marker-icon.png'
import iconShadow from 'leaflet/dist/images/marker-shadow.png'

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
})

L.Marker.prototype.options.icon = DefaultIcon

interface Coordinate {
  lat: number
  lon: number
}

interface MapProps {
  points?: Coordinate[]
  onPointDrag?: (index: number, coord: Coordinate) => void
  onPointDelete?: (index: number) => void
  path?: Coordinate[]
  pathWithSlopes?: any[]  // Path with slope data from backend
  center?: { lat: number; lon: number }
  onMapClick?: (coord: Coordinate) => void
  onMapReady?: (map: L.Map) => void
  showCostSurface?: boolean
  onCloseCostSurface?: () => void
  costSurfaceBounds?: {north: number, south: number, east: number, west: number}
  costPointMode?: boolean
  showTrails?: boolean
  onTrailCount?: (n: number | null) => void
}

function MapClickHandler({ onMapClick }: { onMapClick?: (coord: Coordinate) => void }) {
  useMapEvents({
    click: (e) => {
      if (onMapClick) {
        onMapClick({ lat: e.latlng.lat, lon: e.latlng.lng })
      }
    },
  })
  return null
}

function MapCenterController({ center }: { center?: { lat: number; lon: number } }) {
  const map = useMap()

  useEffect(() => {
    if (center) {
      map.setView([center.lat, center.lon], 14)
    }
  }, [center, map])

  return null
}

function TerrainSlopesController({ setShowSlopes }: { setShowSlopes: (show: boolean) => void }) {
  const map = useMap()

  useEffect(() => {
    const handleOverlayAdd = (e: any) => {
      if (e.name === 'Terrain Slopes') {
        setShowSlopes(true)
      }
    }

    const handleOverlayRemove = (e: any) => {
      if (e.name === 'Terrain Slopes') {
        setShowSlopes(false)
      }
    }

    map.on('overlayadd', handleOverlayAdd)
    map.on('overlayremove', handleOverlayRemove)

    return () => {
      map.off('overlayadd', handleOverlayAdd)
      map.off('overlayremove', handleOverlayRemove)
    }
  }, [map, setShowSlopes])

  return null
}


// Component to handle map ready callback
function MapReadyHandler({ onMapReady }: { onMapReady?: (map: L.Map) => void }) {
  const map = useMap();

  useEffect(() => {
    if (onMapReady && map) {
      onMapReady(map);
    }
  }, [map, onMapReady]);

  return null;
}

export default function Map({ points, onPointDrag, onPointDelete, path, pathWithSlopes, center, onMapClick, onMapReady, showCostSurface, onCloseCostSurface, costSurfaceBounds, costPointMode, showTrails, onTrailCount }: MapProps) {
  // Default center (Utah area)
  const defaultCenter: LatLngExpression = [40.640, -111.570]
  const zoom = 13
  const [showSlopes, setShowSlopes] = useState(false)

  const pts = points ?? []
  const start = pts.length > 0 ? pts[0] : undefined
  const end = pts.length > 1 ? pts[pts.length - 1] : undefined

  return (
    <MapContainer
      center={defaultCenter}
      zoom={zoom}
      className="map-container">
      <LayersControl position="topright">
        {/* Base Layers */}
        <LayersControl.BaseLayer checked name="Street Map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="Topographic">
          <TileLayer
            attribution='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
            maxZoom={17}
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="Satellite">
          <TileLayer
            attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="USGS Topo">
          <TileLayer
            attribution='Map data &copy; <a href="https://www.usgs.gov/">U.S. Geological Survey</a>'
            url="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
            maxZoom={16}
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="Terrain">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            subdomains="abcd"
          />
        </LayersControl.BaseLayer>

        {/* Overlay Layers */}
        <LayersControl.Overlay name="Hillshade">
          <TileLayer
            attribution='Hillshade: <a href="https://github.com/cyclosm/cyclosm-cartocss-style/wiki/Tile-server">CyclOSM</a>'
            url="https://{s}.tile-cyclosm.openstreetmap.fr/hillshading/{z}/{x}/{y}.png"
            opacity={0.5}
          />
        </LayersControl.Overlay>

        <LayersControl.Overlay name="Contour Lines">
          <TileLayer
            attribution='Contours: <a href="https://github.com/cyclosm/cyclosm-cartocss-style/wiki/Tile-server">CyclOSM</a>'
            url="https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png"
            opacity={0.3}
          />
        </LayersControl.Overlay>

        <LayersControl.Overlay name="Terrain Slopes" checked={showSlopes}>
          <LayerGroup />
        </LayersControl.Overlay>
      </LayersControl>

      <SlopeOverlay enabled={showSlopes} />
      <TerrainSlopesController setShowSlopes={setShowSlopes} />

      {/* Trail network the engine routes on — drawn early so it sits beneath
          markers and the computed route. */}
      <TrailsLayer active={showTrails || false} onCount={onTrailCount || (() => {})} />

      {pts.map((p, i) => {
        const isTerminal = i === 0 || i === pts.length - 1
        const label = i === 0 ? 'Start' : i === pts.length - 1 ? 'End' : `Point ${i + 1}`
        const icon = L.divIcon({
          className: 'waypoint-marker',
          html: `<div class="wp-pin ${isTerminal ? 'wp-terminal' : 'wp-via'}">${i + 1}</div>`,
          iconSize: [26, 26],
          iconAnchor: [13, 13],
        })
        return (
          <Marker
            key={i}
            position={[p.lat, p.lon]}
            icon={icon}
            draggable={true}
            eventHandlers={{
              dragend: (e: any) => {
                const ll = e.target.getLatLng()
                onPointDrag?.(i, { lat: ll.lat, lon: ll.lng })
              },
            }}
          >
            <Popup>
              <div>{label}</div>
              <button onClick={() => onPointDelete?.(i)}>Delete point</button>
            </Popup>
          </Marker>
        )
      })}

      {path && path.length > 0 && (
        <PathWithSlopes path={path} pathWithSlopes={pathWithSlopes} />
      )}

      {/* Only enable map click for waypoints when not in cost point mode */}
      {!costPointMode && <MapClickHandler onMapClick={onMapClick} />}
      <MapReadyHandler onMapReady={onMapReady} />
      <MapCenterController center={center} />

      <CostSurfaceExplorer
        startCoord={start ? [start.lat, start.lon] : null}
        endCoord={end ? [end.lat, end.lon] : null}
        visible={showCostSurface || false}
        onClose={onCloseCostSurface || (() => {})}
        bounds={costSurfaceBounds}
      />

      <CostPointExplorer enabled={costPointMode || false} />
    </MapContainer>
  )
}
