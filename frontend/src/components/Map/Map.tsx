import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents, LayersControl, useMap } from 'react-leaflet'
import { useEffect } from 'react'
import L, { LatLngExpression } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import './Map.css'

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
  start?: Coordinate
  end?: Coordinate
  path?: Coordinate[]
  center?: { lat: number; lon: number }
  onMapClick?: (coord: Coordinate) => void
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

export default function Map({ start, end, path, center, onMapClick }: MapProps) {
  // Default center (Utah area)
  const defaultCenter: LatLngExpression = [40.640, -111.570]
  const zoom = 13

  // Convert path to Leaflet format
  const polylinePath: LatLngExpression[] = path?.map(coord => [coord.lat, coord.lon]) || []

  return (
    <MapContainer center={defaultCenter} zoom={zoom} className="map-container">
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
      </LayersControl>

      {start && (
        <Marker position={[start.lat, start.lon]}>
          <Popup>Start Point</Popup>
        </Marker>
      )}

      {end && (
        <Marker position={[end.lat, end.lon]}>
          <Popup>End Point</Popup>
        </Marker>
      )}

      {polylinePath.length > 0 && (
        <Polyline
          positions={polylinePath}
          color="blue"
          weight={3}
          opacity={0.8}
        />
      )}

      <MapClickHandler onMapClick={onMapClick} />
      <MapCenterController center={center} />
    </MapContainer>
  )
}