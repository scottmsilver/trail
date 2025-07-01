import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } from 'react-leaflet'
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

export default function Map({ start, end, path, onMapClick }: MapProps) {
  // Default center (Utah area)
  const defaultCenter: LatLngExpression = [40.640, -111.570]
  const zoom = 13

  // Convert path to Leaflet format
  const polylinePath: LatLngExpression[] = path?.map(coord => [coord.lat, coord.lon]) || []

  return (
    <MapContainer center={defaultCenter} zoom={zoom} className="map-container">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* Terrain layer option */}
      <TileLayer
        attribution='Imagery &copy; <a href="https://www.mapbox.com/">Mapbox</a>'
        url="https://api.mapbox.com/styles/v1/mapbox/outdoors-v12/tiles/{z}/{x}/{y}?access_token={accessToken}"
        accessToken={import.meta.env.VITE_MAPBOX_TOKEN || ''}
        opacity={0.7}
      />

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
    </MapContainer>
  )
}