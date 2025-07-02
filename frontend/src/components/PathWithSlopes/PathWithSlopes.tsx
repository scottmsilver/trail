import React, { useState } from 'react'
import { Polyline, Tooltip } from 'react-leaflet'
import { LatLngExpression } from 'leaflet'
import './PathWithSlopes.css'

interface PathPoint {
  lat: number
  lon: number
  elevation?: number
  slope?: number
}

interface PathWithSlopesProps {
  path: PathPoint[]
  pathWithSlopes?: PathPoint[]
}

function getSegmentColor(slope: number): string {
  const absSlope = Math.abs(slope)
  
  if (absSlope < 5) {
    return '#2ecc71' // Green - easy
  } else if (absSlope < 10) {
    return '#27ae60' // Dark green - moderate easy
  } else if (absSlope < 15) {
    return '#f39c12' // Orange - moderate
  } else if (absSlope < 20) {
    return '#e67e22' // Dark orange - hard
  } else if (absSlope < 25) {
    return '#e74c3c' // Red - very hard
  } else {
    return '#c0392b' // Dark red - extreme
  }
}

function getSlopeDifficulty(slope: number): string {
  const absSlope = Math.abs(slope)
  
  if (absSlope < 5) return 'Easy'
  if (absSlope < 10) return 'Moderate'
  if (absSlope < 15) return 'Challenging'
  if (absSlope < 20) return 'Hard'
  if (absSlope < 25) return 'Very Hard'
  return 'Extreme'
}

export default function PathWithSlopes({ path, pathWithSlopes }: PathWithSlopesProps) {
  const [showLegend, setShowLegend] = useState(true)
  
  // Use pathWithSlopes if available, otherwise use regular path
  const dataPath = pathWithSlopes || path
  
  if (!dataPath || dataPath.length < 2) {
    return null
  }

  // If we don't have slope data, render a simple path
  if (!pathWithSlopes || !pathWithSlopes[0].hasOwnProperty('slope')) {
    const positions: LatLngExpression[] = path.map(p => [p.lat, p.lon])
    return (
      <Polyline
        positions={positions}
        color="blue"
        weight={3}
        opacity={0.8}
      />
    )
  }

  // Render path segments with slope colors
  const segments: JSX.Element[] = []
  
  for (let i = 0; i < pathWithSlopes.length - 1; i++) {
    const start = pathWithSlopes[i]
    const end = pathWithSlopes[i + 1]
    const slope = start.slope || 0
    
    const segmentPath: LatLngExpression[] = [
      [start.lat, start.lon],
      [end.lat, end.lon]
    ]
    
    const color = getSegmentColor(slope)
    const difficulty = getSlopeDifficulty(slope)
    
    segments.push(
      <Polyline
        key={i}
        positions={segmentPath}
        color={color}
        weight={4}
        opacity={0.9}
      >
        <Tooltip sticky>
          <div style={{ minWidth: '150px' }}>
            <strong>Segment {i + 1}</strong><br />
            Slope: {slope.toFixed(1)}°<br />
            Difficulty: {difficulty}<br />
            {start.elevation && (
              <>Elevation: {Math.round(start.elevation)}m<br /></>
            )}
            {slope > 0 ? '↗ Uphill' : slope < 0 ? '↘ Downhill' : '→ Flat'}
          </div>
        </Tooltip>
      </Polyline>
    )
  }

  return (
    <>
      {segments}
      {showLegend && pathWithSlopes && pathWithSlopes[0].hasOwnProperty('slope') && (
        <div className="slope-path-legend">
          <h4>Path Slope Difficulty</h4>
          <div className="slope-legend-items">
            <div className="slope-legend-item">
              <div className="slope-legend-color" style={{ backgroundColor: '#2ecc71' }}></div>
              <span>Easy (0-5°)</span>
            </div>
            <div className="slope-legend-item">
              <div className="slope-legend-color" style={{ backgroundColor: '#27ae60' }}></div>
              <span>Moderate (5-10°)</span>
            </div>
            <div className="slope-legend-item">
              <div className="slope-legend-color" style={{ backgroundColor: '#f39c12' }}></div>
              <span>Challenging (10-15°)</span>
            </div>
            <div className="slope-legend-item">
              <div className="slope-legend-color" style={{ backgroundColor: '#e67e22' }}></div>
              <span>Hard (15-20°)</span>
            </div>
            <div className="slope-legend-item">
              <div className="slope-legend-color" style={{ backgroundColor: '#e74c3c' }}></div>
              <span>Very Hard (20-25°)</span>
            </div>
            <div className="slope-legend-item">
              <div className="slope-legend-color" style={{ backgroundColor: '#c0392b' }}></div>
              <span>Extreme (25°+)</span>
            </div>
          </div>
        </div>
      )}
    </>
  )
}