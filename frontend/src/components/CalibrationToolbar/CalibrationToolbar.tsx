import React, { useState } from 'react'
import { AdjustmentsHorizontalIcon } from '@heroicons/react/24/outline'
import type { RouteOptions, SlopeConfig, CustomPathCosts } from '../../services/api'
import './CalibrationToolbar.css'

interface CalibrationToolbarProps {
  options: RouteOptions
  onChange: (options: RouteOptions) => void
}

const CalibrationToolbar: React.FC<CalibrationToolbarProps> = ({ options, onChange }) => {
  const [activeTab, setActiveTab] = useState<'slopes' | 'paths' | null>(null)
  const [maxSlope, setMaxSlope] = useState<number>(45)
  const [pathCosts, setPathCosts] = useState<CustomPathCosts>({
    trail: 0.2,
    footway: 0.6,
    residential: 0.85,
    off_path: 0.5
  })

  const updateOptions = (updates: Partial<RouteOptions>) => {
    onChange({ ...options, ...updates })
  }

  const handleMaxSlopeChange = (value: number) => {
    setMaxSlope(value)
    updateOptions({ maxSlope: value })
  }

  const handlePathCostChange = (pathType: keyof CustomPathCosts, value: number) => {
    const newCosts = { ...pathCosts, [pathType]: value }
    setPathCosts(newCosts)
    updateOptions({ customPathCosts: newCosts })
  }

  // Preset slope configurations
  const slopePresets = {
    easy: [
      { slope_degrees: 0, cost_multiplier: 1.0 },
      { slope_degrees: 10, cost_multiplier: 2.0 },
      { slope_degrees: 20, cost_multiplier: 5.0 },
      { slope_degrees: 30, cost_multiplier: 10.0 }
    ],
    moderate: [
      { slope_degrees: 0, cost_multiplier: 1.0 },
      { slope_degrees: 15, cost_multiplier: 1.5 },
      { slope_degrees: 30, cost_multiplier: 3.0 },
      { slope_degrees: 45, cost_multiplier: 8.0 }
    ],
    experienced: [
      { slope_degrees: 0, cost_multiplier: 1.0 },
      { slope_degrees: 20, cost_multiplier: 1.2 },
      { slope_degrees: 40, cost_multiplier: 2.0 },
      { slope_degrees: 60, cost_multiplier: 4.0 }
    ]
  }

  const applySlopePreset = (preset: keyof typeof slopePresets) => {
    updateOptions({ customSlopeCosts: slopePresets[preset] })
  }

  return (
    <div className="calibration-toolbar">
      <div className="toolbar-header">
        <AdjustmentsHorizontalIcon className="icon" />
        <span>Route Calibration</span>
      </div>
      
      <div className="toolbar-tabs">
        <button 
          className={`tab ${activeTab === 'slopes' ? 'active' : ''}`}
          onClick={() => setActiveTab(activeTab === 'slopes' ? null : 'slopes')}
        >
          Slope Tolerance
        </button>
        <button 
          className={`tab ${activeTab === 'paths' ? 'active' : ''}`}
          onClick={() => setActiveTab(activeTab === 'paths' ? null : 'paths')}
        >
          Path Preferences
        </button>
      </div>

      {activeTab === 'slopes' && (
        <div className="toolbar-content">
          <div className="preset-buttons">
            <button onClick={() => applySlopePreset('easy')} className="preset-btn">
              Easy
            </button>
            <button onClick={() => applySlopePreset('moderate')} className="preset-btn">
              Moderate
            </button>
            <button onClick={() => applySlopePreset('experienced')} className="preset-btn">
              Experienced
            </button>
          </div>
          
          <div className="slider-control">
            <label>
              <span>Max Slope: {maxSlope}°</span>
              <input
                type="range"
                min="10"
                max="60"
                value={maxSlope}
                onChange={(e) => handleMaxSlopeChange(Number(e.target.value))}
                className="slope-slider"
              />
            </label>
            <div className="slider-labels">
              <span>Gentle</span>
              <span>Steep</span>
              <span>Extreme</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'paths' && (
        <div className="toolbar-content">
          <div className="path-controls">
            <div className="path-control">
              <label>Natural Trails</label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={pathCosts.trail || 0.2}
                onChange={(e) => handlePathCostChange('trail', Number(e.target.value))}
              />
              <span className="value">{(pathCosts.trail || 0.2).toFixed(1)}</span>
            </div>
            
            <div className="path-control">
              <label>Sidewalks</label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={pathCosts.footway || 0.6}
                onChange={(e) => handlePathCostChange('footway', Number(e.target.value))}
              />
              <span className="value">{(pathCosts.footway || 0.6).toFixed(1)}</span>
            </div>
            
            <div className="path-control">
              <label>Roads</label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={pathCosts.residential || 0.85}
                onChange={(e) => handlePathCostChange('residential', Number(e.target.value))}
              />
              <span className="value">{(pathCosts.residential || 0.85).toFixed(1)}</span>
            </div>
            
            <div className="path-control">
              <label>Off Path</label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={pathCosts.off_path || 0.5}
                onChange={(e) => handlePathCostChange('off_path', Number(e.target.value))}
              />
              <span className="value">{(pathCosts.off_path || 0.5).toFixed(1)}</span>
            </div>
          </div>
          
          <div className="path-hint">
            Lower values = more preferred
          </div>
        </div>
      )}
    </div>
  )
}

export default CalibrationToolbar