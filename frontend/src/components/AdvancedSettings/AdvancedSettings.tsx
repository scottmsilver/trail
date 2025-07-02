import React, { useState, useRef, useEffect } from 'react'
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline'
import type { RouteOptions, SlopeConfig, CustomPathCosts } from '../../services/api'
import './AdvancedSettings.css'

interface AdvancedSettingsProps {
  options: RouteOptions
  onChange: (options: RouteOptions) => void
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({ options, onChange }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [slopeConfigs, setSlopeConfigs] = useState<SlopeConfig[]>([
    { slope_degrees: 0, cost_multiplier: 1.0 },
    { slope_degrees: 15, cost_multiplier: 2.0 },
    { slope_degrees: 30, cost_multiplier: 5.0 }
  ])
  const [useCustomSlopes, setUseCustomSlopes] = useState(false)
  const [useCustomPaths, setUseCustomPaths] = useState(false)
  const [maxSlope, setMaxSlope] = useState<number | undefined>(undefined)
  const [pathCosts, setPathCosts] = useState<CustomPathCosts>({
    trail: 0.3,
    footway: 0.5,
    residential: 0.7,
    off_path: 1.0
  })
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => {
        document.removeEventListener('mousedown', handleClickOutside)
      }
    }
  }, [isOpen])

  const updateOptions = (updates: Partial<RouteOptions>) => {
    onChange({ ...options, ...updates })
  }

  const handleSlopeChange = (index: number, field: 'slope_degrees' | 'cost_multiplier', value: number) => {
    const newConfigs = [...slopeConfigs]
    newConfigs[index] = { ...newConfigs[index], [field]: value }
    setSlopeConfigs(newConfigs)
    
    if (useCustomSlopes) {
      updateOptions({ customSlopeCosts: newConfigs })
    }
  }

  const addSlopeConfig = () => {
    const lastSlope = slopeConfigs[slopeConfigs.length - 1]?.slope_degrees || 0
    const newConfigs = [...slopeConfigs, { slope_degrees: lastSlope + 10, cost_multiplier: 2.0 }]
    setSlopeConfigs(newConfigs)
    
    if (useCustomSlopes) {
      updateOptions({ customSlopeCosts: newConfigs })
    }
  }

  const removeSlopeConfig = (index: number) => {
    const newConfigs = slopeConfigs.filter((_, i) => i !== index)
    setSlopeConfigs(newConfigs)
    
    if (useCustomSlopes) {
      updateOptions({ customSlopeCosts: newConfigs })
    }
  }

  const toggleCustomSlopes = (enabled: boolean) => {
    setUseCustomSlopes(enabled)
    updateOptions({ 
      customSlopeCosts: enabled ? slopeConfigs : undefined,
      maxSlope: enabled ? maxSlope : undefined
    })
  }

  const toggleCustomPaths = (enabled: boolean) => {
    setUseCustomPaths(enabled)
    updateOptions({ 
      customPathCosts: enabled ? pathCosts : undefined 
    })
  }

  const handlePathCostChange = (pathType: keyof CustomPathCosts, value: number) => {
    const newCosts = { ...pathCosts, [pathType]: value }
    setPathCosts(newCosts)
    
    if (useCustomPaths) {
      updateOptions({ customPathCosts: newCosts })
    }
  }

  const handleMaxSlopeChange = (value: number | undefined) => {
    setMaxSlope(value)
    if (useCustomSlopes) {
      updateOptions({ maxSlope: value })
    }
  }

  return (
    <div className="advanced-settings" ref={dropdownRef}>
      <button
        className="advanced-settings-toggle"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>Advanced Settings</span>
        {isOpen ? <ChevronUpIcon className="icon" /> : <ChevronDownIcon className="icon" />}
      </button>

      {isOpen && (
        <div className="advanced-settings-content">
          {/* Slope Configuration */}
          <div className="setting-section">
            <div className="section-header">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useCustomSlopes}
                  onChange={(e) => toggleCustomSlopes(e.target.checked)}
                />
                <span>Custom Slope Penalties</span>
              </label>
            </div>

            {useCustomSlopes && (
              <>
                <div className="slope-configs">
                  {slopeConfigs.map((config, index) => (
                    <div key={index} className="slope-config-row">
                      <div className="input-group">
                        <label>Slope (°)</label>
                        <input
                          type="number"
                          min="0"
                          max="90"
                          step="1"
                          value={config.slope_degrees}
                          onChange={(e) => handleSlopeChange(index, 'slope_degrees', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="input-group">
                        <label>Cost ×</label>
                        <input
                          type="number"
                          min="0.1"
                          max="100"
                          step="0.1"
                          value={config.cost_multiplier}
                          onChange={(e) => handleSlopeChange(index, 'cost_multiplier', parseFloat(e.target.value))}
                        />
                      </div>
                      {slopeConfigs.length > 1 && (
                        <button
                          className="remove-btn"
                          onClick={() => removeSlopeConfig(index)}
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                  <button className="add-btn" onClick={addSlopeConfig}>
                    + Add Slope Point
                  </button>
                </div>

                <div className="max-slope-section">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={maxSlope !== undefined}
                      onChange={(e) => handleMaxSlopeChange(e.target.checked ? 30 : undefined)}
                    />
                    <span>Maximum Slope Limit</span>
                  </label>
                  {maxSlope !== undefined && (
                    <div className="input-group inline">
                      <input
                        type="number"
                        min="1"
                        max="90"
                        step="1"
                        value={maxSlope}
                        onChange={(e) => handleMaxSlopeChange(parseFloat(e.target.value))}
                      />
                      <span>degrees</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Path Preferences */}
          <div className="setting-section">
            <div className="section-header">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useCustomPaths}
                  onChange={(e) => toggleCustomPaths(e.target.checked)}
                />
                <span>Custom Path Preferences</span>
              </label>
            </div>

            {useCustomPaths && (
              <div className="path-costs">
                <div className="path-cost-row">
                  <label>Trails</label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={pathCosts.trail || 1}
                    onChange={(e) => handlePathCostChange('trail', parseFloat(e.target.value))}
                  />
                  <span className="value">{pathCosts.trail?.toFixed(1)}</span>
                </div>
                <div className="path-cost-row">
                  <label>Sidewalks</label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={pathCosts.footway || 1}
                    onChange={(e) => handlePathCostChange('footway', parseFloat(e.target.value))}
                  />
                  <span className="value">{pathCosts.footway?.toFixed(1)}</span>
                </div>
                <div className="path-cost-row">
                  <label>Streets</label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={pathCosts.residential || 1}
                    onChange={(e) => handlePathCostChange('residential', parseFloat(e.target.value))}
                  />
                  <span className="value">{pathCosts.residential?.toFixed(1)}</span>
                </div>
                <div className="path-cost-row">
                  <label>Off-path</label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={pathCosts.off_path || 1}
                    onChange={(e) => handlePathCostChange('off_path', parseFloat(e.target.value))}
                  />
                  <span className="value">{pathCosts.off_path?.toFixed(1)}</span>
                </div>
                <div className="path-cost-help">
                  <small>Lower values = preferred, Higher values = avoided</small>
                </div>
              </div>
            )}
          </div>

          {/* Preset Examples */}
          <div className="preset-examples">
            <h4>Quick Presets:</h4>
            <div className="preset-buttons">
              <button
                className="preset-btn"
                onClick={() => {
                  setSlopeConfigs([
                    { slope_degrees: 0, cost_multiplier: 1.0 },
                    { slope_degrees: 3, cost_multiplier: 2.0 },
                    { slope_degrees: 5, cost_multiplier: 10.0 }
                  ])
                  setMaxSlope(5)
                  setUseCustomSlopes(true)
                  toggleCustomSlopes(true)
                }}
              >
                Wheelchair
              </button>
              <button
                className="preset-btn"
                onClick={() => {
                  setSlopeConfigs([
                    { slope_degrees: 0, cost_multiplier: 1.0 },
                    { slope_degrees: 20, cost_multiplier: 1.2 },
                    { slope_degrees: 40, cost_multiplier: 2.0 }
                  ])
                  setMaxSlope(undefined)
                  setUseCustomSlopes(true)
                  toggleCustomSlopes(true)
                }}
              >
                Mountain Goat
              </button>
              <button
                className="preset-btn"
                onClick={() => {
                  setPathCosts({
                    trail: 0.1,
                    footway: 0.8,
                    residential: 2.0,
                    off_path: 1.5
                  })
                  setUseCustomPaths(true)
                  toggleCustomPaths(true)
                }}
              >
                Trail Lover
              </button>
              <button
                className="preset-btn"
                onClick={() => {
                  setPathCosts({
                    trail: 0.9,
                    footway: 0.2,
                    residential: 0.4,
                    off_path: 3.0
                  })
                  setUseCustomPaths(true)
                  toggleCustomPaths(true)
                }}
              >
                City Walker
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AdvancedSettings