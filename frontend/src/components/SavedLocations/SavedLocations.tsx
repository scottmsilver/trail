import { useState } from 'react'
import './SavedLocations.css'
import type { SavedLocation } from '../../hooks/useSavedLocations'

interface SavedLocationsProps {
  presets: SavedLocation[]
  recents: SavedLocation[]
  /** The point offered to "Save current point" — start, end, or map center. */
  currentPoint: { lat: number; lon: number } | null
  onUseAsStart: (loc: SavedLocation) => void
  onUseAsEnd: (loc: SavedLocation) => void
  onAddPreset: (name: string, lat: number, lon: number) => void
  onUpdatePreset: (id: string, name: string) => void
  onDeletePreset: (id: string) => void
  onPromoteRecent: (recent: SavedLocation) => void
}

function coordLabel(loc: SavedLocation): string {
  return `${loc.lat.toFixed(4)}, ${loc.lon.toFixed(4)}`
}

export default function SavedLocations({
  presets,
  recents,
  currentPoint,
  onUseAsStart,
  onUseAsEnd,
  onAddPreset,
  onUpdatePreset,
  onDeletePreset,
  onPromoteRecent,
}: SavedLocationsProps) {
  const [collapsed, setCollapsed] = useState(false)

  const handleSaveCurrent = () => {
    if (!currentPoint) return
    const name = window.prompt('Name this preset:')
    if (name && name.trim()) {
      onAddPreset(name.trim(), currentPoint.lat, currentPoint.lon)
    }
  }

  const handleRename = (loc: SavedLocation) => {
    const name = window.prompt('Rename preset:', loc.name)
    if (name && name.trim()) {
      onUpdatePreset(loc.id, name.trim())
    }
  }

  const renderRow = (loc: SavedLocation, isPreset: boolean) => (
    <li key={loc.id} className="saved-loc-row">
      <div className="saved-loc-info">
        <span className="saved-loc-name">{loc.name}</span>
        <span className="saved-loc-coord">{coordLabel(loc)}</span>
      </div>
      <div className="saved-loc-actions">
        <button
          className="saved-loc-btn start"
          onClick={() => onUseAsStart(loc)}
          aria-label={`Use as start: ${loc.name}`}
          title="Use as start point"
        >
          Start
        </button>
        <button
          className="saved-loc-btn end"
          onClick={() => onUseAsEnd(loc)}
          aria-label={`Use as end: ${loc.name}`}
          title="Use as end point"
        >
          End
        </button>
        {isPreset ? (
          <>
            <button
              className="saved-loc-icon"
              onClick={() => handleRename(loc)}
              aria-label={`Rename ${loc.name}`}
              title="Rename"
            >
              ✎
            </button>
            <button
              className="saved-loc-icon danger"
              onClick={() => onDeletePreset(loc.id)}
              aria-label={`Delete ${loc.name}`}
              title="Delete"
            >
              ✕
            </button>
          </>
        ) : (
          <button
            className="saved-loc-icon"
            onClick={() => onPromoteRecent(loc)}
            aria-label={`Save ${loc.name} as preset`}
            title="Save as preset"
          >
            ★
          </button>
        )}
      </div>
    </li>
  )

  return (
    <div className="saved-locations">
      <div className="saved-locations-header">
        <button
          className="saved-locations-toggle"
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
        >
          {collapsed ? '▸' : '▾'} Saved Locations
        </button>
        <button
          className="saved-loc-save-current"
          onClick={handleSaveCurrent}
          disabled={!currentPoint}
          title={currentPoint ? 'Save the current point as a preset' : 'Set a point first'}
        >
          + Save current point
        </button>
      </div>

      {!collapsed && (
        <div className="saved-locations-body">
          <section>
            <h4 className="saved-loc-section-title">Presets</h4>
            {presets.length === 0 ? (
              <p className="saved-loc-empty">No presets yet.</p>
            ) : (
              <ul className="saved-loc-list">{presets.map((p) => renderRow(p, true))}</ul>
            )}
          </section>

          <section>
            <h4 className="saved-loc-section-title">Recent</h4>
            {recents.length === 0 ? (
              <p className="saved-loc-empty">No recent locations.</p>
            ) : (
              <ul className="saved-loc-list">{recents.map((r) => renderRow(r, false))}</ul>
            )}
          </section>
        </div>
      )}
    </div>
  )
}
