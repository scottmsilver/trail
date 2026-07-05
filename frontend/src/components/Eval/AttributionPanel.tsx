import React from 'react'
import type { ScoredPath } from '../../services/evalApi'
import { formatCost } from '../../services/evalApi'
import './AttributionPanel.css'

interface AttributionPanelProps {
  scored: ScoredPath
  onHoverSegment?: (i: number | null) => void
}

// Stable color per factor name so bars read consistently across segments.
const FACTOR_COLORS: Record<string, string> = {
  base: '#94a3b8',
  terrain: '#f59e0b',
  slope: '#ef4444',
  sustained: '#8b5cf6',
  deviation: '#3b82f6',
}
const FACTOR_FALLBACK = '#64748b'
const factorColor = (name: string) => FACTOR_COLORS[name] ?? FACTOR_FALLBACK

/** One row per scored segment: index, rounded cost, a stacked bar of the
 *  segment's factors sized by their share of the cost, and the dominant
 *  factor label. Hovering a row reports its index via onHoverSegment. */
const AttributionPanel: React.FC<AttributionPanelProps> = ({ scored, onHoverSegment }) => {
  return (
    <div className="attribution-panel">
      {scored.segments.map((seg, i) => {
        const total = Object.values(seg.factors).reduce((a, b) => a + b, 0)
        return (
          <div
            className="segment-row"
            data-testid="segment-row"
            key={i}
            onMouseEnter={() => onHoverSegment?.(i)}
            onMouseLeave={() => onHoverSegment?.(null)}
          >
            <span className="segment-index">{i + 1}</span>
            <div className="segment-body">
              <div className="segment-bar">
                {Object.entries(seg.factors).map(([name, value]) => (
                  <div
                    key={name}
                    className="segment-slice"
                    title={`${name}: ${Math.round(value)}`}
                    style={{
                      width: total > 0 ? `${(value / total) * 100}%` : '0%',
                      background: factorColor(name),
                    }}
                  />
                ))}
              </div>
              <span className="segment-dominant">{seg.dominantFactor}</span>
            </div>
            <span className="segment-cost">{formatCost(seg.cost)}</span>
          </div>
        )
      })}
    </div>
  )
}

export default AttributionPanel
