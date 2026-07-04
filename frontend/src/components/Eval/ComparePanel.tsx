import React from 'react'
import type { ScoredPath } from '../../services/evalApi'
import './ComparePanel.css'

interface ComparePanelProps {
  optimal: ScoredPath | null
  drawn: ScoredPath | null
}

const fmt = (n: number) => n.toLocaleString('en-US')

/** Two-column comparison (Optimal vs Yours) of distance, elevation gain, and
 *  total cost, plus the percent difference of the drawn route's cost against
 *  the optimal. Presentational only — parents supply the scored paths. */
const ComparePanel: React.FC<ComparePanelProps> = ({ optimal, drawn }) => {
  const km = (p: ScoredPath | null) => (p ? `${(p.distanceM / 1000).toFixed(2)} km` : '—')
  const gain = (p: ScoredPath | null) => (p ? `${fmt(Math.round(p.elevationGainM))} m` : '—')
  const cost = (p: ScoredPath | null) => (p ? fmt(Math.round(p.totalCost)) : '—')

  let pctLabel = '—'
  if (optimal && drawn && optimal.totalCost !== 0) {
    const pct = ((drawn.totalCost - optimal.totalCost) / optimal.totalCost) * 100
    const rounded = Math.round(pct)
    pctLabel = `${rounded >= 0 ? '+' : ''}${rounded}%`
  }

  const rows: Array<{ label: string; opt: string; drw: string }> = [
    { label: 'Distance', opt: km(optimal), drw: km(drawn) },
    { label: 'Elevation gain', opt: gain(optimal), drw: gain(drawn) },
    { label: 'Total cost', opt: cost(optimal), drw: cost(drawn) },
  ]

  return (
    <div className="compare-panel">
      <div className="compare-header">
        <span className="compare-col-label">Metric</span>
        <span className="compare-col-label">Optimal</span>
        <span className="compare-col-label">Yours</span>
      </div>
      {rows.map((r) => (
        <div className="compare-row" key={r.label}>
          <span className="compare-metric">{r.label}</span>
          <span className="compare-value">{r.opt}</span>
          <span className="compare-value">{r.drw}</span>
        </div>
      ))}
      <div className="compare-diff">
        {optimal && drawn ? (
          <span
            className={`compare-diff-value ${pctLabel.startsWith('+') ? 'worse' : 'better'}`}
          >
            {pctLabel} vs optimal
          </span>
        ) : (
          <span className="compare-diff-placeholder">Run a route</span>
        )}
      </div>
    </div>
  )
}

export default ComparePanel
