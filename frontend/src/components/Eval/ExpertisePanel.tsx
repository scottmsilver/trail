import './ExpertisePanel.css'
import type { RouteVariant } from '../../services/evalApi'
import { variantsForDisplay, levelLabel, LEVEL_META } from './expertise'

interface ExpertisePanelProps {
  /** Family from POST /api/routes/variants, or null before it has been run. */
  variants: RouteVariant[] | null
  /** level → shown on the map. A level absent from the map (undefined) is shown. */
  visible: Record<string, boolean>
  onToggle: (level: string) => void
  onRun: () => void
  /** No start/end set yet. */
  disabled: boolean
  running: boolean
}

/** Presentational: a "Run family" button plus a legend of the expertise levels
 *  with per-level color, SAC/budget, a visibility checkbox, and a note for
 *  levels that duplicate an easier line or found no route. All routing/state
 *  lives in EvalPage; this only renders and reports clicks. */
export default function ExpertisePanel({
  variants,
  visible,
  onToggle,
  onRun,
  disabled,
  running,
}: ExpertisePanelProps) {
  const rows = variants ? variantsForDisplay(variants, () => true) : []

  return (
    <section className="expertise-panel" data-testid="expertise-panel">
      <h3 className="expertise-title">Route family</h3>
      <button
        className="eval-btn eval-btn-primary expertise-run"
        onClick={onRun}
        disabled={disabled || running}
      >
        {running ? 'Running…' : 'Run family'}
      </button>

      {rows.length > 0 && (
        <ul className="expertise-legend">
          {rows.map((d) => {
            const level = d.variant.level
            const drawable = d.hasRoute && !d.duplicateOf
            const budget = d.variant.scrambleBudgetM
            const sac = LEVEL_META[level]?.sac
            return (
              <li key={level} className="expertise-row">
                <input
                  type="checkbox"
                  aria-label={`show ${levelLabel(level)}`}
                  checked={drawable && visible[level] !== false}
                  disabled={!drawable}
                  onChange={() => onToggle(level)}
                />
                <span
                  className="expertise-swatch"
                  style={{ background: drawable ? d.color : 'transparent', borderColor: d.color }}
                />
                <span className="expertise-name">{levelLabel(level)}</span>
                <span className="expertise-meta">
                  {sac ? `${sac} · ` : ''}
                  {budget} m
                </span>
                {!d.hasRoute && <span className="expertise-note">no route</span>}
                {d.hasRoute && d.duplicateOf && (
                  <span className="expertise-note">same as {levelLabel(d.duplicateOf)}</span>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
