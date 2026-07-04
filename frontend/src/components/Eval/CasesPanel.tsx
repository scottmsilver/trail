import { useEffect, useState } from 'react'
import './CasesPanel.css'
import type { Coordinate, RouteOptions } from '../../services/api'
import type { EvalCase, EvalLabel } from '../../services/evalApi'
import { listCases, saveCase, deleteCase } from '../../services/evalApi'

/** Turn a case name into a filesystem/id-safe slug (mirrors the backend's
 *  sanitization so save→reload round-trips predictably). */
export function slug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export interface CasesPanelProps {
  current: { start: Coordinate | null; end: Coordinate | null; options: RouteOptions; referencePath: Coordinate[] }
  onLoad: (c: EvalCase) => void
}

/** Save the current eval setup as a named case, list saved cases, load one back,
 *  delete, and apply a manual pass/fail/unsure label on re-run. */
export default function CasesPanel({ current, onLoad }: CasesPanelProps) {
  const [cases, setCases] = useState<EvalCase[]>([])
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [error, setError] = useState('')

  const refresh = async () => {
    try {
      setCases(await listCases())
    } catch (e) {
      setError('Could not load cases: ' + (e as Error).message)
    }
  }

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const cs = await listCases()
        if (alive) setCases(cs)
      } catch (e) {
        if (alive) setError('Could not load cases: ' + (e as Error).message)
      }
    })()
    return () => {
      alive = false
    }
  }, [])

  const canSave = name.trim() !== '' && current.start != null && current.end != null

  const handleSave = async () => {
    if (!canSave || !current.start || !current.end) return
    const c: EvalCase = {
      id: slug(name),
      name: name.trim(),
      notes: notes.trim(),
      start: current.start,
      end: current.end,
      options: current.options,
      referencePath: current.referencePath,
      labels: [],
    }
    try {
      await saveCase(c)
      setName('')
      setNotes('')
      await refresh()
    } catch (e) {
      setError('Save failed: ' + (e as Error).message)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteCase(id)
      await refresh()
    } catch (e) {
      setError('Delete failed: ' + (e as Error).message)
    }
  }

  const handleLabel = async (c: EvalCase, verdict: EvalLabel['verdict']) => {
    const label: EvalLabel = { ts: new Date().toISOString(), verdict }
    try {
      await saveCase({ ...c, labels: [...c.labels, label] })
      await refresh()
    } catch (e) {
      setError('Label failed: ' + (e as Error).message)
    }
  }

  return (
    <section className="cases-panel" data-testid="cases-panel">
      <h3>Cases</h3>

      <div className="cases-save">
        <input
          className="cases-input"
          placeholder="Case name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          aria-label="Case name"
        />
        <input
          className="cases-input"
          placeholder="Notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          aria-label="Case notes"
        />
        <button className="cases-btn" onClick={handleSave} disabled={!canSave}>
          Save current
        </button>
      </div>

      {error && <div className="cases-error">{error}</div>}

      <ul className="cases-list">
        {cases.map((c) => {
          const last = c.labels[c.labels.length - 1]
          return (
            <li key={c.id} className="cases-item" data-testid="cases-item">
              <div className="cases-item-head">
                <span className="cases-name">{c.name}</span>
                {last && <span className={`cases-verdict cases-verdict-${last.verdict}`}>{last.verdict}</span>}
              </div>
              <div className="cases-item-actions">
                <button className="cases-btn cases-btn-sm" onClick={() => onLoad(c)}>
                  Load
                </button>
                <button className="cases-btn cases-btn-sm" onClick={() => handleLabel(c, 'pass')} title="Mark good">
                  ✓
                </button>
                <button className="cases-btn cases-btn-sm" onClick={() => handleLabel(c, 'fail')} title="Mark wrong">
                  ✗
                </button>
                <button className="cases-btn cases-btn-sm" onClick={() => handleLabel(c, 'unsure')} title="Unsure">
                  ?
                </button>
                <button className="cases-btn cases-btn-sm cases-btn-del" onClick={() => handleDelete(c.id)}>
                  Delete
                </button>
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
