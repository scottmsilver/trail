import { useState } from 'react'
import App from './App'
import EvalPage from './components/Eval/EvalPage'
import './AppShell.css'

/** Top-level tab switch between the existing routing UI and the Eval workbench.
 *  The app has no router, so a simple view toggle keeps things dependency-free. */
export default function AppShell() {
  const [tab, setTab] = useState<'route' | 'eval'>('route')
  return (
    <div className="app-shell">
      <nav className="app-shell-nav">
        <button
          className={tab === 'route' ? 'active' : ''}
          onClick={() => setTab('route')}
          aria-pressed={tab === 'route'}
        >
          Route
        </button>
        <button
          className={tab === 'eval' ? 'active' : ''}
          onClick={() => setTab('eval')}
          aria-pressed={tab === 'eval'}
        >
          Eval
        </button>
      </nav>
      {tab === 'route' ? <App /> : <EvalPage />}
    </div>
  )
}
