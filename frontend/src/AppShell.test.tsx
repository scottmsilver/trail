import { test, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AppShell from './AppShell'

// App boots Leaflet/maps which are heavy in jsdom; stub it so this test
// isolates the shell's tab-switching behavior.
vi.mock('./App', () => ({ default: () => <div data-testid="route-page">route</div> }))

test('renders Route tab by default and switches to Eval', () => {
  render(<AppShell />)
  expect(screen.getByTestId('route-page')).toBeInTheDocument()
  expect(screen.queryByTestId('eval-page')).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', { name: /eval/i }))
  expect(screen.getByTestId('eval-page')).toBeInTheDocument()
  expect(screen.queryByTestId('route-page')).not.toBeInTheDocument()
})
