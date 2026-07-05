import { test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ComparePanel from './ComparePanel'
import { MOCK_SCORED_PATH } from './mockScoredPath'

test('shows total costs and percent difference', () => {
  const optimal = { ...MOCK_SCORED_PATH, totalCost: 1240 }
  render(<ComparePanel optimal={optimal} drawn={MOCK_SCORED_PATH} />)
  expect(screen.getByText(/1,240/)).toBeInTheDocument()
  expect(screen.getByText(/1,610/)).toBeInTheDocument()
  expect(screen.getByText(/\+30%|30%/)).toBeInTheDocument()
})
