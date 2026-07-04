import { test, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AttributionPanel from './AttributionPanel'
import { MOCK_SCORED_PATH } from './mockScoredPath'

test('one row per segment, dominant factor shown, hover fires', () => {
  const onHover = vi.fn()
  render(<AttributionPanel scored={MOCK_SCORED_PATH} onHoverSegment={onHover} />)
  const rows = screen.getAllByTestId('segment-row')
  expect(rows).toHaveLength(2)
  expect(screen.getByText(/slope/i)).toBeInTheDocument()
  fireEvent.mouseEnter(rows[0]); expect(onHover).toHaveBeenCalledWith(0)
  fireEvent.mouseLeave(rows[0]); expect(onHover).toHaveBeenCalledWith(null)
})
