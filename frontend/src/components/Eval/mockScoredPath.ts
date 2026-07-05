import type { ScoredPath } from '../../services/evalApi'

/** Fixture matching the ScoredPath contract, used to build panels (T5) in
 *  parallel with the real backend scorer (T1). */
export const MOCK_SCORED_PATH: ScoredPath = {
  path: [
    { lat: 37.74, lon: -119.53 },
    { lat: 37.742, lon: -119.532 },
    { lat: 37.745, lon: -119.535 },
  ],
  snapped: false,
  totalCost: 1610,
  distanceM: 1800,
  elevationGainM: 240,
  segments: [
    {
      from: { lat: 37.74, lon: -119.53 },
      to: { lat: 37.742, lon: -119.532 },
      cost: 900,
      factors: { base: 600, slope: 260, terrain: 40 },
      dominantFactor: 'slope',
    },
    {
      from: { lat: 37.742, lon: -119.532 },
      to: { lat: 37.745, lon: -119.535 },
      cost: 710,
      factors: { base: 600, terrain: 110 },
      dominantFactor: 'terrain',
    },
  ],
}
