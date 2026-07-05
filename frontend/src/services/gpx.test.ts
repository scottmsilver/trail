import { test, expect } from 'vitest'
import { parseGpx, downsamplePath, chooseGpxComparison, buildGpx } from './gpx'

const GPX = `<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk><name>Test</name><trkseg>
    <trkpt lat="40.6250" lon="-111.5700"><ele>2100</ele></trkpt>
    <trkpt lat="40.6260" lon="-111.5680"></trkpt>
    <trkpt lat="40.6270" lon="-111.5660"></trkpt>
  </trkseg></trk>
</gpx>`

test('parseGpx extracts ordered track points with elevation', () => {
  const pts = parseGpx(GPX)
  expect(pts).toHaveLength(3)
  expect(pts[0]).toEqual({ lat: 40.625, lon: -111.57, ele: 2100 }) // has <ele>
  expect(pts[2]).toEqual({ lat: 40.627, lon: -111.566 }) // no <ele>
})

test('parseGpx falls back to rtept when there is no track', () => {
  const rte = `<gpx><rte><rtept lat="1" lon="2"/><rtept lat="3" lon="4"/></rte></gpx>`
  expect(parseGpx(rte)).toEqual([{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }])
})

test('parseGpx rejects invalid xml and empty tracks', () => {
  expect(() => parseGpx('not xml <<<')).toThrow()
  expect(() => parseGpx('<gpx><trk><trkseg></trkseg></trk></gpx>')).toThrow(/fewer than 2/)
})

test('parseGpx skips out-of-range and junk coordinates', () => {
  const bad = `<gpx><trk><trkseg>
    <trkpt lat="40.62" lon="-111.57"/>
    <trkpt lat="999" lon="-111.57"/>
    <trkpt lat="47abc" lon="-111.57"/>
    <trkpt lat="40.63" lon="-111.56"/>
  </trkseg></trk></gpx>`
  const pts = parseGpx(bad)
  expect(pts).toEqual([{ lat: 40.62, lon: -111.57 }, { lat: 40.63, lon: -111.56 }])
})

test('downsamplePath caps points at max and keeps first + last', () => {
  const path = Array.from({ length: 5000 }, (_, i) => ({ lat: 40 + i * 1e-5, lon: -111 }))
  const out = downsamplePath(path, 1500)
  expect(out.length).toBeLessThanOrEqual(1500)
  expect(out[0]).toEqual(path[0])
  expect(out[out.length - 1]).toEqual(path[path.length - 1])
})

test('downsamplePath is a no-op below the cap', () => {
  const path = [{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }]
  expect(downsamplePath(path, 1500)).toBe(path)
})

test('chooseGpxComparison uses a point-to-point track whole', () => {
  const pts = [
    { lat: 40.62, lon: -111.57 },
    { lat: 40.63, lon: -111.56 },
    { lat: 40.64, lon: -111.55 }, // ~far from start
  ]
  const c = chooseGpxComparison(pts)
  expect(c.note).toBe('')
  expect(c.start).toEqual({ lat: 40.62, lon: -111.57 })
  expect(c.end).toEqual({ lat: 40.64, lon: -111.55 })
  expect(c.track).toHaveLength(3)
})

test('chooseGpxComparison splits a loop at its highest point', () => {
  // start ≈ end (a loop); the high point is in the middle.
  const pts = [
    { lat: 40.655, lon: -111.569, ele: 2600 }, // trailhead
    { lat: 40.648, lon: -111.573, ele: 2800 },
    { lat: 40.642, lon: -111.577, ele: 2991 }, // summit (highest)
    { lat: 40.648, lon: -111.574, ele: 2790 },
    { lat: 40.6552, lon: -111.5692, ele: 2605 }, // back near trailhead
  ]
  const c = chooseGpxComparison(pts)
  expect(c.note).toMatch(/Loop detected.*highest point \(2991 m\)/)
  expect(c.end).toEqual({ lat: 40.642, lon: -111.577 }) // the summit
  expect(c.track).toHaveLength(3) // outbound leg: trailhead..summit
})

test('chooseGpxComparison falls back to farthest point for an elevation-less loop', () => {
  const pts = [
    { lat: 40.655, lon: -111.569 },
    { lat: 40.642, lon: -111.577 }, // farthest from start
    { lat: 40.6552, lon: -111.5692 },
  ]
  const c = chooseGpxComparison(pts)
  expect(c.note).toMatch(/farthest point/)
  expect(c.end).toEqual({ lat: 40.642, lon: -111.577 })
})

test('buildGpx round-trips lat/lon through parseGpx', () => {
  const pts = [
    { lat: 40.625, lon: -111.57 },
    { lat: 40.626, lon: -111.568 },
    { lat: 40.627, lon: -111.566 },
  ]
  const parsed = parseGpx(buildGpx(pts, 'Drawn route'))
  expect(parsed).toEqual(pts)
})

test('buildGpx emits <ele> only for points that carry a finite elevation', () => {
  const xml = buildGpx([
    { lat: 40.625, lon: -111.57, ele: 2100 },
    { lat: 40.627, lon: -111.566 },
  ])
  expect(xml).toContain('<ele>2100</ele>')
  // Exactly one <ele> element — the elevation-less point omits it.
  expect(xml.match(/<ele>/g)).toHaveLength(1)
  const parsed = parseGpx(xml)
  expect(parsed[0]).toEqual({ lat: 40.625, lon: -111.57, ele: 2100 })
  expect(parsed[1]).toEqual({ lat: 40.627, lon: -111.566 })
})

test('buildGpx escapes the track name for XML', () => {
  const xml = buildGpx(
    [
      { lat: 1, lon: 2 },
      { lat: 3, lon: 4 },
    ],
    'A & B <"tricky">',
  )
  expect(xml).toContain('<name>A &amp; B &lt;&quot;tricky&quot;&gt;</name>')
  // Still valid XML after escaping.
  expect(() => parseGpx(xml)).not.toThrow()
})

test('buildGpx allows a single point but throws on an empty path', () => {
  expect(() => buildGpx([])).toThrow(/at least one point/)
  expect(buildGpx([{ lat: 1, lon: 2 }])).toContain('<trkpt lat="1" lon="2">')
})

test('buildGpx rejects non-finite / out-of-range coordinates', () => {
  expect(() => buildGpx([{ lat: NaN, lon: -111 }])).toThrow(/invalid coordinate/)
  expect(() => buildGpx([{ lat: 91, lon: 0 }])).toThrow(/invalid coordinate/)
  expect(() => buildGpx([{ lat: 40, lon: 200 }])).toThrow(/invalid coordinate/)
})
