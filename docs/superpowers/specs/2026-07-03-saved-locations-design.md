# Saved Locations — Presets + Recents

**Date:** 2026-07-03
**Status:** Approved design, ready for implementation planning

## Goal

Let the UI remember locations so the user doesn't re-click the same points every
session. Two kinds of remembered locations:

- **Presets** — user-managed, named locations (e.g. "Home — 91 White Pine Canyon").
  Editable from the UI. Seeded from a **git-ignored** file so a real home address
  never gets committed.
- **Recents** — automatically captured locations the user has actually used as a
  start or end point, so recent work is one click away.

Selecting a saved location sets it as the route **start** or **end** point (and
recenters the map).

## Non-goals (YAGNI)

- No backend involvement. Frontend-only.
- No cross-device sync.
- No import/export UI.
- Caching strategy rules do not apply — this is UI preference state, not terrain data.

## Data model

```ts
type SavedLocation = {
  id: string          // stable unique id (e.g. crypto.randomUUID())
  name: string        // "Home — 91 White Pine Canyon"
  lat: number
  lon: number
  kind: 'preset' | 'recent'
}
```

## Storage

Two `localStorage` keys:

- `trail.presets` — array of `SavedLocation` (kind `'preset'`). User-managed.
- `trail.recents` — array of `SavedLocation` (kind `'recent'`), capped at **8**,
  newest first, deduped by **~50 m proximity** (a new recent within ~50 m of an
  existing one replaces/promotes it to the front rather than adding a duplicate).

### Seeding presets

On first load, if `trail.presets` is absent/empty, seed it from a git-ignored
config file:

- Real file (git-ignored): `frontend/src/config/presets.local.json`
- Committed example: `frontend/src/config/presets.local.example.json`
- `.gitignore` excludes `presets.local.json`.

The file is imported **optionally** via
`import.meta.glob('./presets.local.json', { eager: true })` so the build succeeds
whether or not the file exists. If present, its entries seed `trail.presets` on
first run. After first run, localStorage is authoritative (user edits win); the
seed file is only consulted when `trail.presets` has never been populated.

The home address stays out of git two ways: the seed file is git-ignored, and live
edits persist only to localStorage.

`presets.local.example.json` format:

```json
[
  { "name": "Home — Example St", "lat": 40.6461, "lon": -111.4980 },
  { "name": "Favorite Trailhead", "lat": 40.6000, "lon": -111.5400 }
]
```

(`id`/`kind` are assigned at seed time; the file only needs name/lat/lon.)

## Hook: `useSavedLocations`

Owns all reads/writes. Exposes:

- `presets: SavedLocation[]`
- `recents: SavedLocation[]`
- `addPreset(name, lat, lon): void`
- `updatePreset(id, name): void` — rename
- `deletePreset(id): void`
- `recordRecent(lat, lon, name?): void` — cap/dedup/order logic lives here
- `promoteRecentToPreset(recent, name?): void` — save a recent as a preset

All mutations persist to localStorage synchronously.

## UI: `SavedLocations` panel

Collapsible panel in the left controls column (near the profile selector). Two
sections: **Presets** and **Recent**.

Each row: name + compact `lat, lon`, plus two buttons **Start** and **End**.
Clicking sets that point in `App` state and recenters the map.

Preset management:
- **"Save current point"** button — enabled when a start or end is set (falls back
  to map center if neither). Prompts for a name, calls `addPreset`.
- Each preset row: small **edit** (rename) and **delete** controls.
- Recents are read-only, but each has a **★** button to promote it to a preset.

## Wiring in `App.tsx`

- New handlers `useAsStart(loc)` / `useAsEnd(loc)`: set `start` / `end` and
  `mapCenter`, reusing existing state.
- `recordRecent(...)` fires whenever a point is set — from map click, SearchBox
  select, or a saved-location button — so recents capture real usage regardless of
  source.
- The existing `route`-blocks-clicks guard in `handleMapClick` stays as-is. Saved
  Location Start/End buttons bypass that guard, so the user can retarget start/end
  from saved locations even while a route is displayed.

## Testing (TDD, Vitest + testing-library)

Matches existing setup used by `Map.test.tsx`.

Unit tests — `useSavedLocations`:
- seeds presets from seed data when `trail.presets` is empty
- does not re-seed once presets exist (user edits win)
- `addPreset` / `updatePreset` / `deletePreset` mutate and persist
- `recordRecent` caps at 8, newest-first ordering
- `recordRecent` dedup by ~50 m proximity (near-duplicate promotes, not appends)
- `promoteRecentToPreset` creates a preset

Component tests — `SavedLocations`:
- renders presets and recents
- Start / End buttons fire the right callbacks with the location
- "Save current point" → addPreset flow
- rename and delete flows
- ★ promote flow

## Files touched

- `frontend/src/config/presets.local.example.json` (new, committed)
- `frontend/src/config/presets.local.json` (new, git-ignored — dev creates locally)
- `frontend/src/hooks/useSavedLocations.ts` (new)
- `frontend/src/hooks/useSavedLocations.test.ts` (new)
- `frontend/src/components/SavedLocations/SavedLocations.tsx` (new)
- `frontend/src/components/SavedLocations/SavedLocations.css` (new)
- `frontend/src/components/SavedLocations/SavedLocations.test.tsx` (new)
- `frontend/src/App.tsx` (wire in panel + handlers)
- `.gitignore` (add `presets.local.json`)
