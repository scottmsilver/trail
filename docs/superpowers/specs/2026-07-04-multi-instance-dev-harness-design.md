# Multi-Instance Dev Harness — Design

**Date:** 2026-07-04
**Status:** Design approved; implementation plan drafted
**Author:** Scott + Claude

## Problem

We want to run several dev versions of the trail app at the same time — e.g. one
worktree experimenting with map-grid zoom/classification visualization, while
other feature branches keep running untouched. Two hard requirements:

1. **Don't re-fetch OSM/DEM.** The expensive, slow data (OpenStreetMap via
   Overpass + DEM elevation downloads) must be fetched once and shared across all
   instances. Today each checkout resolves its caches relative to its own working
   directory, so a fresh worktree starts with empty caches and re-hits OSM.
2. **Private external reachability.** Each instance should be reachable from
   anywhere (e.g. phone) but visible only to Scott — Cloudflare tunnel + simple
   auth that is logged into occasionally, not every hit.

Out of scope: the map-grid visualization feature itself. That is the motivating
example and gets its own spec/plan once this harness exists.

## Current state (as found)

- Backend: FastAPI on `:9001`, run via `uvicorn app.main:app --reload`.
- Frontend: Vite/React on `:9002`. API base is
  `import.meta.env.VITE_API_URL ?? 'http://localhost:9001'`
  (`frontend/src/services/api.ts:7`). An explicit empty `VITE_API_URL` yields
  `API_BASE === ''`, i.e. same-origin relative calls (`${API_BASE}/api/...`).
- Caches (all resolved relative to the backend working dir). **Corrected after
  investigation** — the 2.1 GB OSM cache is osmnx's own Overpass response cache
  (`ox.settings.cache_folder`, default `./cache`), NOT the HyRiver/DEM cache:
  | Cache | Size | Contents | Env knob today |
  |---|---|---|---|
  | `backend/cache/*.json` | ~2.1 GB | **osmnx Overpass/OSM responses — the expensive one** | none (osmnx default `cache_folder="./cache"`) |
  | `backend/cache/aiohttp_cache.sqlite` | (same dir) | DEM py3dep/HyRiver HTTP downloads | `HYRIVER_CACHE_NAME` (`main.py:44`, `dem_tile_cache.py:81`) |
  | `dem_data/` | 174 MB | DEM composites | none (hardcoded `os.path.abspath("dem_data")`, `dem_tile_cache.py:78`) |
  | `tile_cache/` | 633 MB | computed cost-surface tiles | none (default arg `tile_cache`, `tiled_dem_cache.py:61`) |
  | `path_cache_v2/` | 31 MB | v2 path results + `osmtile_*.pkl` (v2 engine's own OSM tiles) | `TRAIL_V2_PATH_CACHE_DIR` (`engine_v2/service.py:50`) |
- OSM fetching goes through osmnx (`ox.features_from_polygon`) in three places:
  `engine_v2/path_layer.py` (`_default_fetch`), `main.py:773`,
  `dem_tile_cache.py:603`. Overpass mirrors/timeout already come from env
  (`OVERPASS_URLS`, `OVERPASS_TIMEOUT`); `cache_folder` is the missing knob.
- Cloudflare: `cloudflared` 2026.3.0 installed; domain `oursilverfamily.com`
  already onboarded (`~/.cloudflared/cert.pem` + many named tunnels). Wildcard DNS
  and tunnels are fully CLI-scriptable. **Cloudflare Access is NOT** managed by
  `cloudflared` or `wrangler` — only via the Cloudflare API / Terraform.
- Worktrees already the norm (`.claude/worktrees/*`).
- Python venv: `trail_env/` at repo root. Tests run from `backend/` as
  `PYTHONPATH=. ../../../../trail_env/bin/python -m pytest ...` (path relative to
  a worktree; from the main checkout it is `../trail_env/bin/python`).

## Decisions (all approved 2026-07-04)

1. **Share all caches** read-write from a single canonical dir. On a single-user
   box the write-race risk on computed tiles is negligible.
2. **Add three env knobs** — `OSM_CACHE_DIR` (osmnx `cache_folder`, the critical
   OSM one), `TRAIL_DEM_DATA_DIR`, `TRAIL_TILE_CACHE_DIR` — alongside the two that
   already exist (`HYRIVER_CACHE_NAME`, `TRAIL_V2_PATH_CACHE_DIR`). TDD'd.
3. **Random subdomains under a wildcard** `*.t.oursilverfamily.com`, all behind
   **one Cloudflare Access application** scoped to `scottmsilver@gmail.com`.
4. **Access created via the Cloudflare API**, fully scripted, after a one-time
   dashboard bootstrap (mint an API token + enable Zero Trust once).
5. Shared cache at `~/development/trail-shared/`; subdomain prefix `*.t.`;
   dedicated new `trail` tunnel.

## Architecture

### 1. Shared cache

```
~/development/trail-shared/
  osm/            osmnx Overpass responses  -> OSM_CACHE_DIR            (new knob)
  http/           DEM py3dep downloads      -> HYRIVER_CACHE_NAME       (exists; a file path)
  dem_data/       DEM composites            -> TRAIL_DEM_DATA_DIR       (new knob)
  tile_cache/     cost-surface tiles        -> TRAIL_TILE_CACHE_DIR     (new knob)
  path_cache_v2/  v2 paths + osmtile pkls   -> TRAIL_V2_PATH_CACHE_DIR  (exists)
  trail-shared.env    sourced by every instance
  logs/               per-instance backend/frontend logs
```

- Seed once by **moving** the existing populated caches from the main checkout
  into `trail-shared/` (move, not copy — avoids duplicating ~3 GB).
- `trail-shared.env` sets the five vars to absolute paths under `trail-shared/`.
- **Validation gate:** a second instance pointed at a pre-warmed shared cache
  serves a known area with **zero** outbound Overpass requests (assert the osmnx
  cache dir already contains the tile and network is not hit).

### 2. App changes (TDD)

- **`OSM_CACHE_DIR`** — set `ox.settings.cache_folder` from the env var wherever
  osmnx is configured. Centralize in one helper called at each of the three osmnx
  sites so it is DRY and CLI/test paths are covered.
- **`TRAIL_DEM_DATA_DIR`** — `dem_tile_cache.py:78`
  `os.path.abspath("dem_data")` → `os.path.abspath(os.environ.get("TRAIL_DEM_DATA_DIR", "dem_data"))`.
- **`TRAIL_TILE_CACHE_DIR`** — `dem_tile_cache.py:75` construction of
  `TiledDEMCache(...)` passes `cache_dir=os.environ.get("TRAIL_TILE_CACHE_DIR", "tile_cache")`.
- All defaults unchanged when the vars are unset (backwards compatible).

### 3. Per-instance runtime

An instance = a git worktree + a stable port pair + one public hostname.

- **Port pair** derived deterministically from the instance name (hash → offset),
  backend `91xx` / frontend `92xx`. Recorded in `~/.trail-instances.json`.
- **Same-origin API.** Frontend calls a relative `/api`; **Vite's dev server
  proxies `/api` → `http://localhost:<backend_port>`**. `VITE_API_URL` is left
  empty so `API_BASE === ''`. Consequences: one hostname per instance, no CORS,
  no hardcoded server URL (satisfies the "never hardcode a server URL" rule), and
  identical behavior at `localhost:<fp>` and through the tunnel.
- `vite.config.ts` reads `PORT` and `BACKEND_PORT` from env for port + proxy
  target, and sets `server.allowedHosts` to include `.oursilverfamily.com` (Vite
  otherwise blocks requests whose Host header is the public domain).
- Backend `uvicorn --reload --port <bp>`; frontend `vite --port <fp>`. Both
  inherit the shared-cache env. Logs to `trail-shared/logs/<name>.{be,fe}.log`.

### 4. Cloudflare (scripted)

**One-time bootstrap (dashboard, ~2 min):** enable Zero Trust (team name); mint an
API token scoped `Access: Apps and Policies: Edit` + `Zone: DNS: Edit` for
`oursilverfamily.com`; store in `~/development/trail-shared/trail-cf.env`
(chmod 600, never committed).

**One-time scripted setup:** `cloudflared tunnel create trail`;
`cloudflared tunnel route dns trail '*.t.oursilverfamily.com'`; `curl` the Access
API to create one self-hosted app on `*.t.oursilverfamily.com` with an allow
policy `email == scottmsilver@gmail.com`, Google login.

**Per-instance (scripted in the launcher):** append
`<name>.t.oursilverfamily.com → http://localhost:<fp>` to the `trail` tunnel's
ingress YAML (regenerated from `~/.trail-instances.json`) and reload the tunnel.
Wildcard DNS + wildcard Access app mean no per-instance DNS/Access calls.

### 5. Launcher CLI — `scripts/trail-instance`

- `trail-instance up <name> [branch]` — ensure/create worktree; allocate/lookup
  port pair; write registry; source `trail-shared.env`; start backend + frontend
  (backgrounded); regenerate + reload tunnel ingress; print the public URL + ports.
- `trail-instance down <name>` — stop procs, remove from registry + ingress, reload.
- `trail-instance list` — table of instances, ports, URLs, up/down status.

## Concurrency & safety notes

- osmnx Overpass cache and DEM HTTP cache are content-hash keyed (distinct files
  per request) → concurrent reads/writes across instances are safe.
- Computed caches (tiles, composites, path/pkl results) are geography-keyed; two
  instances computing an identical tile at the identical moment could race one
  file. Accepted for a single-user dev box; atomic temp+rename is a later option.
- Secrets: `trail-cf.env` (API token) is chmod 600 and git-ignored. No server URL
  is baked into code. The app stays unauthenticated behind Access.

## Out of scope

- The map-grid zoom/classification visualization feature (separate spec/plan).
- Production/multi-user deployment, autoscaling. Cloudflare terminates TLS.
