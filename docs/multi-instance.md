# Multi-Instance Dev Harness

Run several dev versions of the trail app at once. Each instance is a git
worktree with its own live-reloading backend + frontend on a derived port pair.
All instances share **one** copy of the expensive OSM/DEM caches, so OSM is
fetched once, ever. Each instance is reachable privately at
`https://<name>.t.oursilverfamily.com`, gated by Cloudflare Access to one email.

## One-time setup

### 1. Shared cache

```bash
tools/seed-shared-cache.sh
```

Copies the existing populated caches from the main checkout into
`~/development/trail-shared/` (non-destructive — the main checkout keeps its own)
and writes `trail-shared.env`. The five cache env vars it sets:

| Var | Holds |
|---|---|
| `OSM_CACHE_DIR` | osmnx Overpass responses (the ~2 GB one) |
| `HYRIVER_CACHE_NAME` | DEM py3dep/HyRiver HTTP downloads (sqlite file) |
| `TRAIL_DEM_DATA_DIR` | DEM composites |
| `TRAIL_TILE_CACHE_DIR` | computed cost-surface tiles |
| `TRAIL_V2_PATH_CACHE_DIR` | v2 path results + `osmtile_*.pkl` OSM tiles |

The backend honors all five with no hardcoded paths; unset ⇒ old working-dir
defaults.

### 2. Cloudflare (token bootstrap, then scripted)

In the dashboard, once: enable Zero Trust (pick a team name, free plan) and mint
a **Custom API token** with a single scope — **Account → Access: Apps and
Policies → Edit**, scoped to the `oursilverfamily` account. No Zone/DNS scope is
needed: the wildcard DNS is created by `cloudflared` using its existing
`cert.pem`, not the token. Then:

```bash
cat > ~/development/trail-shared/trail-cf.env <<'EOF'
export CLOUDFLARE_API_TOKEN="<token>"
export CF_ACCOUNT_ID="2ff88e1aab448260d945962395baf6f1"
export TRAIL_ACCESS_EMAIL="scottmsilver@gmail.com"
EOF
chmod 600 ~/development/trail-shared/trail-cf.env

tools/trail-cf-setup.sh
```

This creates the `trail` tunnel, the wildcard DNS record
`*.t.oursilverfamily.com`, and one Access app + email policy. New instances need
zero further Cloudflare changes.

**Login method:** the default One-time PIN needs no identity provider — Cloudflare
emails a code to `TRAIL_ACCESS_EMAIL` and the session lasts 30 days. "Sign in with
Google" is optional and requires configuring a Google IdP in Zero Trust once.

## Daily use

```bash
tools/trail-instance up grid            # new worktree "grid" (branch worktree-grid)
tools/trail-instance up grid my-branch  # ...or an existing branch
tools/trail-instance list
tools/trail-instance down grid
```

`up` prints the public URL and local ports and tails logs to
`~/development/trail-shared/logs/<name>.{be,fe}.log`.

## How it stays single-origin (no hardcoded server URL)

The frontend calls a relative `/api`; the launcher writes `frontend/.env.local`
with an empty `VITE_API_URL` so `API_BASE === ''`. Vite's dev server proxies
`/api` (and `/docs`, `/openapi.json`) to that instance's backend. So one hostname
serves both, there is no CORS, and it works identically at `localhost:<port>` and
through the tunnel. `vite.config.ts` reads `PORT` / `BACKEND_PORT` from the env
and allows the `.oursilverfamily.com` Host header.

## Secrets

`trail-cf.env` (the API token) lives outside the repo, `chmod 600`, and is
git-ignored. The app itself is unauthenticated — Cloudflare Access is the only
auth surface exposed publicly.
