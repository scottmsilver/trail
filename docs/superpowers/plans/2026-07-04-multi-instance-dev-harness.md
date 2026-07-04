# Multi-Instance Dev Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run several dev versions of the trail app at once — each an isolated
worktree with its own live-reloading backend+frontend — that all share one copy
of the expensive OSM/DEM caches and are each reachable at a private,
Access-gated URL.

**Architecture:** Redirect all five caches to a canonical `~/development/trail-shared/`
via env vars (three new knobs added to the backend). Each instance runs its own
uvicorn+vite on a derived port pair, with Vite proxying `/api` to its backend so
the frontend is single-origin. One Cloudflare `trail` tunnel + wildcard
`*.t.oursilverfamily.com` behind one Access policy exposes each instance. A
`scripts/trail-instance` CLI ties it together.

**Tech Stack:** Python 3 / FastAPI / uvicorn, osmnx + py3dep, Vite/React/TypeScript,
`cloudflared`, Cloudflare API (curl), bash.

## Global Constraints

- **Never hardcode a server URL in code** — hosts/ports come from env or config.
  (Copied from `~/.claude/CLAUDE.md`.)
- **Do NOT `git commit` or `git push` automatically.** Commits are gated on Scott
  typing the password `1234`; pushes require explicit per-time approval. Where a
  step says "Commit", **stage only and pause for Scott's authorization.**
- **Python always via the repo venv** `trail_env` — never system/anaconda python.
  Absolute interpreter: `/home/ssilver/development/trail/trail_env/bin/python`.
  Backend tests run from `backend/` with `PYTHONPATH=.`.
- **TDD** — failing test first, minimal code, green, then stage. Do not disable a
  test to make it pass.
- Env-var defaults must be **backwards compatible**: unset var ⇒ current behavior.
- Canonical shared dir: `~/development/trail-shared/`. Subdomain prefix `*.t.`.
  Domain `oursilverfamily.com`. Access policy email `scottmsilver@gmail.com`.

## File Structure

- Create: `backend/app/services/osm_settings.py` — osmnx cache-folder config helper.
- Modify: `backend/app/services/dem_tile_cache.py` — `TRAIL_DEM_DATA_DIR`,
  `TRAIL_TILE_CACHE_DIR` knobs + call osm_settings helper at the osmnx site.
- Modify: `backend/app/engine_v2/path_layer.py` — call osm_settings helper.
- Modify: `backend/app/main.py` — call osm_settings helper at the osmnx site.
- Create: `backend/tests/unit/test_osm_settings.py`, `backend/tests/unit/test_cache_env_dirs.py`.
- Modify: `frontend/vite.config.ts` — env-driven port + `/api` proxy + allowedHosts.
- Create: `scripts/trail-instance` — launcher CLI (up/down/list).
- Create: `scripts/trail-cf-setup.sh` — one-time Cloudflare tunnel/DNS/Access setup.
- Create: `scripts/lib/trail-registry.sh` — shared registry/port/ingress helpers.
- Create: `docs/multi-instance.md` — operator docs incl. one-time token bootstrap.
- Modify: `.gitignore` — ignore `trail-cf.env` and any local instance state.

---

### Task 1: `OSM_CACHE_DIR` knob (osmnx Overpass cache — the 2.1 GB one)

**Files:**
- Create: `backend/app/services/osm_settings.py`
- Create: `backend/tests/unit/test_osm_settings.py`
- Modify: `backend/app/engine_v2/path_layer.py` (in `_default_fetch`, after `import osmnx as ox`)
- Modify: `backend/app/main.py:753-771` (after `import osmnx as ox`)
- Modify: `backend/app/services/dem_tile_cache.py:601-603` (after `import osmnx as ox`)

**Interfaces:**
- Produces: `osm_settings.osm_cache_dir(default: str = "cache") -> str` (absolute path);
  `osm_settings.apply_osm_settings(ox) -> None` (sets `ox.settings.cache_folder`).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_osm_settings.py`:

```python
import os
from app.services import osm_settings


def test_osm_cache_dir_defaults_to_local_cache(monkeypatch):
    monkeypatch.delenv("OSM_CACHE_DIR", raising=False)
    assert osm_settings.osm_cache_dir() == os.path.abspath("cache")


def test_osm_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OSM_CACHE_DIR", str(tmp_path / "osm"))
    assert osm_settings.osm_cache_dir() == os.path.abspath(str(tmp_path / "osm"))


def test_apply_osm_settings_sets_cache_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("OSM_CACHE_DIR", str(tmp_path / "osm"))

    class FakeSettings:
        cache_folder = "cache"

    class FakeOx:
        settings = FakeSettings()

    ox = FakeOx()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.cache_folder == os.path.abspath(str(tmp_path / "osm"))
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `backend/`):
```bash
cd /home/ssilver/development/trail/.claude/worktrees/multi-instance-harness/backend
PYTHONPATH=. /home/ssilver/development/trail/trail_env/bin/python -m pytest tests/unit/test_osm_settings.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.osm_settings'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/services/osm_settings.py`:

```python
"""osmnx settings helper.

osmnx caches Overpass responses under ``ox.settings.cache_folder`` (default
``./cache``). To share that cache across dev instances we redirect it via the
``OSM_CACHE_DIR`` env var without hardcoding any path in code.
"""
import os


def osm_cache_dir(default: str = "cache") -> str:
    """Absolute path osmnx should use for its Overpass response cache."""
    return os.path.abspath(os.environ.get("OSM_CACHE_DIR", default))


def apply_osm_settings(ox) -> None:
    """Point osmnx's cache at ``OSM_CACHE_DIR`` (no-op change when unset)."""
    ox.settings.cache_folder = osm_cache_dir()
```

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command as Step 2. Expected: 3 passed.

- [ ] **Step 5: Wire the helper into the three osmnx sites**

In `backend/app/engine_v2/path_layer.py`, inside `_default_fetch`, immediately
after `import osmnx as ox` (currently line ~209):

```python
    import osmnx as ox

    from app.services.osm_settings import apply_osm_settings
    apply_osm_settings(ox)
```

In `backend/app/main.py`, in the block that does `import osmnx as ox` (~line 753),
right after the import:

```python
        import osmnx as ox

        from app.services.osm_settings import apply_osm_settings
        apply_osm_settings(ox)
```

In `backend/app/services/dem_tile_cache.py`, at the osmnx use (~line 601-603),
right after `import osmnx as ox`:

```python
            import osmnx as ox

            from app.services.osm_settings import apply_osm_settings
            apply_osm_settings(ox)
```

- [ ] **Step 6: Run the fast smoke suite to confirm nothing broke**

Run:
```bash
cd /home/ssilver/development/trail/.claude/worktrees/multi-instance-harness
./run_tests.sh
```
Expected: PASS (same set as before, plus the new module imports cleanly).

- [ ] **Step 7: Stage (do NOT commit — await Scott's `1234`)**

```bash
git add backend/app/services/osm_settings.py backend/tests/unit/test_osm_settings.py \
        backend/app/engine_v2/path_layer.py backend/app/main.py \
        backend/app/services/dem_tile_cache.py
# Pause here for authorization before: git commit -m "feat(cache): OSM_CACHE_DIR knob for osmnx Overpass cache"
```

---

### Task 2: `TRAIL_DEM_DATA_DIR` + `TRAIL_TILE_CACHE_DIR` knobs

**Files:**
- Modify: `backend/app/services/dem_tile_cache.py:75` and `:78`
- Create: `backend/tests/unit/test_cache_env_dirs.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DEMTileCache().dem_data_dir` honors `TRAIL_DEM_DATA_DIR`;
  `DEMTileCache().tiled_cache.cache_dir` honors `TRAIL_TILE_CACHE_DIR`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_cache_env_dirs.py`:

```python
import os
from app.services.dem_tile_cache import DEMTileCache


def test_dem_data_dir_defaults(monkeypatch):
    monkeypatch.delenv("TRAIL_DEM_DATA_DIR", raising=False)
    cache = DEMTileCache()
    assert cache.dem_data_dir == os.path.abspath("dem_data")


def test_dem_data_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIL_DEM_DATA_DIR", str(tmp_path / "dem"))
    cache = DEMTileCache()
    assert cache.dem_data_dir == os.path.abspath(str(tmp_path / "dem"))


def test_tile_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIL_TILE_CACHE_DIR", str(tmp_path / "tiles"))
    cache = DEMTileCache()
    assert cache.tiled_cache.cache_dir == os.path.abspath(str(tmp_path / "tiles"))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/ssilver/development/trail/.claude/worktrees/multi-instance-harness/backend
PYTHONPATH=. /home/ssilver/development/trail/trail_env/bin/python -m pytest tests/unit/test_cache_env_dirs.py -v
```
Expected: FAIL — `test_dem_data_dir_honors_env` and `test_tile_cache_dir_honors_env`
fail (paths still resolve to the working-dir defaults).

- [ ] **Step 3: Write minimal implementation**

In `backend/app/services/dem_tile_cache.py`, change the `TiledDEMCache`
construction (line ~75):

```python
        # Initialize tiled cache for cost surfaces
        self.tiled_cache = TiledDEMCache(
            tile_size_degrees=0.01,
            cache_dir=os.environ.get("TRAIL_TILE_CACHE_DIR", "tile_cache"),
        )  # ~1km tiles
```

and the DEM data dir (line ~78):

```python
        # Get absolute path for DEM data directory
        self.dem_data_dir = os.path.abspath(
            os.environ.get("TRAIL_DEM_DATA_DIR", "dem_data")
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command as Step 2. Expected: 3 passed.

- [ ] **Step 5: Run the fast smoke suite**

Run:
```bash
cd /home/ssilver/development/trail/.claude/worktrees/multi-instance-harness
./run_tests.sh
```
Expected: PASS.

- [ ] **Step 6: Stage (do NOT commit — await `1234`)**

```bash
git add backend/app/services/dem_tile_cache.py backend/tests/unit/test_cache_env_dirs.py
# Pause for authorization before: git commit -m "feat(cache): TRAIL_DEM_DATA_DIR and TRAIL_TILE_CACHE_DIR knobs"
```

---

### Task 3: Env-driven Vite config (port + `/api` proxy + allowedHosts)

**Files:**
- Modify: `frontend/vite.config.ts`

**Interfaces:**
- Consumes env at dev-server start: `PORT` (frontend port), `BACKEND_PORT`
  (uvicorn port to proxy `/api` to).
- Produces: a Vite dev server that serves the app on `PORT`, proxies `/api` and
  `/docs`/`/openapi.json` to `http://localhost:${BACKEND_PORT}`, and accepts the
  public Host header.

- [ ] **Step 1: Replace `frontend/vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Ports and the backend proxy target come from the environment so many
// instances can run at once without hardcoding a server URL.
const PORT = Number(process.env.PORT ?? 9002)
const BACKEND_PORT = Number(process.env.BACKEND_PORT ?? 9001)
const backend = `http://localhost:${BACKEND_PORT}`

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: PORT,
    strictPort: true,
    host: 'localhost',
    // Requests arrive through the Cloudflare tunnel with a public Host header;
    // allow the whole family domain so Vite does not reject them.
    allowedHosts: ['.oursilverfamily.com', 'localhost'],
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/openapi.json': { target: backend, changeOrigin: true },
      '/docs': { target: backend, changeOrigin: true },
    },
  },
})
```

- [ ] **Step 2: Verify same-origin proxy works end-to-end**

Ensure a backend is running on 9001, then:
```bash
cd /home/ssilver/development/trail/.claude/worktrees/multi-instance-harness/frontend
VITE_API_URL= PORT=9002 BACKEND_PORT=9001 npm run dev &
sleep 4
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9002/api/health
```
Expected: `200` (the request hit Vite on 9002 and was proxied to the backend).
Then stop the dev server (`kill %1`).

- [ ] **Step 3: Stage (do NOT commit — await `1234`)**

```bash
git add frontend/vite.config.ts
# Pause for authorization before: git commit -m "feat(frontend): env-driven vite port + /api proxy for multi-instance"
```

---

### Task 4: Shared cache dir, seed, and `trail-shared.env`

**Files:**
- Create (outside the repo): `~/development/trail-shared/` tree + `trail-shared.env`.
- Create: `scripts/seed-shared-cache.sh` (idempotent seeding helper, committed).

**Interfaces:**
- Produces: `~/development/trail-shared/trail-shared.env` exporting `OSM_CACHE_DIR`,
  `HYRIVER_CACHE_NAME`, `TRAIL_DEM_DATA_DIR`, `TRAIL_TILE_CACHE_DIR`,
  `TRAIL_V2_PATH_CACHE_DIR` as absolute paths.

- [ ] **Step 1: Create `scripts/seed-shared-cache.sh`**

```bash
#!/usr/bin/env bash
# Seed the canonical shared cache by MOVING existing populated caches out of the
# main checkout. Idempotent: skips anything already present in the shared dir.
set -euo pipefail

SHARED="${TRAIL_SHARED_DIR:-$HOME/development/trail-shared}"
MAIN="${TRAIL_MAIN_CHECKOUT:-$HOME/development/trail}"

mkdir -p "$SHARED"/{osm,dem_data,tile_cache,path_cache_v2,logs}

move_once() { # $1 = src dir/file, $2 = dest dir
  local src="$1" dest="$2"
  if [[ -e "$src" && -z "$(ls -A "$dest" 2>/dev/null || true)" ]]; then
    echo "moving $src -> $dest"
    mv "$src"/* "$dest"/ 2>/dev/null || true
  else
    echo "skip $src (missing or dest non-empty)"
  fi
}

# osmnx Overpass cache (the big one) lives in $MAIN/backend/cache/*.json
mkdir -p "$SHARED/osm"
if [[ -d "$MAIN/backend/cache" && -z "$(ls -A "$SHARED/osm" 2>/dev/null || true)" ]]; then
  echo "moving osmnx json cache -> $SHARED/osm"
  find "$MAIN/backend/cache" -maxdepth 1 -name '*.json' -exec mv {} "$SHARED/osm"/ \;
fi
# DEM HyRiver sqlite (if present) -> $SHARED/http/
mkdir -p "$SHARED/http"
[[ -f "$MAIN/backend/cache/aiohttp_cache.sqlite" ]] && \
  mv "$MAIN/backend/cache/aiohttp_cache.sqlite" "$SHARED/http/" || true

move_once "$MAIN/dem_data"          "$SHARED/dem_data"
move_once "$MAIN/backend/tile_cache" "$SHARED/tile_cache"
move_once "$MAIN/backend/path_cache_v2" "$SHARED/path_cache_v2"

cat > "$SHARED/trail-shared.env" <<EOF
# Shared cache locations for all trail dev instances. Source before launching.
export OSM_CACHE_DIR="$SHARED/osm"
export HYRIVER_CACHE_NAME="$SHARED/http/aiohttp_cache.sqlite"
export TRAIL_DEM_DATA_DIR="$SHARED/dem_data"
export TRAIL_TILE_CACHE_DIR="$SHARED/tile_cache"
export TRAIL_V2_PATH_CACHE_DIR="$SHARED/path_cache_v2"
EOF
echo "wrote $SHARED/trail-shared.env"
```

- [ ] **Step 2: Make it executable and run it**

```bash
chmod +x scripts/seed-shared-cache.sh
./scripts/seed-shared-cache.sh
du -sh ~/development/trail-shared/* ; cat ~/development/trail-shared/trail-shared.env
```
Expected: `osm/` holds the bulk (~2 GB); `trail-shared.env` lists 5 absolute paths.

- [ ] **Step 3: Validation gate — prove a fresh instance does NOT re-hit OSM**

Pick a small area already represented in the OSM cache. With the shared env
sourced and `OVERPASS_URLS` pointed at an unreachable host (so any *real* fetch
would fail), a cached area must still resolve:

```bash
cd /home/ssilver/development/trail/.claude/worktrees/multi-instance-harness/backend
source ~/development/trail-shared/trail-shared.env
OVERPASS_URLS="http://127.0.0.1:9/none" \
  PYTHONPATH=. /home/ssilver/development/trail/trail_env/bin/python - <<'PY'
import os
# Count osm cache files before; resolve a known-cached grid; expect no failure
# and no new network fetch (osmnx serves from cache_folder).
before = len(os.listdir(os.environ["OSM_CACHE_DIR"]))
# Import here and exercise the v2 PathLayer over a KNOWN cached tile.
# (Executor: substitute a bbox you confirmed is already cached, e.g. from an
#  osmtile_*.pkl name or a recent route.) Assert it returns a grid.
print("osm cache files:", before)
PY
```
Expected: resolves from cache with the Overpass endpoint unreachable, proving the
shared osmnx cache is honored. If it instead tries to fetch, `OSM_CACHE_DIR` is
not wired correctly — return to Task 1.

- [ ] **Step 4: Stage the helper (do NOT commit — await `1234`)**

```bash
git add scripts/seed-shared-cache.sh
# Pause for authorization before: git commit -m "feat(cache): shared-cache seeding script"
```

---

### Task 5: One-time Cloudflare setup — tunnel, wildcard DNS, Access app

**Prerequisite (manual, one-time, dashboard):** Scott enables Zero Trust (picks a
team name) and mints an API token scoped `Access: Apps and Policies: Edit` +
`Zone: DNS: Edit` for `oursilverfamily.com`. Store it:

```bash
mkdir -p ~/development/trail-shared
cat > ~/development/trail-shared/trail-cf.env <<'EOF'
export CLOUDFLARE_API_TOKEN="<paste token>"
export CF_ACCOUNT_ID="<account id>"
export CF_ZONE_ID="<zone id for oursilverfamily.com>"
export TRAIL_ACCESS_EMAIL="scottmsilver@gmail.com"
EOF
chmod 600 ~/development/trail-shared/trail-cf.env
```

**Files:**
- Create: `scripts/trail-cf-setup.sh`

- [ ] **Step 1: Create `scripts/trail-cf-setup.sh`**

```bash
#!/usr/bin/env bash
# One-time: create the `trail` tunnel, wildcard DNS, and a single Access app
# gating *.t.oursilverfamily.com to one email. Requires trail-cf.env sourced.
set -euo pipefail
source ~/development/trail-shared/trail-cf.env
: "${CLOUDFLARE_API_TOKEN:?}" "${CF_ACCOUNT_ID:?}" "${TRAIL_ACCESS_EMAIL:?}"

WILDCARD="*.t.oursilverfamily.com"
API="https://api.cloudflare.com/client/v4"
auth=(-H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json")

# 1) Tunnel (idempotent: reuse if it exists)
if ! cloudflared tunnel list 2>/dev/null | grep -qw trail; then
  cloudflared tunnel create trail
fi
# 2) Wildcard DNS -> tunnel
cloudflared tunnel route dns --overwrite-dns trail "$WILDCARD"

# 3) Access application on the wildcard, with an allow-by-email policy
app_payload=$(cat <<JSON
{"name":"trail-dev","domain":"$WILDCARD","type":"self_hosted",
 "session_duration":"720h","app_launcher_visible":false}
JSON
)
app_id=$(curl -s "${auth[@]}" -X POST \
  "$API/accounts/$CF_ACCOUNT_ID/access/apps" --data "$app_payload" \
  | /home/ssilver/development/trail/trail_env/bin/python -c 'import sys,json;print(json.load(sys.stdin)["result"]["id"])')
echo "Access app: $app_id"

policy_payload=$(cat <<JSON
{"name":"only-scott","decision":"allow",
 "include":[{"email":{"email":"$TRAIL_ACCESS_EMAIL"}}]}
JSON
)
curl -s "${auth[@]}" -X POST \
  "$API/accounts/$CF_ACCOUNT_ID/access/apps/$app_id/policies" \
  --data "$policy_payload" >/dev/null
echo "Access policy attached for $TRAIL_ACCESS_EMAIL"
```

> Note: Cloudflare's Access API shape has evolved (inline `policies` vs the
> separate `/policies` endpoint, and account-level reusable policies). The
> executor should confirm the current schema at
> `https://developers.cloudflare.com/api/` for `Access Applications` /
> `Access Policies` and adjust the two payloads if the account rejects them.

- [ ] **Step 2: Run it and verify**

```bash
chmod +x scripts/trail-cf-setup.sh
source ~/development/trail-shared/trail-cf.env
./scripts/trail-cf-setup.sh
cloudflared tunnel list | grep trail
```
Expected: tunnel `trail` exists; wildcard CNAME created; Access app + policy
created (verify in the Zero Trust dashboard once, then never again).

- [ ] **Step 3: Stage (do NOT commit — await `1234`)**

```bash
git add scripts/trail-cf-setup.sh
# Pause for authorization before: git commit -m "feat(cf): one-time trail tunnel + wildcard DNS + Access app"
```

---

### Task 6: `scripts/trail-instance` launcher (up / down / list)

**Files:**
- Create: `scripts/lib/trail-registry.sh` (registry + port derivation + ingress regen)
- Create: `scripts/trail-instance`

**Interfaces:**
- Registry file: `~/.trail-instances.json` — `{ "<name>": {"branch","backend_port",
  "frontend_port","hostname","be_pid","fe_pid"} }`.
- Tunnel ingress file: `~/.cloudflared/config-trail.yml`, regenerated from the
  registry each up/down.

- [ ] **Step 1: Create `scripts/lib/trail-registry.sh`**

```bash
#!/usr/bin/env bash
# Shared helpers for the trail-instance launcher.
set -euo pipefail
REG="$HOME/.trail-instances.json"
SHARED="${TRAIL_SHARED_DIR:-$HOME/development/trail-shared}"
MAIN="${TRAIL_MAIN_CHECKOUT:-$HOME/development/trail}"
TUNNEL_CFG="$HOME/.cloudflared/config-trail.yml"
PY=/home/ssilver/development/trail/trail_env/bin/python

[[ -f "$REG" ]] || echo '{}' > "$REG"

# Deterministic port pair from the instance name.
ports_for() { # $1=name -> prints "BACKEND FRONTEND"
  local h; h=$(printf '%s' "$1" | cksum | cut -d' ' -f1)
  local off=$(( h % 80 ))          # 0..79
  echo "$((9100 + off)) $((9200 + off))"
}

reg_set() { # $1=name $2=json-object
  "$PY" - "$REG" "$1" "$2" <<'PY'
import json,sys
reg_path,name,obj=sys.argv[1],sys.argv[2],sys.argv[3]
reg=json.load(open(reg_path)); reg[name]=json.loads(obj)
json.dump(reg,open(reg_path,'w'),indent=2)
PY
}
reg_del() { "$PY" - "$REG" "$1" <<'PY'
import json,sys
reg=json.load(open(sys.argv[1])); reg.pop(sys.argv[2],None)
json.dump(reg,open(sys.argv[1],'w'),indent=2)
PY
}
reg_get() { "$PY" - "$REG" "$1" <<'PY'
import json,sys
print(json.dumps(json.load(open(sys.argv[1])).get(sys.argv[2],{})))
PY
}

# Regenerate the tunnel ingress from the registry and reload the tunnel.
regen_ingress() {
  "$PY" - "$REG" "$TUNNEL_CFG" <<'PY'
import json,sys
reg=json.load(open(sys.argv[1]))
lines=["tunnel: trail",
       "credentials-file: %s/.cloudflared/trail.json"%__import__('os').path.expanduser('~'),
       "ingress:"]
for name,info in reg.items():
    lines.append("  - hostname: %s" % info["hostname"])
    lines.append("    service: http://localhost:%d" % info["frontend_port"])
lines.append("  - service: http_status:404")
open(sys.argv[2],'w').write("\n".join(lines)+"\n")
PY
  # Reload: restart the trail tunnel process (systemd --user unit if present,
  # else kill+run). Executor: wire to however cloudflared is supervised here.
  pkill -f "cloudflared.*config-trail.yml" 2>/dev/null || true
  nohup cloudflared tunnel --config "$TUNNEL_CFG" run trail \
    >> "$SHARED/logs/tunnel.log" 2>&1 &
}
```

- [ ] **Step 2: Create `scripts/trail-instance`**

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/lib/trail-registry.sh"
source "$SHARED/trail-shared.env"

cmd="${1:-}"; name="${2:-}"

up() {
  local branch="${3:-worktree-$name}"
  local wt="$MAIN/.claude/worktrees/$name"
  [[ -d "$wt" ]] || git -C "$MAIN" worktree add "$wt" -b "$branch" 2>/dev/null \
    || git -C "$MAIN" worktree add "$wt" "$branch"
  read -r BP FP < <(ports_for "$name")
  local host="$name.t.oursilverfamily.com"
  mkdir -p "$SHARED/logs"
  # Backend
  ( cd "$wt/backend" && PYTHONPATH=. nohup \
    /home/ssilver/development/trail/trail_env/bin/python -m uvicorn app.main:app \
    --reload --port "$BP" >> "$SHARED/logs/$name.be.log" 2>&1 & echo $! > "/tmp/trail-$name.be" )
  # Frontend (single-origin: empty VITE_API_URL, proxy to backend)
  ( cd "$wt/frontend" && [[ -d node_modules ]] || npm install
    VITE_API_URL= PORT="$FP" BACKEND_PORT="$BP" nohup npm run dev \
    >> "$SHARED/logs/$name.fe.log" 2>&1 & echo $! > "/tmp/trail-$name.fe" )
  reg_set "$name" "$(printf '{"branch":"%s","backend_port":%d,"frontend_port":%d,"hostname":"%s","be_pid":%s,"fe_pid":%s}' \
    "$branch" "$BP" "$FP" "$host" "$(cat /tmp/trail-$name.be)" "$(cat /tmp/trail-$name.fe)")"
  regen_ingress
  echo "UP  $name  https://$host  (frontend :$FP  backend :$BP)"
}

down() {
  local info; info="$(reg_get "$name")"
  for pid in $("$PY" -c 'import json,sys;i=json.loads(sys.argv[1]);print(i.get("be_pid",""),i.get("fe_pid",""))' "$info"); do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  reg_del "$name"; regen_ingress
  echo "DOWN $name"
}

list() {
  "$PY" - "$REG" <<'PY'
import json,sys
reg=json.load(open(sys.argv[1]))
w=max([len(k) for k in reg]+[4])
print(f'{"NAME":<{w}}  FRONT  BACK   URL')
for n,i in reg.items():
    print(f'{n:<{w}}  {i["frontend_port"]:<5}  {i["backend_port"]:<5}  https://{i["hostname"]}')
PY
}

case "$cmd" in
  up)   [[ -n "$name" ]] || { echo "usage: trail-instance up <name> [branch]"; exit 1; }; up "$@";;
  down) [[ -n "$name" ]] || { echo "usage: trail-instance down <name>"; exit 1; }; down;;
  list) list;;
  *) echo "usage: trail-instance {up <name> [branch]|down <name>|list}"; exit 1;;
esac
```

- [ ] **Step 3: Make executable and smoke-test `up`/`list`/`down`**

```bash
chmod +x scripts/trail-instance scripts/lib/trail-registry.sh
./scripts/trail-instance up demo
sleep 6
./scripts/trail-instance list
curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:$(./scripts/trail-instance list | awk '/^demo/{print $2}')/api/health"
./scripts/trail-instance down demo
```
Expected: `up` prints the URL + ports; `/api/health` returns `200` via the Vite
proxy; `list` shows `demo`; `down` stops it and clears the row.

- [ ] **Step 4: Verify external + Access**

In a browser (or phone) open `https://demo.t.oursilverfamily.com` while `demo` is
up. Expected: Cloudflare Access login (Google) → after auth, the trail app loads;
a different Google account is denied.

- [ ] **Step 5: Stage (do NOT commit — await `1234`)**

```bash
git add scripts/trail-instance scripts/lib/trail-registry.sh
# Pause for authorization before: git commit -m "feat: trail-instance launcher (up/down/list)"
```

---

### Task 7: Operator docs + `.gitignore`

**Files:**
- Create: `docs/multi-instance.md`
- Modify: `.gitignore`

- [ ] **Step 1: Append secrets/state to `.gitignore`**

Add these lines to `.gitignore`:

```
# multi-instance harness (local secrets / state live outside the repo)
trail-cf.env
```

- [ ] **Step 2: Write `docs/multi-instance.md`**

Include: the one-time bootstrap (token scopes, `trail-cf.env` fields, Zero Trust
team name, `trail-cf-setup.sh`), `seed-shared-cache.sh`, the five cache env vars
and what each holds, and the `trail-instance up/down/list` usage with the URL
pattern `https://<name>.t.oursilverfamily.com`. Note the same-origin design
(Vite proxies `/api`) and that no server URL is hardcoded.

- [ ] **Step 3: Stage (do NOT commit — await `1234`)**

```bash
git add .gitignore docs/multi-instance.md
# Pause for authorization before: git commit -m "docs: multi-instance dev harness operator guide"
```

---

## Self-Review

- **Spec coverage:** cache sharing (Tasks 1,2,4), the corrected OSM knob (Task 1),
  same-origin frontend (Task 3), instance lifecycle/ports (Task 6), Cloudflare
  tunnel+wildcard+Access (Task 5), token bootstrap + docs + gitignore (Tasks 5,7).
  Validation gate for "no re-hit OSM" is Task 4 Step 3. All spec sections mapped.
- **Placeholders:** the only deliberately-open items are (a) the exact bbox for the
  Task 4 validation (executor supplies a known-cached area) and (b) the Cloudflare
  Access API payload schema (executor confirms against current docs) — both flagged
  inline with how to resolve, not silent TODOs.
- **Type/name consistency:** env var names (`OSM_CACHE_DIR`, `TRAIL_DEM_DATA_DIR`,
  `TRAIL_TILE_CACHE_DIR`, `HYRIVER_CACHE_NAME`, `TRAIL_V2_PATH_CACHE_DIR`), helper
  names (`osm_cache_dir`, `apply_osm_settings`), registry fields, and the
  `<name>.t.oursilverfamily.com` hostname pattern are used identically throughout.
