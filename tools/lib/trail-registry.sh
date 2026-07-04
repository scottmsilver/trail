#!/usr/bin/env bash
# Shared helpers for the trail-instance launcher: registry, deterministic port
# derivation, and Cloudflare tunnel ingress regeneration.
set -euo pipefail

REG="$HOME/.trail-instances.json"
SHARED="${TRAIL_SHARED_DIR:-$HOME/development/trail-shared}"
# shellcheck disable=SC2034  # MAIN is consumed by scripts that source this lib
MAIN="${TRAIL_MAIN_CHECKOUT:-$HOME/development/trail}"
TUNNEL_CFG="$HOME/.cloudflared/config-trail.yml"
PY=/home/ssilver/development/trail/trail_env/bin/python

[[ -f "$REG" ]] || echo '{}' > "$REG"

# Deterministic port pair from the instance name.
ports_for() { # $1=name -> prints "BACKEND FRONTEND"
  local h off
  h=$(printf '%s' "$1" | cksum | cut -d' ' -f1)
  off=$(( h % 80 ))            # 0..79
  echo "$((9100 + off)) $((9200 + off))"
}

reg_set() { # $1=name $2=json-object
  "$PY" - "$REG" "$1" "$2" <<'PY'
import json, sys
reg_path, name, obj = sys.argv[1], sys.argv[2], sys.argv[3]
reg = json.load(open(reg_path))
reg[name] = json.loads(obj)
json.dump(reg, open(reg_path, "w"), indent=2)
PY
}

reg_del() { # $1=name
  "$PY" - "$REG" "$1" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
reg.pop(sys.argv[2], None)
json.dump(reg, open(sys.argv[1], "w"), indent=2)
PY
}

reg_get() { # $1=name -> prints JSON object ({} if absent)
  "$PY" - "$REG" "$1" <<'PY'
import json, sys
print(json.dumps(json.load(open(sys.argv[1])).get(sys.argv[2], {})))
PY
}

# UUID of the `trail` tunnel, or empty if it does not exist yet.
trail_tunnel_uuid() {
  cloudflared tunnel list -o json 2>/dev/null | "$PY" -c '
import json, sys
try:
    for t in json.load(sys.stdin):
        if t.get("name") == "trail":
            print(t["id"]); break
except Exception:
    pass
'
}

CF_API="https://api.cloudflare.com/client/v4"
_cf_env() {
  local f="$HOME/development/trail-shared/trail-cf.env"
  # shellcheck source=/dev/null
  [[ -f "$f" ]] && source "$f"
}

# Create the single-level DNS CNAME for a host (uses cloudflared cert.pem, not
# the API token). Idempotent via --overwrite-dns.
cf_route_dns() { # $1=host
  cloudflared tunnel route dns --overwrite-dns trail "$1" >/dev/null 2>&1 \
    && echo "dns: routed $1" || echo "dns: could not route $1 (check cert.pem)"
}

# Ensure a self-hosted Access app + allow-by-email policy exists for a host.
# Idempotent (reuses an existing app for that exact domain). Echoes the app id,
# or empty when no token is configured (instance then runs UNGATED).
cf_access_ensure() { # $1=host
  _cf_env
  [[ -n "${CLOUDFLARE_API_TOKEN:-}" ]] || { echo ""; return 0; }
  local host="$1" aid
  aid=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
    "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps" \
    | "$PY" -c "import sys,json
apps=json.load(sys.stdin).get('result') or []
print(next((a['id'] for a in apps if a.get('domain')=='$host'),''))")
  if [[ -z "$aid" ]]; then
    aid=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" \
      -X POST "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps" \
      --data "{\"name\":\"trail-$host\",\"domain\":\"$host\",\"type\":\"self_hosted\",\"session_duration\":\"720h\",\"app_launcher_visible\":false}" \
      | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["result"]["id"])')
    curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" \
      -X POST "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps/$aid/policies" \
      --data "{\"name\":\"only-scott\",\"decision\":\"allow\",\"include\":[{\"email\":{\"email\":\"$TRAIL_ACCESS_EMAIL\"}}]}" >/dev/null
  fi
  echo "$aid"
}

# Delete an Access app by id (best-effort).
cf_access_delete() { # $1=app_id
  _cf_env
  [[ -n "${CLOUDFLARE_API_TOKEN:-}" && -n "${1:-}" ]] || return 0
  curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
    -X DELETE "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps/$1" >/dev/null
}

# Regenerate the tunnel ingress from the registry and (best-effort) reload.
# Safe to call before Cloudflare is set up: it writes the config and skips the
# reload when no `trail` tunnel/credentials exist yet.
regen_ingress() {
  local uuid cred
  uuid="$(trail_tunnel_uuid)"
  cred="$HOME/.cloudflared/${uuid}.json"
  mkdir -p "$SHARED/logs"

  "$PY" - "$REG" "$TUNNEL_CFG" "$uuid" "$cred" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
cfg, uuid, cred = sys.argv[2], sys.argv[3], sys.argv[4]
lines = []
if uuid:
    lines += [f"tunnel: {uuid}", f"credentials-file: {cred}"]
lines.append("ingress:")
for name, info in reg.items():
    lines.append(f"  - hostname: {info['hostname']}")
    lines.append(f"    service: http://localhost:{info['frontend_port']}")
lines.append("  - service: http_status:404")
open(cfg, "w").write("\n".join(lines) + "\n")
PY

  if [[ -z "$uuid" || ! -f "$cred" ]]; then
    echo "note: no 'trail' tunnel/credentials yet — wrote $TUNNEL_CFG but skipped reload."
    echo "      run tools/trail-cf-setup.sh once to create the tunnel + Access app."
    return 0
  fi
  # Reload: restart the trail tunnel process. Track it by pidfile rather than
  # `pkill -f` (a broad pattern can match unrelated processes — even the caller).
  local pidfile="$SHARED/trail-tunnel.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    kill "$(cat "$pidfile")" 2>/dev/null || true
    sleep 1
  fi
  nohup cloudflared tunnel --config "$TUNNEL_CFG" run trail \
    >> "$SHARED/logs/tunnel.log" 2>&1 &
  echo "$!" > "$pidfile"
  echo "reloaded trail tunnel (pid $!)"
}
