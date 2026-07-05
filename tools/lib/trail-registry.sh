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

# True when a Cloudflare API token is configured.
cf_has_token() { _cf_env; [[ -n "${CLOUDFLARE_API_TOKEN:-}" ]]; }

# curl to the Cloudflare API with the bearer token supplied via a mode-600
# config file, so the token never appears in process argv (ps / /proc/*/cmdline).
# printf is a bash builtin, so writing the file doesn't expose the token either.
_cf_curl() {
  _cf_env
  local kf; kf=$(mktemp "${TMPDIR:-/tmp}/trailcf.XXXXXX"); chmod 600 "$kf"
  printf 'header = "Authorization: Bearer %s"\n' "$CLOUDFLARE_API_TOKEN" > "$kf"
  curl -s -K "$kf" "$@"; local rc=$?
  rm -f "$kf"; return $rc
}

# Create the single-level DNS CNAME for a host (uses cloudflared cert.pem, not
# the API token). Idempotent via --overwrite-dns.
cf_route_dns() { # $1=host
  cloudflared tunnel route dns --overwrite-dns trail "$1" >/dev/null 2>&1 \
    && echo "dns: routed $1" || echo "dns: could not route $1 (check cert.pem)"
}

# Ensure a self-hosted Access app WITH an allow-by-email policy exists for a host.
# Idempotent (reuses an existing app for that exact domain, and adds the policy if
# missing). Echoes the app id ONLY when an allow policy is confirmed present;
# echoes empty on no-token / creation failure / policy failure so callers fail
# CLOSED (never expose an ungated host). All Cloudflare-bound values are passed to
# Python via argv/env and JSON is built with json.dumps — never string-interpolated
# into source or payloads.
cf_access_ensure() { # $1=host
  cf_has_token || { echo ""; return 0; }
  local host="$1" aid npol ok
  aid=$(_cf_curl "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps" \
    | "$PY" -c 'import sys,json
apps=json.load(sys.stdin).get("result") or []
print(next((a["id"] for a in apps if a.get("domain")==sys.argv[1]),""))' "$host")
  if [[ -z "$aid" ]]; then
    aid=$(_cf_curl -H "Content-Type: application/json" -X POST \
      "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps" \
      --data "$("$PY" -c 'import json,sys;print(json.dumps({"name":"trail-"+sys.argv[1],"domain":sys.argv[1],"type":"self_hosted","session_duration":"720h","app_launcher_visible":False}))' "$host")" \
      | "$PY" -c 'import sys,json; print((json.load(sys.stdin).get("result") or {}).get("id",""))')
    [[ -n "$aid" ]] || { echo ""; return 0; }   # app creation failed -> fail closed
  fi
  # Confirm an allow policy exists; create one if the app has none.
  npol=$(_cf_curl "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps/$aid/policies" \
    | "$PY" -c 'import sys,json; print(len(json.load(sys.stdin).get("result") or []))')
  if [[ "$npol" == "0" ]]; then
    ok=$(_cf_curl -H "Content-Type: application/json" -X POST \
      "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps/$aid/policies" \
      --data "$("$PY" -c 'import json,os;print(json.dumps({"name":"only-owner","decision":"allow","include":[{"email":{"email":os.environ["TRAIL_ACCESS_EMAIL"]}}]}))')" \
      | "$PY" -c 'import sys,json; print("ok" if json.load(sys.stdin).get("success") else "")')
    [[ "$ok" == "ok" ]] || { echo ""; return 0; }  # policy creation failed -> fail closed
  fi
  echo "$aid"
}

# Delete an Access app by id (best-effort).
cf_access_delete() { # $1=app_id
  cf_has_token || return 0
  [[ -n "${1:-}" ]] || return 0
  _cf_curl -X DELETE "$CF_API/accounts/$CF_ACCOUNT_ID/access/apps/$1" >/dev/null
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
