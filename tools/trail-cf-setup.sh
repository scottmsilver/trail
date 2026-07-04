#!/usr/bin/env bash
# One-time Cloudflare setup: create the `trail` tunnel, a wildcard DNS record,
# and a single Access application gating *.t.oursilverfamily.com to one email.
#
# Prerequisite (manual, once): enable Zero Trust and mint an API token, then put
# it in ~/development/trail-shared/trail-cf.env (chmod 600) with:
#   export CLOUDFLARE_API_TOKEN="..."   # scope: Account -> Access: Apps and Policies: Edit
#   export CF_ACCOUNT_ID="2ff88e1aab448260d945962395baf6f1"
#   export TRAIL_ACCESS_EMAIL="scottmsilver@gmail.com"
# The wildcard DNS below uses cloudflared's cert.pem (not the API token), so no
# Zone/DNS scope and no zone id are required.
set -euo pipefail

# shellcheck source=/dev/null
source ~/development/trail-shared/trail-cf.env
: "${CLOUDFLARE_API_TOKEN:?}" "${CF_ACCOUNT_ID:?}" "${TRAIL_ACCESS_EMAIL:?}"

PY=/home/ssilver/development/trail/trail_env/bin/python
WILDCARD="*.t.oursilverfamily.com"
API="https://api.cloudflare.com/client/v4"
auth=(-H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json")

# 1) Tunnel (idempotent: reuse if it exists)
if ! cloudflared tunnel list 2>/dev/null | grep -qw trail; then
  cloudflared tunnel create trail
fi

# 2) Wildcard DNS -> tunnel
cloudflared tunnel route dns --overwrite-dns trail "$WILDCARD"

# 3) Access application on the wildcard, with an allow-by-email policy.
# Idempotent: reuse an existing app for this domain instead of duplicating it.
app_id=$(curl -s "${auth[@]}" "$API/accounts/$CF_ACCOUNT_ID/access/apps" \
  | "$PY" -c "import sys,json
apps=json.load(sys.stdin).get('result') or []
print(next((a['id'] for a in apps if a.get('domain')=='$WILDCARD'), ''))")
if [[ -z "$app_id" ]]; then
  app_payload=$(cat <<JSON
{"name":"trail-dev","domain":"$WILDCARD","type":"self_hosted",
 "session_duration":"720h","app_launcher_visible":false}
JSON
)
  app_id=$(curl -s "${auth[@]}" -X POST \
    "$API/accounts/$CF_ACCOUNT_ID/access/apps" --data "$app_payload" \
    | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["result"]["id"])')
  echo "Access app created: $app_id"
else
  echo "Access app already exists for $WILDCARD: $app_id (reusing)"
fi

# Skip policy creation if this app already has an allow policy.
has_policy=$(curl -s "${auth[@]}" \
  "$API/accounts/$CF_ACCOUNT_ID/access/apps/$app_id/policies" \
  | "$PY" -c "import sys,json; print(len(json.load(sys.stdin).get('result') or []))")
if [[ "$has_policy" != "0" ]]; then
  echo "Access policy already present ($has_policy) — skipping."
  echo "done."
  exit 0
fi

policy_payload=$(cat <<JSON
{"name":"only-scott","decision":"allow",
 "include":[{"email":{"email":"$TRAIL_ACCESS_EMAIL"}}]}
JSON
)
curl -s "${auth[@]}" -X POST \
  "$API/accounts/$CF_ACCOUNT_ID/access/apps/$app_id/policies" \
  --data "$policy_payload" >/dev/null
echo "Access policy attached for $TRAIL_ACCESS_EMAIL"

# NOTE: Cloudflare's Access API shape has evolved (inline `policies` vs the
# separate /policies endpoint, and account-level reusable policies). If the
# account rejects the payloads above, confirm the current schema at
# https://developers.cloudflare.com/api/ (Access Applications / Access Policies)
# and adjust the two JSON bodies.
echo "done. Tunnel 'trail' + wildcard DNS + Access app configured."
