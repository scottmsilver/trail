#!/usr/bin/env bash
# One-time Cloudflare setup: create the `trail` tunnel.
#
# Per-instance DNS records and Access apps are created by `trail-instance up`
# (single-level trail-<name>.oursilverfamily.com hosts, covered by free
# Universal SSL). So this one-time step only needs the tunnel to exist.
#
# Prerequisite (manual, once): enable Zero Trust and mint an API token, then put
# it in ~/development/trail-shared/trail-cf.env (chmod 600) with:
#   export CLOUDFLARE_API_TOKEN="..."   # scope: Account -> Access: Apps and Policies: Edit
#   export CF_ACCOUNT_ID="2ff88e1aab448260d945962395baf6f1"
#   export TRAIL_ACCESS_EMAIL="scottmsilver@gmail.com"
# DNS is created by cloudflared's cert.pem (not the token), so no Zone/DNS scope.
set -euo pipefail

# shellcheck source=/dev/null
source ~/development/trail-shared/trail-cf.env
: "${CLOUDFLARE_API_TOKEN:?}" "${CF_ACCOUNT_ID:?}" "${TRAIL_ACCESS_EMAIL:?}"

# Tunnel (idempotent: reuse if it exists).
if cloudflared tunnel list 2>/dev/null | grep -qw trail; then
  echo "tunnel 'trail' already exists — reusing"
else
  cloudflared tunnel create trail
fi

# Verify the token is valid and can manage Access (fail fast otherwise).
verify=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/access/apps" \
  | grep -o '"success":[a-z]*' | head -1)
echo "access-api reachable: ${verify:-unknown}"
echo "done. 'trail-instance up <name>' now handles per-instance DNS + Access."
