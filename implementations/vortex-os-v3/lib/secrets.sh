# lib/secrets.sh — Improvement #11: Per-Secret Approval Matrix
# config.json -> secret_approvers ; agents declare needs_secret_approval[].
set -euo pipefail
LIB_PREFIX="[secrets]"

SECRETS_ALLOWLIST="${ORCH_SECRETS_ALLOWLIST:-./secrets.allowlist.json}"

secrets_allowlist_init() {
  mkdir -p "$(dirname "$SECRETS_ALLOWLIST")"
  [[ -f $SECRETS_ALLOWLIST ]] || cat > "$SECRETS_ALLOWLIST" <<'JSON'
{ "version":"2.0","items":[] }
JSON
}

# Approve: id, agents (glob), approver_email, rotation_days
secrets_approve() {
  local id="$1" agents_globs="$2" approver="$3" rotation="${4:-30}"
  tmp=$(mktemp)
  jq --arg id "$id" --arg agents "$agents_globs" --arg app "$approver" --argjson rot "$rotation" \
    '.items += [{id:$id,agents:$agents,approver:$app,rotation_days:$rot}]' \
    "$SECRETS_ALLOWLIST" > "$tmp" && mv "$tmp" "$SECRETS_ALLOWLIST"
}

# Approve-or-deny for a (agent, secret_id) pair.
secrets_authorize() {
  local agent="$1" secret_id="$2"
  jq -r --arg a "$agent" --arg id "$secret_id" '
    .items[] | select(.id == $id)
    | (.agents | split(",")) as $g
    | if ($g | map(. as $p | ($a | test($p))) | any) then "APPROVED" else empty end
  ' "$SECRETS_ALLOWLIST"
  [[ -z $(secrets_authorize "$agent" "$secret_id" 2>/dev/null) ]] && echo "DENIED"
}

# Strip any reference of a secret_id from a stream of text.
secrets_redact() {
  local secret_id="$1"
  sed -E "s/${secret_id}=\"[^\"]*\"/${secret_id}=\"***REDACTED***/g; s/${secret_id}:[ \t]*[A-Za-z0-9_\-]+/${secret_id}: ***REDACTED***/g"
}
