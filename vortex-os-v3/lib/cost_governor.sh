# lib/cost_governor.sh — Improvement #18: Cost Governor
# Plan-level $ cap with warn/abort percent triggers.
set -euo pipefail
LIB_PREFIX="[cost_governor]"

COSTGOVERNOR_FILE="${ORCH_COST_FILE:-./.orchestr8r/cost.json}"

cost_governor_init() {
  mkdir -p "$(dirname "$COSTGOVERNOR_FILE")"
  echo '{"spent_usd":0,"budget_usd":5.0,"warn_pct":80,"abort_pct":100}' > "$COSTGOVERNOR_FILE"
}

# add_usd is the cost of the latest task.
cost_governor_check() {
  local add_usd="$1"
  local budget="${2:-5.0}"
  local warn="${3:-80}"
  local abort="${4:-100}"
  tmp=$(mktemp)
  jq --argjson add "$add_usd" --argjson b "$budget" --argjson w "$warn" --argjson a "$abort" \
    '.spent_usd += $add | .budget_usd=$b | .warn_pct=$w | .abort_pct=$a
     | .pct_used = (.spent_usd*100/$b)
     | .status = (if .pct_used >= $a then "abort"
                   elif .pct_used >= $w then "warn"
                   else "ok" end)' "$COSTGOVERNOR_FILE" > "$tmp" \
    && mv "$tmp" "$COSTGOVERNOR_FILE"
  cost_governor_status="$(jq -r '.status' "$COSTGOVERNOR_FILE")"
  echo "$cost_governor_status"
}

cost_governor_degrade() {
  # At warn, prepend the "cheapest" adapter to all subsequent http agents.
  [[ "$cost_governor_status" == "warn" ]]
}

cost_governor_abort() { [[ "$cost_governor_status" == "abort" ]]; }
