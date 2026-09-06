# lib/graceful_degradation.sh — Bonus C: Graceful Degradation Matrix
# Per-agent fallback chain (haiku -> sonnet -> opus -> escalate_human).
# The supervisor never *fails*; it *degrades*.
set -euo pipefail
LIB_PREFIX="[degrade]"
[[ -n "${__LIB_DEGRADE_LOADED:-}" ]] && return 0
__LIB_DEGRADE_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"      2>/dev/null || true
source "$SKILL_SCRIPT_DIR/lib/learning.sh"  2>/dev/null || true

# Default fallback ladder, ordered cheapest -> most-expensive.
DEGRADE_DEFAULT_CHAIN=("haiku" "sonnet" "opus" "escalate_human")

# degrade_chain_for <agent_kind> — return JSON array of tiers.
degrade_chain_for() {
  local kind="$1"
  local cfg="${ORCH_DEGRADE_FILE:-./.orchestr8r/degradation.json}"
  if [[ -f "$cfg" ]] && jq -e --arg k "$kind" '.chains[$k]' "$cfg" >/dev/null; then
    jq -c --arg k "$kind" '.chains[$k]' "$cfg"
    return 0
  fi
  printf '%s\n' "${DEGRADE_DEFAULT_CHAIN[@]}" | jq -R 'split("\n") | map(select(length>0))'
}

# degrade_next_tier <agent_kind> <last_tier> — returns the next tier or "stop".
degrade_next_tier() {
  local kind="$1" last="${2:-}"
  local chain; chain="$(degrade_chain_for "$kind")"
  if [[ -z "$last" ]]; then
    jq -r '.[0]' <<<"$chain"
    return 0
  fi
  local n; n=$(jq -r --arg l "$last" 'index($l) // -1' <<<"$chain")
  local next=$(( n + 1 ))
  local size; size=$(jq 'length' <<<"$chain")
  if (( next >= size )); then
    echo "stop"; return 0
  fi
  jq -r --argjson i "$next" '.[$i]' <<<"$chain"
}

# degrade_should_fallback <task_id> <fail_count> — returns 0 if we should escalate.
degrade_should_fallback() {
  local task_id="$1" fails="${2:-1}"
  local cap="${ORCH_DEGRADE_CAP:-3}"
  if (( fails >= cap )); then return 0; fi
  return 1
}

# degrade_record_event <task_id> <from_tier> <to_tier> <reason>
degrade_record_event() {
  local tid="$1" from="$2" to="$3" reason="$4"
  local af="${ORCH_DEGRADE_LOG:-./.orchestr8r/degradation.jsonl}"
  mkdir -p "$(dirname "$af")"
  jq -c -n \
    --arg ts "$(now_iso)" \
    --arg t  "$tid" \
    --arg fr "$from" \
    --arg to "$to" \
    --arg r  "$reason" \
    '{ts:$ts,task:$t,from:$fr,to:$to,reason:$r}' >> "$af"
}
