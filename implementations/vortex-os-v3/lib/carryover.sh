# lib/carryover.sh — Improvement #17: Cross-Plan Memory Carryover
# Sub-plan inherits parent's episodic+semantic memory namespace.
set -euo pipefail
LIB_PREFIX="[carryover]"

carryover_inherit() {
  local parent_id="$1" child_ns="$2"
  memory_v2_inherit_from_parent "$parent_id" "$child_ns"
  echo "[carryover] inherited memory from $parent_id into $child_ns"
}

carryover_set_inherit() {
  local task_id="$1" inherit="$2"
  local f="${ORCH_STATE_FILE:-./state.json}"
  tmp=$(mktemp)
  jq --arg t "$task_id" --argjson i "$inherit" \
    '(.carryover //= {})[$t] = $i' "$f" > "$tmp" && mv "$tmp" "$f"
}
