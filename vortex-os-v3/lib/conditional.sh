# lib/conditional.sh — Improvement #22: Conditional Routing
# Plan tasks may carry a "condition" object; supervisor evaluates it.
set -euo pipefail
LIB_PREFIX="[conditional]"

# Evaluate a JSON condition against the current state.
conditional_eval() {
  local cond_json="$1"
  jq -e "$cond_json" "${ORCH_STATE_FILE:-./state.json}" >/dev/null
}

# For a task, return 0 if dispatched-once rule matches, else 1.
conditional_should_dispatch() {
  local task_id="$1"
  local f="${ORCH_STATE_FILE:-./state.json}"
  local cond
  cond=$(jq -r --arg t "$task_id" '
    (.tasks // []) | map(select(.id==$t))[0].condition // "true"
  ' "${ORCH_PLAN_FILE:-./plan.json}")
  conditional_eval "$cond"
}
