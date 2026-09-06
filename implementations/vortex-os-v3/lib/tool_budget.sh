# lib/tool_budget.sh — Improvement #26: Tool-Call Budget per Task
# Wrap any tool-calling agent; abort when budget is exceeded.
set -euo pipefail
LIB_PREFIX="[tool_budget]"

# Initialise a per-task budget counter file.
tool_budget_init() {
  local task_id="$1" budget="${2:-20}"
  local f="${ORCH_BUDGETS_DIR:-./.orchestr8r/budgets}/${task_id}"
  mkdir -p "$(dirname "$f")"
  echo "$budget" > "${f}.budget"
  echo 0 > "${f}.spent"
}

tool_budget_check_and_consume() {
  local task_id="$1" n="${2:-1}"
  local bdir="${ORCH_BUDGETS_DIR:-./.orchestr8r/budgets}"
  local fb="$bdir/${task_id}.budget" fs="$bdir/${task_id}.spent"
  [[ -f $fb && -f $fs ]] || return 1
  local budget; budget=$(cat "$fb")
  local spent;  spent=$(cat "$fs")
  if (( spent + n > budget )); then
    jq -nc --argjson b "$budget" --argjson s "$spent" \
      '{ok:false, abort_reason:"tool_budget_exceeded", budget:$b, spent:$s}' >&2
    return 2
  fi
  echo $((spent + n)) > "$fs"
  jq -nc --argjson b "$budget" --argjson s "$((spent+n))" \
    '{ok:true, budget:$b, spent:$s}'
}

tool_budget_diagnose() {
  local task_id="$1"
  local bdir="${ORCH_BUDGETS_DIR:-./.orchestr8r/budgets}"
  local fb="$bdir/${task_id}.budget" fs="$bdir/${task_id}.spent"
  jq -nc --argjson b "$(cat "$fb")" --argjson s "$(cat "$fs")" \
    '{budget:$b, spent:$s, history:.tool_call_history // []}'
}
