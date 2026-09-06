# lib/skills_matrix.sh — Improvement #14: Soft-Skills Matrix Routing
# 3-axis capability vector: capability/confidence/cost_per_call
set -euo pipefail
LIB_PREFIX="[skills_matrix]"

SKILLS_FILE="${ORCH_SKILLS_FILE:-./.orchestr8r/skills.json}"

skills_init() {
  mkdir -p "$(dirname "$SKILLS_FILE")"
  [[ -f $SKILLS_FILE ]] || echo '{"agents":{},"history":[]}' > "$SKILLS_FILE"
}

skills_set() {
  local agent="$1" capability="$2" confidence="$3" cost="$4"
  tmp=$(mktemp)
  jq --arg a "$agent" --argjson c "$capability" --argjson k "$confidence" --argjson u "$cost" \
    '.agents[$a] = {capability:$c,confidence:$k,cost_per_call:$u}' "$SKILLS_FILE" > "$tmp" \
    && mv "$tmp" "$SKILLS_FILE"
}

# Pick the Pareto-optimal agent for a task with required capability_score.
skills_pick() {
  local task_capability_min="$1" max_cost="${2:-0.1}"
  jq -r --argjson m "$task_capability_min" --argjson mc "$max_cost" '
    .agents | to_entries
    | map(select(.value.capability >= $m and .value.cost_per_call <= $mc))
    | sort_by(-.value.confidence)
    | .[0].key // empty
  ' "$SKILLS_FILE"
}
