# lib/learning.sh — Improvement #13: Learning Router
# ε-greedy bandit over (agent, task_type) success_rate / cost / latency.
set -euo pipefail
LIB_PREFIX="[learning]"

LEARNING_FILE="${ORCH_LEARNING_FILE:-./.orchestr8r/learning.jsonl}"

learning_init() {
  mkdir -p "$(dirname "$LEARNING_FILE")"
  : > "$LEARNING_FILE"
}

# Record an outcome.
learning_record() {
  local agent="$1" task_type="$2" ok="$3" cost="${4:-0}" latency_s="${5:-0}"
  jq -nc --arg a "$agent" --arg t "$task_type" --argjson ok "$ok" \
    --argjson cost "$cost" --argjson lat "$latency_s" \
    '{agent:$a,task_type:$t,ok:$ok,cost:$cost,latency_s:$lat,ts:now|todate}' \
    >> "$LEARNING_FILE"
}

# Compute aggregate per (agent, task_type).
learning_stats() {
  local task_type="$1"
  jq -s --arg t "$task_type" '
    map(select(.task_type==$t))
    | group_by(.agent)
    | map({agent:.[0].agent, n:length,
            success_rate:(map(select(.ok))|length)/length,
            mean_cost:(map(.cost)|add)/length,
            mean_lat:(map(.latency_s)|add)/length})
  ' "$LEARNING_FILE"
}

# Pick the next agent: ε-greedy.
learning_pick() {
  local task_type="$1" epsilon="${2:-0.1}"
  local r=$((RANDOM % 1000))
  local is_explore
  is_explore=$(awk -v e="$epsilon" -v r="$r" 'BEGIN{print (r/1000 < e) ? 1 : 0}')
  if [[ $is_explore -eq 1 ]]; then
    # Explore: random agent from config.agents.
    jq -r '.agents | keys[]' "${ORCH_CONFIG:-./config.json}" 2>/dev/null | shuf -n1
  else
    learning_stats "$task_type" \
      | jq -r 'sort_by(-.success_rate)[0].agent // empty' \
      | head -1
  fi
}

# Quarantine agent whose recent success_rate dips below threshold.
learning_quarantine_check() {
  local threshold="${1:-0.6}"
  learning_stats "_all" 2>/dev/null | \
    jq -r --argjson t "$threshold" \
      '.[] | select(.n >= 5 and .success_rate < $t) | .agent'
}
