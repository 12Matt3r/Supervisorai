# lib/shadow.sh — Improvement #25: Side-by-Side Shadow Routing
# Run two candidate agents in parallel; one is shadow, used only for comparison.
set -euo pipefail
LIB_PREFIX="[shadow]"

SHADOW_DIR="${ORCH_SHADOW_DIR:-./.orchestr8r/shadow}"

shadow_init() { mkdir -p "$SHADOW_DIR"; }

shadow_run() {
  local primary_agent="$1" shadow_agent="$2" input="$3"
  local out_p="$SHADOW_DIR/primary.$$.json"
  local out_s="$SHADOW_DIR/shadow.$$.json"
  {
    llm_adapter_invoke "$primary_agent" "$input" 0 > "$out_p" &
    pid_p=$!
    llm_adapter_invoke "$shadow_agent" "$input" 0 > "$out_s" &
    pid_s=$!
    wait "$pid_p"
    wait "$pid_s"
  } 2>/dev/null
  # Promote the result of primary.
  cat "$out_p"
}

shadow_compare() {
  local primary="$1" shadow="$2"
  jq -s '{
    primary: .[0],
    shadow:   .[1],
    parity:   ((.[0].ok == .[1].ok) and (.[0].quality.score // 0) >= (.[1].quality.score // 0))
  }' "$primary" "$shadow"
}
