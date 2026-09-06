# lib/preflight.sh — Improvement #23: Predicted-Cost Pre-Flight
# Ask an LLM-driven "Haiku"-tier estimator to predict cost before invoking.
set -euo pipefail
LIB_PREFIX="[preflight]"

PREFLIGHT_THRESHOLD="${ORCH_PREFLIGHT_USD:-0.50}"

# Estimate: tokens_in/3 + tokens_out + llm cost. Body of envelope.in.input.
preflight_estimate() {
  local agent="$1" input_json="$2"
  # Naive estimator (override per project with a real model call).
  local bytes; bytes=$(printf '%s' "$input_json" | wc -c)
  local est_tokens_out=512
  awk -v b="$bytes" -v to="$est_tokens_out" 'BEGIN {
    ti = b/4; printf "%d", int((ti*0.000003) + (to*0.000015))
  }'
}

preflight_authorize() {
  local predicted="$1" threshold="${2:-$PREFLIGHT_THRESHOLD}"
  awk -v p "$predicted" -v t "$threshold" 'BEGIN { exit (p <= t ? 0 : 1) }'
}
