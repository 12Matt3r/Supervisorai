# lib/backpressure.sh — Improvement #15: Backpressure / Queue-Depth Awareness
# Caps in-flight tasks per kind; warns when queue depth grows.
set -euo pipefail
LIB_PREFIX="[backpressure]"

BP_FILE="${ORCH_BP_FILE:-./.orchestr8r/backpressure.json}"

backpressure_init() {
  mkdir -p "$(dirname "$BP_FILE")"
  echo '{"in_flight":{},"warned":false}' > "$BP_FILE"
}

# Returns 0 if can dispatch, 1 if at capacity.
backpressure_acquire_slot() {
  local kind="$1" max="${2:-4}"
  local d="$(dirname "$BP_FILE")"
  flock "$d" -c '
    f="'"$BP_FILE"'"
    cur=$(jq -r --arg k "'"$kind"'" ".in_flight[\$k] // 0" "$f")
    if [[ "$cur" -ge '"$max"' ]]; then exit 1; fi
    tmp=$(mktemp)
    jq --arg k "'"$kind"'" --argjson cur "$cur" \
       ".in_flight[\$k] = \$cur + 1" "$f" > "$tmp" && mv "$tmp" "$f"
  '
}

backpressure_release_slot() {
  local kind="$1"
  local d="$(dirname "$BP_FILE")"
  flock "$d" -c '
    f="'"$BP_FILE"'"
    tmp=$(mktemp)
    jq --arg k "'"$kind"'" ".in_flight[\$k] = (.in_flight[\$k] // 0) - 1
       | if .in_flight[\$k] < 0 then .in_flight[\$k] = 0 else . end" "$f" > "$tmp" && mv "$tmp" "$f"
  '
}

# Warn if queue depth exceeds threshold; emit to notification layer.
backpressure_check_warn() {
  local warn="${1:-8}"
  local depth
  depth=$(jq '[.in_flight[]] | add // 0' "$BP_FILE")
  if (( depth > warn )); then
    echo "[backpressure] queue depth ${depth} > warn=${warn}"
    return 1
  fi
}
