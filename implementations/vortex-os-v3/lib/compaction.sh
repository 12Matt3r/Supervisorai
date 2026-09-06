# lib/compaction.sh — Improvement #27: Mid-Run Context Compaction
# Replace raw context with summary pointers; reduce state.json bloat.
set -euo pipefail
LIB_PREFIX="[compaction]"

COMPACTION_EVERY_N="${ORCH_COMPACT_EVERY:-5}"

compaction_should_run() {
  local n="$1" every="${2:-$COMPACTION_EVERY_N}"
  (( n > 0 && n % every == 0 ))
}

# Build a summary from the current plan + state and replace raw context.
compaction_run() {
  local every="${1:-$COMPACTION_EVERY_N}" summary_file="$2"
  {
    echo "# Compaction @ $(date -Iseconds)"
    echo
    echo "## Plan Overview"
    jq -r '
      . as $p
      | ($p.tasks // [])
      | "Total tasks: \(length)",
        "Complete: \(map(select(.status=="complete"))|length)",
        "Failed:   \(map(select(.status=="failed"))|length)"
    ' "${ORCH_PLAN_FILE:-./plan.json}"
    echo
    echo "## Recent completions"
    jq -r '
      (.tasks // [])
      | map(select(.status=="complete"))
      | .[-5:]
      | .[]
      | "- \(.id): \(.description)"
    ' "${ORCH_PLAN_FILE:-./plan.json}"
  } > "$summary_file"
  echo "[compaction] wrote summary -> $summary_file"
}
