# lib/dlq.sh — Improvement #16: Dead-Letter Queue + Auto-Postmortem
# Failed tasks (after retry exhausted) land in dlq/<id>/.
set -euo pipefail
LIB_PREFIX="[dlq]"

DLQ_DIR="${ORCH_DLQ_DIR:-./.orchestr8r/dlq}"

dlq_init() { mkdir -p "$DLQ_DIR"; }

dlq_dump() {
  local task_id="$1" envelope_path="$2" reason="$3"
  local d="$DLQ_DIR/$task_id"
  mkdir -p "$d"
  cp "$envelope_path" "$d/last_envelope.json" 2>/dev/null || true
  jq -nc --arg t "$task_id" --arg r "$reason" \
    '{task:$t,reason:$r,at:now|todate}' > "$d/postmortem.seed.json"
  echo "[dlq] dumped task $task_id -> $d"
}

# Auto-draft a postmortem stub from a failed envelope.
dlq_postmortem_draft() {
  local task_id="$1"
  local d="$DLQ_DIR/$task_id"
  [[ -f "$d/last_envelope.json" ]] || { echo "no envelope" >&2; return 1; }
  {
    echo "# Postmortem: $task_id"
    echo
    echo "- Failed at: $(date -Iseconds)"
    echo "- Reason: $(jq -r '.reason' "$d/postmortem.seed.json")"
    echo "- Last status:"
    jq -r '"  - ok: \(.ok)\n  - duration_ms: \(.duration_ms)\n  - errors: \(.errors|join(\", \"))"' \
       "$d/last_envelope.json"
    echo
    echo "## Suggested next steps"
    echo "- Inspect retry_policy and lower max_attempts"
    echo "- Add a richer quality gate"
    echo "- If transient: schedule a daily retry"
  } > "$d/postmortem.md"
  echo "[dlq] drafted postmortem at $d/postmortem.md"
}

dlq_list() {
  [[ -d "$DLQ_DIR" ]] && find "$DLQ_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n"
}
