# lib/streaming.sh — Improvement #3: Streaming Chunk Emission
# NDJSON deltas so the dashboard can render partial progress.
set -euo pipefail
LIB_PREFIX="[streaming]"

stream_open() {
  local task_id="$1" out_dir="${2:-./.orchestr8r/streams}"
  mkdir -p "$out_dir"
  local f="$out_dir/${task_id}.chunks.ndjson"
  : > "$f"
  printf '%s' "$f"
}

stream_emit() {
  local chunks_file="$1" kind="$2" payload="$3"
  jq -nc --arg k "$kind" --arg p "$payload" \
    '{kind:$k,payload:$p,ts:now|todate}' >> "$chunks_file"
}

stream_close() {
  local chunks_file="$1" final_summary="$2"
  stream_emit "$chunks_file" "final" "$final_summary"
}

stream_subscribe() {
  # Tail the NDJSON stream, emit each line to stdout (for SSE/dashboard).
  local chunks_file="$1" since_line="${2:-0}"
  tail -n "+$((since_line+1))" -F "$chunks_file" 2>/dev/null
}
