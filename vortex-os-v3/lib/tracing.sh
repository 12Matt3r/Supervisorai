# lib/tracing.sh — Improvement #10: OpenTelemetry GenAI Span Emission
# Every agent invocation emits OTLP-ready GenAI spans.
set -euo pipefail
LIB_PREFIX="[tracing]"

TRACES_DIR="${ORCH_TRACES_DIR:-./.orchestr8r/traces}"
TRACES_EXPORT="${ORCH_TRACES_EXPORT:-console}"

tracing_init() {
  mkdir -p "$TRACES_DIR"
}

# Generate a 128-bit trace/span id (hex).
tracing_new_id() {
  od -An -tx1 -N16 /dev/urandom | tr -d ' \n' | head -c 32
}

tracing_span_start() {
  local trace_id="$1" name="$2" parent_id="${3:-}"
  local span_id; span_id=$(tracing_new_id)
  local f="$TRACES_DIR/${trace_id}.${span_id}.span.json"
  jq -nc --arg n "$name" --arg t "$trace_id" --arg s "$span_id" --arg p "$parent_id" \
    '{name:$n,trace_id:$t,span_id:$s,parent_id:$p,start_ns:(now|todate)}' > "$f"
  printf '%s' "$span_id"
}

tracing_span_end() {
  local trace_id="$1" span_id="$2" attrs="${3:-{}}"
  local f="$TRACES_DIR/${trace_id}.${span_id}.span.json"
  [[ -f $f ]] || return 1
  tmp=$(mktemp)
  jq --argjson a "$attrs" '. + {end_ns:(now|todate),attributes:$a}' "$f" > "$tmp" && mv "$tmp" "$f"
}

tracing_export_console() {
  local out="$TRACES_DIR/export.ndjson"
  for f in "$TRACES_DIR"/*.span.json; do
    [[ -f $f ]] && cat "$f" >> "$out"
  done
  echo "[tracing] exported to $out"
}

tracing_export_otlp() {
  local url="$1"
  curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @"$TRACES_DIR/export.ndjson" "$url" || true
  echo "[tracing] posted OTLP to $url"
}

tracing_export() {
  case "$TRACES_EXPORT" in
    console) tracing_export_console ;;
    otlp)    tracing_export_otlp "${ORCH_OTLP_ENDPOINT:-http://localhost:4318/v1/traces}" ;;
    *)       tracing_export_console ;;
  esac
}
