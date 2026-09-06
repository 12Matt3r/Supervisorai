# lib/canary.sh — Improvement #28: Periodic Canary Agent ("canary-doctor")
# Hidden 31st agent. Daily micro-tasks assert the supervisor is healthy.
# Also emits three-channel metrics (counters/gauges -> metrics.jsonl,
# audit -> audit.jsonl, OTel-GenAI trace).
set -euo pipefail
LIB_PREFIX="[canary]"

# Source observability helpers if they aren't already loaded so we can
# audit_log / obs_record / obs_trace_for_task gracefully.  We only
# attempt the chain when CONFIG is set; canary is sometimes invoked
# before the orchestrator state is initialised.
if [[ -n "${CONFIG:-}" ]] && [[ -f "$SKILL_SCRIPT_DIR/lib/observability.sh" ]]; then
  source "$SKILL_SCRIPT_DIR/lib/observability.sh" 2>/dev/null || true
fi
# Provide a minimal now_iso() if not already in scope.
if ! declare -F now_iso >/dev/null; then
  now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
fi

CANARY_FILE="${ORCH_CANARY:-./.orchestr8r/canary.json}"

canary_init() {
  mkdir -p "$(dirname "$CANARY_FILE")"
  cat > "$CANARY_FILE" <<'JSON'
{
  "version":"2.0",
  "last_run":null,
  "results":[],
  "drift_detected":false
}
JSON
}

canary_assert() {
  local name="$1" cmd="$2"
  local tmp; tmp=$(mktemp)
  local ok=false
  if eval "$cmd" >/dev/null 2>&1; then ok=true; fi
  jq --arg n "$name" --argjson ok "$ok" '.results += [{name:$n, ok:$ok, at:now|todate}]' \
     "$CANARY_FILE" > "$tmp" && mv "$tmp" "$CANARY_FILE"
}

# Run the canned health checks.
canary_run_all() {
  canary_init
  canary_assert "config_valid"        "jq -e . '.paths' >/dev/null < \"$ORCH_CONFIG\""
  canary_assert "plan_valid"          "jq -e . '.tasks' >/dev/null < \"${ORCH_PLAN_FILE:-./plan.json}\""
  canary_assert "state_present"       "test -f \"${ORCH_STATE_FILE:-./state.json}\""
  canary_assert "audit_appending"     "test -w \"${ORCH_AUDIT_FILE:-./audit.jsonl}\" || test -f \"${ORCH_AUDIT_FILE:-./audit.jsonl}\""
  canary_assert "bus_socket_alive"    "test -S \"${ORCH_BUS_SOCKET:-./.orchestr8r/bus.sock}\""
  jq '.last_run = now|todate' "$CANARY_FILE" > "${CANARY_FILE}.tmp" && mv "${CANARY_FILE}.tmp" "$CANARY_FILE"
  local fails; fails=$(jq '[.results[]|select(.ok==false)]|length' "$CANARY_FILE")
  if (( fails > 0 )); then
    jq '.drift_detected = true' "$CANARY_FILE" > "${CANARY_FILE}.tmp" && mv "${CANARY_FILE}.tmp" "$CANARY_FILE"
  fi
  canary_emit_metrics
}

canary_drift?() { jq -e '.drift_detected' "$CANARY_FILE" >/dev/null; }

# ---------------------------------------------------------------------------
# Metrics emission (Improvement #28)
# Three channels, in order:
#   1. metrics.jsonl   — counter / gauge time series (scrapable by Prom/OTel)
#   2. audit.jsonl     — structured event log via audit_log if available
#   3. obs_trace span  — single OTel-GenAI-shaped trace for the canary run
# Every channel degrades gracefully if its target is unavailable.
# ---------------------------------------------------------------------------

# Path to the metrics time series. Defaults to .orchestr8r/ next to the project.
CANARY_METRICS_FILE="${ORCH_METRICS_FILE:-./.orchestr8r/canary.metrics.jsonl}"

# canary_emit_metrics — summarise canary.json into time-series metrics.
# Safe to call repeatedly; produces one NDJSON line per emission.
canary_emit_metrics() {
  [[ -f "$CANARY_FILE" ]] || { echo "${LIB_PREFIX} no canary.json yet; skipping" >&2; return 0; }
  mkdir -p "$(dirname "$CANARY_METRICS_FILE")"

  # Counters and gauges we care about.
  local total passes fails drift first_seen last_seen
  total=$(jq '.results|length'   "$CANARY_FILE")
  passes=$(jq '[.results[]|select(.ok==true)]|length'  "$CANARY_FILE")
  fails=$(jq '[.results[]|select(.ok==false)]|length' "$CANARY_FILE")
  drift=$(jq '.drift_detected'   "$CANARY_FILE")
  first_seen=$(jq -r '.results[0].at // empty' "$CANARY_FILE")
  last_seen=$(jq -r '.results[-1].at // empty' "$CANARY_FILE")

  # Pass ratio (0.0–1.0).  If no assertions ran, default to 1.0.
  local ratio
  if (( total == 0 )); then ratio="1.0"; else
    ratio=$(awk -v p="$passes" -v t="$total" 'BEGIN{printf "%.4f", p/t}')
  fi

  # Time-series emission (NDJSON, one line per metric).
  cat >> "$CANARY_METRICS_FILE" <<JSON
$(now_iso) canary.assessments      counter 1
$(now_iso) canary.assertions_total counter $total
$(now_iso) canary.passes_total     counter $passes
$(now_iso) canary.fails_total      counter $fails
$(now_iso) canary.pass_ratio       gauge   $ratio
$(now_iso) canary.drift_detected   gauge   $drift
$(now_iso) canary.first_seen       info    "$first_seen"
$(now_iso) canary.last_seen        info    "$last_seen"
JSON

  # Mirror into the audit log if obs_record is available.
  if declare -F obs_record >/dev/null; then
    obs_record "canary.run" \
      "total=$total" "passes=$passes" "fails=$fails" \
      "ratio=$ratio" "drift=$drift"
  elif declare -F audit_log >/dev/null; then
    audit_log "canary.run" \
      "total=$total" "passes=$passes" "fails=$fails" \
      "ratio=$ratio" "drift=$drift"
  fi

  # Emit an OTel-GenAI-shaped trace stub if the obs_tracing helper exists.
  if declare -F obs_trace_for_task >/dev/null; then
    obs_trace_for_task "canary-doctor" || true
  fi
}

# canary_metrics_show — pretty-print the time series.
canary_metrics_show() {
  [[ -f "$CANARY_METRICS_FILE" ]] || { echo "${LIB_PREFIX} no metrics at $CANARY_METRICS_FILE yet"; return 0; }
  awk '
    { ts=$1; $1=""; $2=""; printf("%-25s %-18s %-10s %s\n", ts, $2, $3, $4) }
    { sub(/^[ \t]+/, "") }
  ' "$CANARY_METRICS_FILE" | tail -n 50
}

# canary_metrics_summary — single-grep dump for dashboards.
canary_metrics_summary() {
  [[ -f "$CANARY_METRICS_FILE" ]] || { echo "0"; return 0; }
  local latest_ratio latest_drift
  latest_ratio=$(grep "canary.pass_ratio"     "$CANARY_METRICS_FILE" | tail -1 | awk '{print $NF}')
  latest_drift=$(grep "canary.drift_detected" "$CANARY_METRICS_FILE" | tail -1 | awk '{print $NF}')
  echo "pass_ratio=${latest_ratio:-1.0} drift=${latest_drift:-false}"
}
