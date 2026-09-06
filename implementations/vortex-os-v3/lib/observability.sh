# lib/observability.sh — audit log, watch, cost tracking, OTLP trace
# shellcheck shell=bash

[[ -n "${__LIB_OBS_LOADED:-}" ]] && return 0
__LIB_OBS_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"

# Append an observability record.  Extra args are k,v pairs.
obs_record() {
  local event="$1"; shift
  audit_log "$event" "$@"
}

# Stream state.json by polling.
obs_watch() {
  local sp; sp=$(state_path)
  [[ -f "$sp" ]] || { err "No state file: $sp"; return 1; }
  if [[ "${OBS_FORMAT:-plain}" == "json" ]]; then
    local cur=""
    while true; do
      local next; next="$(python3 -m json.tool "$sp" 2>/dev/null)"
      if [[ "$next" != "$cur" ]]; then
        echo "----- $(now_iso) -----"
        echo "$next"
        cur="$next"
      fi
      sleep "${OBS_INTERVAL:-2}"
    done
  else
    tail -F "$sp"
  fi
}

# Emit the cost report aggregated across all task records.
obs_cost_report() {
  local pp; pp=$(plan_path)
  python3 - "$pp" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
total = sum(t.get("task_cost_usd", 0) for t in plan.get("tasks", []))
tokens = sum(t.get("task_tokens_used", 0) for t in plan.get("tasks", []))
print(f"Total cost   : ${total:.4f} USD")
print(f"Total tokens : {tokens}")
print()
print(f"{'TASK ID':<32} {'AGENT':<24} {'COST (USD)':>12} {'TOKENS':>10}")
print('-'*82)
for t in plan.get("tasks", []):
    print(f"{t['id']:<32} {t.get('agent_role',''):<24} "
          f"{t.get('task_cost_usd',0):>12.4f} {t.get('task_tokens_used',0):>10d}")
PY
}

# Emit OTLP-JSONL trace stub for a task. Useful for piping into Jaeger/Tempo.
obs_trace_for_task() {
  local task_id="$1"
  python3 - <<PY
import json, time
trace = {
  "trace_id": "$task_id-$(date +%s)",
  "span_name": "orchestrator.task",
  "attributes": {
    "task.id":   "$task_id",
    "agent.role": next((t.get('agent_role','') for t in __import__("json").load(open("$(plan_path)"))['tasks'] if t['id'] == "$task_id"), "")
  },
  "start_time_ns": int(time.time()*1e9),
  "status": "ok"
}
print(json.dumps(trace))
PY
}

# Emit the audit log as pretty-printed text.
obs_audit_show() {
  local af; af="$(audit_path)"
  [[ -f "$af" ]] || { warn "No audit log at $af"; return 0; }
  python3 - "$af" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        try:
            o = json.loads(line)
        except Exception:
            print(f"{i}: <bad line> {line.strip()!r}"); continue
        keys = " ".join(f"{k}={v}" for k,v in o.items() if k != "ts")
        print(f"{i:>4} {o.get('ts','')} {keys}")
PY
}
