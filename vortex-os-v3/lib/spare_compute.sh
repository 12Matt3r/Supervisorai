# lib/spare_compute.sh — Bonus D: Spare-Compute Mode
# When the orchestrator host is under load, non-critical agents defer to
# a quiet queue.  The supervisor measures load (1-min loadavg / CPU count)
# and parks non-critical tasks.
set -euo pipefail
LIB_PREFIX="[spare-compute]"
[[ -n "${__LIB_SPARE_LOADED:-}" ]] && return 0
__LIB_SPARE_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh" 2>/dev/null || true

SPARE_QUEUE_FILE="${ORCH_SPARE_QUEUE:-./.orchestr8r/spare_queue.jsonl}"

# spare_load_ratio — echo a number in [0.0, ...].  1.0 means CPU is exactly busy.
spare_load_ratio() {
  local cores load1
  cores=$(nproc 2>/dev/null || echo 1)
  load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
  awk -v l="$load1" -v c="$cores" 'BEGIN{printf "%.2f", l/c}'
}

# spare_should_park <agent_kind> — returns 0 (true) if this agent should park.
# Default policy: anything in low-priority families defers once load exceeds 0.8.
spare_should_park() {
  local kind="$1"
  local ratio; ratio=$(spare_load_ratio)
  local cap="${ORCH_SPARE_THRESHOLD:-0.8}"
  local low_priority="${ORCH_SPARE_KINDS:-writer.commit writer.docs support.triager pm.prioritizer}"
  awk -v r="$ratio" -v c="$cap" 'BEGIN{exit !(r+0 > c+0)}' || return 1
  # kind match against low_priority list
  [[ " $low_priority " == *" $kind "* ]] || return 1
  return 0
}

# spare_enqueue <task_id> <kind> — defer a task to the quiet queue.
spare_enqueue() {
  local tid="$1" kind="$2"
  local ratio; ratio=$(spare_load_ratio)
  mkdir -p "$(dirname "$SPARE_QUEUE_FILE")"
  jq -c -n \
    --arg ts "$(now_iso)" \
    --arg t  "$tid" \
    --arg k  "$kind" \
    --argjson r "$ratio" \
    '{ts:$ts, task:$t, kind:$k, load:$r, parked:true}' >> "$SPARE_QUEUE_FILE"
}

# spare_drain — emit (stdout) NDJSON of all parked tasks and clear the queue.
spare_drain() {
  [[ -f "$SPARE_QUEUE_FILE" ]] || return 0
  cat "$SPARE_QUEUE_FILE"
  : > "$SPARE_QUEUE_FILE"
}

# spare_status — one-line human summary.
spare_status() {
  local ratio parked
  ratio=$(spare_load_ratio)
  parked=$([[ -f "$SPARE_QUEUE_FILE" ]] && wc -l < "$SPARE_QUEUE_FILE" || echo 0)
  echo "load=${ratio}x   parked=${parked}"
}
