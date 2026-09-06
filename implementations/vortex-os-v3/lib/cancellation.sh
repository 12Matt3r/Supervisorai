# lib/cancellation.sh — Improvement #9: Cooperative Cancellation Token
# Long-running agents check a touch file each iteration and exit cleanly.
set -euo pipefail
LIB_PREFIX="[cancellation]"

CANCEL_DIR="${ORCH_CANCEL_DIR:-./.orchestr8r/cancel}"

cancellation_init() { mkdir -p "$CANCEL_DIR"; }

# Mint a fresh cancel token for a task id, return the file path.
cancellation_open() {
  local task_id="$1"
  local f="$CANCEL_DIR/${task_id}.cancel"
  rm -f "$f"
  printf '%s' "$f"
}

cancellation_request() {
  local task_id="$1" reason="${2:-user}"
  local f="$CANCEL_DIR/${task_id}.cancel"
  jq -nc --arg t "$task_id" --arg r "$reason" \
    '{task:$t,reason:$r,at:now|todate}' > "$f"
}

cancellation_check() {
  local token_file="$1"
  [[ -f $token_file ]]
}

# Run a command but wrap with cancel polling. Period calls cancel_check.
cancellation_run() {
  local cancel_token="$1" period="${2:-1}"
  shift 2
  (
    while true; do
      if cancellation_check "$cancel_token"; then
        echo "cancelled"
        exit 130
      fi
      "$@" || exit $?
      exit 0
    done
  ) &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if cancellation_check "$cancel_token"; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 130
    fi
    sleep "$period"
  done
  wait "$pid" 2>/dev/null || true
}
