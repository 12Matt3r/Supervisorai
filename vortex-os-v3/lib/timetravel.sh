# lib/timetravel.sh — Bonus A: Time-Travel Debug
# Snapshot state.json per task; --rewind <task_id> jumps back.
set -euo pipefail
LIB_PREFIX="[timetravel]"

SNAPSHOT_DIR="${ORCH_SNAPSHOT_DIR:-./.orchestr8r/snapshots}"

timetravel_init() { mkdir -p "$SNAPSHOT_DIR"; }

timetravel_snapshot() {
  local task_id="$1"
  local f="$SNAPSHOT_DIR/$task_id.json"
  cp "${ORCH_STATE_FILE:-./state.json}" "$f"
  cp "${ORCH_PLAN_FILE:-./plan.json}" "$f.plan"
  echo "[timetravel] snapshotted $task_id -> $f"
}

timetravel_rewind() {
  local task_id="$1"
  local f="$SNAPSHOT_DIR/$task_id.json"
  [[ -f $f ]] || { echo "no snapshot for $task_id" >&2; return 1; }
  cp "$f" "${ORCH_STATE_FILE:-./state.json}"
  cp "$f.plan" "${ORCH_PLAN_FILE:-./plan.json}"
  echo "[timetravel] rewound to $task_id"
}
