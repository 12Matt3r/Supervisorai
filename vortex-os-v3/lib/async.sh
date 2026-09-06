#!/usr/bin/env bash
###############################################################################
#  async.sh — Asynchronous Yielding & Deep-Sleep Suspension Library (V3)
#  ------------------------------------------------------------------
#  Provides cmd_task_yield which lets a running agent persist its state
#  to disk and return a 202 exit code so the master dispatch loop can
#  safely hand control back to the orchestrator.
#
#  On resume (cmd_task_resume) the scratchpad is reinjected into the
#  task and execution continues from the saved resume_at_index.
###############################################################################
set -euo pipefail

# ----------------------------------------------------------------------------
# cmd_task_yield <task_id> <current_index> <scratchpad_json>
#     Persists task state to memory/suspended/task_<id>.state and signals
#     the master loop with exit code 202.
# ----------------------------------------------------------------------------
cmd_task_yield() {
  local task_id="$1"
  local current_index="$2"
  local scratchpad_data="$3"
  local state_file="memory/suspended/task_${task_id}.state"
  mkdir -p "$(dirname "$state_file")"
  # Write the scratchpad verbatim (it is already a JSON fragment).
  printf '%s\n' "$scratchpad_data" > "$state_file"
  # Exit 202 signals the master loop that this task is suspended.
  return 202
}

# ----------------------------------------------------------------------------
# cmd_task_resume <task_id>
#     Re-dispatches a previously suspended task, injecting the saved
#     scratchpad and resume index, then deletes the state file.
# ----------------------------------------------------------------------------
cmd_task_resume() {
  local task_id="$1"
  local state_file="memory/suspended/task_${task_id}.state"
  [[ -f "$state_file" ]] || { echo "[error] No suspended task: $task_id" >&2; return 1; }

  local resume_index scratchpad
  resume_index=$(jq -r '.resume_at_index // 0' "$state_file" 2>/dev/null || echo 0)
  scratchpad=$(cat "$state_file" 2>/dev/null || echo "")

  if declare -F dispatch_single_task >/dev/null 2>&1; then
    dispatch_single_task "$task_id" --resume-index "$resume_index" --inject-scratchpad "$scratchpad" || true
  fi
  rm -f "$state_file"
}
