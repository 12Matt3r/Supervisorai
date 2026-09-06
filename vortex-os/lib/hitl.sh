#!/bin/bash
# ----------------------------------------------------------------------------
# V4 Module: Chat-Based Human-in-the-Loop Checkpointing
# ----------------------------------------------------------------------------
# When high-stakes actions are detected, this module halts execution,
# writes a checkpoint file, and emits a conversational message that flows
# back to the user via exit code 203. The user can then approve/deny and
# the pipeline resumes via cmd_resume_approved_task.
# ----------------------------------------------------------------------------

cmd_yield_for_approval() {
  local task_id="$1"
  local proposed_action="$2"
  local severity_level="${3:-HIGH}"

  local checkpoint_file="state/pending_approvals/${task_id}.json"
  mkdir -p "state/pending_approvals"

  jq -n \
    --arg tid "$task_id" \
    --arg act "$proposed_action" \
    --arg sev "$severity_level" \
    --arg ts "$(date +%s)" \
    '{ task_id: $tid, status: "PENDING_HUMAN", severity: $sev, proposed_action: $act, timestamp: $ts }' > "$checkpoint_file"

  # Direct conversational message output designed to route straight to the user
  echo -e "\n**Approval Required (Task Context: $task_id) [Severity: $severity_level]**"
  echo -e "Execution halted by the architecture safety gate. A critical action requires your verification:"
  echo -e "> **Proposed Action:** $proposed_action"
  echo -e "\nPlease reply directly with **'Approve $task_id'** to authorize execution, or **'Deny'** to abort the sequence."

  exit 203
}

cmd_resume_approved_task() {
  local task_id="$1"
  local checkpoint_file="state/pending_approvals/${task_id}.json"

  [[ -f "$checkpoint_file" ]] || return 1

  local status
  status=$(jq -r '.status' "$checkpoint_file")

  if [[ "$status" == "APPROVED" ]]; then
    echo -e "**Authorization Verified.** Resuming pipeline execution for Task $task_id..."
    rm "$checkpoint_file"
    dispatch_v4_pipeline "$task_id" "$(plan_get_task_agent "$task_id")"
  else
    echo -e "Cannot resume. Task $task_id is still flagged as: $status"
    return 1
  fi
}
