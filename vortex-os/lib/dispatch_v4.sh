#!/bin/bash
# ----------------------------------------------------------------------------
# V4 Module: Master Pipeline Dispatcher
# ----------------------------------------------------------------------------
# The V4 pipeline is hierarchical and governance-aware. The order is:
#   1. Governance pre-check (HITL gate for high-stakes actions)
#   2. Hierarchical routing (Tier 1/2 supervisors vs Tier 3 workers)
#   3. Standard sandboxed execution
#   4. Cross-cutting audit (token budget + loop detection)
#   5. Continuity enforcement + self-healing optimizer
#   6. Pipeline finalization
# ----------------------------------------------------------------------------

dispatch_v4_pipeline() {
  local task_id="$1"
  local agent_name="$2"

  # 1. GOVERNANCE PRE-CHECK: Intercept High-Stakes Actions before hitting workers
  if task_is_high_stakes "$task_id"; then
    cmd_yield_for_approval "$task_id" "Agent ${agent_name} is requesting permission to execute an unverified high-stakes operational step." "MODERATE"
    local hitl_exit=$?
    if [[ $hitl_exit -eq 203 ]]; then
       return 0 # Halted gracefully, state saved, control yielded back to conversation
    fi
  fi

  # 2. HIERARCHICAL ROUTING: Determine if task belongs to a Supervisor or Swarm
  if [[ "$agent_name" == "supervisor.store" ]]; then
     local master_obj
     master_obj=$(jq -r '.objective' "tasks/${task_id}.json")
     cmd_spawn_swarm "$task_id" "$master_obj"
     return 0
  fi

  # 3. STANDARD EXECUTION PASS (V3 Component Hand-off)
  execute_sandboxed_agent_workload "$task_id" "$agent_name"

  # Extract metrics from raw execution outputs
  local run_tokens
  run_tokens=$(jq -r '.metrics.tokens_out' "tmp/raw_output_${task_id}.json" 2>/dev/null || echo "0")

  # 4. CROSS-CUTTING AUDIT: Route output variables straight to the Inspector Tier
  if ! cmd_inspect_execution "$task_id" "$agent_name" "$run_tokens"; then
    return 1 # Pipeline structural drop out due to budget enforcement
  fi

  # 5. CONTINUITY ENFORCEMENT & SELF-HEALING AUTOMATION
  if ! cmd_continuity_check "tmp/raw_output_${task_id}.json"; then
    # The Shift Supervisor layer catches the invalid generation format or rule break
    cmd_optimize_agent "$agent_name" "$(cat tmp/raw_output_${task_id}.json)" "Failed to preserve critical system invariants or style guide."
    return 1
  fi

  # 6. PIPELINE FINALIZATION
  record_behavioral_fingerprint "$agent_name"
  finalize_task_state "$task_id"
}
