#!/bin/bash
# ----------------------------------------------------------------------------
# V4 Module: Governance, Telemetry, and Compute Budget Auditing
# ----------------------------------------------------------------------------
# Tracks token consumption per dispatch. When a single run exceeds the
# budget threshold (15,000 tokens), triggers a native LLM audit to detect
# hallucination/retry storms. Halt decisions route through cmd_yield_for_approval.
# ----------------------------------------------------------------------------

cmd_inspect_execution() {
  local task_id="$1"
  local agent_name="$2"
  local tokens_used_this_run="$3"

  local budget_file="state/active_budgets.json"
  mkdir -p state
  [[ -f "$budget_file" ]] || echo '{"total_tokens": 0}' > "$budget_file"

  # Accumulate token metrics deterministically
  local current_total
  current_total=$(jq -r '.total_tokens' "$budget_file")
  local new_total=$((current_total + tokens_used_this_run))

  jq --arg nt "$new_total" '.total_tokens = ($nt | tonumber)' "$budget_file" > "${budget_file}.tmp" && mv "${budget_file}.tmp" "$budget_file"

  # Audit Rule: If a worker burns more than 15,000 tokens on a single dispatch run, initiate inspection
  if [[ "$tokens_used_this_run" -gt 15000 ]]; then
    echo -e "\033[0;31m[INSPECTOR] Anomalous token burn velocity detected ($tokens_used_this_run tokens). Evaluating loop telemetry... \033[0m"

    local recent_telemetry
    if [[ -f "memory/audit.jsonl" ]]; then
      recent_telemetry=$(tail -n 5 "memory/audit.jsonl")
    else
      recent_telemetry="No historical logs found."
    fi

    # Heuristic audit utilizing native evaluation
    local audit_query="As a Governance Inspector, audit this log trace: '${recent_telemetry}'. The agent '${agent_name}' expended ${tokens_used_this_run} tokens. Is this agent executing productively, or is it caught in a hallucination/retry storm? Respond exactly with either 'APPROVED' or 'HALT: <explicit reason>'."

    local audit_verdict
    audit_verdict=$(query_native_coder "$audit_query")

    if [[ "$audit_verdict" == HALT* ]]; then
      echo -e "\033[0;31m[INSPECTOR INTERVENTION] Flagging execution loop anomaly: ${audit_verdict}\033[0m"
      cmd_yield_for_approval "$task_id" "The Inspector Tier forced a halt due to compute budget inefficiency: ${audit_verdict}" "CRITICAL_BUDGET"
      return 1
    fi
  fi

  return 0
}
