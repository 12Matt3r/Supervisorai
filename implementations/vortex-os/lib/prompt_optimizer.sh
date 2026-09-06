#!/bin/bash
# ----------------------------------------------------------------------------
# V4 Module: DSPy-Inspired Prompt Self-Optimization & Healing
# ----------------------------------------------------------------------------
# When an agent breaches a constraint, this module queries the native LLM
# to rewrite the agent's description to be stricter, then saves the new version.
# This implements a closed-loop self-healing system inspired by DSPy optimizers.
# ----------------------------------------------------------------------------

cmd_optimize_agent() {
  local agent_name="$1"
  local failed_output="$2"
  local failure_reason="$3"

  local agent_file="agents/${agent_name}.json"
  [[ -f "$agent_file" ]] || return 1

  echo -e "\033[0;33m[OPTIMIZER] Agent '${agent_name}' breached constraint. Initiating self-healing... \033[0m"

  local current_desc
  current_desc=$(jq -r '.description' "$agent_file")

  # Trigger native inference to rewrite the description based on the specific failure context
  local optimization_query="The agent '${agent_name}' with baseline instruction: '${current_desc}' failed constraints. Reason: '${failure_reason}'. Erroneous Output: '${failed_output}'. Rewrite the agent's core description string to be significantly stricter, explicitly forbidding this failure mode while preserving capabilities. Output ONLY the raw description text."

  local new_desc
  new_desc=$(query_native_coder "$optimization_query")

  if [[ -n "$new_desc" && "$new_desc" != "null" ]]; then
    jq --arg nd "$new_desc" '.description = $nd | .version = (.version + "-optimized")' "$agent_file" > "${agent_file}.tmp"
    mv "${agent_file}.tmp" "$agent_file"
    echo -e "\033[0;32m[OPTIMIZER] Successfully updated prompt architecture for ${agent_name}.\033[0m"
    return 0
  fi

  return 1
}
