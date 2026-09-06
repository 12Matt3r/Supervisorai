#!/bin/bash
# ----------------------------------------------------------------------------
# V4 Module: Hierarchical Swarm Topologies & Workspaces
# ----------------------------------------------------------------------------
# Creates isolated workspaces (swarms) for each Tier 2 supervisor.
# Each swarm gets its own directory structure, plan, and memory database.
# This prevents cross-project context contamination and supports parallel work.
# ----------------------------------------------------------------------------

cmd_spawn_swarm() {
  local swarm_id="$1"
  local master_objective="$2"

  echo -e "\033[0;32m[SWARM] Store Supervisor spawning Tier 2 Shift Supervisor workspace for swarm: ${swarm_id}\033[0m"

  local swarm_dir="swarms/active_${swarm_id}"
  mkdir -p "$swarm_dir"/{agents,memory,deliverables,state}

  # Delegate planning down the hierarchy to the Shift Supervisor layer
  local swarm_plan
  swarm_plan=$(query_native_coder "As a Shift Supervisor, decompose the macro-phase objective into 3 discrete worker tickets. Objective: ${master_objective}. Output valid JSON matching the task schema format.")

  echo "$swarm_plan" > "$swarm_dir/plan.json"

  # Isolate local vector memory database tracking for this specific swarm context
  sqlite3 "$swarm_dir/memory/memory.db" "CREATE TABLE IF NOT EXISTS project_embeddings (id INTEGER PRIMARY KEY, chunk_text TEXT, embedding BLOB);" 2>/dev/null || true

  # --------------------------------------------------------------------------
  # T2 → T3: dispatch every planned phase to its worker (real MiniMax-M3).
  # Each worker output is continuity-checked; a violation triggers one
  # self-healing retry before the phase is accepted.
  # --------------------------------------------------------------------------
  local phase_keys
  phase_keys=$(jq -r 'keys_unsorted[]' "$swarm_dir/plan.json" 2>/dev/null)
  if [[ -z "$phase_keys" ]]; then
    echo -e "\033[0;33m[SWARM] Plan had no parseable phases; skipping worker dispatch.\033[0m"
    echo "$swarm_dir"; return 0
  fi

  local pnum=0
  while IFS= read -r pk; do
    [[ -z "$pk" ]] && continue
    pnum=$((pnum+1))
    local pagent pobj pdel out_name
    pagent=$(jq -r --arg k "$pk" '.[$k].assigned_agent // "coder.javascript"' "$swarm_dir/plan.json")
    pobj=$(jq -r --arg k "$pk"   '.[$k].objective // ""' "$swarm_dir/plan.json")
    pdel=$(jq -r --arg k "$pk"   '.[$k].deliverable // ""' "$swarm_dir/plan.json")
    out_name=$(basename "${pdel%% *}")
    [[ -z "$out_name" || "$out_name" == "null" ]] && out_name="${pk}.txt"

    echo -e "\033[0;36m[T2] Phase ${pnum} (${pk}) → ${pagent}\033[0m"
    execute_sandboxed_agent_workload "${swarm_id}_${pk}" "$pagent" "$pobj" "$out_name"

    # Continuity enforcement + one self-healing retry.
    if ! cmd_continuity_check "$ROOT_DIR/tmp/raw_output_${swarm_id}_${pk}.json"; then
      echo -e "\033[0;31m[T2] Continuity violation on ${pk} — self-healing and retrying.\033[0m"
      cmd_optimize_agent "$pagent" "$(cat "$ROOT_DIR/tmp/raw_output_${swarm_id}_${pk}.json")" "Continuity violation on ${pk}"
      execute_sandboxed_agent_workload "${swarm_id}_${pk}" "$pagent" \
        "$pobj (STRICT: obey every continuity rule; introduce no anachronisms or forbidden elements)" "$out_name"
    fi
  done <<< "$phase_keys"

  echo -e "\033[0;32m[SWARM] All ${pnum} phases dispatched. Deliverables in $ROOT_DIR/deliverables/\033[0m"
  echo "$swarm_dir"
}
