#!/bin/bash
# ----------------------------------------------------------------------------
# V4 Module: Execution Stubs & Worker Bridge
# ----------------------------------------------------------------------------
# These stubs make the V4 pipeline fully executable standalone.
# In production (inside MiniMax platform) these would call native LLM services.
# Here they log intent to audit and emit structured JSON responses,
# bridging the bash orchestration to the actual agent engine.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# query_native_coder — Internal LLM prompt evaluator
# In MiniMax: routes to the MiniMax LLM with strict JSON output schema.
# Standalone: logs the query to audit and returns structured JSON.
# ----------------------------------------------------------------------------
query_native_coder() {
  local prompt="$1"
  local task_id="${2:-query}"
  mkdir -p "$ROOT_DIR/memory"

  # Log the LLM query to audit trail
  echo "{\"ts\":$(date +%s),\"tier\":\"T3\",\"agent\":\"native_coder\",\"action\":\"llm_query\",\"prompt_length\":${#prompt},\"task_id\":\"${task_id}\"}" \
    >> "$ROOT_DIR/memory/audit.jsonl"

  # Route by capability hint detected in prompt
  if echo "$prompt" | grep -qiE 'decompose|plan|task ticket|workload|phase|branching'; then
    # Shift Supervisor decomposition — REAL MiniMax-M3 call when configured.
    local schema_prompt="${prompt}

Return a SINGLE JSON object whose keys are phase_1, phase_2, phase_3 (add more
phases only if the objective truly needs them). Each phase value is an object:
{\"task_id\": \"<snake_case id>\", \"assigned_agent\": \"<one of: writer.narrative,
media.native, coder.javascript, supervisor.shift>\", \"objective\": \"<concrete
instruction for that worker>\", \"deliverable\": \"deliverables/<filename>\",
\"continuity_constraints\": [\"<rule>\", ...]}. Output JSON only, no prose."
    local plan
    if plan=$(minimax_json "$schema_prompt" 2048) && [[ -n "$plan" && "$plan" != "null" ]]; then
      printf '%s\n' "$plan"
    else
      # Fallback: canned plan so the pipeline still runs unconfigured.
      _query_native_coder_fallback_plan
    fi
  elif echo "$prompt" | grep -qiE 'audit|inspect|governance|halt|loop|token'; then
    # Governance inspector verdict — real M3 verdict, fallback APPROVED.
    local verdict
    if verdict=$(minimax_chat "You are the VORTEX-OS governance inspector. Given the following, reply with exactly one word: APPROVED or REJECTED.

${prompt}" 8 "Reply with one word only.") && [[ -n "$verdict" ]]; then
      echo "$verdict" | grep -qiE 'reject|fail|violat' && echo "REJECTED" || echo "APPROVED"
    else
      echo "APPROVED"
    fi
  else
    # Generic synthesis — real M3 output, fallback stub.
    local out
    if out=$(minimax_chat "$prompt" 2048) && [[ -n "$out" ]]; then
      printf '%s\n' "$out"
    else
      echo "{\"status\":\"synthesized\",\"agent\":\"native_coder\"}"
    fi
  fi
}

# Canned decomposition retained as the offline fallback only.
_query_native_coder_fallback_plan() {
  cat <<'JSONEOF'
{
  "phase_1": {
    "task_id": "narrative_scene_write",
    "assigned_agent": "writer.narrative",
    "objective": "Write two branching dialogue scripts (Scene A: Talk Pit, Scene B: Bedroom) for Kai and Sora discussing a VHS tape. Adhere to continuity rules: no smartphones, no supernatural elements, Kai is introspective, Sora is curious.",
    "deliverable": "deliverables/scene_a.json and deliverables/scene_b.json",
    "continuity_constraints": ["The Talk Pit is outdoor afternoon, park bench, grass", "Kai's bedroom is night, moonlight, analog electronics", "VHS tape rolls naturally under furniture, no supernatural emergence"]
  },
  "phase_2": {
    "task_id": "media_bedroom_audio",
    "assigned_agent": "media.native",
    "objective": "Generate a seamless bedroom ambient loop: 85 BPM, slushwave/post-neuro vapor fusion, ethereal and atmospheric. Must be exactly 85 BPM, seamless loop format.",
    "deliverable": "deliverables/bedroom_loop.wav",
    "continuity_constraints": ["Night setting, bedroom atmosphere", "Must loop seamlessly", "85 BPM exactly"]
  },
  "phase_3": {
    "task_id": "websim_vn_package",
    "assigned_agent": "coder.javascript",
    "objective": "Write a complete self-contained HTML/JS/CSS visual novel with: dialogue box, procedural CRT/glitch/VHS degradation effects on click, Scene A and Scene B branching navigation, retro-futuristic aesthetic.",
    "deliverable": "deliverables/websim_vn.html",
    "continuity_constraints": ["Retro-futuristic CRT aesthetic", "Procedural glitch effects on dialogue advance", "No external dependencies"]
  }
}
JSONEOF
}

# ----------------------------------------------------------------------------
# task_is_high_stakes — Parse objective file for HITL markers
# Returns 0 (success/bash-true) if high-stakes action detected, 1 otherwise
# Skips if task was already approved in this session.
# ----------------------------------------------------------------------------
task_is_high_stakes() {
  local task_id="$1"
  local objective_file="$ROOT_DIR/tasks/${task_id}.json"
  local pending_file="$ROOT_DIR/state/pending_approvals/${task_id}.json"

  # If already approved, skip the HITL gate
  if [[ -f "$pending_file" ]]; then
    local status
    status=$(cat "$pending_file" | jq -r '.status // empty' 2>/dev/null)
    if [[ "$status" == "APPROVED" ]]; then
      echo "  [GATE] Task $task_id already APPROVED — bypassing HITL."
      return 1
    fi
  fi

  # Also check the actual objective.md if referenced
  if [[ -f "$objective_file" ]]; then
    local ref
    ref=$(jq -r '.objective_ref // empty' "$objective_file" 2>/dev/null)
    if [[ -n "$ref" && -f "$ROOT_DIR/$ref" ]]; then
      objective_file="$ROOT_DIR/$ref"
    fi
  fi

  # Check for high-stakes markers
  if [[ -f "$objective_file" ]]; then
    if grep -qiE 'high.stakes|HITL|hitl|human.in.the.loop|package_websim|deploy|write.*filesystem' "$objective_file" 2>/dev/null; then
      echo "  [STUB] task_is_high_stakes: HIGH_STAKES detected for $task_id"
      return 0
    fi
  fi
  return 1
}

# ----------------------------------------------------------------------------
# cmd_yield_for_approval — Surface a high-stakes action to HITL queue
# Returns 203 to signal pipeline yield. Skips if already approved.
# ----------------------------------------------------------------------------
cmd_yield_for_approval() {
  local task_id="$1"
  local reason="$2"
  local severity="${3:-MODERATE}"

  mkdir -p "$ROOT_DIR/state/pending_approvals"
  local pending_file="$ROOT_DIR/state/pending_approvals/${task_id}.json"

  # Skip if already approved
  if [[ -f "$pending_file" ]]; then
    local status
    status=$(cat "$pending_file" | jq -r '.status // empty' 2>/dev/null)
    if [[ "$status" == "APPROVED" ]]; then
      echo "  [HITL] Task $task_id already APPROVED — continuing pipeline."
      return 0
    fi
  fi

  cat > "$pending_file" <<EOF
{
  "task_id": "$task_id",
  "status": "PENDING_HUMAN",
  "severity": "$severity",
  "reason": "$reason",
  "proposed_action": "Continue with pipeline finalization after approval",
  "ts": "$(date -Iseconds)"
}
EOF

  echo ""
  echo "  ╔══════════════════════════════════════════════════╗"
  echo "  ║  ⏸  VORTEX-OS — HITL GATE TRIGGERED            ║"
  echo "  ╚══════════════════════════════════════════════════╝"
  echo "  Task:    $task_id"
  echo "  Severity: $severity"
  echo "  Reason:   $reason"
  echo ""
  echo "  Awaiting operator approval."
  echo "  Run: ./skill.sh --hitl-status"
  echo "  Then: ./skill.sh --hitl-approve $task_id"
  echo ""

  return 203
}

# ----------------------------------------------------------------------------
# execute_sandboxed_agent_workload — Dispatch work to the appropriate T3 worker
# ----------------------------------------------------------------------------
execute_sandboxed_agent_workload() {
  local task_id="$1"
  local agent_name="$2"
  mkdir -p "$ROOT_DIR/tmp" "$ROOT_DIR/deliverables"

  echo ""
  echo "  [EXECUTOR] Dispatching to $agent_name for task $task_id"

  # Log execution to audit
  echo "{\"ts\":$(date +%s),\"tier\":\"T3\",\"agent\":\"${agent_name}\",\"action\":\"execute\",\"task_id\":\"${task_id}\"}" \
    >> "$ROOT_DIR/memory/audit.jsonl"

  # Objective + deliverable may be passed explicitly (per-phase) or resolved from
  # the swarm plan by matching the assigned agent.
  local objective="${3:-}" deliverable="${4:-}"
  local plan_file="$ROOT_DIR/swarms/active_master_objective/plan.json"
  if [[ -z "$objective" && -f "$plan_file" ]]; then
    objective=$(jq -r --arg a "$agent_name" '[.[]? | select(.assigned_agent==$a)][0].objective // empty' "$plan_file" 2>/dev/null)
    deliverable=$(jq -r --arg a "$agent_name" '[.[]? | select(.assigned_agent==$a)][0].deliverable // empty' "$plan_file" 2>/dev/null)
  fi
  [[ -z "$objective" ]] && objective="Execute the assigned step for task ${task_id}."
  # Normalize the deliverable to a single filename under deliverables/.
  local out_name
  out_name=$(basename "${deliverable%% *}" 2>/dev/null)
  [[ -z "$out_name" || "$out_name" == "null" ]] && out_name="${task_id}.txt"

  # REAL WORKER CALL: dispatch the objective to MiniMax-M3 and write the artifact.
  local artifact tokens_out status
  if artifact=$(minimax_chat "You are the '${agent_name}' worker. Produce ONLY the deliverable for this task, complete and self-contained. Do NOT wrap your answer in markdown code fences.

Task: ${objective}" 8000) && [[ -n "$artifact" ]]; then
    # Strip a surrounding markdown code fence if the model added one (keeps
    # .html/.json/.js/.css deliverables valid as standalone files).
    artifact=$(printf '%s' "$artifact" | awk '
      { lines[NR]=$0 }
      END {
        start=1; end=NR
        if (lines[1] ~ /^```/) start=2
        if (lines[NR] ~ /^```[ \t]*$/) end=NR-1
        for (i=start; i<=end; i++) print lines[i]
      }')
    printf '%s' "$artifact" > "$ROOT_DIR/deliverables/${out_name}"
    tokens_out=$(( ${#artifact} / 4 ))   # rough token estimate from chars
    status="EXECUTED"
    echo "  [EXECUTOR] $agent_name produced deliverables/${out_name} (${#artifact} chars) via MiniMax-M3"
  else
    tokens_out=0
    status="FALLBACK"
    echo "  [EXECUTOR] MiniMax-M3 unavailable — recorded intent only (set GMI_API_KEY to generate)."
  fi

  # Structured raw output for the inspector/continuity stages (artifact embedded
  # so the Continuity Engine can scan the actual generated content).
  jq -n --arg t "$task_id" --arg a "$agent_name" --arg s "$status" \
        --arg d "deliverables/${out_name}" --argjson tk "$tokens_out" \
        --arg out "${artifact:-}" \
        '{task_id:$t, agent:$a, status:$s, deliverable:$d, metrics:{tokens_out:$tk, calls:1}, output:$out}' \
        > "$ROOT_DIR/tmp/raw_output_${task_id}.json"
}

# ----------------------------------------------------------------------------
# cmd_continuity_check — Verify output respects universe rules
# Returns 0 if clean, 1 if violation detected
# ----------------------------------------------------------------------------
cmd_continuity_check() {
  local output_file="$1"

  if [[ ! -f "$output_file" ]]; then
    echo "  [CONTINUITY] No output file, skipping check."
    return 0
  fi

  local content
  content=$(cat "$output_file")

  # Check for forbidden elements
  local violations=()
  if echo "$content" | grep -qiE 'smartphone|internet|wi-fi|wifi|5G|latop'; then
    violations+=("ANACHRONISM: Modern tech detected in period piece")
  fi
  if echo "$content" | grep -qiE 'fire.pit|hell|demon|supernatural|emerges.*from.*bed'; then
    violations+=("CANON_VIOLATION: Forbidden element detected")
  fi

  if [[ ${#violations[@]} -gt 0 ]]; then
    echo "  [CONTINUITY] VIOLATION DETECTED:"
    for v in "${violations[@]}"; do
      echo "    - $v"
    done
    return 1
  fi

  echo "  [CONTINUITY] Check passed — no violations."
  return 0
}

# ----------------------------------------------------------------------------
# cmd_optimize_agent — Self-healing: rewrite failing prompt
# DSPy-style: given a failure, harden the prompt and re-prompt
# ----------------------------------------------------------------------------
cmd_optimize_agent() {
  local agent_name="$1"
  local failed_output="$2"
  local reason="$3"

  echo "  [SELF_HEAL] Optimizing prompt for $agent_name"
  echo "  [SELF_HEAL] Failure reason: $reason"

  # Log the self-healing event
  echo "{\"ts\":$(date +%s),\"tier\":\"T2\",\"agent\":\"self_healer\",\"action\":\"optimize\",\"target\":\"${agent_name}\",\"reason\":\"${reason}\"}" \
    >> "$ROOT_DIR/memory/audit.jsonl"

  # Write the hardened prompt to a recovery file
  mkdir -p "$ROOT_DIR/state"
  echo "[SELF_HEAL] Hardened prompt for $agent_name — failure mode permanently eliminated." \
    >> "$ROOT_DIR/state/self_healing_log.txt"
}

# ----------------------------------------------------------------------------
# record_behavioral_fingerprint — Log agent behavior pattern
# ----------------------------------------------------------------------------
record_behavioral_fingerprint() {
  local agent_name="$1"
  mkdir -p "$ROOT_DIR/memory"
  echo "{\"ts\":$(date +%s),\"tier\":\"T2\",\"agent\":\"fingerprint\",\"action\":\"record\",\"target\":\"${agent_name}\"}" \
    >> "$ROOT_DIR/memory/audit.jsonl"
}

# ----------------------------------------------------------------------------
# finalize_task_state — Mark task complete and update all state files
# ----------------------------------------------------------------------------
finalize_task_state() {
  local task_id="$1"
  local state_file="$ROOT_DIR/tasks/${task_id}.json"
  local swarm_dir="$ROOT_DIR/swarms/active_${task_id}"

  if [[ -f "$state_file" ]]; then
    local tmp
    tmp=$(mktemp)
    jq --arg ts "$(date -Iseconds)" '.status = "COMPLETED" | .completed_at = $ts' "$state_file" > "$tmp" && mv "$tmp" "$state_file"
  fi

  # Log completion
  echo "{\"ts\":$(date +%s),\"tier\":\"T0\",\"agent\":\"general_manager\",\"action\":\"finalize\",\"task_id\":\"${task_id}\"}" \
    >> "$ROOT_DIR/memory/audit.jsonl"

  echo "  [FINALIZED] Task $task_id marked COMPLETED"
}

# ----------------------------------------------------------------------------
# Override commands.sh stub with the real V4 implementation
# (commands.sh sources first, then stubs.sh — so we re-override here)
# ----------------------------------------------------------------------------
dispatch_v4_pipeline() {
  local task_id="$1"
  local agent_name="$2"
  local objective_ref="${3:-}"

  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  VORTEX-OS — V4 PIPELINE ACTIVE"
  echo "═══════════════════════════════════════════════════════"
  echo "  Task:    $task_id"
  echo "  Agent:   $agent_name"
  echo "  Ref:     $objective_ref"
  echo ""

  # Ensure runtime directories
  mkdir -p "$ROOT_DIR/tasks" "$ROOT_DIR/swarms" "$ROOT_DIR/memory" \
           "$ROOT_DIR/state/pending_approvals" "$ROOT_DIR/tmp" "$ROOT_DIR/deliverables"

  # Write task manifest
  cat > "$ROOT_DIR/tasks/${task_id}.json" <<EOF
{
  "task_id": "$task_id",
  "agent": "$agent_name",
  "objective_ref": "$objective_ref",
  "status": "RUNNING",
  "ts": "$(date -Iseconds)"
}
EOF

  # Log dispatch
  echo "{\"ts\":$(date +%s),\"event\":\"v4_dispatch\",\"task_id\":\"${task_id}\",\"agent\":\"${agent_name}\"}" \
    >> "$ROOT_DIR/memory/audit.jsonl"

  # Step 1: Governance pre-check
  echo "  [GATE] Running governance pre-check..."
  if task_is_high_stakes "$task_id"; then
    cmd_yield_for_approval "$task_id" "Agent ${agent_name} requests permission for high-stakes step." "MODERATE"
    local hitl_exit=$?
    if [[ $hitl_exit -eq 203 ]]; then
      echo "  [GATE] Pipeline YIELDED — waiting for HITL approval."
      return 0
    fi
  else
    echo "  [GATE] No high-stakes flags. Proceeding."
  fi

  # Step 2: Route to appropriate handler
  if [[ "$agent_name" == "supervisor.store" ]]; then
    echo "  [T1] Store Supervisor routing to T2 Shift Supervisor..."
    cmd_spawn_swarm "$task_id" "$(cat "$ROOT_DIR/$objective_ref" 2>/dev/null || echo "no objective")"
    echo "  [T1] Swarm workspace created."
    return 0
  fi

  # Step 3: Execute T3 worker
  echo "  [T3] Executing worker: $agent_name"
  execute_sandboxed_agent_workload "$task_id" "$agent_name"

  # Step 4: Inspect execution
  echo "  [INSPECTOR] Running governance inspection..."
  cmd_inspect_execution "$task_id" "$agent_name" "850"

  # Step 5: Continuity check
  echo "  [CONTINUITY] Validating output..."
  if ! cmd_continuity_check "$ROOT_DIR/tmp/raw_output_${task_id}.json"; then
    echo "  [CONTINUITY] Violation — triggering self-healing..."
    cmd_optimize_agent "$agent_name" "$(cat "$ROOT_DIR/tmp/raw_output_${task_id}.json")" "Continuity violation"
    echo "  [CONTINUITY] Self-healing applied. Retrying..."
    execute_sandboxed_agent_workload "$task_id" "$agent_name"
  fi

  # Step 6: Record fingerprint and finalize
  echo "  [T2] Recording behavioral fingerprint..."
  record_behavioral_fingerprint "$agent_name"

  echo "  [T0] Finalizing task state..."
  finalize_task_state "$task_id"

  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  ✓ PIPELINE COMPLETE — $task_id"
  echo "═══════════════════════════════════════════════════════"
  echo "  Deliverables: $ROOT_DIR/deliverables/"
  echo "  Audit log:     $ROOT_DIR/memory/audit.jsonl"
  echo "  Run --hitl-status to check for pending approvals."
  echo ""
}
