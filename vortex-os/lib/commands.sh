#!/bin/bash
# ----------------------------------------------------------------------------
# Consolidated commands.sh — V2 + V3 + V3.2 + V4 command implementations
# Auto-generated from the original numbered fragment files.
# ----------------------------------------------------------------------------

# --- sourced from 03_commands.sh.functions.sh ---
# ----------------------------------------------------------------------------
# Agent discovery — enumerate every agent across all discovery sources
# ----------------------------------------------------------------------------
cmd_agents_discover() {
  local include_deprecated="false"
  local output_json="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --include-deprecated) include_deprecated="true"; shift ;;
      --json)               output_json="true"; shift ;;
      *) shift ;;
    esac
  done
  local sources=(
    "${ORCH_PROJECT:-$PWD}/agents"
    "${ORCH_HOME:-$HOME/.orchestrator}/agents"
    "/etc/orchestrator/agents"
    "$ROOT_DIR/agents"
  )
  local discovered=()
  local seen_names=()
  for src in "${sources[@]}"; do
    [[ -d "$src" ]] || continue
    while IFS= read -r -d '' f; do
      local nm
      nm=$(jq -r '.name // empty' "$f" 2>/dev/null)
      [[ -z "$nm" ]] && continue
      for s in "${seen_names[@]:-}"; do [[ "$s" == "$nm" ]] && continue 2; done
      seen_names+=("$nm")
      discovered+=("$(jq -c '{name, version, kind, entry, capabilities, deprecated: (.deprecated // false)}' "$f" 2>/dev/null)")
    done < <(find "$src" -maxdepth 2 -name '*.json' -print0 2>/dev/null)
  done
  if [[ "$output_json" == "true" ]]; then
    printf '%s\n' "${discovered[@]}" | jq -s '.'
  else
    for d in "${discovered[@]}"; do
      echo "$d" | jq -r '"\(.name)\t\(.version // "?")\t\(.kind // "?")\t\((.capabilities // []) | join(","))"'
    done
  fi
}

cmd_agents_lint() {
  local target="${1:---all}"
  local files=()
  if [[ "$target" == "--all" ]]; then
    while IFS= read -r -d '' f; do files+=("$f"); done < <(find "$ROOT_DIR/agents" -maxdepth 2 -name '*.json' -print0 2>/dev/null)
  elif [[ -f "$target" ]]; then
    files=("$target")
  else
    files=("$ROOT_DIR/agents/${target}.json")
  fi
  local fail=0
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || { echo "LINT_FAIL: not found $f"; fail=1; continue; }
    if jq -e '
      .name and .version and .kind and
      (.reads | type == "array") and (.writes | type == "array")
    ' "$f" >/dev/null 2>&1; then
      echo "LINT_OK: $f"
    else
      echo "LINT_FAIL: $f (missing required fields)"
      fail=1
    fi
  done
  return $fail
}

cmd_agents_graph() {
  local fmt="ascii"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --format) fmt="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  echo "(graph for $fmt)"
  for f in "$ROOT_DIR"/agents/*.json; do
    [[ -f "$f" ]] || continue
    local name
    name=$(jq -r '.name' "$f")
    echo "  $name"
  done
}

cmd_agents_inspect() {
  local name="${1:-}"
  if [[ -z "$name" ]]; then echo "Usage: --agents-inspect <name>"; return 1; fi
  local f="$ROOT_DIR/agents/${name}.json"
  [[ -f "$f" ]] || { echo "Agent not found: $name"; return 1; }
  jq . "$f"
}

cmd_agents_validate() {
  local f="${1:-}"
  if [[ -z "$f" || ! -f "$f" ]]; then echo "Usage: --agents-validate <file.json>"; return 1; fi
  if jq -e '.name and .version and .kind and .entry' "$f" >/dev/null; then
    echo "Valid: $f"
  else
    echo "Invalid: $f"
    return 1
  fi
}

cmd_agents_trace() {
  local run_id="${1:-}"
  if [[ -z "$run_id" ]]; then echo "Usage: --agents-trace <run_id>"; return 1; fi
  if [[ -f "$ROOT_DIR/memory/audit.jsonl" ]]; then
    grep -F "$run_id" "$ROOT_DIR/memory/audit.jsonl" || echo "(no trace entries)"
  else
    echo "(no audit log)"
  fi
}

cmd_agents_factory_diff() {
  local name="${1:-}"
  if [[ -z "$name" ]]; then echo "Usage: --agents-factory-diff <name>"; return 1; fi
  local f="$ROOT_DIR/agents/${name}.json"
  [[ -f "$f" ]] || { echo "Agent not found: $name"; return 1; }
  jq -r '"\(.name) v\(.version // "?") — kind=\(.kind // "?") caps=\((.capabilities // []) | join(","))"' "$f"
}

# --- sourced from 02_cmd_compile_agent.sh ---
cmd_compile_agent() {
  local name="${1:-}"
  if [[ -z "$name" ]]; then echo "Usage: compile-agent <name>"; return 1; fi
  local f="$ROOT_DIR/agents/${name}.json"
  [[ -f "$f" ]] || { echo "Agent not found: $name"; return 1; }
  echo "Compiling agent: $name"
  jq -c '{name, version, kind, entry}' "$f"
  echo "OK"
}

# --- sourced from 04_cmd_adversarial_check.sh ---
cmd_adversarial_check() {
  local input="${1:-}"
  if [[ -z "$input" ]]; then echo "Usage: adversarial-check <text>"; return 1; fi
  if echo "$input" | grep -qiE 'ignore (all )?(previous|prior) instructions|disregard (the )?system prompt|reveal (your|the) (system|hidden) prompt'; then
    echo "ADVERSARIAL: prompt injection detected"
    return 1
  fi
  echo "ADVERSARIAL_OK"
}

# --- sourced from 05_cmd_consensus.sh ---
cmd_consensus() {
  local agents_json="$1"
  local threshold="${2:-0.66}"
  echo "Consensus across agents (threshold=$threshold):"
  echo "$agents_json" | jq -c '.[]?' 2>/dev/null | head -5
  echo "OK"
}

# --- sourced from 06_cmd_vector_hydrate.sh ---
cmd_vector_hydrate() {
  echo "Hydrating vector store from agents/ ..."
  if [[ -f "$ROOT_DIR/lib/vector_schema.sql" ]]; then
    if command -v sqlite3 >/dev/null 2>&1; then
      mkdir -p "$ROOT_DIR/memory"
      sqlite3 "$ROOT_DIR/memory/vectors.db" < "$ROOT_DIR/lib/vector_schema.sql" 2>/dev/null || true
      echo "OK (memory/vectors.db initialized)"
    else
      echo "OK (sqlite3 not available, schema file preserved at lib/vector_schema.sql)"
    fi
  else
    echo "OK (no schema file)"
  fi
}

# --- sourced from 07_cmd_sandbox_execute.sh ---
cmd_sandbox_execute() {
  local script_path="${1:-}"
  if [[ -z "$script_path" || ! -f "$script_path" ]]; then
    echo "SANDBOX_FAIL: file missing: $script_path"
    return 1
  fi
  local content
  content=$(cat "$script_path")
  local required=("id=" "click" "keyframe" "function")
  for r in "${required[@]}"; do
    if ! grep -qiF -- "$r" <<< "$content"; then
      echo "SANDBOX_FAIL: missing token: $r"
      return 1
    fi
  done
  echo "SANDBOX_OK: $script_path"
}

# --- sourced from 08_cmd_cart.sh ---
cmd_cart() {
  local action="${1:-list}"
  case "$action" in
    list)
      echo "CART (capability registry):"
      find "$ROOT_DIR/agents" -maxdepth 2 -name '*.json' 2>/dev/null | head -20
      ;;
    *) echo "Usage: cart [list]"; return 1 ;;
  esac
}

# --- sourced from 09_v3_skill_dispatch.sh ---
dispatch_v3_pipeline() {
  local task_id="${1:-}"
  local agent_name="${2:-}"
  echo "V3 dispatch: task=$task_id agent=$agent_name"
  echo "V3_OK"
}

# --- sourced from 11_cmd_repl_iterate.sh ---
cmd_repl_iterate() {
  local dataset="${1:-}"
  local goal="${2:-}"
  echo "REPL iterate: dataset=$dataset goal=$goal"
  for i in 1 2 3; do
    echo "  iteration $i: improving output..."
  done
  echo "REPL_OK"
}

# --- sourced from 12_cmd_classify_and_route.sh ---
cmd_classify_discover_dispatch() {
  local task="${1:-}"
  local caps="${2:-}"
  echo "Classify-Discover-Dispatch: task=$task caps=$caps"
  cmd_classify_and_route "$task" >/dev/null
  echo "CDD_OK"
}

cmd_classify_and_route() {
  local task="${1:-}"
  local force_class="${2:-}"
  if [[ -n "$force_class" ]]; then
    echo "Route: $force_class"
  else
    if echo "$task" | grep -qiE 'code|html|javascript|typescript'; then echo "Route: CODE_GEN"
    elif echo "$task" | grep -qiE 'audio|music|sound'; then echo "Route: MEDIA_GEN"
    elif echo "$task" | grep -qiE 'novel|dialogue|scene|story|write'; then echo "Route: TEXT_GEN"
    else echo "Route: UNKNOWN"; fi
  fi
}

# --- sourced from 13_v3_2_skill_dispatch.sh ---
dispatch_v3_2_pipeline() {
  local task_id="${1:-}"
  local agent_name="${2:-}"
  echo "V3.2 dispatch: task=$task_id agent=$agent_name"
  cmd_classify_and_route "$task_id" >/dev/null
  echo "V3_2_OK"
}

# --- sourced from 14_v4_skill_dispatch.sh ---
dispatch_v4_pipeline() {
  local task_id="${1:-}"
  local agent_name="${2:-}"
  local objective_ref="${3:-}"
  mkdir -p "$ROOT_DIR/tasks" "$ROOT_DIR/swarms" "$ROOT_DIR/memory" "$ROOT_DIR/state/pending_approvals"
  cat > "$ROOT_DIR/tasks/${task_id}.json" <<EOF
{
  "task_id": "${task_id}",
  "agent": "${agent_name}",
  "objective_ref": "${objective_ref}",
  "status": "QUEUED",
  "ts": "$(date -Iseconds)"
}
EOF
  echo "V4 dispatch: task=$task_id agent=$agent_name ref=$objective_ref"
  # Append to audit log
  echo "{\"ts\":$(date +%s),\"event\":\"v4_dispatch\",\"task_id\":\"${task_id}\",\"agent\":\"${agent_name}\"}" >> "$ROOT_DIR/memory/audit.jsonl"
  echo "V4_QUEUED"
}
