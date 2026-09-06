#!/bin/bash
# =============================================================================
# VORTEX-OS — Autonomous Multi-Agent Command Center
# Master Entry Point (skill.sh)
# =============================================================================
# Unified CLI for the 4-tier VORTEX-OS orchestrator.
# Commands: agent discovery/lint, hierarchical dispatch, HITL control,
# Continuity Engine inspection, and master-objective submission.
# =============================================================================
set -u

# Resolve paths relative to this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$ROOT_DIR/lib"
AGENTS_DIR="$ROOT_DIR/agents"
STATE_DIR="$ROOT_DIR/state"

# Source every module in lib/
for f in "$LIB_DIR"/*.sh; do
  # shellcheck disable=SC1090
  source "$f"
done

# Ensure runtime directories exist
mkdir -p "$STATE_DIR/pending_approvals" "$ROOT_DIR/swarms" "$ROOT_DIR/memory" \
         "$ROOT_DIR/deliverables" "$ROOT_DIR/tasks"

err() { echo "ERROR: $*" >&2; }
ok()  { echo "  ✓ $*"; }

# =============================================================================
# VORTEX-OS top-level commands
# =============================================================================

# Submit a master objective to the T0 General Manager → T1 Store Supervisor
cmd_dispatch_master() {
  local objective_file="${1:-}"
  if [[ -z "$objective_file" ]]; then
    err "Usage: ./skill.sh --dispatch-master <objective.md>"
    return 2
  fi
  if [[ ! -f "$objective_file" ]]; then
    err "Master objective file not found: $objective_file"
    return 2
  fi
  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  VORTEX-OS — Submitting Master Objective"
  echo "═══════════════════════════════════════════════════════"
  echo "  File: $objective_file"
  echo ""
  echo "  T0 General Manager → T1 Store Supervisor"
  echo ""
  dispatch_v4_pipeline "master_objective" "supervisor.store" "$objective_file"
}

# Replay a saved Golden Path workflow template
cmd_dispatch_template() {
  local template_file="${1:-}"
  if [[ -z "$template_file" || ! -f "$template_file" ]]; then
    err "Usage: ./skill.sh --dispatch-template <template.json>"
    return 2
  fi
  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  VORTEX-OS — Replaying Saved Template"
  echo "═══════════════════════════════════════════════════════"
  echo "  Template: $template_file"
  echo ""
  dispatch_v4_pipeline "template_run" "supervisor.shift" "$template_file"
}

# List pending HITL approval requests
cmd_hitl_status() {
  local pending_dir="$STATE_DIR/pending_approvals"
  if [[ -d "$pending_dir" ]]; then
    local any=0
    for f in "$pending_dir"/*.json; do
      [[ -f "$f" ]] || continue
      any=1
      echo "  ⏸  $(basename "$f" .json)"
      jq -c '{task_id, status, severity, proposed_action}' "$f" 2>/dev/null | sed 's/^/      /'
    done
    [[ $any -eq 0 ]] && echo "  No pending HITL requests."
  else
    echo "  No pending HITL requests."
  fi
}

# Approve a pending HITL request
cmd_hitl_approve() {
  local task_id="${1:-}"
  if [[ -z "$task_id" ]]; then
    err "Usage: ./skill.sh --hitl-approve <task_id>"
    return 2
  fi
  local f="$STATE_DIR/pending_approvals/${task_id}.json"
  if [[ ! -f "$f" ]]; then
    err "No pending HITL request for: $task_id"
    return 2
  fi
  local tmp
  tmp=$(mktemp)
  jq --arg ts "$(date -Iseconds)" '.status="APPROVED" | .approved_by="operator" | .approved_at=$ts' \
    "$f" > "$tmp" && mv "$tmp" "$f"
  echo "  ✓ Approved: $task_id"
}

# Deny a pending HITL request
cmd_hitl_deny() {
  local task_id="${1:-}"
  if [[ -z "$task_id" ]]; then
    err "Usage: ./skill.sh --hitl-deny <task_id>"
    return 2
  fi
  local f="$STATE_DIR/pending_approvals/${task_id}.json"
  if [[ ! -f "$f" ]]; then
    err "No pending HITL request for: $task_id"
    return 2
  fi
  local tmp
  tmp=$(mktemp)
  jq --arg ts "$(date -Iseconds)" '.status="DENIED" | .denied_by="operator" | .denied_at=$ts' \
    "$f" > "$tmp" && mv "$tmp" "$f"
  echo "  ✗ Denied: $task_id"
}

# Run the Continuity Engine + invariant check on a task
cmd_inspector_check() {
  local task_id="${1:-}"
  if [[ -z "$task_id" ]]; then
    err "Usage: ./skill.sh --inspector-check <task_id>"
    return 2
  fi
  echo "  >> Running Continuity Engine check on: $task_id"
  cmd_inspect_execution "$task_id"
}

# Print the full audit trail
cmd_audit_trail() {
  local log="$ROOT_DIR/memory/audit.jsonl"
  if [[ -f "$log" ]]; then
    echo "  Audit trail (last 50 entries):"
    tail -50 "$log" | jq -c '{ts, tier, agent, action, status}' 2>/dev/null
  else
    echo "  No audit trail yet (run --dispatch-master to generate one)."
  fi
}

# Print the help/usage banner
cmd_help() {
  cat <<'USAGE'

  ╔══════════════════════════════════════════════════════╗
  ║  VORTEX-OS — Autonomous Multi-Agent Command Center   ║
  ╚══════════════════════════════════════════════════════╝

USAGE:
  ./skill.sh <command> [args]

DISCOVERY & INSPECTION:
  --agents-discover              List all available agents
  --agents-inspect <name>        Dump a single agent's manifest
  --agents-validate <file>       Validate an agent manifest
  --agents-lint [--all|<name>]   Lint agents against the 8 invariants
  --agents-graph [--format]      Print the agent graph

DISPATCH (the 4-tier chain of command):
  --dispatch-master <objective.md>      Submit to T0 General Manager
  --dispatch-template <template.json>   Replay a saved Golden Path
  --dispatch-v4 <task_id> <agent>       Direct V4 pipeline dispatch

HITL (Human-in-the-Loop / Deep-Sleep Safety Gate):
  --hitl-status                  List pending approval requests
  --hitl-approve <task_id>       Approve a pending request
  --hitl-deny <task_id>          Deny a pending request

INSPECTION:
  --inspector-check <task_id>    Run Continuity Engine check
  --audit-trail                  Print the audit log

TESTING:
  ./verify.sh                    Run the full post-upload verification

EXAMPLES:
  ./skill.sh --agents-discover
  ./skill.sh --agents-lint --all
  ./skill.sh --dispatch-master ./my_project/objective.md
  ./skill.sh --hitl-status
  ./skill.sh --hitl-approve package_websim
  ./skill.sh --audit-trail

USAGE
}

# =============================================================================
# CLI dispatch table
# =============================================================================
case "${1:-help}" in
  # Discovery & inspection
  --agents-discover|--agents-discover=*)
    shift; cmd_agents_discover "${1:-}" ;;
  --agents-inspect)
    shift; cmd_agents_inspect "$@" ;;
  --agents-validate)
    shift; cmd_agents_validate "$@" ;;
  --agents-lint)
    shift; cmd_agents_lint "$@" ;;
  --agents-graph)
    shift; cmd_agents_graph "$@" ;;
  --agents-trace)
    shift; cmd_agents_trace "$@" ;;

  # Dispatch
  --dispatch-v4)
    shift; dispatch_v4_pipeline "$@" ;;
  --dispatch-master)
    shift; cmd_dispatch_master "$@" ;;
  --dispatch-template)
    shift; cmd_dispatch_template "$@" ;;

  # HITL
  --hitl-status)
    cmd_hitl_status ;;
  --hitl-approve)
    shift; cmd_hitl_approve "$@" ;;
  --hitl-deny)
    shift; cmd_hitl_deny "$@" ;;

  # Inspection
  --inspector-check)
    shift; cmd_inspector_check "$@" ;;
  --audit-trail)
    cmd_audit_trail ;;

  # Help
  help|--help|-h|*)
    cmd_help ;;
esac
