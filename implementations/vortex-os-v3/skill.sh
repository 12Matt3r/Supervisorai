#!/usr/bin/env bash
###############################################################################
#  skill.sh — The Universal Sub-Agent Orchestration Skill (v2.0.0)
#  ------------------------------------------------------------------
#  A project-agnostic, configurable orchestration layer for delegating
#  work to a pool of specialized sub-agents. Now supports:
#    - Hierarchical sub-plans (supervisor-of-supervisors)
#    - Quality gates, retries with exponential backoff
#    - Cost & token tracking, audit trail, OTel-compatible traces
#    - Plan templates, export/import, replay, diff
#    - Cross-plan memory store
#    - Pass-5 notifications
#    - Optional LLM dynamic-routing mode
#    - Hard-seal / unseal contract
###############################################################################
set -euo pipefail

# Determine our own directory once.
SKILL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_VERSION="2.0.0"
SKILL_NAME="Universal Sub-Agent Orchestrator Skill"

# Default config / docs
DEFAULT_CONFIG="$SKILL_SCRIPT_DIR/config.minimax.json"
EXAMPLE_CONFIG="$SKILL_SCRIPT_DIR/config.example.json"

# Override via env var
CONFIG="${SKILL_CONFIG:-$DEFAULT_CONFIG}"
VERBOSE="${SKILL_VERBOSE:-0}"

# Load the library modules in dependency order.
source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/commands.sh"
source "$SKILL_SCRIPT_DIR/lib/minimax.sh"   # MiniMax-M3 (GMI) bridge — real LLM calls
source "$SKILL_SCRIPT_DIR/lib/v3.sh"
source "$SKILL_SCRIPT_DIR/lib/llm_stubs.sh"
source "$SKILL_SCRIPT_DIR/lib/jit_bridge.sh"

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
# Tracks position in argument list
ARG_POS=1
ARGS=("$@")

cmd_unknown() {
  err "Unknown argument: $1"
  err "Run '$0 --help' for usage."
  exit 1
}

# Replace positional argv with what commands expect
dispatch() {
  local arg="${ARGS[0]:-}"
  case "$arg" in
    ""                 ) cmd_help ;;
    -h|--help          ) cmd_help ;;
    -V|--version       ) cmd_version ;;
    -v|--verbose       ) VERBOSE=1; shift_args; dispatch ;;
    -c|--config        ) shift_args; CONFIG="${ARGS[0]:-$DEFAULT_CONFIG}"; shift_args;
                         ok "Using config: $CONFIG"; dispatch ;;
    --init             ) shift_args; cmd_init "${ARGS[0]:-}" ;;

    # Plan & state
    --plan             ) shift_args; cmd_plan ;;
    --plan-json        ) shift_args; cmd_plan_json ;;
    --status           ) shift_args; cmd_status ;;
    --add-task         ) shift_args; cmd_add_task "${ARGS[0]:-}" ;;
    --add-task-file    ) shift_args; cmd_add_task_file "${ARGS[0]:-}" ;;
    --import-template  ) shift_args; cmd_import_template "${ARGS[@]}" ;;
    --apply-template   ) shift_args; cmd_apply_template "${ARGS[@]}" ;;
    --list-templates   ) shift_args; cmd_list_templates ;;

    # Execution flow
    --ready            ) shift_args; cmd_ready ;;
    --next-task        ) shift_args; cmd_next_task ;;
    --dispatch-ready   ) shift_args; cmd_dispatch_ready ;;
    --mark-done        ) shift_args; cmd_mark_done "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --mark-failed      ) shift_args; cmd_mark_failed "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --escalate         ) shift_args; cmd_escalate "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --reset            ) shift_args; cmd_reset "${ARGS[0]:-}" ;;
    --skip             ) shift_args; cmd_skip "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --validate         ) shift_args; FORCE=1 cmd_validate ;;
    --aggregate        ) shift_args; cmd_aggregate ;;
    --seal             ) shift_args; cmd_seal ;;
    --unseal           ) shift_args; cmd_unseal ;;
    --checkpoint       ) shift_args; cmd_checkpoint ;;
    --recover          ) shift_args; cmd_recover ;;

    # Analysis
    --topo             ) shift_args; cmd_topo ;;
    --critical-path    ) shift_args; cmd_critical_path ;;
    --stats            ) shift_args; cmd_stats ;;
    --dot              ) shift_args; cmd_dot ;;
    --dag              ) shift_args; cmd_dag ;;
    --diff             ) shift_args; cmd_diff "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --doctor           ) shift_args; cmd_doctor ;;

    # Quality
    --quality-gate     ) shift_args; cmd_quality_gate "${ARGS[0]:-}" ;;
    --judge            ) shift_args; cmd_judge "${ARGS[0]:-}" "${ARGS[1]:-}" "${ARGS[2]:-}" ;;

    # Observability
    --watch            ) shift_args; cmd_watch "${ARGS[0]:-}" ;;
    --audit            ) shift_args; cmd_audit ;;
    --cost-report      ) shift_args; cmd_cost_report ;;
    --trace            ) shift_args; cmd_trace "${ARGS[0]:-}" ;;

    # Hierarchy
    --spawn-subplan    ) shift_args; cmd_spawn_subplan "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --list-subplans    ) shift_args; cmd_list_subplans ;;

    # Export / Import / Replay
    --export           ) shift_args; cmd_export "${ARGS[0]:-}" ;;
    --import           ) shift_args; cmd_import "${ARGS[0]:-}" ;;
    --replay           ) shift_args; cmd_replay ;;

    # Memory
    --memory-set       ) shift_args; cmd_memory_set "${ARGS[0]:-}" "${ARGS[1]:-}" "${ARGS[2]:-}" ;;
    --memory-get       ) shift_args; cmd_memory_get "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --memory-list      ) shift_args; cmd_memory_list "${ARGS[0]:-}" ;;
    --memory-delete    ) shift_args; cmd_memory_delete "${ARGS[0]:-}" "${ARGS[1]:-}" ;;
    --memory-snippet   ) shift_args; cmd_memory_snippet "${ARGS[0]:-}" "${ARGS[1]:-}" ;;

    # Dynamic
    --auto-plan        ) shift_args; cmd_auto_plan "${ARGS[0]:-}" "${ARGS[1]:-}" ;;

    # Notifications
    --notify           ) shift_args; cmd_notify "${ARGS[0]:-}" ;;

    # Resource management
    --agent-load       ) shift_args; cmd_agent_load ;;
    --record-load      ) shift_args; cmd_record_load "${ARGS[0]:-}" "${ARGS[1]:-}" ;;

    # Dynamic agent factory
    --agents-factory-create      ) shift_args; cmd_agents_factory_create    "${ARGS[0]:-}" "${ARGS[1]:-}" "${ARGS[2]:-}" ;;
    --agents-factory-persona     ) shift_args; cmd_agents_factory_persona   "${ARGS[0]:-}" "${ARGS[1]:-}" "${ARGS[2]:-}" "${ARGS[3]:-}" ;;
    --agents-factory-synthesize  ) shift_args; cmd_agents_factory_synthesize "${ARGS[0]:-}" "${ARGS[1]:-}" "${ARGS[2]:-}" ;;
    --agents-factory-list        ) shift_args; cmd_agents_factory_list ;;
    --agents-factory-show        ) shift_args; cmd_agents_factory_show    "${ARGS[0]:-}" ;;
    --agents-factory-tick        ) shift_args; cmd_agents_factory_tick    "${ARGS[0]:-}" ;;
    --agents-factory-remove      ) shift_args; cmd_agents_factory_remove  "${ARGS[0]:-}" ;;

    # ----- V2 Agent Discovery / Lint / Graph (Block 4) -----
    --agents-discover)
      shift_args
      cmd_agents_discover "${ARGS[0]:-false}"
      exit $?
      ;;
    --agents-discover=*)
      shift_args
      cmd_agents_discover "${ARGS[0]:-false}"
      exit $?
      ;;

    --agents-inspect)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --agents-inspect <name>"; exit 1; }
      local_path=$(agent_resolve_name "${ARGS[0]}")
      [[ -f "$local_path" ]] || { err "Agent not found: ${ARGS[0]}"; exit 1; }
      jq . "$local_path"
      exit $?
      ;;

    --agents-validate)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --agents-validate <file.json>"; exit 1; }
      [[ -f "${ARGS[0]}" ]] || { err "File not found: ${ARGS[0]}"; exit 1; }
      if jq -e '
        .name and .version and .kind and .entry and
        .input_schema and .output_schema and
        (.writes | type == "array") and (.reads | type == "array")
      ' "${ARGS[0]}" >/dev/null; then
        echo "Valid: ${ARGS[0]}"
        exit 0
      else
        echo "Invalid: ${ARGS[0]}"
        jq . "${ARGS[0]}"
        exit 1
      fi
      ;;

    --agents-lint)
      shift_args
      cmd_agents_lint "${ARGS[@]}"
      exit $?
      ;;

    --agents-graph)
      shift_args
      cmd_agents_graph "${ARGS[@]}"
      exit $?
      ;;

    --agents-trace)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --agents-trace <run_id>"; exit 1; }
      cmd_agents_trace "${ARGS[0]}"
      exit $?
      ;;

    --agents-factory-diff)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --agents-factory-diff <name>"; exit 1; }
      cmd_agents_factory_diff "${ARGS[0]}"
      exit $?
      ;;

    # ----- V3 Agent Compilation / V3 Pipeline (Block 11) -----
    --agents-compile)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --agents-compile <name> [--force]"; exit 1; }
      local force="false"
      [[ "${ARGS[1]:-}" == "--force" ]] && force="true"
      cmd_compile_agent "${ARGS[0]}" "$force"
      exit $?
      ;;

    --agents-compile-all)
      cmd_agents_compile_all
      exit $?
      ;;

    --adversarial-check)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --adversarial-check <context> [agent_name]"; exit 1; }
      cmd_adversarial_check "${ARGS[0]}" "${ARGS[1]:-manual}"
      exit $?
      ;;

    --consensus)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --consensus <task_description> [capability]"; exit 1; }
      cmd_consensus "${ARGS[0]}" "${ARGS[1]:-}"
      exit $?
      ;;

    --memory-init)
      cmd_memory_init
      exit $?
      ;;

    --memory-ingest)
      shift_args
      [[ ${#ARGS[@]} -ge 2 ]] || { err "Usage: --memory-ingest <text> <source> [chunk_id]"; exit 1; }
      cmd_memory_ingest "${ARGS[0]}" "${ARGS[1]}" "${ARGS[2]:-}"
      exit $?
      ;;

    --memory-hydrate)
      shift_args
      [[ ${#ARGS[@]} -ge 2 ]] || { err "Usage: --memory-hydrate <agent_name> <query>"; exit 1; }
      cmd_memory_hydrate "${ARGS[0]}" "${ARGS[1]}"
      exit $?
      ;;

    --sandbox-execute)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --sandbox-execute <code>"; exit 1; }
      cmd_sandbox_execute "${ARGS[0]}"
      exit $?
      ;;

    --cart)
      shift_args
      local dry_run="false"
      [[ "${ARGS[0]:-}" == "--dry-run" ]] && dry_run="true"
      cmd_cart "$dry_run"
      exit $?
      ;;

    --cart-test)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --cart-test <agent_name> [--dry-run]"; exit 1; }
      local dry_run="false"
      [[ "${ARGS[1]:-}" == "--dry-run" ]] && dry_run="true"
      cmd_cart_test_agent "${ARGS[0]}" "$dry_run"
      exit $?
      ;;

    --aesthetic-check)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --aesthetic-check <agent_name>"; exit 1; }
      local agent_path
      agent_path=$(agent_resolve_name "${ARGS[0]}")
      if [[ -f "state/compiled_prompt_${ARGS[0]}.txt" ]]; then
        echo "Compiled aesthetic-aware prompt for ${ARGS[0]}:"
        echo "---"
        cat "state/compiled_prompt_${ARGS[0]}.txt"
      else
        err "No compiled prompt found for ${ARGS[0]}. Run --agents-compile first."
        exit 1
      fi
      exit $?
      ;;

    --dispatch-v3)
      shift_args
      [[ ${#ARGS[@]} -ge 2 ]] || { err "Usage: --dispatch-v3 <task_id> <agent_name>"; exit 1; }
      dispatch_v3_pipeline "${ARGS[0]}" "${ARGS[1]}"
      exit $?
      ;;

    --llm-stub)
      # V3 — native LLM stubs diagnostic.  Shows which LLM stub functions are loaded.
      echo "Native LLM stubs (heuristic implementations):"
      echo "  query_security_monitor  -> $(declare -F query_security_monitor >/dev/null && echo OK || echo MISSING)"
      echo "  query_fast_llm          -> $(declare -F query_fast_llm >/dev/null && echo OK || echo MISSING)"
      echo "  evaluate_consensus      -> $(declare -F evaluate_consensus >/dev/null && echo OK || echo MISSING)"
      echo "  get_embedding           -> $(declare -F get_embedding >/dev/null && echo OK || echo MISSING)"
      exit 0
      ;;

    --jit-bridge)
      shift_args
      [[ ${#ARGS[@]} -ge 1 ]] || { err "Usage: --jit-bridge <capability>"; exit 1; }
      cmd_jit_bridge "${ARGS[0]}"
      exit $?
      ;;

    --jit-status)
      # Diagnostics for the JIT bridge: request log, response log, generated wrappers.
      echo "=== JIT Request Log (last 5) ==="
      tail -n 5 state/jit_requests.log 2>/dev/null || echo "  (empty)"
      echo "=== JIT Response Log (last 5) ==="
      tail -n 5 state/jit_responses.log 2>/dev/null || echo "  (empty)"
      echo "=== JIT-Synthesised Agent Wrappers ==="
      ls -la agents/minimax.native.*.json agents/auto.synthesized.*.json 2>/dev/null || echo "  (none yet)"
      exit 0
      ;;

    --force            ) shift_args; FORCE=1; dispatch ;;

    --json             ) shift_args; OUTPUT_FORMAT=json; dispatch ;;

    # passthrough / unknown
    *                  ) cmd_unknown "$arg" ;;
  esac
}

shift_args() {
  if [[ ${#ARGS[@]} -gt 0 ]]; then
    ARGS=("${ARGS[@]:1}")
  fi
}

dispatch
