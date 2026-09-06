# lib/commands.sh — all cmd_* implementations
# shellcheck shell=bash

[[ -n "${__LIB_COMMANDS_LOADED:-}" ]] && return 0
__LIB_COMMANDS_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/locking.sh"
source "$SKILL_SCRIPT_DIR/lib/schema.sh"
source "$SKILL_SCRIPT_DIR/lib/graph.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"
source "$SKILL_SCRIPT_DIR/lib/quality.sh"
source "$SKILL_SCRIPT_DIR/lib/retry.sh"
source "$SKILL_SCRIPT_DIR/lib/observability.sh"
source "$SKILL_SCRIPT_DIR/lib/subplan.sh"
source "$SKILL_SCRIPT_DIR/lib/dynamic_agents.sh"
source "$SKILL_SCRIPT_DIR/lib/templates.sh"
source "$SKILL_SCRIPT_DIR/lib/export_import.sh"
source "$SKILL_SCRIPT_DIR/lib/memory.sh"
source "$SKILL_SCRIPT_DIR/lib/notification.sh"
source "$SKILL_SCRIPT_DIR/lib/aliases.sh"

# ----------------------------------------------------------------------------
# Help / version
# ----------------------------------------------------------------------------
cmd_version() {
  echo "${SKILL_NAME:-Universal Sub-Agent Orchestrator Skill} v${SKILL_VERSION:-2.0.0}"
}

cmd_help() {
  cat <<'EOF'
Universal Sub-Agent Orchestrator Skill — v2

CORE COMMANDS
  --init <dir>                          Bootstrap a new project
  --plan                                Show the current plan (pretty)
  --plan-json                           Show plan.json (raw)
  --status                              Show orchestration status
  --add-task <json>                     Add one task from a JSON string
  --add-task-file <path>                Add one task from a JSON file (safe)
  --import-template <name>              Add tasks from a template (use --template-var k=v)
  --list-templates                      List available templates
  --ready                               List ready tasks
  --next-task                           Print the next ready task as JSON
  --dispatch-ready                      Atomically dispatch all ready tasks
  --mark-done <id> <output>             Mark a task complete, record its output
  --mark-failed <id> <reason>           Mark a task failed
  --escalate <id> <target_role>         Escalate a task to a stronger role
  --reset <id>                          Send a task back to pending
  --skip <id> <reason>                  Skip a task with justification
  --validate                            Validate plan
  --aggregate                           List all completed tasks and their outputs
  --seal                                Seal the plan (sets sealed=true)
  --unseal                              Reopen the plan (rotates an unseal record)
  --checkpoint                          Save a state checkpoint
  --recover                             Print the last state checkpoint

ANALYSIS COMMANDS
  --topo                                Topological ordering of all tasks
  --critical-path                       Longest dependency chain
  --stats                               Counts by status + cost & tokens
  --dot                                 Emit Graphviz DOT (pipe to `dot -Tpng`)
  --dag                                 ASCII dependency graph
  --diff <plan_a> <plan_b>              Compare two plans
  --doctor                              Pre-flight environment checks

EXECUTION POLICY
  --quality-gate <id>                   Run validation on a task's output
  --judge <id> <score> <reason>         Apply a manual LLM judge verdict

OBSERVABILITY
  --watch [--json]                      Stream state changes
  --audit                               Print audit log
  --cost-report                         Aggregate costs and tokens
  --trace <id>                          Emit OTLP trace stub for a task

HIERARCHY (SUPERVISOR-OF-SUPERVISORS)
  --spawn-subplan <parent_id> <goal>    Spawn a nested plan from a task
  --list-subplans                       Tree view of parent + sub-plans

EXPORT / IMPORT / REPLAY
  --export <file.json>                  Bundle plan + state + audit
  --import <file.json>                  Replace plan from a bundle
  --replay                              Bring sealed-but-complete tasks back to complete

MEMORY STORE
  --memory-set <ns> <key> <json>        Set a value in the shared memory
  --memory-get <ns> <key>               Get a value from the shared memory
  --memory-list <ns>                    List keys in a namespace
  --memory-delete <ns> <key>            Delete a key
  --memory-snippet <name> <body>        Save a snippet

DYNAMIC ROUTING
  --auto-plan "<goal>" --model <m>      Use an LLM to expand a goal into tasks

PASS 5: NOTIFICATIONS
  --notify <event>                      Fire configured notification channels

RESOURCE MANAGEMENT
  --agent-load                          Show agent load (from agent_load.json)
  --record-load <role> <pct>            Update agent_load.json

DYNAMIC AGENT FACTORY  (requires dynamic_agents.enabled=true)
  --agents-factory-create <name> <parent> [reason]
                                      Clone an existing agent as <name>
  --agents-factory-persona <name> <base> <persona> [reason]
                                      Create a persona variant
  --agents-factory-synthesize <name> <gap> [reason]
                                      Synthesize an agent from a capability gap
  --agents-factory-list                List all spawned dynamic agents
  --agents-factory-show <name>         Show details of one dynamic agent
  --agents-factory-tick <name>         Age out one quarantine run
  --agents-factory-remove <name>       Remove a dynamic agent

GENERIC
  --config <path>                       Use a specific config file
  --verbose                             Enable diagnostic logging
  --version                             Print skill version
  --help                                This help

ENVIRONMENT
  SKILL_CONFIG          Path to the JSON config file
  SKILL_VERBOSE         Set to 1 for verbose logs
  ORCHESTRATOR_MODEL    LLM model used by --auto-plan (default gpt-4o-mini)

EXAMPLES
  skill --init ./proj
  skill --add-task '{"id":"r1","description":"market research","agent_role":"research"}'
  skill --add-task-file task.json
  skill --import-template competitor_landscape --template-var INDUSTRY="AI"
  skill --ready
  skill --dispatch-ready
  skill --mark-done r1 ./out/r.md
  skill --watch
  skill --seal
EOF
}

# ----------------------------------------------------------------------------
# Init / show / status
# ----------------------------------------------------------------------------
cmd_init() {
  local target="${1:-./orchestrator_skill}"
  if [[ -d "$target" ]]; then
    warn "Directory already exists: $target (skipping init)"
  else
    mkdir -p "$target"
    cp -n "$SKILL_SCRIPT_DIR/config.example.json" "$target/config.json" 2>/dev/null || true
    ok "Initialized orchestrator in: $target"
  fi
  cat <<EOF
Next steps:
  1. Edit $target/config.json
  2. Run: skill --config $target/config.json --plan
  3. Run: skill --config $target/config.json --add-task-file tasks.json
     (or: skill --import-template competitor_landscape)
EOF
}

cmd_plan() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  if [[ "${OUTPUT_FORMAT:-pretty}" == "json" ]]; then
    python3 -m json.tool "$pp"
    return 0
  fi
  header "Current plan: $(basename "$pp")"
  python3 - "$pp" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
print(f"Plan ID       : {plan['plan_id']}")
print(f"Goal          : {plan.get('goal','')}")
print(f"Created       : {plan.get('created_at','')}")
print(f"Sealed        : {plan.get('sealed', False)}")
print(f"Parent plan   : {plan.get('parent_plan_id','-')}")
print(f"Sub-plans     : {len(plan.get('sub_plan_ids',[]) or [])}")
print(f"Tasks         : {len(plan.get('tasks',[]))}")
print()
if plan.get('tasks'):
    print(f"{'ID':<28} {'AGENT':<24} {'STATUS':<10} {'DEPENDS_ON'}")
    print('-'*92)
    for t in plan['tasks']:
        deps = ','.join(t.get('depends_on',[])) or '-'
        print(f"{t['id']:<28} {t.get('agent_role',''):<24} {t.get('status',''):<10} {deps}")
PY
}

cmd_plan_json() {
  ensure_plan "$(gen_plan_id)" || return 1
  cat "$(plan_path)"
}

cmd_status() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp sp; pp=$(plan_path); sp=$(state_path)
  header "Orchestration status"
  python3 - "$pp" "$sp" <<'PY'
import json, os, sys
plan = json.load(open(sys.argv[1]))
state_path = sys.argv[2]
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
tasks = plan.get("tasks", [])
by_status = {}
for t in tasks:
    s = t.get("status","pending"); by_status[s] = by_status.get(s, 0) + 1
print(f"Plan  : {plan['plan_id']}")
print(f"Goal  : {plan.get('goal','')}")
print(f"Sealed: {plan.get('sealed', False)}")
print()
print(f"Total tasks: {len(tasks)}")
for s in ["pending","ready","dispatched","running","complete","failed","sealed","skipped","escalated"]:
    if s in by_status:
        color = {'complete':'\033[0;32m','sealed':'\033[0;32m',
                 'failed':'\033[0;31m','pending':'\033[1;33m',
                 'dispatched':'\033[0;34m'}.get(s,'')
        rst = '\033[0m' if color else ''
        print(f"  {color}{s:<12}{rst}: {by_status[s]}")
print()
if state:
    print('State file:')
    print(f"  Current pass   : {state.get('current_pass')}")
    print(f"  Tokens used    : {state.get('context_tokens_used',0)}")
    print(f"  Total cost USD : {state.get('total_cost_usd', 0)}")
    print(f"  Last checkpoint: {state.get('last_checkpoint','never')}")
PY
}

# ----------------------------------------------------------------------------
# Task mutation
# ----------------------------------------------------------------------------
cmd_add_task() {
  local task_json="$1"
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" "$task_json" <<'PY'
import json, sys, time
pp, tjson = sys.argv[1], sys.argv[2]
plan = json.load(open(pp))
try:
    task = json.loads(tjson)
except Exception as e:
    print(f"bad json: {e}", file=sys.stderr); sys.exit(2)
task.setdefault("status", "pending")
task.setdefault("depends_on", [])
task.setdefault("attempts", 0)
task.setdefault("notes", "")
if "id" not in task:
    task["id"] = f"task_{int(time.time())}_{__import__('random').randint(1000,9999)}"
plan["tasks"].append(task)
json.dump(plan, open(pp, "w"), indent=2)
print(f"Added task: {task['id']} (agent={task.get('agent_role','?')})")
PY
}

# Add a task from a file: avoids shell-escaping pitfalls.
cmd_add_task_file() {
  local fp="$1"
  [[ -f "$fp" ]] || { err "Not found: $fp"; return 1; }
  python3 -c "import json,sys; json.load(open('$fp'))" >/dev/null \
    || { err "Bad JSON: $fp"; return 1; }
  cmd_add_task "$(cat "$fp")"
}

cmd_ready() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
tasks = plan.get('tasks',[])
done_ids = {t['id'] for t in tasks if t.get('status') in ('complete','sealed','skipped')}
ready = [t for t in tasks if t.get('status') in ('pending','ready') and
         all(d in done_ids for d in t.get('depends_on', []))]
print(f'Ready tasks: {len(ready)}')
for t in ready:
    print(f"  {t['id']:<28} -> {t.get('agent_role','?'):<24} : {t.get('description','')[:80]}")
PY
}

cmd_next_task() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
tasks = plan.get('tasks',[])
done_ids = {t['id'] for t in tasks if t.get('status') in ('complete','sealed','skipped')}
for t in tasks:
    if t.get('status') in ('pending','ready') and \
       all(d in done_ids for d in t.get('depends_on', [])):
        print(json.dumps(t, indent=2)); sys.exit(0)
sys.exit(1)
PY
}

# ----------------------------------------------------------------------------
# Dispatch / mark-done / mark-failed
# ----------------------------------------------------------------------------
cmd_dispatch_ready() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp sp af
  pp=$(plan_path); sp=$(state_path); af=$(audit_path)

  # ------------------------------------------------------------------
  # 0. JIT BRIDGE PRE-PASS (V3.1)
  #    Walk every ready task.  If the local similarity band for the
  #    declared role is "propose" or "block", OR if the role has no
  #    static agent JSON, route through the JIT bridge (native -> web
  #    research) and rewrite the task to the newly-acquired wrapper.
  # ------------------------------------------------------------------
  if [[ -f "$pp" ]] && declare -F cmd_dispatch_jit >/dev/null; then
    local _jit_tid _jit_role _jit_band _jit_best _jit_score
    while IFS=$'\t' read -r _jit_tid _jit_role; do
      [[ -z "$_jit_tid" || -z "$_jit_role" ]] && continue

      # Fast path: if a static agent JSON with this name exists, skip.
      if [[ -f "agents/${_jit_role}.json" ]]; then
        continue
      fi

      # Find the best local agent by computing weighted Jaccard between
      # the role name and every static agent JSON.
      _jit_best=""
      _jit_score=0
      local _a _s
      for _a in agents/*.json; do
        [[ -f "$_a" ]] || continue
        _s=$(agent_similarity_score "$_a" "$_jit_role" 2>/dev/null || echo "0")
        awk -v a="$_s" -v b="$_jit_score" 'BEGIN{exit !(a+0 > b+0)}' && {
          _jit_score="$_s"
          _jit_best="$_a"
        }
      done

      _jit_band=$(agent_similarity_band "$_jit_score" 2>/dev/null || echo "block")

      # Trigger JIT if the band is weak (propose/block) OR if the
      # declared role is genuinely unknown to the local registry.
      local _jit_trigger="no"
      case "$_jit_band" in
        propose|block) _jit_trigger="yes" ;;
      esac
      # If the best match scored below reuse_max and the role name
      # itself isn't a known agent, treat as JIT territory.
      local _reuse_max
      _reuse_max=$(config_get_or_default dynamic_agents.similarity_threshold.reuse_max 0.15)
      awk -v s="$_jit_score" -v r="$_reuse_max" 'BEGIN{exit !(s+0 < r+0)}' \
        && [[ -z "$_jit_best" ]] && _jit_trigger="yes"

      if [[ "$_jit_trigger" == "yes" ]]; then
        local _jit_wrapper
        if _jit_wrapper=$(cmd_dispatch_jit "$_jit_tid" "$_jit_role" 2>/dev/null); then
          [[ -f "$_jit_wrapper" ]] && {
            local _jit_name
            _jit_name=$(basename "$_jit_wrapper" .json)
            plan_set_task_agent "$_jit_tid" "$_jit_name" 2>/dev/null || true
            log_info "JIT rewrote $_jit_tid -> $_jit_name (band=$_jit_band, score=$_jit_score)"
          }
        else
          log_warn "JIT bridge could not resolve $_jit_tid ($_jit_role); leaving as-is"
        fi
      fi
    done < <(jq -r '.tasks[]? | select(.status=="pending" or .status=="ready") | "\(.id)\t\(.agent_role // .resolved_agent // "?")"' "$pp" 2>/dev/null)
  fi

  # ------------------------------------------------------------------
  # 1. Auto-spawn dynamic agents for any task whose role has no
  #    matching entry in config.agents AND no static agent in
  #    $SKILL_SCRIPT_DIR/agents/<role>.json.  Runs once before the
  #    normal dispatch loop.
  # ------------------------------------------------------------------
  if factory_enabled && ! factory_static_only; then
    cmd__factory_autospawn_pending || true
  fi

  python3 - "$pp" "$sp" "$af" <<'PY'
import json, os, sys, time
pp, sp, af = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(pp))
state = json.load(open(sp)) if os.path.exists(sp) else {}
af_handle = open(af, "a") if af and os.path.exists(os.path.dirname(af) or ".") else None

# Reusable handoff-mode rules:
HANDOFF_DEFAULT = "bidirectional"

done_ids = {t['id'] for t in plan.get('tasks',[])
            if t.get('status') in ('complete','sealed','skipped')}
ready = [t for t in plan.get('tasks',[])
         if t.get('status') in ('pending','ready')
         and all(d in done_ids for d in t.get('depends_on', []))]

# Group by agent_role and respect max_parallel cap from config.
config = json.load(open("$CONFIG")) if os.path.exists("$CONFIG") else {}
agents_cfg = config.get("agents", {})

# Cluster tasks by role; within a role, drop any whose dispatch would
# exceed the agent's max_parallel cap for in-flight count.
buckets = {}
for t in ready:
    role = t.get("agent_role", "?")
    buckets.setdefault(role, []).append(t)
dispatched = []
for role, tasks in buckets.items():
    cfg = agents_cfg.get(role, {})
    cap = cfg.get("max_parallel", 1)
    sel = tasks[:cap]  # simplest heuristic: take first `cap`
    for t in sel:
        t["status"] = "dispatched"
        t["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t["attempts"] = t.get("attempts", 0) + 1
        # Default handoff behavior
        t.setdefault("handoff", HANDOFF_DEFAULT)
        # Resolve the agent's underlying primary (alias-friendly)
        primary = cfg.get("primary", role)
        t["resolved_agent"] = primary
        dispatched.append(t["id"])
        if af_handle:
            af_handle.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "dispatch",
                "plan_id": plan["plan_id"],
                "task_id": t["id"],
                "agent_role": role,
                "primary": primary,
                "handoff": t["handoff"],
            }) + "\n")

json.dump(plan, open(pp, "w"), indent=2)

state["plan_id"] = plan["plan_id"]
state["current_pass"] = 2
state["last_checkpoint"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
state.setdefault("dispatched_tasks", [])
for tid in dispatched:
    state["dispatched_tasks"].append(tid)
state["current_executing_task"] = dispatched[0] if dispatched else None
json.dump(state, open(sp, "w"), indent=2)
print(f"Dispatching {len(dispatched)} task(s).")
for t in [t for tt in buckets.values() for t in tt if t['id'] in dispatched]:
    print(f"  -> {t['id']} (role={t.get('agent_role','?')} primary={t.get('resolved_agent','?')})")
if af_handle: af_handle.close()
PY

  # Auto-tick quarantine for every dispatched dynamic agent.  Runs once
  # per dispatch, after the normal Python dispatch loop completes.
  cmd__factory_autotick_dispatched || true
}

# Internal helper: for each just-dispatched task whose resolved_agent
# is a registered dynamic agent, decrement its quarantine_remaining
# counter.  After `default_quarantine` runs the agent is "trusted".
cmd__factory_autotick_dispatched() {
  [[ -f "${ORCH_PLAN_FILE:-}" ]] || return 0
  local pp; pp=$(plan_path)
  [[ -f "$pp" ]] || return 0
  factory_index_ensure

  local ticked=0
  while IFS= read -r role; do
    [[ -z "$role" ]] && continue
    # If the role matches a registered dynamic agent, tick it.
    if jq -e --arg r "$role" '.agents[$r]' "$FACTORY_INDEX_FILE" >/dev/null 2>&1; then
      factory_quarantine_tick "$role" || true
      ticked=$((ticked + 1))
    fi
  done < <(jq -r '.tasks[]? | select(.status=="dispatched") | .resolved_agent // .agent_role' "$pp")

  [[ $ticked -gt 0 ]] && log "factory: auto-ticked quarantine on $ticked dynamic agent(s)"
  return 0
}

# ----------------------------------------------------------------------------
# Internal helper: auto-spawn dynamic agents for any pending task whose role
# has no static agent and no entry in config.agents.  This is the on-the-fly
# hook that closes the loop: "the supervisor creates a new sub-agent right
# there when it arises".
# ----------------------------------------------------------------------------
cmd__factory_autospawn_pending() {
  local pp; pp=$(plan_path)
  [[ -f "$pp" ]] || return 0

  # Pull the unique set of pending task roles from the plan.
  local roles
  roles=$(jq -r '
    .tasks[]? | select(.status=="pending" or .status=="ready") | .agent_role
  ' "$pp" | sort -u | sed '/^$/d')
  [[ -z "$roles" ]] && return 0

  local static_agents_dir="$SKILL_SCRIPT_DIR/agents"
  local spawned=0
  for role in $roles; do
    # Skip if already a dynamic agent (idempotent re-dispatch).
    if factory_index_ensure && jq -e --arg r "$role" '.agents[$r]' "$FACTORY_INDEX_FILE" >/dev/null 2>&1; then
      continue
    fi
    # Skip if a static agent exists for this role.
    [[ -f "$static_agents_dir/$role.json" ]] && continue
    # Skip if config.agents has an entry for this role.
    if [[ -n "${CONFIG:-}" && -f "$CONFIG" ]]; then
      jq -e --arg r "$role" '.agents[$r]' "$CONFIG" >/dev/null 2>&1 && continue
    fi
    # Skip if the role is the literal "?" unknown sentinel.
    [[ "$role" == "?" ]] && continue
    # Cap check.
    if factory_at_cap; then
      warn "factory: at max_dynamic cap; skipping autospawn for role=$role"
      continue
    fi
    # We need to know what the task actually wants so the factory can pick
    # a sensible family.  Pull the first description we can find.
    local gap
    gap=$(jq -r --arg r "$role" '
      .tasks[]? | select(.agent_role==$r) | (.description // .task_description // "")
    ' "$pp" | head -1)
    [[ -z "$gap" ]] && gap="$role"

    local new_name="auto.${role}.$(date +%s)"
    local out
    out=$(factory_synthesize "$new_name" "$gap" "auto-spawn: no static match for role=$role") || {
      warn "factory: auto-spawn failed for role=$role"
      continue
    }
    ok "Auto-spawned dynamic agent: $new_name  (for role=$role)"
    spawned=$((spawned + 1))
  done
  [[ $spawned -gt 0 ]] && ok "Auto-spawned $spawned dynamic agent(s) this pass"
  return 0
}

cmd_mark_done() {
  local task_id="$1" output="${2:-}"
  ensure_plan "$(gen_plan_id)" || return 1
  local pp sp af
  pp=$(plan_path); sp=$(state_path); af=$(audit_path)
  python3 - "$pp" "$sp" "$af" "$task_id" "$output" <<'PY'
import json, os, sys, time
pp, sp, af, tid, output = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
plan = json.load(open(pp))
state = json.load(open(sp)) if os.path.exists(sp) else {}

found = False
for t in plan['tasks']:
    if t['id'] == tid:
        t['status'] = 'complete'
        t['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        if output: t['output_path'] = output
        # Task-level cost/tokens as args (col 3+): t=<tokens> c=<usd> d=<ms>
        # Example: --mark-done <id> <output> --tokens 1234 --cost 0.012 --duration 4500
        found = True
        break
if not found:
    print(f'Task not found: {tid}'); sys.exit(1)

json.dump(plan, open(pp, "w"), indent=2)
state['plan_id'] = plan['plan_id']
state.setdefault('completed_tasks', [])
state['completed_tasks'].append(tid)
state['last_checkpoint'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
state['current_executing_task'] = None
json.dump(state, open(sp, "w"), indent=2)

if af and os.path.exists(os.path.dirname(af) or "."):
    with open(af, "a") as f:
        f.write(json.dumps({
          "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          "event": "mark_done",
          "plan_id": plan['plan_id'],
          "task_id": tid,
          "output_path": output,
        }) + "\n")
print(f"Marked complete: {tid} -> {output}")
PY
}

cmd_mark_failed() {
  local task_id="$1" reason="$2"
  ensure_plan "$(gen_plan_id)" || return 1
  local pp sp af
  pp=$(plan_path); sp=$(state_path); af=$(audit_path)
  python3 - "$pp" "$sp" "$af" "$task_id" "$reason" <<'PY'
import json, os, sys, time
pp, sp, af, tid, reason = sys.argv[1:6]
plan = json.load(open(pp))
state = json.load(open(sp)) if os.path.exists(sp) else {}
found = False
for t in plan['tasks']:
    if t['id'] == tid:
        t['status'] = 'failed'
        t['notes'] = reason
        found = True
        break
if not found:
    print(f'Task not found: {tid}'); sys.exit(1)

json.dump(plan, open(pp, "w"), indent=2)
state['plan_id'] = plan['plan_id']
state.setdefault('failed_tasks', [])
state['failed_tasks'].append(tid)
state['last_checkpoint'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
state['current_executing_task'] = None
json.dump(state, open(sp, "w"), indent=2)

if af and os.path.exists(os.path.dirname(af) or "."):
    with open(af, "a") as f:
        f.write(json.dumps({
          "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          "event": "mark_failed",
          "plan_id": plan['plan_id'],
          "task_id": tid,
          "reason": reason,
        }) + "\n")
print(f'Marked failed: {tid} (reason: {reason})')
PY
}

cmd_escalate() {
  local task_id="$1" target_role="$2"
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" "$task_id" "$target_role" <<'PY'
import json, sys
pp, tid, target = sys.argv[1:4]
plan = json.load(open(pp))
for t in plan['tasks']:
    if t['id'] == tid:
        t['status'] = 'escalated'
        t['notes'] = f"escalated to {target}"
        t['agent_role'] = target
        break
json.dump(plan, open(pp, "w"), indent=2)
PY
}

cmd_reset() {
  local task_id="$1"
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" "$task_id" <<'PY'
import json, sys
pp, tid = sys.argv[1], sys.argv[2]
plan = json.load(open(pp))
for t in plan['tasks']:
    if t['id'] == tid:
        # If plan is sealed, refuse without --unseal first.
        if plan.get('sealed'):
            print('Plan is sealed. Run --unseal first.'); sys.exit(2)
        t['status'] = 'pending'
        t['notes'] = (t.get('notes','') + '|[reset]').strip('|')
        break
json.dump(plan, open(pp, "w"), indent=2)
PY
}

cmd_skip() {
  local task_id="$1" reason="$2"
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" "$task_id" "$reason" <<'PY'
import json, sys
pp, tid, reason = sys.argv[1:4]
plan = json.load(open(pp))
for t in plan['tasks']:
    if t['id'] == tid:
        t['status'] = 'skipped'
        t['notes'] = reason
        break
json.dump(plan, open(pp, "w"), indent=2)
PY
}

# ----------------------------------------------------------------------------
# Validate / aggregate / seal / unseal
# ----------------------------------------------------------------------------
cmd_validate() {
  if ! validate_plan_and_tasks; then return 1; fi
  graph_cycle_check "$(plan_path)" >/dev/null || { err "Dependency cycle detected"; return 1; }
  python3 - "$(plan_path)" "$CONFIG" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
cfg = json.load(open(sys.argv[2]))
agents = set(cfg.get('agents', {}).keys())
errors = []
for t in plan.get('tasks', []):
    if t.get('agent_role') not in agents:
        errors.append(f"Task {t['id']}: unknown agent_role '{t.get('agent_role')}'")
    if t.get('status') not in ("pending","ready","dispatched","running","complete","failed","sealed","skipped","escalated"):
        errors.append(f"Task {t['id']}: invalid status '{t.get('status')}'")
    if not isinstance(t.get('depends_on', []), list):
        errors.append(f"Task {t['id']}: depends_on must be a list")
    for d in t.get('depends_on', []):
        if d not in {x['id'] for x in plan.get('tasks', [])}:
            errors.append(f"Task {t['id']}: depends on missing task '{d}'")
    handoff = t.get('handoff')
    if handoff is not None and handoff not in ("one_way","bidirectional","fan_in","none"):
        errors.append(f"Task {t['id']}: invalid handoff mode '{handoff}'")
if errors:
    print("VALIDATION FAILED:")
    for e in errors: print('  -', e)
    sys.exit(1)
print("Validation OK: plan and all tasks are well-formed.")
PY
}

cmd_aggregate() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  header "Aggregated outputs"
  python3 - "$pp" <<'PY'
import json, os, sys
plan = json.load(open(sys.argv[1]))
tasks = plan.get('tasks', [])
total = len(tasks)
complete = [t for t in tasks if t.get('status') == 'complete']
print(f'Complete: {len(complete)} / {total}')
print()
for t in complete:
    op = t.get('output_path','?')
    exists = '✓' if os.path.exists(op) else '✗'
    cost = t.get('task_cost_usd', 0)
    tokens = t.get('task_tokens_used', 0)
    print(f"  {exists} {t['id']:<28} -> {op:<50} cost=${cost:.4f} tokens={tokens}")
PY
}

cmd_seal() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp sp af
  pp=$(plan_path); sp=$(state_path); af=$(audit_path)

  # Refuse to seal unless all tasks are complete or skipped (unless --force).
  if [[ "${FORCE:-0}" != "1" ]]; then
    python3 - "$pp" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
allowed = ("complete","skipped")
incomplete = [t['id'] for t in plan.get('tasks',[])
              if t.get('status') not in allowed]
if incomplete:
    print(f"Cannot seal: {len(incomplete)} task(s) not complete/skipped:")
    for i in incomplete: print(f"  - {i}")
    print("Pass --force to override.")
    sys.exit(1)
PY
    [[ $? -ne 0 ]] && return 1
  fi

  python3 - "$pp" "$sp" "$af" "$(json_get "$CONFIG" "passes.pass_4_logging_sealing.sealed_marker")" <<'PY'
import json, os, sys, time
pp, sp, af, marker = sys.argv[1:5]
plan = json.load(open(pp))
state = json.load(open(sp)) if os.path.exists(sp) else {}
plan['sealed'] = True
plan['sealed_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

state['sealed_at'] = plan['sealed_at']
state['sealed_marker'] = marker or "[STATUS: ORCHESTRATION COMPLETE.]"
state.setdefault("sealed_unsealed_history", [])
state["sealed_unsealed_history"].append({
  "at": plan["sealed_at"], "by": os.environ.get("USER","?"), "action": "seal",
})
state['current_pass'] = 4
complete = [t['id'] for t in plan.get('tasks',[]) if t.get('status') == 'complete']
state['completed_tasks'] = complete
state['plan_id'] = plan['plan_id']
json.dump(plan, open(pp, "w"), indent=2)
json.dump(state, open(sp, "w"), indent=2)
if af and os.path.exists(os.path.dirname(af) or "."):
    with open(af, "a") as f:
        f.write(json.dumps({"ts":plan['sealed_at'],"event":"seal","plan_id":plan['plan_id']}) + "\n")
print(f'SEALED: {plan["plan_id"]}')
print(f'  complete: {len(complete)}')
print(f'  marker  : {state["sealed_marker"]}')
PY
  notification_fire "sealed"
}

cmd_unseal() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp sp
  pp=$(plan_path); sp=$(state_path)
  python3 - "$pp" "$sp" <<'PY'
import json, sys, os, time
pp, sp = sys.argv[1], sys.argv[2]
plan = json.load(open(pp))
state = json.load(open(sp)) if os.path.exists(sp) else {}
if not plan.get('sealed'):
    print("Plan is not sealed."); sys.exit(0)
plan['sealed'] = False
state.setdefault('sealed_unsealed_history', [])
state['sealed_unsealed_history'].append({
  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "by": os.environ.get("USER","?"),
  "action": "unseal",
})
json.dump(plan, open(pp, "w"), indent=2)
json.dump(state, open(sp, "w"), indent=2)
print("Unsealed plan.")
PY
}

cmd_checkpoint() {
  ensure_state "$(gen_plan_id)" || return 1
  local sp
  sp=$(state_path)
  python3 - "$sp" <<'PY'
import json, time, sys
sp = sys.argv[1]
state = json.load(open(sp)) if __import__('os').path.exists(sp) else {}
state['last_checkpoint'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(state, open(sp, "w"), indent=2)
print("Checkpoint saved.")
PY
}

cmd_recover() {
  local sp; sp=$(state_path)
  if [[ ! -f "$sp" ]]; then warn "No state file."; return 1; fi
  header "Recovery"
  if [[ "${OUTPUT_FORMAT:-pretty}" == "json" ]]; then
    cat "$sp"
  else
    cat "$sp" | python3 -m json.tool
  fi
  echo ""
  echo "Ready to resume. Use --ready to see next tasks."
}

# ----------------------------------------------------------------------------
# Analysis commands
# ----------------------------------------------------------------------------
cmd_topo() { ensure_plan "$(gen_plan_id)" >/dev/null; graph_topological_order "$(plan_path)" | python3 -c "import json,sys; print('\n'.join(json.loads(sys.stdin.read())))"; }
cmd_critical_path() { ensure_plan "$(gen_plan_id)" >/dev/null; graph_critical_path "$(plan_path)" | python3 -c "import json,sys; print('\n'.join(json.loads(sys.stdin.read())))"; }
cmd_stats() { ensure_plan "$(gen_plan_id)" >/dev/null; graph_stats "$(plan_path)" | python3 -m json.tool; }
cmd_dot() { ensure_plan "$(gen_plan_id)" >/dev/null; graph_to_dot "$(plan_path)"; }
cmd_dag() {
  ensure_plan "$(gen_plan_id)" || return 1
  local pp; pp=$(plan_path)
  python3 - "$pp" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
nodes = {t["id"]: t for t in plan.get("tasks",[])}
# Topo order then print levels.
graph = {t["id"]: t.get("depends_on",[]) for t in plan.get("tasks",[])}
levels = {}
def depth(n):
    if n in levels: return levels[n]
    deps = graph.get(n,[])
    levels[n] = 1 + max((depth(d) for d in deps), default=0)
    return levels[n]
for n in nodes:
    depth(n)
order = sorted(levels, key=lambda x: (-levels[x], x))
maxd = max(levels.values()) if levels else 0
buckets = [[] for _ in range(maxd+1)]
for n in order:
    buckets[levels[n]].append(n)
for i, layer in enumerate(buckets):
    if not layer: continue
    print(f"L{i}: " + " | ".join(f"[{nodes[n].get('status','?')[:1]}] {n}" for n in layer))
PY
}

cmd_diff() { plan_diff "${1:-}" "${2:-}"; }

# ----------------------------------------------------------------------------
# Doctor
# ----------------------------------------------------------------------------
cmd_doctor() {
  header "Doctor — environment checks"
  local ok=0 bad=0
  for bin in bash python3 date flock; do
    if command -v "$bin" >/dev/null; then
      echo "  - $bin: OK ($(command -v "$bin"))"
      ok=$((ok+1))
    else
      warn "$bin: NOT FOUND"
      bad=$((bad+1))
    fi
  done
  echo ""
  echo "Config:"
  echo "  - $CONFIG"
  if [[ -f "$CONFIG" ]]; then echo "    exists: OK"; else echo "    missing!"; fi
  echo ""
  echo "Plan / State:"
  local pp sp
  pp=$(plan_path); sp=$(state_path)
  echo "  - plan:  $pp"
  echo "  - state: $sp"
  echo ""
  if (( bad == 0 )); then ok "$((ok)) check(s) passed"; else warn "$((bad)) check(s) failed"; fi
}

# ----------------------------------------------------------------------------
# Quality gate / judge
# ----------------------------------------------------------------------------
cmd_quality_gate() {
  local tid="$1"
  local out; out="$(quality_gate "$tid")"
  echo "$out" | python3 -m json.tool
  local score; score=$(echo "$out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read()).get('score') or 1.0)")
  local ok; ok=$(echo "$out" | python3 -c "import json,sys;print('1' if json.loads(sys.stdin.read()).get('ok') else '0')")
  if [[ "$ok" == "1" ]]; then
    quality_apply "$tid" "$score" "automated gate passed"
  fi
}

cmd_judge() {
  local tid="$1" score="$2" reason="$3"
  quality_apply "$tid" "$score" "$reason"
}

# ----------------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------------
cmd_watch() {
  [[ "${1:-}" == "--json" ]] && OBS_FORMAT=json
  obs_watch
}
cmd_audit() { obs_audit_show; }
cmd_cost_report() { obs_cost_report; }
cmd_trace() { obs_trace_for_task "$1"; }

# ----------------------------------------------------------------------------
# Subplans
# ----------------------------------------------------------------------------
cmd_spawn_subplan() {
  local parent_id="$1" sub_goal="$2"
  subplan_spawn "$parent_id" "$sub_goal" "[]"
}
cmd_list_subplans() { subplan_walk; }

# ----------------------------------------------------------------------------
# Templates / Import / Export / Replay
# ----------------------------------------------------------------------------
cmd_import_template() {
  cmd_apply_template "$@"
}
cmd_apply_template() {
  local name="$1"; shift
  templates_apply "$name" "$@"
}
cmd_list_templates() { templates_list; }
cmd_export() { plan_export "$1"; }
cmd_import() { plan_import "$1"; }
cmd_replay() { plan_replay; }

# ----------------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------------
cmd_memory_set() { memory_set "$1" "$2" "$3"; }
cmd_memory_get() { memory_get "$1" "$2" | python3 -m json.tool 2>/dev/null || memory_get "$1" "$2"; }
cmd_memory_list() { memory_list "$1"; }
cmd_memory_delete() { memory_delete "$1" "$2"; }
cmd_memory_snippet() { memory_snippet_set "$1" "$2"; }

# ----------------------------------------------------------------------------
# Dynamic routing
# ----------------------------------------------------------------------------
cmd_auto_plan() {
  local goal="$1"; shift
  local model="${1:-${ORCHESTRATOR_MODEL:-gpt-4o-mini}}"
  agent_dynamic_route "$goal" "$model"
}

# ----------------------------------------------------------------------------
# Notifications
# ----------------------------------------------------------------------------
cmd_notify() { notification_fire "${1:-event}"; }

# ----------------------------------------------------------------------------
# Resource management
# ----------------------------------------------------------------------------
cmd_record_load() {
  local role="$1" pct="$2"
  local base; base="$(config_dir)"
  local f="$base/agent_load.json"
  ensure_parent "$f"
  python3 - "$f" "$role" "$pct" <<'PY'
import json, os, sys, time
f, role, pct = sys.argv[1], sys.argv[2], float(sys.argv[3])
data = {}
if os.path.exists(f): data = json.load(open(f))
data[role] = {"load_pct": pct, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(data, open(f, "w"), indent=2)
PY
  ok "Recorded load for $role = $pct%"
}
cmd_agent_load() {
  local base; base="$(config_dir)"
  local f="$base/agent_load.json"
  [[ -f "$f" ]] || { warn "No load data"; return 0; }
  python3 -m json.tool "$f"
}

# ----------------------------------------------------------------------------
# Dynamic Agent Factory
# ----------------------------------------------------------------------------
#
# These subcommands let you spawn, list, inspect, age, and remove
# sub-agents on the fly.  They map 1-to-1 onto the lib/dynamic_agents.sh
# API.  Every spawn is audited to <config_dir>/factory_audit.jsonl.
#
# Examples:
#   skill --agents-factory-create        auto.coder.go coder.python "needed a Go agent"
#   skill --agents-factory-persona       auto.coder.zen coder.python writer "calmer voice"
#   skill --agents-factory-synthesize    auto.ops.k8s "deploy a Kubernetes manifest"
#   skill --agents-factory-list
#   skill --agents-factory-show          auto.coder.go
#   skill --agents-factory-tick          auto.coder.go
#   skill --agents-factory-remove        auto.coder.go
#
cmd_agents_factory_create() {
  local new_name="${1:?usage: --agents-factory-create <new_name> <parent> [reason]}"
  local parent="${2:?usage: --agents-factory-create <new_name> <parent> [reason]}"
  local reason="${3:-on-demand}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false in $CONFIG"
    err "       set 'dynamic_agents.enabled=true' in your config to enable"
    return 2
  fi
  if factory_static_only; then
    err "factory: dynamic_agents.static_only=true; refusing to spawn"
    return 2
  fi
  # Allow the caller to pass either a bare agent name (resolved against
  # the static agents/ dir) or an absolute / relative path.
  local parent_path
  if [[ -f "$parent" ]]; then
    parent_path="$parent"
  elif [[ -f "$SKILL_SCRIPT_DIR/agents/$parent.json" ]]; then
    parent_path="$SKILL_SCRIPT_DIR/agents/$parent.json"
  else
    err "factory: parent agent not found: $parent (looked in $parent and $SKILL_SCRIPT_DIR/agents/)"
    return 1
  fi
  local out
  out=$(factory_create_clone "$new_name" "$parent_path" "$reason") || return $?
  ok "Created dynamic agent: $new_name"
  echo "$out"
}

cmd_agents_factory_persona() {
  local new_name="${1:?usage: --agents-factory-persona <new_name> <base> <persona> [reason]}"
  local base="${2:?usage: --agents-factory-persona <new_name> <base> <persona> [reason]}"
  local persona="${3:?usage: --agents-factory-persona <new_name> <base> <persona> [reason]}"
  local reason="${4:-persona-shift}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false in $CONFIG"; return 2
  fi
  local base_path
  if [[ -f "$base" ]]; then
    base_path="$base"
  elif [[ -f "$SKILL_SCRIPT_DIR/agents/$base.json" ]]; then
    base_path="$SKILL_SCRIPT_DIR/agents/$base.json"
  else
    err "factory: base agent not found: $base"; return 1
  fi
  local out
  out=$(factory_create_persona "$new_name" "$base_path" "$persona" "$reason") || return $?
  ok "Created persona variant: $new_name (voice=$persona)"
  echo "$out"
}

cmd_agents_factory_synthesize() {
  local new_name="${1:?usage: --agents-factory-synthesize <new_name> <gap_description> [reason]}"
  local gap="${2:?usage: --agents-factory-synthesize <new_name> <gap_description> [reason]}"
  local reason="${3:-capability-gap}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false in $CONFIG"; return 2
  fi
  local out
  out=$(factory_synthesize "$new_name" "$gap" "$reason") || return $?
  ok "Synthesized dynamic agent: $new_name"
  echo "$out"
}

cmd_agents_factory_list() {
  if ! factory_enabled; then
    warn "factory: dynamic_agents.enabled=false; nothing to show"
    return 0
  fi
  factory_print_dynamic_table
}

cmd_agents_factory_show() {
  local name="${1:?usage: --agents-factory-show <name>}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false"; return 2
  fi
  factory_index_ensure
  jq --arg n "$name" '.agents[$n]' "$FACTORY_INDEX_FILE"
}

cmd_agents_factory_tick() {
  local name="${1:?usage: --agents-factory-tick <name>}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false"; return 2
  fi
  factory_quarantine_tick "$name"
  local rem; rem=$(factory_quarantine_remaining "$name")
  ok "Quarantine age-out for $name; remaining=$rem"
}

cmd_agents_factory_remove() {
  local name="${1:?usage: --agents-factory-remove <name>}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false"; return 2
  fi
  factory_index_ensure
  local path; path=$(jq -r --arg n "$name" '.agents[$n].path // empty' "$FACTORY_INDEX_FILE")
  if [[ -z "$path" ]]; then
    err "factory: $name is not a registered dynamic agent"
    return 1
  fi
  rm -f "$path" && factory_index_remove "$name" >/dev/null
  ok "Removed dynamic agent: $name"
}

# Default action
cmd_default() { cmd_help; }
# Block 3 — lib/commands.sh additions
# Append everything below to your existing lib/commands.sh
# ----------------------------------------------------------------------------

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
    "$SKILL_SCRIPT_DIR/agents"
  )
  local discovered=()
  local seen_names=()

  for src in "${sources[@]}"; do
    [[ -d "$src" ]] || continue
    for f in "$src"/*.json; do
      [[ -f "$f" ]] || continue
      local parsed
      parsed=$(jq -c '
        {
          name: .name,
          version: .version,
          kind: .kind,
          entry: .entry,
          source: input_filename,
          description: (.description // ""),
          deprecated: (.deprecated // false),
          deprecation_message: (.deprecation_message // null),
          invariants_compliant: (.invariants_compliant // []),
          invariants_provisional: (.invariants_provisional // []),
          composes_with: (.composes_with // []),
          capabilities: {
            reads: (.reads // []),
            writes: (.writes // []),
            resources: (.resources // {})
          }
        }
      ' "$f" 2>/dev/null) || {
        log_warn "Skipping invalid agent JSON: $f"
        continue
      }

      local name deprecated
      name=$(echo "$parsed" | jq -r '.name')
      deprecated=$(echo "$parsed" | jq -r '.deprecated')

      if printf '%s\n' "${seen_names[@]}" 2>/dev/null | grep -qx "$name"; then
        log_warn "Duplicate agent '$name' — keeping higher-precedence copy"
        continue
      fi

      if [[ "$deprecated" == "true" && "$include_deprecated" != "true" ]]; then
        seen_names+=("$name")
        continue
      fi

      seen_names+=("$name")
      discovered+=("$parsed")
    done
  done

  if [[ "$output_json" == "true" ]]; then
    printf '%s\n' "${discovered[@]}" | jq -s '.'
  else
    printf '%-40s %-10s %-12s %-10s %s\n' "NAME" "KIND" "VERSION" "DEPRECATED" "SOURCE"
    printf '%-40s %-10s %-12s %-10s %s\n' "----" "----" "-------" "----------" "------"
    for d in "${discovered[@]}"; do
      printf '%-40s %-10s %-12s %-10s %s\n' \
        "$(echo "$d" | jq -r '.name')" \
        "$(echo "$d" | jq -r '.kind')" \
        "$(echo "$d" | jq -r '.version')" \
        "$(echo "$d" | jq -r '.deprecated')" \
        "$(echo "$d" | jq -r '.source')"
    done
    echo
    echo "Total: ${#discovered[@]} active agent(s) across ${#sources[@]} source(s)"
    local hidden
    hidden=$(printf '%s\n' "${seen_names[@]}" "${discovered[@]}" 2>/dev/null | sort -u | wc -l)
    if [[ "$hidden" -gt "${#discovered[@]}" ]]; then
      echo "Hidden: $((hidden - ${#discovered[@]})) deprecated agent(s) — use --include-deprecated to show"
    fi
  fi
}

# ----------------------------------------------------------------------------
# Similarity scoring — weighted Jaccard over schema/description/capabilities
# ----------------------------------------------------------------------------
agent_similarity_score() {
  local a_name="$1"  # path to agent A's JSON
  local b_text="$2"  # task description text (or path to B if b_is_file=true)
  local b_is_file="${3:-false}"

  # Load field weights from config (with sane defaults)
  local w_schema w_desc w_caps
  w_schema=$(config_get_or_default dynamic_agents.weights.schema 0.5)
  w_desc=$(config_get_or_default  dynamic_agents.weights.description 0.25)
  w_caps=$(config_get_or_default  dynamic_agents.weights.capabilities 0.25)

  local a_text
  a_text=$(jq -r '
    "SCHEMA " + (.input_schema.required // [] | tostring) + " " +
    "SCHEMA " + (.output_schema.required // [] | tostring) + " " +
    "DESC " + (.description // "") + " " +
    "CAPS " + ((.writes // []) | tostring) + " " +
    "CAPS " + ((.reads // []) | tostring)
  ' "$a_name" 2>/dev/null)

  local a_schema a_desc a_caps
  a_schema=$(echo "$a_text" | grep -oP 'SCHEMA \K.*' | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)
  a_desc=$(echo "$a_text" | grep -oP 'DESC \K.*' | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)
  a_caps=$(echo "$a_text" | grep -oP 'CAPS \K.*' | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)

  local b_schema b_desc b_caps
  if [[ "$b_is_file" == "true" ]]; then
    b_schema=$(jq -r '(.input_schema.required // [] | tostring) + " " + (.output_schema.required // [] | tostring)' "$b_text" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)
    b_desc=$(jq -r '.description // ""' "$b_text" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)
    b_caps=$(jq -r '((.writes // []) | tostring) + " " + ((.reads // []) | tostring)' "$b_text" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)
  else
    b_schema=$(echo "$b_text" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | sort -u)
    b_desc="$b_schema"
    b_caps="$b_schema"
  fi

  _weighted_jaccard() {
    local set_a="$1" set_b="$2" weight="$3"
    if [[ -z "$set_a" || -z "$set_b" ]]; then echo "0.000"; return; fi
    local inter union
    inter=$(comm -12 <(echo "$set_a") <(echo "$set_b") 2>/dev/null | wc -l)
    union=$(cat <(echo "$set_a") <(echo "$set_b") | sort -u | wc -l)
    if [[ "$union" -eq 0 ]]; then echo "0.000"; return; fi
    awk -v i="$inter" -v u="$union" -v w="$weight" 'BEGIN{printf "%.3f\n", (i/u)*w}'
  }

  local s_schema s_desc s_caps
  s_schema=$(_weighted_jaccard "$a_schema" "$b_schema" "$w_schema")
  s_desc=$(_weighted_jaccard   "$a_desc"   "$b_desc"   "$w_desc")
  s_caps=$(_weighted_jaccard   "$a_caps"   "$b_caps"   "$w_caps")

  awk -v s="$s_schema" -v d="$s_desc" -v c="$s_caps" 'BEGIN{printf "%.3f\n", s+d+c}'
}

# Returns: reuse | reuse_log | propose | block
agent_similarity_band() {
  local score="$1"
  local rmax rlmax pmax
  rmax=$(config_get_or_default  dynamic_agents.similarity_threshold.reuse_max 0.15)
  rlmax=$(config_get_or_default dynamic_agents.similarity_threshold.reuse_log_max 0.40)
  pmax=$(config_get_or_default  dynamic_agents.similarity_threshold.propose_max 0.70)
  awk -v s="$score" -v r="$rmax" -v rl="$rlmax" -v p="$pmax" 'BEGIN{
    if (s < r)       { print "reuse" }
    else if (s < rl) { print "reuse_log" }
    else if (s < p)  { print "propose" }
    else             { print "block" }
  }'
}

# ----------------------------------------------------------------------------
# Lint agents against the 8 invariants
# ----------------------------------------------------------------------------
cmd_agents_lint() {
  local target="${1:-}"
  local files=()
  local total=0 passed=0 failed=0

  if [[ -z "$target" || "$target" == "--all" ]]; then
    while IFS= read -r f; do
      files+=("$f")
    done < <(find "${ORCH_PROJECT:-$PWD}/agents" \
                  "${ORCH_HOME:-$HOME/.orchestrator}/agents" \
                  "/etc/orchestrator/agents" \
                  "$SKILL_SCRIPT_DIR/agents" \
                  -name '*.json' 2>/dev/null | sort -u)
  elif [[ -f "$target" ]]; then
    files=("$target")
  else
    while IFS= read -r f; do files+=("$f"); done < <(agent_resolve_name "$target")
  fi

  for f in "${files[@]}"; do
    total=$((total + 1))
    local errors=()

    # I5: Sealed envelope
    jq -e '.name and .version and .kind and .entry and .input_schema and .output_schema and .writes and .reads' "$f" >/dev/null \
      || errors+=("I5: missing required envelope fields")

    # I3/I4: writes/reads must be arrays
    jq -e '.writes | type == "array"' "$f" >/dev/null || errors+=("I3: writes is not an array")
    jq -e '.reads  | type == "array"' "$f" >/dev/null || errors+=("I4: reads is not an array")

    # I2: resources object present
    jq -e '.resources | type == "object"' "$f" >/dev/null || errors+=("I2: resources missing")

    # I8: invariants_compliant declared
    jq -e '.invariants_compliant | type == "array" and length > 0' "$f" >/dev/null \
      || errors+=("I8: invariants_compliant not declared")

    # Bonus: deprecation consistency
    local dep msg
    dep=$(jq -r '.deprecated // false' "$f")
    msg=$(jq -r '.deprecation_message // ""' "$f")
    if [[ "$dep" == "true" && -z "$msg" ]]; then
      errors+=("WARN: deprecated=true without deprecation_message")
    fi

    if [[ ${#errors[@]} -eq 0 ]]; then
      passed=$((passed + 1))
      printf '  PASS  %s\n' "$f"
    else
      failed=$((failed + 1))
      printf '  FAIL  %s\n' "$f"
      for e in "${errors[@]}"; do printf '        - %s\n' "$e"; done
    fi
  done

  echo
  echo "Linted $total agent(s): $passed passed, $failed failed"
  return $(( failed > 0 ? 1 : 0 ))
}

# ----------------------------------------------------------------------------
# Render the cross-agent composition graph
# ----------------------------------------------------------------------------
cmd_agents_graph() {
  local format="${1:-ascii}"
  local nodes=()

  case "$format" in
    dot)
      echo "digraph agents {"
      echo "  rankdir=LR;"
      echo "  node [shape=box, style=rounded];"
      while IFS= read -r f; do
        local name kind
        name=$(jq -r '.name' "$f")
        kind=$(jq -r '.kind' "$f")
        printf '  "%s" [label="%s\\n(%s)"];\n' "$name" "$name" "$kind"
        nodes+=("$name")
        jq -r '.composes_with // [] | .[]' "$f" | while read -r target; do
          printf '  "%s" -> "%s";\n' "$name" "$target"
        done
      done < <(find "${ORCH_PROJECT:-$PWD}/agents" "$SKILL_SCRIPT_DIR/agents" -name '*.json' 2>/dev/null | sort -u)
      echo "}"
      ;;
    ascii|*)
      echo "Agent Composition Graph"
      echo "======================="
      while IFS= read -r f; do
        local name children
        name=$(jq -r '.name' "$f")
        children=$(jq -r '.composes_with // [] | join(", ")' "$f")
        [[ -n "$children" ]] && printf '%-32s -> %s\n' "$name" "$children"
      done < <(find "${ORCH_PROJECT:-$PWD}/agents" "$SKILL_SCRIPT_DIR/agents" -name '*.json' 2>/dev/null | sort -u)
      ;;
  esac
}

# ----------------------------------------------------------------------------
# JSON diff between a dynamic agent and its parent
# ----------------------------------------------------------------------------
cmd_agents_factory_diff() {
  local dynamic_name="$1"
  local dynamic_path
  dynamic_path=$(agent_resolve_name "$dynamic_name")
  [[ -f "$dynamic_path" ]] || { err "Dynamic agent not found: $dynamic_name"; return 1; }

  local parent
  parent=$(jq -r '.spawned_from // empty' "$dynamic_path")
  [[ -n "$parent" ]] || { err "Agent $dynamic_name has no parent (not a dynamic agent?)"; return 1; }

  local parent_path
  parent_path=$(agent_resolve_name "$parent")

  diff --color=always -u \
    <(jq -S . "$parent_path") \
    <(jq -S . "$dynamic_path") \
    | head -200
}

# ----------------------------------------------------------------------------
# Auto-tick quarantine counter after every dynamic-agent invocation
# ----------------------------------------------------------------------------
agent_maybe_auto_tick() {
  local dynamic_agent_name="$1"
  [[ "$(config_get dynamic_agents.auto_tick_quarantine)" == "true" ]] || return 0
  cmd_agents_factory_tick "$dynamic_agent_name" >/dev/null
}

# ----------------------------------------------------------------------------
# Quarantine promotion — the lint-or-orphan gate
# Called when quarantine_remaining reaches zero. Replaces the simple
# counter-decrement that was previously the only behavior.
# ----------------------------------------------------------------------------
agent_quarantine_promote() {
  local dynamic_agent_name="$1"
  local dynamic_path
  dynamic_path=$(agent_resolve_name "$dynamic_agent_name")
  [[ -f "$dynamic_path" ]] || return 1

  log_info "Quarantine expired for $dynamic_agent_name — running mandatory lint"

  # Run lint and capture exit code
  if cmd_agents_lint "$dynamic_path" >/dev/null 2>&1; then
    log_info "Lint passed — promoting $dynamic_agent_name to trusted status"
    jq '. + {quarantine_status: "trusted", promoted_at: now | todate}' \
       "$dynamic_path" > "$dynamic_path.tmp" && mv "$dynamic_path.tmp" "$dynamic_path"
    audit_log_event "agent_promoted" "{\"agent\":\"$dynamic_agent_name\"}"
    return 0
  else
    log_warn "Lint FAILED — orphaning $dynamic_agent_name"
    local orphan_dir
    orphan_dir=$(config_get_or_default dynamic_agents.orphan_dir "agents/orphaned/")
    mkdir -p "$orphan_dir"
    mv "$dynamic_path" "$orphan_dir/$(basename "$dynamic_path")"
    audit_log_event "agent_orphaned" "{\"agent\":\"$dynamic_agent_name\",\"reason\":\"failed_lint_on_quarantine_expiry\"}"
    return 1
  fi
}

# ----------------------------------------------------------------------------
# Replay a past agent invocation step-by-step
# ----------------------------------------------------------------------------
cmd_agents_trace() {
  local run_id="$1"
  [[ -n "$run_id" ]] || { err "Usage: skill --agents-trace <run_id>"; return 1; }

  local audit_file
  audit_file=$(config_get paths.audit_file)
  [[ -f "$audit_file" ]] || { err "Audit file not found: $audit_file"; return 1; }

  echo "Trace for run: $run_id"
  echo "Source: $audit_file"
  echo "=============================================="
  jq -c --arg rid "$run_id" 'select(.run_id == $rid)' "$audit_file" \
    | jq -r '
        "[\(.ts // "?")] \(.event // "?") -- \(.agent // .task // "-")\n  \(.details // {} | tostring)"
      '
}

# ----------------------------------------------------------------------------
# Helper: resolve a named agent to its file path across all sources
# ----------------------------------------------------------------------------
agent_resolve_name() {
  local target_name="$1"
  local sources=(
    "${ORCH_PROJECT:-$PWD}/agents"
    "${ORCH_HOME:-$HOME/.orchestrator}/agents"
    "/etc/orchestrator/agents"
    "$SKILL_SCRIPT_DIR/agents"
  )
  for src in "${sources[@]}"; do
    [[ -d "$src" ]] || continue
    local found
    found=$(find "$src" -maxdepth 1 -name '*.json' -exec jq -r --arg n "$target_name" \
      'select(.name == $n) | input_filename' {} \; 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return 0; }
  done
  return 1
}

# ----------------------------------------------------------------------------
# Helper: config_get_or_default (fallback when key missing)
# ----------------------------------------------------------------------------
config_get_or_default() {
  local key="$1"
  local default="$2"
  local val
  val=$(config_get "$key" 2>/dev/null)
  if [[ -z "$val" || "$val" == "null" ]]; then
    echo "$default"
  else
    echo "$val"
  fi
}
