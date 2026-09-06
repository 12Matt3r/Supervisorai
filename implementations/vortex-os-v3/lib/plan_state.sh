# lib/plan_state.sh — unified, safe, locked CRUD on plan.json and state.json
# shellcheck shell=bash

[[ -n "${__LIB_PLAN_LOADED:-}" ]] && return 0
__LIB_PLAN_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/locking.sh"
source "$SKILL_SCRIPT_DIR/lib/schema.sh"
source "$SKILL_SCRIPT_DIR/lib/graph.sh"

PLAN_SCHEMA="${PLAN_SCHEMA:-$SKILL_SCRIPT_DIR/schemas/plan.schema.json}"
STATE_SCHEMA="${STATE_SCHEMA:-$SKILL_SCRIPT_DIR/schemas/state.schema.json}"

# Ensure plan and state files exist; create a valid skeleton if not.
ensure_plan() {
  ensure_file_with "$1" "$(plan_path)" _plan_skeleton
}
ensure_state() {
  ensure_file_with "$1" "$(state_path)" _state_skeleton
}

# Ensure file $2 exists; if not, write the output of $3 (with $1 = plan_id).
ensure_file_with() {
  local pid="$1" file="$2" skeleton_fn="$3"
  if [[ -f "$file" ]]; then return 0; fi
  ensure_parent "$file" || return 1
  "$skeleton_fn" "$file" "$pid"
}

_plan_skeleton() {
  local file="$1" pid="$2"
  cat > "$file" <<EOF
{
  "plan_id": "$pid",
  "goal": "Untitled plan",
  "created_at": "$(now_iso)",
  "tasks": []
}
EOF
  ok "Created new plan: $file (id=$pid)"
}

_state_skeleton() {
  local file="$1" pid="$2"
  cat > "$file" <<EOF
{
  "plan_id": "$pid",
  "current_pass": 0,
  "dispatched_tasks": [],
  "completed_tasks": [],
  "failed_tasks": []
}
EOF
}

# Read plan as JSON, mutate it via a Python expression, write it back, all
# under a lock. The first arg is the Python code to run, with `plan` bound.
# Usage: with_locked_plan "plan['goal'] = 'X'; ..."
with_locked_plan() {
  local pp; pp=$(plan_path)
  ensure_plan "$(gen_plan_id)" || return 1
  local lock; lock="$(lock_acquire "$pp")" || return 1
  python3 - "$pp" <<PY
import json,sys
try:
    p = json.load(open("$pp"))
except Exception as e:
    sys.stderr.write("Cannot read plan: %s\n" % e); sys.exit(2)
$1
json.dump(p, open("$pp","w"), indent=2)
PY
  local rc=$?
  lock_release "$lock"
  return $rc
}

# Same for state.
with_locked_state() {
  local sp; sp=$(state_path)
  ensure_state "$(gen_plan_id)" || return 1
  local lock; lock="$(lock_acquire "$sp")" || return 1
  python3 - "$sp" <<PY
import json,sys
try:
    s = json.load(open("$sp"))
except Exception:
    s = {}
$1
json.dump(s, open("$sp","w"), indent=2)
PY
  local rc=$?
  lock_release "$lock"
  return $rc
}

# Audit append (JSONL).  Each entry: {ts, event, plan_id, task_id, ...extra}
audit_log() {
  local event="$1"; shift
  local af; af="$(audit_path)"
  [[ -z "$af" ]] && return 0
  ensure_parent "$af"
  python3 - "$af" "$event" "$@" <<'PY'
import json, sys, time
af = sys.argv[1]
entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "event": sys.argv[2]}
for k, v in zip(sys.argv[3::2], sys.argv[4::2]):
    entry[k] = v
with open(af, "a") as f:
    f.write(json.dumps(entry) + "\n")
PY
}

# validate_plan_and_tasks: returns 0 ok, 1 invalid.
validate_plan_and_tasks() {
  local pp; pp=$(plan_path)
  [[ -f "$pp" ]] || { err "No plan file at $pp"; return 1; }
  if [[ -f "$PLAN_SCHEMA" ]]; then
    if ! validate_json_file "$pp" "$PLAN_SCHEMA" >/dev/null; then
      validate_json_file "$pp" "$PLAN_SCHEMA"
      return 1
    fi
  fi
  graph_validate_task_shape "$pp" | python3 -c "
import json,sys
o=json.loads(sys.stdin.read())
print('Validation OK.' if o['ok'] else 'VALIDATION FAILED:')
for e in o['errors']: print('  -', e)
sys.exit(0 if o['ok'] else 1)
"
}

# Find an agent role in the config by name.
agent_role_config() {
  local role="$1"
  json_get "$CONFIG" "agents.$role.primary"
  return 0
}

# Returns a stamp string usable for filename suffixes.
run_stamp() { date -u +"%Y%m%dT%H%M%SZ"; }
