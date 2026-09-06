# lib/subplan.sh — hierarchical orchestration (supervisor-of-supervisors)
# shellcheck shell=bash

[[ -n "${__LIB_SUBPLAN_LOADED:-}" ]] && return 0
__LIB_SUBPLAN_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"

# Spawn a sub-plan from a parent task.
#
# Usage: subplan_spawn PARENT_TASK_ID SUB_GOAL [TASKS_JSON]
subplan_spawn() {
  local parent_task="$1" sub_goal="$2" tasks_json="${3:-[]}"

  local pp; pp=$(plan_path)
  ensure_plan "$(gen_plan_id)" || return 1

  # Allocate a sub-plan id and compute the sub-plan path.
  local sub_id; sub_id="sub_$(gen_plan_id)"
  local base; base="$(config_dir)"
  local sub_file="$base/${sub_id}.plan.json"

  python3 - "$pp" "$sub_file" "$sub_id" "$sub_goal" "$tasks_json" "$parent_task" <<'PY'
import json, os, sys
parent_path, sub_path, sub_id, sub_goal, tasks_json, parent_task = sys.argv[1:7]
parent = json.load(open(parent_path))
tasks = json.loads(tasks_json)
# Make all sub-task ids qualified with the sub_id so they're globally unique.
qualified = []
for t in tasks:
    t = dict(t)
    t["id"] = f"{sub_id}:{t['id']}"
    qualified.append(t)
sub_plan = {
  "plan_id": sub_id,
  "parent_plan_id": parent["plan_id"],
  "goal": sub_goal,
  "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
  "tasks": qualified,
}
# On the parent task, mark it pending a sub-plan and add depends_on to it.
parent["sub_plan_ids"] = list(parent.get("sub_plan_ids") or []) + [sub_id]
for t in parent["tasks"]:
    if t["id"] == parent_task:
        t["sub_plan_id"] = sub_id
        # After the sub-plan completes, the parent task flips to complete.
        # Until then, the parent task waits.
        t["status"] = "pending"
        t["notes"] = "delegated to sub-plan: " + sub_id
json.dump(sub_plan, open(sub_path, "w"), indent=2)
json.dump(parent, open(parent_path, "w"), indent=2)
print(sub_id)
PY
}

# Mark a sub-plan as complete; flips the parent task to complete.
subplan_complete() {
  local sub_id="$1"
  local base; base="$(config_dir)"
  local sub_file="$base/${sub_id}.plan.json"
  [[ -f "$sub_file" ]] || { err "No sub-plan at $sub_file"; return 1; }

  python3 - "$sub_file" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
tasks = plan.get("tasks", [])
for t in tasks:
    t["status"] = t.get("status","pending")
incomplete = [t["id"] for t in tasks if t.get("status") not in ("complete","sealed","skipped")]
if incomplete:
    print(json.dumps({"ok": False, "incomplete": incomplete})); sys.exit(1)
plan["sealed"] = True
plan["sealed_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
json.dump(plan, open(sys.argv[1], "w"), indent=2)
print(json.dumps({"ok": True, "id": plan["plan_id"]}))
PY
}

subplan_walk() {
  # Print a tree of the plan + its sub-plans.
  python3 - <<PY
import json, os
base = "$(config_dir)"
plan = json.load(open("$(plan_path)"))
def walk(plan_path, depth=0):
    indent = "  " * depth
    print(f"{indent}- {plan_path['plan_id']}: {plan_path.get('goal','')[:60]}")
    for s in plan_path.get("sub_plan_ids") or []:
        sp = os.path.join(base, s + ".plan.json")
        if os.path.exists(sp):
            walk(json.load(open(sp)), depth + 1)
walk(plan)
PY
}
