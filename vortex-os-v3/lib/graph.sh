# lib/graph.sh — dependency graph operations on plan.json
# shellcheck shell=bash

[[ -n "${__LIB_GRAPH_LOADED:-}" ]] && return 0
__LIB_GRAPH_LOADED=1

# All operations are pure: they read plan.json and emit JSON to stdout.
# No side effects. Callers wrap with_lock if needed.

graph_cycle_check() {
  # Echo "CYCLE" or "OK". Exit 1 if cycle.
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
graph = {t["id"]: t.get("depends_on", []) for t in plan.get("tasks", [])}
WHITE, GRAY, BLACK = 0, 1, 2
color = {n: WHITE for n in graph}
def dfs(n):
    color[n] = GRAY
    for m in graph.get(n, []):
        c = color.get(m, WHITE)
        if c == GRAY: return True
        if c == WHITE and dfs(m): return True
    color[n] = BLACK
    return False
for n in list(graph):
    if color[n] == WHITE and dfs(n):
        print("CYCLE"); sys.exit(1)
print("OK")
PY
}

graph_ready_set() {
  # Emit IDs of tasks whose deps are all met and status == pending.
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
done = {t["id"] for t in plan.get("tasks", []) if t.get("status") in ("complete","sealed","skipped")}
ready = [t["id"] for t in plan.get("tasks", [])
         if t.get("status") == "pending"
         and all(d in done for d in t.get("depends_on", []))]
print(json.dumps(ready))
PY
}

graph_topological_order() {
  # Emit a stable topological order of all task IDs (by deps).
  # If a cycle exists, exit 2.
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
graph = {t["id"]: t.get("depends_on", []) for t in plan.get("tasks", [])}
WHITE, GRAY, BLACK = 0, 1, 2
color = {n: WHITE for n in graph}
order = []
def visit(n):
    if color.get(n, WHITE) == BLACK: return
    if color.get(n, WHITE) == GRAY:
        sys.stderr.write("CYCLE\n"); sys.exit(2)
    color[n] = GRAY
    for m in graph.get(n, []):
        visit(m)
    color[n] = BLACK
    order.append(n)
# Visit in insertion order to keep output stable
for t in plan.get("tasks", []):
    visit(t["id"])
print(json.dumps(order))
PY
}

graph_critical_path() {
  # Echo the longest dependency chain (by node count).
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
graph = {t["id"]: t.get("depends_on", []) for t in plan.get("tasks", [])}
memo = {}
def longest(n):
    if n in memo: return memo[n]
    best = 1
    for m in graph.get(n, []):
        best = max(best, longest(m) + 1)
    memo[n] = best
    return best
ends = max(graph, key=lambda x: longest(x)) if graph else None
def chain(n):
    c = [n]
    for m in graph.get(n, []):
        if longest(m) + 1 == longest(n):
            c = chain(m) + [n]; break
    return c
print(json.dumps(chain(ends) if ends else []))
PY
}

graph_to_dot() {
  # Emit Graphviz DOT for `dot -Tpng > plan.png`.
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
print("digraph plan {")
print('  rankdir="LR";')
status_color = {
  "pending":   "lightyellow",
  "ready":     "lightyellow",
  "dispatched":"lightblue",
  "running":   "lightblue",
  "complete":  "lightgreen",
  "failed":    "salmon",
  "sealed":    "lightgreen",
  "skipped":   "lightgray",
  "escalated": "orange",
}
for t in plan.get("tasks", []):
    s = t.get("status", "pending")
    color = status_color.get(s, "white")
    label = f"{t['id']}\\n[{t.get('agent_role','?')}|{s}]"
    print(f'  "{t["id"]}" [label="{label}" style=filled fillcolor={color}];')
for t in plan.get("tasks", []):
    for d in t.get("depends_on", []):
        print(f'  "{d}" -> "{t["id"]}";')
print("}")
PY
}

graph_stats() {
  # Emit a small JSON object with counts.
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
tasks = plan.get("tasks", [])
by_status = {}
for t in tasks:
    s = t.get("status","pending"); by_status[s] = by_status.get(s, 0) + 1
total_cost = sum(t.get("task_cost_usd", 0) for t in tasks)
total_tokens = sum(t.get("task_tokens_used", 0) for t in tasks)
print(json.dumps({
  "total": len(tasks),
  "by_status": by_status,
  "total_cost_usd": round(total_cost, 4),
  "total_tokens": total_tokens,
}))
PY
}

graph_validate_task_shape() {
  # Apply the task-level schema in addition to the plan-level schema.
  python3 - "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
errors = []
ids = []
for t in plan.get("tasks", []):
    for req in ("id","agent_role","status"):
        if req not in t:
            errors.append(f"task missing '{req}'")
    ids.append(t.get("id"))
dupes = {i for i in ids if ids.count(i) > 1}
for d in set(dupes):
    errors.append(f"duplicate task id: {d}")
known_statuses = {"pending","ready","dispatched","running","complete","failed","sealed","skipped","escalated"}
for t in plan.get("tasks", []):
    if t.get("status") not in known_statuses:
        errors.append(f"task {t.get('id')}: invalid status '{t.get('status')}'")
print(json.dumps({"ok": not errors, "errors": errors}))
PY
}
