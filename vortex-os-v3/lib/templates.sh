# lib/templates.sh — plan templates (instantiable goal packs)
# shellcheck shell=bash

[[ -n "${__LIB_TEMPLATES_LOADED:-}" ]] && return 0
__LIB_TEMPLATES_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"

TEMPLATES_DIR="${TEMPLATES_DIR:-$SKILL_SCRIPT_DIR/templates}"

# List available templates
templates_list() {
  if [[ -d "$TEMPLATES_DIR" ]]; then
    for f in "$TEMPLATES_DIR"/*.json; do
      [[ -f "$f" ]] || continue
      python3 - "$f" <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))
print(f"- {o.get('name','?')}: {o.get('description','')[:80]}")
PY
    done
  fi
}

# Apply a template: copy its tasks into the current plan
# Variably substitute placeholders via --template-vars.
templates_apply() {
  local name="$1"; shift
  local tpath="$TEMPLATES_DIR/${name}.json"
  [[ -f "$tpath" ]] || { err "Template not found: $name"; return 1; }

  local pp; pp=$(plan_path)
  ensure_plan "$(gen_plan_id)" || return 1
  # Read vars and pass them as JSON env var
  local vars_json="{}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --template-var) vars_json="$vars_json"
        IFS='=' read -r k v <<< "$2"
        vars_json="$(python3 -c "import json,sys; o=json.loads('$vars_json'); o['$k']='$v'; print(json.dumps(o))")"
        shift 2
        ;;
      *) shift ;;
    esac
  done

  python3 - "$tpath" "$pp" "$vars_json" <<'PY'
import json, sys
tpl_path, plan_path, vars_json = sys.argv[1], sys.argv[2], sys.argv[3]
template = json.load(open(tpl_path))
plan = json.load(open(plan_path))
vars = json.loads(vars_json)
def substitute(o):
    if isinstance(o, str):
        for k, v in vars.items():
            o = o.replace("{{" + k + "}}", str(v))
        return o
    if isinstance(o, list):
        return [substitute(x) for x in o]
    if isinstance(o, dict):
        return {k: substitute(v) for k, v in o.items()}
    return o
applied = []
for t in template.get("tasks", []):
    t = substitute(t)
    # Ensure required defaults
    t.setdefault("status", "pending")
    t.setdefault("depends_on", [])
    t.setdefault("attempts", 0)
    t.setdefault("notes", "")
    plan["tasks"].append(t)
    applied.append(t["id"])
# Update goal if not set
if not plan.get("goal") or plan["goal"] == "Untitled plan":
    plan["goal"] = template.get("name", "Applied template")
json.dump(plan, open(plan_path, "w"), indent=2)
print(json.dumps({"applied": applied}))
PY
}
