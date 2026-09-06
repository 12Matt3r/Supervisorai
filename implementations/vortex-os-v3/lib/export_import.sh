# lib/export_import.sh — plan export / import, replay, diff
# shellcheck shell=bash

[[ -n "${__LIB_XPORT_LOADED:-}" ]] && return 0
__LIB_XPORT_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"

plan_export() {
  local out="$1"
  local pp sp af
  pp=$(plan_path); sp=$(state_path); af=$(audit_path)
  local bundle="$out"
  ensure_parent "$bundle"

  python3 - "$pp" "$sp" "$af" "$bundle" <<'PY'
import json, zipfile, os, sys
pp, sp, af, bundle = sys.argv[1:5]
def slurp(p):
    return open(p).read() if os.path.exists(p) else None
out = {}
out["plan.json"]  = slurp(pp)
out["state.json"] = slurp(sp)
out["audit.jsonl"] = slurp(af)
# Write as JSON bundle (one file), not a zip — easy to diff and email
json.dump(out, open(bundle, "w"), indent=2)
PY
  ok "Exported plan bundle to $bundle"
}

plan_import() {
  local src="$1"
  [[ -f "$src" ]] || { err "Import source not found: $src"; return 1; }
  local pp; pp=$(plan_path)
  python3 - "$src" "$pp" <<'PY'
import json, sys
src, pp = sys.argv[1], sys.argv[2]
bundle = json.load(open(src))
plan = json.loads(bundle["plan.json"])
# Add a note that this is an import so reviewers can see provenance.
plan["notes"] = "imported from " + sys.argv[1]
json.dump(plan, open(pp, "w"), indent=2)
print("Imported plan:", plan.get("plan_id"))
PY
}

plan_diff() {
  local a="$1" b="$2"
  python3 - "$a" "$b" <<'PY'
import json, sys
ap = json.load(open(sys.argv[1])) if sys.argv[1] != "-" else None
bp = json.load(open(sys.argv[2])) if sys.argv[2] != "-" else None
print("# Goal")
print(f"- A: {ap.get('goal','')}")
print(f"+ B: {bp.get('goal','')}")
print()
ai = {t["id"]: t for t in ap.get("tasks", [])}
bi = {t["id"]: t for t in bp.get("tasks", [])}
added = set(bi) - set(ai)
removed = set(ai) - set(bi)
common = set(ai) & set(bi)
for k in sorted(added):
    print(f"+ task: {k} ({bi[k].get('status','?')})")
for k in sorted(removed):
    print(f"- task: {k}")
for k in sorted(common):
    a2, b2 = ai[k], bi[k]
    changes = {kk: (a2.get(kk), b2.get(kk)) for kk in set(a2)|set(b2)
               if a2.get(kk) != b2.get(kk) and kk not in ("notes",)}
    if changes:
        print(f"~ task: {k}")
        for kk, (av, bv) in changes.items():
            print(f"    {kk}: {av} -> {bv}")
PY
}

# Replay: mark all sealed tasks whose output_path still exists as complete.
plan_replay() {
  local pp; pp=$(plan_path)
  with_locked_plan "
import os, time
for t in p['tasks']:
    if t.get('status') in ('sealed','complete'):
        op = t.get('output_path','')
        if op and os.path.exists(op) and not op.endswith('.plan.json'):
            t['status'] = 'complete'
            t['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
" >/dev/null
}
