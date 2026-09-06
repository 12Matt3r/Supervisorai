# lib/quality.sh — output quality gates
# shellcheck shell=bash

[[ -n "${__LIB_QUALITY_LOADED:-}" ]] && return 0
__LIB_QUALITY_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"

# Run a quality gate on a task.
#
# Three gate types are supported:
#   - validate_cmd:   run a shell command; pass if exit==0
#   - validate_tool:  emit JSON {"score":0..1, "reason":"..."} via a tool
#   - validate_rubric: a free-form description passed to a downstream judge
#
# A task is kept in `complete` only if every gate passes. Otherwise
# its status reverts to `failed` with notes set to the gate messages.
quality_gate() {
  local task_id="$1" judge_cmd="${2:-}"

  local pp; pp=$(plan_path)
  python3 - "$pp" "$task_id" "$judge_cmd" <<'PY'
import json, os, subprocess, sys, time, re

pp, task_id, judge_cmd = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(pp))
task = next((t for t in plan["tasks"] if t["id"] == task_id), None)
if not task:
    print(json.dumps({"ok": False, "reason": "task not found"})); sys.exit(0)

v = task.get("validation") or {}

# 1) validate_cmd
rc = 0
note_lines = []
if v.get("validate_cmd"):
    cmd = v["validate_cmd"]
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        rc = proc.returncode
        note_lines.append(f"validate_cmd exit={rc}: {cmd}")
        if proc.stdout.strip():
            note_lines.append("stdout: " + proc.stdout.strip()[:500])
        if proc.stderr.strip():
            note_lines.append("stderr: " + proc.stderr.strip()[:500])
    except subprocess.TimeoutExpired:
        rc = -1
        note_lines.append("validate_cmd timeout")

ok = (rc == 0)

# 2) validate_tool -> a user-supplied shell command that prints JSON {score, reason}
score = None
reason = None
if v.get("validate_tool") and ok:
    try:
        proc = subprocess.run(v["validate_tool"], shell=True,
                              env={**os.environ, "TASK_ID": task_id,
                                   "OUTPUT_PATH": task.get("output_path","")},
                              capture_output=True, text=True, timeout=180)
        if proc.returncode == 0:
            try:
                o = json.loads(proc.stdout)
                score = o.get("score")
                reason = o.get("reason","")
            except Exception as e:
                note_lines.append(f"validate_tool: bad JSON output: {e}")
                ok = False
        else:
            ok = False
            note_lines.append(f"validate_tool exit={proc.returncode}")
    except subprocess.TimeoutExpired:
        ok = False
        note_lines.append("validate_tool timeout")

# 3) min_score check
if score is not None:
    mn = v.get("validate_min_score", 0.0)
    if score < mn:
        ok = False
        note_lines.append(f"score {score} < min_score {mn}")

# 4) optional LLM judge fallback (skipped here — parent can run via --judge command)

print(json.dumps({
  "ok": ok,
  "task_id": task_id,
  "score": score,
  "reason": reason,
  "notes": note_lines,
}))
PY
}

# Manually mark a task quality-judged, applying the result.
quality_apply() {
  local task_id="$1" score="$2" reason="$3"
  with_locked_plan "
t = next((t for t in p['tasks'] if t['id'] == '$task_id'), None)
if not t:
    print('Task not found: $task_id'); sys.exit(1)
v = t.get('validation') or {}
mn = v.get('validate_min_score', 0.0)
ok = float('$score') >= mn
if ok:
    t['status'] = 'complete'
    t['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
else:
    t['status'] = 'failed'
t['notes'] = (t.get('notes','') + '|' + '$reason').strip('|')
" >/dev/null
}
