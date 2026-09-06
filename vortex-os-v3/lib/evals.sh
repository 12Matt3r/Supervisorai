# lib/evals.sh — Improvement #7: Per-Agent Eval Suite
# Gold-set + program-of-thought grading + online eval traces.
set -euo pipefail
LIB_PREFIX="[evals]"

EVALS_DIR="${ORCH_EVALS_DIR:-./agents/evals}"

evals_init() {
  mkdir -p "$EVALS_DIR"
}

evals_register() {
  local agent="$1"
  local agent_dir="$EVALS_DIR/$agent"
  mkdir -p "$agent_dir"
  [[ -f "$agent_dir/gold_set.json" ]] || echo '{"cases":[]}' > "$agent_dir/gold_set.json"
  [[ -f "$agent_dir/oracle_judge.py" ]] || cat > "$agent_dir/oracle_judge.py" <<'PY'
#!/usr/bin/env python3
# Default oracle judge. Override per agent.
import json,sys
gold = json.load(open(sys.argv[1]))["cases"]
out = json.load(sys.stdin)
score = 0.0
for c in gold:
    exp = c.get("expected_features", [])
    got = []
    for f in exp:
        if f in out.get("stdout_excerpt","") or f in out.get("diff",""):
            got.append(f)
    score += len(got) / max(1, len(exp))
print(json.dumps({"score": score / max(1, len(gold)), "matched": got}))
PY
  chmod +x "$agent_dir/oracle_judge.py"
}

evals_add_case() {
  local agent="$1" case_json="$2"
  local f="$EVALS_DIR/$agent/gold_set.json"
  tmp=$(mktemp)
  jq --argjson c "$case_json" '.cases += [$c]' "$f" > "$tmp" && mv "$tmp" "$f"
}

evals_run() {
  local agent="$1" envelope_path="$2"
  local dir="$EVALS_DIR/$agent"
  [[ -f "$dir/oracle_judge.py" ]] || { echo '{"score":0,"note":"no oracle"}'; return; }
  python3 "$dir/oracle_judge.py" "$dir/gold_set.json" < "$envelope_path"
}

evals_record_online() {
  local agent="$1" envelope_path="$2"
  local f="$EVALS_DIR/$agent/online_traces.jsonl"
  cp "$envelope_path" "$f" 2>/dev/null || true
  echo "$(date -Iseconds) $(sha256sum "$envelope_path" | cut -c1-12)" >> "${f}.idx"
}
