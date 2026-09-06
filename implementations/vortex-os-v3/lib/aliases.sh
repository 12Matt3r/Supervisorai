# lib/aliases.sh — decouple orchestrator roles from underlying agent names
# shellcheck shell=bash

[[ -n "${__LIB_ALIASES_LOADED:-}" ]] && return 0
__LIB_ALIASES_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"

# Resolve a role via the agent_aliases map; if no alias, return the role itself.
agent_resolve() {
  local role="$1"
  local primary; primary=$(json_get "$CONFIG" "agents.$role.primary")
  if [[ -n "$primary" && "$primary" != "null" ]]; then
    echo "$primary"
    return 0
  fi
  echo "$role"
}

# Rewrite an existing plan to use aliases. Idempotent.
agent_rewrite_plan() {
  local pp; pp=$(plan_path)
  with_locked_plan "
import os
for t in p['tasks']:
    t['agent_role'] = agent_resolve(t.get('agent_role',''))
" >/dev/null 2>&1 || true
}

# Apply a configured 'dynamic' routing model: turn a goal into tasks using the
# LLM. Currently uses a JSON LLM client stub that calls OpenAI-compatible /v1/chat.
# Set OPENAI_API_KEY or ANTHROPIC_API_KEY and the orchestrator will auto-route.
agent_dynamic_route() {
  local goal="$1"
  local model="${ORCHESTRATOR_MODEL:-gpt-4o-mini}"
  python3 - "$goal" "$model" <<'PY'
import json, os, sys, urllib.request
goal, model = sys.argv[1], sys.argv[2]
api_key = os.environ.get("OPENAI_API_KEY","")
if not api_key:
    print("OPENAI_API_KEY not set; dynamic routing skipped.")
    sys.exit(0)
prompt = (
  "Decompose the following high-level goal into a JSON plan. "
  "Use these agent roles if applicable: research, documentation, design_web, "
  "design_ppt, build_static_site, build_interactive_app, build_fullstack_app, "
  "build_ppt, batch_technical, build_mcp. "
  "Each task must have id, description, agent_role, depends_on (list).\n\n"
  f"GOAL: {goal}\n\n"
  "Output ONLY valid JSON in the form {\"tasks\": [...]}."
)
body = json.dumps({
    "model": model,
    "messages": [{"role":"user","content":prompt}],
    "response_format": {"type":"json_object"},
    "temperature": 0.2,
}).encode()
req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                             data=body,
                             headers={"Authorization":f"Bearer {api_key}",
                                      "Content-Type":"application/json"},
                             method="POST")
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
        text = out["choices"][0]["message"]["content"]
        plan = json.loads(text)
        for t in plan.get("tasks", []):
            t.setdefault("status","pending")
            t.setdefault("depends_on",[])
            t.setdefault("attempts",0)
            t.setdefault("notes","")
        print(json.dumps({"tasks": plan.get("tasks", [])}))
except Exception as e:
    print(f"dynamic routing failed: {e}", file=sys.stderr); sys.exit(1)
PY
}
