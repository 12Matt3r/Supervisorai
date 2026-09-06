# lib/tools_registry.sh — Improvement #4: Shared MCP Tool Registry
# Tools live in tools/registry.json; agents declare tools_required[].
set -euo pipefail
LIB_PREFIX="[tools_registry]"

TOOLS_REGISTRY_PATH="${ORCH_TOOLS_REGISTRY:-./tools/registry.json}"

tools_registry_load() {
  mkdir -p "$(dirname "$TOOLS_REGISTRY_PATH")"
  [[ -f $TOOLS_REGISTRY_PATH ]] || cat > "$TOOLS_REGISTRY_PATH" <<'JSON'
{
  "$schema": "https://orchestrator.local/schemas/tools_registry.v2.json",
  "version": "2.0",
  "tools": {
    "web_search":    { "provider": "tavily",    "cost_per_call": 0.005, "net": "online",  "sandbox_ok": false },
    "fs_read":       { "provider": "local",     "cost_per_call": 0.0,   "net": "offline", "sandbox_ok": true },
    "fs_write":      { "provider": "local",     "cost_per_call": 0.0,   "net": "offline", "sandbox_ok": true },
    "shell_exec":    { "provider": "local",     "cost_per_call": 0.0,   "net": "offline", "sandbox_ok": true },
    "http_post":     { "provider": "local",     "cost_per_call": 0.0,   "net": "online",  "sandbox_ok": true },
    "llm_call":      { "provider": "anthropic", "cost_per_call": 0.01,  "net": "online",  "sandbox_ok": true }
  }
}
JSON
}

tools_registry_get() {
  local tool_name="$1"
  jq -er --arg n "$tool_name" '.tools[$n]' "$TOOLS_REGISTRY_PATH"
}

tools_registry_resolve() {
  # Resolve a list of required tools into JSON {name: spec}.
  local agent="$1"
  shift
  local reqs=("$@")
  jq -nc --arg agent "$agent" \
    '(.tools) as $all | ($ARGS.positional) as $names
     | reduce $names[] as $n ({}; .[$n] = $all[$n])' \
    --args "${reqs[@]}"
}

tools_registry_suitable() {
  # Given sandbox type & net state, return only the tools that work.
  local sandbox="$1" net="$2"
  jq -r --arg s "$sandbox" --arg n "$net" \
    '.tools | to_entries[]
     | select((.value.net=="offline" or $n=="online")
              and ((.value.sandbox_ok==true) or $s=="host"))
     | .key' "$TOOLS_REGISTRY_PATH"
}
