# lib/cache_aware.sh — Improvement #1: Prompt-Cache Awareness
# Tracks cache prefixes per agent, computes hit ratios.
set -euo pipefail
LIB_NAME="cache_aware"
LIB_PREFIX="[cache_aware]"

cache_aware_init() {
  local cfg="$1"
  local audit="${ORCH_AUDIT_FILE:-./audit.jsonl}"
  CACHE_PREFIX_DIR="${ORCH_CACHE_DIR:-$HOME/.orchestr8r/cache}"
  mkdir -p "$CACHE_PREFIX_DIR"
  : > "$CACHE_PREFIX_DIR/index.json"
}

# Compute a stable cache prefix for an agent+persona combo.
# Stable => the LLM provider can cache across calls.
cache_aware_prefix() {
  local agent_name="$1" persona="${2:-default}" version="${3:-v1}"
  local stable="system: agent=$agent_name persona=$persona ver=$version"
  printf '%s' "$stable" | sha256sum | cut -c1-16 | { read h; echo "${agent_name}-${persona}-${h}"; }
}

# Send prefix + record expected/actual cache hits.
cache_aware_record_hit() {
  local agent="$1" hit="0" total="0"
  hit="${2:-0}"; total="${3:-1}"
  local f="${CACHE_PREFIX_DIR}/${agent}.jsonl"
  jq -nc --arg a "$agent" --argjson hit "$hit" --argjson total "$total" \
    '{agent:$a,hit:$hit,total:$total,ratio:(if $total>0 then ($hit/$total) else 0 end),ts:now|todate}' >> "$f"
}

# Aggregate ratio over the last N records.
cache_aware_ratio() {
  local agent="$1" window="${2:-100}"
  local f="${CACHE_PREFIX_DIR}/${agent}.jsonl"
  if [[ ! -f $f ]]; then echo "0.0"; return; fi
  tail -n "$window" "$f" | jq -s '
    {h:map(.hit)|add, t:map(.total)|add}
    | if .t>0 then (.h/.t) else 0 end
  '
}

# Emit the cache_hit_ratio into an envelope metrics block.
cache_aware_envelope() {
  local agent="$1"
  jq -nc --arg a "$agent" '{cache_hit_ratio:('"$(cache_aware_ratio "$agent" 100)"')}'
}
