# lib/memory_v2.sh — Improvement #2: Five-Tier Memory Plumbing
# working / episodic / semantic / procedural / shared
set -euo pipefail
LIB_NAME="memory_v2"
LIB_PREFIX="[memory_v2]"

MEM_TIER_DIR="${ORCH_MEMORY_DIR:-$HOME/.orchestr8r/memory}"
MEM_NAMESPACES=("working" "episodic" "semantic" "procedural" "shared")

memory_v2_init() {
  for n in "${MEM_NAMESPACES[@]}"; do
    mkdir -p "${MEM_TIER_DIR}/${n}"
  done
}

memory_v2_set() {
  local tier="$1" ns="$2" key="$3" value="$4"
  [[ " ${MEM_NAMESPACES[*]} " == *" $tier "* ]] || { echo "unknown tier: $tier" >&2; return 1; }
  local d="${MEM_TIER_DIR}/${tier}"
  mkdir -p "${d}/${ns}"
  printf '%s' "$value" > "${d}/${ns}/${key}.json"
}

memory_v2_get() {
  local tier="$1" ns="$2" key="$3"
  local f="${MEM_TIER_DIR}/${tier}/${ns}/${key}.json"
  [[ -f $f ]] && cat "$f" || return 1
}

memory_v2_list() {
  local tier="$1" ns="${2:-}"
  local d
  if [[ -n $ns ]]; then d="${MEM_TIER_DIR}/${tier}/${ns}/"
  else                  d="${MEM_TIER_DIR}/${tier}/"; fi
  [[ -d $d ]] && find "$d" -type f -name "*.json" | sort
}

memory_v2_delete() {
  local tier="$1" ns="$2" key="$3"
  rm -f "${MEM_TIER_DIR}/${tier}/${ns}/${key}.json"
}

# Carry-over: pull parent's episodic into current sub-plan.
memory_v2_inherit_from_parent() {
  local parent_id="$1" ns="$2"
  local src="${MEM_TIER_DIR}/episodic/${parent_id}"
  [[ -d $src ]] || return 0
  local dst="${MEM_TIER_DIR}/episodic/${ns}"
  mkdir -p "$dst"
  cp -ru "$src/" "$dst/" 2>/dev/null || true
}

memory_v2_compact() {
  # Compress episodic to semantic via simple summarisation (extend with LLM call).
  local ns="$1"
  local src="${MEM_TIER_DIR}/episodic/${ns}"
  local dst="${MEM_TIER_DIR}/semantic/${ns}.md"
  [[ -d $src ]] || return 0
  {
    echo "# Summary: ${ns} @ $(date -Iseconds)"
    echo
    jq -r 'select(.kind=="note") | "- " + .body' "$src"/*.json 2>/dev/null | head -200 || true
    echo
  } > "$dst"
}

# Pub: publish a topic to the shared tier.
# Sub: list subscribers for a topic.
memory_v2_publish() {
  local topic="$1" payload="$2"
  local f="${MEM_TIER_DIR}/shared/${topic}.jsonl"
  printf '%s\n' "$payload" >> "$f"
}

memory_v2_subscribers() {
  local topic="$1"
  cat "${MEM_TIER_DIR}/shared/${topic}.subs" 2>/dev/null || true
}
