# lib/memory.sh — cross-plan memory store
# shellcheck shell=bash

[[ -n "${__LIB_MEMORY_LOADED:-}" ]] && return 0
__LIB_MEMORY_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/locking.sh"

# Persistent key/value store, optionally sharded by namespace.
# Files live under memory_dir and are JSON.

memory_path_for() {
  local ns="${1:-default}" key="${2:-}"
  local md; md="$(memory_path)"
  [[ -d "$md" ]] || mkdir -p "$md"
  if [[ -n "$key" ]]; then
    echo "$md/$ns/$key.json"
  else
    echo "$md/$ns/"
  fi
}

memory_set() {
  local ns="$1" key="$2" value_json="$3"
  local md; md="$(memory_path_for "$ns")"
  mkdir -p "$md"
  local file="$md/$key.json"
  echo "$value_json" > "$file"
  ok "Memory[$ns/$key] set"
}

memory_get() {
  local ns="$1" key="$2"
  local file; file="$(memory_path_for "$ns" "$key")"
  [[ -f "$file" ]] || { echo ""; return 0; }
  cat "$file"
}

memory_list() {
  local ns="$1"
  local md; md="$(memory_path_for "$ns")"
  [[ -d "$md" ]] || { echo "(no memory)"; return 0; }
  ls -1 "$md"
}

memory_delete() {
  local ns="$1" key="$2"
  local file; file="$(memory_path_for "$ns" "$key")"
  rm -f "$file"
  ok "Memory[$ns/$key] deleted"
}

# Compact a memory namespace: keep only the last N entries (by mtime).
memory_compact() {
  local ns="$1" keep="${2:-100}"
  local md; md="$(memory_path_for "$ns")"
  [[ -d "$md" ]] || return 0
  ls -1t "$md" | tail -n +$((keep+1)) | while read -r f; do rm -f "$md/$f"; done
}

# Snippet store: a named text snippet (e.g. a persona brief) injectable into tasks.
memory_snippet_set() {
  local name="$1" body="$2"
  memory_set "snippets" "$name" "$(python3 -c 'import json,sys; print(json.dumps({"name":sys.argv[1],"body":sys.argv[2]}))' "$name" "$body")"
}
