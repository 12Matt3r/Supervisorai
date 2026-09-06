# lib/core.sh — logging, paths, and JSON helpers
# shellcheck shell=bash

# Guard against double-sourcing
[[ -n "${__LIB_CORE_LOADED:-}" ]] && return 0
__LIB_CORE_LOADED=1

# Determine script_dir once. Idempotent.
if [[ -z "${SKILL_SCRIPT_DIR:-}" ]]; then
  # BASH_SOURCE[0] may be a lib file when sourced via the parent skill.sh;
  # the parent always sets SKILL_SCRIPT_DIR; fall back gracefully.
  SKILL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
fi

# ----------------------------------------------------------------------------
# Logging (terse, color-capable)
# ----------------------------------------------------------------------------
_log_init() {
  if [[ -t 1 ]]; then
    C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'
    C_BLUE='\033[0;34m'; C_MAGENTA='\033[0;35m'; C_CYAN='\033[0;36m'
    C_BOLD='\033[1m'; C_RST='\033[0m'
  else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_MAGENTA=''; C_CYAN=''
    C_BOLD=''; C_RST=''
  fi
}
_log_init

log()   { [[ "${VERBOSE:-0}" == "1" ]] && echo -e "${C_BLUE}[orch]${C_RST} $*" >&2 || true; }
warn()  { echo -e "${C_YELLOW}[warn]${C_RST}  $*" >&2; }
err()   { echo -e "${C_RED}[error]${C_RST} $*" >&2; }
ok()    { echo -e "${C_GREEN}[ok]${C_RST}    $*"; }
header(){ echo -e "\n${C_BOLD}=== $* ===${C_RST}"; }
debug() { [[ "${VERBOSE:-0}" == "1" ]] && echo -e "${C_MAGENTA}[debug]${C_RST} $*" >&2 || true; }

# Verbose-aware helpers used across libraries (V3+ modules).
log_info()  { [[ "${VERBOSE:-0}" == "1" ]] && echo -e "${C_BLUE}[info]${C_RST}  $*" >&2 || true; }
log_warn()  { echo -e "${C_YELLOW}[warn]${C_RST}  $*" >&2; }
log_err()   { echo -e "${C_RED}[error]${C_RST} $*" >&2; }

# JSON output for every command (so a parent agent can `jq` everything)
emit_json() {
  python3 -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read() or '{}'), indent=2))" || true
}

# ----------------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------------
resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then echo "$p"; else echo "$SKILL_SCRIPT_DIR/$p"; fi
}

require_file() { [[ -f "$1" ]] || { err "File not found: $1"; return 1; }; }
require_dir()  { [[ -d "$1" ]] || { err "Directory not found: $1"; return 1; }; }

# ----------------------------------------------------------------------------
# Time / IDs
# ----------------------------------------------------------------------------
now_iso()      { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
now_ms()       { date +%s%3N; }
gen_plan_id()  { echo "plan_$(date -u +%Y_%m_%d)_$(date -u +%H%M%S)_$$"; }
gen_task_id()  { echo "task_$(date -u +%s)_$RANDOM"; }
gen_run_id()   { echo "run_$(date -u +%s)_$RANDOM"; }

# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------

# Read a JSON value via a Python one-liner with a stable signature.
# Usage: json_get FILE KEY   (KEY is dotted, e.g. "paths.plan_file")
json_get() {
  local file="$1" key="$2"
  [[ -f "$file" ]] || { echo ""; return 0; }
  python3 - "$file" "$key" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    for k in sys.argv[2].split('.'):
        if isinstance(data, dict):
            data = data.get(k, "")
        else:
            data = ""; break
    if isinstance(data, (dict, list)):
        print(json.dumps(data))
    else:
        print(data if data is not None else "")
except Exception:
    print("", end="")
PY
}

# Whole-document pretty printer
json_pp() {
  [[ -f "$1" ]] || { err "json_pp: missing $1"; return 1; }
  python3 -m json.tool "$1"
}

# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------
plan_path()  { resolve_path "$(json_get "$CONFIG" "paths.plan_file")"; }
state_path() { resolve_path "$(json_get "$CONFIG" "paths.state_file")"; }
memory_path(){ resolve_path "$(json_get "$CONFIG" "paths.memory_dir")"; }
audit_path() { resolve_path "$(json_get "$CONFIG" "paths.audit_file")"; }
config_dir() { echo "$(dirname "$CONFIG")"; }

# Best-effort creation of a sub-path under the project's output_dir
ensure_parent() {
  local f="$1"
  local d; d="$(dirname "$f")"
  [[ -d "$d" ]] || mkdir -p "$d" || { err "Cannot create $d"; return 1; }
}
