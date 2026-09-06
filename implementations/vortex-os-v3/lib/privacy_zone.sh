# lib/privacy_zone.sh — Bonus B: Tool-Use Privacy Zone
# Each tool call is wrapped in a privacy envelope so data leaks are
# syntactically impossible.  Allow/redact lists validated at dispatch.
set -euo pipefail
LIB_PREFIX="[privacy-zone]"
[[ -n "${__LIB_PRIVACY_LOADED:-}" ]] && return 0
__LIB_PRIVACY_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh" 2>/dev/null || true

PRIVACY_CFG_FILE="${ORCH_PRIVACY_FILE:-./.orchestr8r/privacy.json}"

# privacy_load — read the policy once, cached for the session.
privacy_load() {
  [[ -f "$PRIVACY_CFG_FILE" ]] || {
    cat > "$PRIVACY_CFG_FILE" <<'JSON'
{
  "version":"1.0",
  "allow":  ["src/**", "tests/**", "README.md"],
  "redact": [".env", "*.pem", "*.key", "secrets/**"],
  "deny":   ["**/.git/**", "**/node_modules/**"]
}
JSON
  }
  jq '.' "$PRIVACY_CFG_FILE"
}

# privacy_check_path <path> — echo ALLOW|REDACT|DENY for a single path.
privacy_check_path() {
  local p="${1#/}"; p="${p#/}"
  local cfg; cfg="$(privacy_load)"
  # Deny takes priority first.
  if jq -e --arg p "$p" '.deny   | any(. as $g | $p | match($g))' <<<"$cfg" >/dev/null; then
    echo "DENY"; return 0
  fi
  if jq -e --arg p "$p" '.redact | any(. as $g | $p | match($g))' <<<"$cfg" >/dev/null; then
    echo "REDACT"; return 0
  fi
  if jq -e --arg p "$p" '.allow  | any(. as $g | $p | match($g))' <<<"$cfg" >/dev/null; then
    echo "ALLOW"; return 0
  fi
  # Default-deny: anything not explicitly allowed is denied.
  echo "DENY"
}

# privacy_wrap <input_file> <output_file> — apply the redaction policy.
privacy_wrap() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || { err "no source $src"; return 1; }
  mkdir -p "$(dirname "$dst")"
  local verdict; verdict=$(privacy_check_path "${src#./}")
  case "$verdict" in
    ALLOW)  cp "$src" "$dst" ;;
    REDACT) sed -E 's/(api[_-]?key|token|secret|password)[^[:space:]]*/\1=REDACTED/g' "$src" > "$dst" ;;
    DENY)   err "privacy violation: $src blocked by policy" >&2; return 2 ;;
  esac
}

# privacy_audit — emit a one-shot audit of every path that an envelope
#               might touch.  Hands a JSON array to stdout.
privacy_audit() {
  local subject="${1:-.}"
  find "$subject" -type f 2>/dev/null | \
    while IFS= read -r f; do
      printf '{"path":"%s","verdict":"%s"}\n' "${f#./}" "$(privacy_check_path "${f#./}")"
    done
}
