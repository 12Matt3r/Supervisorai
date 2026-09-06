#!/bin/bash
# =============================================================================
# VORTEX-OS — Post-Upload Verification
# =============================================================================
# Run after uploading to confirm everything works.
# Exits 0 on full success, 1 on any failure.
# =============================================================================
set +e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  VORTEX-OS — Post-Upload Verification                ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

failed=0

step() { echo -e "${BOLD}${CYAN}▶ $1${NC}"; }
ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
err()  { echo -e "  ${RED}✗ $1${NC}"; failed=$((failed + 1)); }

# -----------------------------------------------------------------------------
# 1. File presence (only the files we actually ship)
# -----------------------------------------------------------------------------
step "1. File presence"
for f in skill.sh SKILL.md _meta.json README.md INSTRUCTIONS.md verify.sh \
         lib/swarm.sh lib/hitl.sh lib/inspector.sh lib/prompt_optimizer.sh \
         lib/dispatch_v4.sh lib/commands.sh \
         agents/supervisor.store.json agents/supervisor.shift.json \
         agents/inspector.governance.json; do
  if [[ -f "$f" ]]; then ok "$f"; else err "$f MISSING"; fi
done

# -----------------------------------------------------------------------------
# 2. Tool availability
# -----------------------------------------------------------------------------
step "2. Tool check"
for t in bash jq python3 sqlite3; do
  if command -v "$t" >/dev/null 2>&1; then
    ok "$t: $(command -v $t)"
  else
    err "$t not installed"
  fi
done

# -----------------------------------------------------------------------------
# 3. Bash syntax check (all shipped scripts)
# -----------------------------------------------------------------------------
step "3. Bash syntax check"
for f in skill.sh verify.sh lib/*.sh; do
  if bash -n "$f" 2>/dev/null; then
    ok "syntax: $f"
  else
    err "syntax error in: $f"
  fi
done

# -----------------------------------------------------------------------------
# 4. JSON validity
# -----------------------------------------------------------------------------
step "4. JSON validity"
for f in _meta.json agents/*.json; do
  [[ -f "$f" ]] || continue
  if jq empty "$f" >/dev/null 2>&1; then
    ok "json: $f"
  else
    err "json invalid: $f"
  fi
done

# -----------------------------------------------------------------------------
# 5. _meta.json validation
# -----------------------------------------------------------------------------
step "5. _meta.json validation"
if [[ -f _meta.json ]]; then
  if jq -e '.skill_id and .name and .version and .entry_point' _meta.json >/dev/null 2>&1; then
    skill_id=$(jq -r '.skill_id' _meta.json)
    skill_name=$(jq -r '.name' _meta.json)
    display_name=$(jq -r '.display_name' _meta.json)
    ok "_meta.json valid: skill_id=$skill_id, name=$skill_name"
    if echo "$display_name $skill_name" | grep -qi "vortex"; then
      ok "VORTEX-OS branding present in _meta.json"
    else
      err "VORTEX-OS branding missing from _meta.json"
    fi
  else
    err "_meta.json missing required fields (skill_id, name, version, entry_point)"
  fi
else
  err "_meta.json MISSING (required for platform registration)"
fi

# -----------------------------------------------------------------------------
# 6. SKILL.md branding & trigger semantics
# -----------------------------------------------------------------------------
step "6. SKILL.md branding"
if [[ -f SKILL.md ]]; then
  if grep -qi "VORTEX-OS" SKILL.md; then
    ok "VORTEX-OS branding present in SKILL.md"
  else
    err "VORTEX-OS branding missing from SKILL.md"
  fi
  if grep -qE "TRIGGER when:|DO NOT TRIGGER when:" SKILL.md; then
    ok "trigger semantics present in SKILL.md frontmatter"
  else
    err "trigger semantics missing from SKILL.md frontmatter"
  fi
else
  err "SKILL.md MISSING"
fi

# -----------------------------------------------------------------------------
# 7. Agent discovery
# -----------------------------------------------------------------------------
step "7. Agent discovery"
if [[ -x skill.sh ]]; then
  disc_output=$(./skill.sh --agents-discover 2>&1)
  if echo "$disc_output" | grep -qE "supervisor\.|inspector\."; then
    echo "$disc_output" | head -5 | sed 's/^/    /'
    ok "discovery works"
  else
    err "discovery failed — no agents found"
  fi
else
  err "skill.sh not executable (run: chmod +x skill.sh)"
fi

# -----------------------------------------------------------------------------
# 8. Agent lint
# -----------------------------------------------------------------------------
step "8. Agent lint"
if ./skill.sh --agents-lint --all 2>&1 | grep -q "LINT_OK"; then
  ok "all agents pass lint"
else
  err "some agents failed lint"
fi

# -----------------------------------------------------------------------------
# 9. Help banner
# -----------------------------------------------------------------------------
step "9. Help banner"
if ./skill.sh help 2>&1 | grep -qi "VORTEX-OS"; then
  ok "help banner shows VORTEX-OS branding"
else
  err "help banner missing VORTEX-OS branding"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${NC}"
if [[ $failed -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}  ✓ ALL VERIFICATION CHECKS PASSED — skill is ready to deploy.${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${NC}"
  exit 0
else
  echo -e "${RED}${BOLD}  ✗ $failed CHECK(S) FAILED — see above for details.${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${NC}"
  exit 1
fi
