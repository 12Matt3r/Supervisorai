# lib/personas.sh — Improvement #6: Persona & Voice Pins
# Reusable personas that multiple agents can pin to.
set -euo pipefail
LIB_PREFIX="[personas]"

PERSONAS_DIR="${ORCH_PERSONAS_DIR:-./personas}"

personas_init() {
  mkdir -p "$PERSONAS_DIR"
  [[ -f "$PERSONAS_DIR/staff_engineer.md" ]] || cat > "$PERSONAS_DIR/staff_engineer.md" <<'MD'
# Persona: staff_engineer
You are a staff-level engineer. Be specific, cite files, write tests, and never paper over bugs.
Write idiomatic, production-ready code. Prefer clarity over cleverness.
MD
  [[ -f "$PERSONAS_DIR/librarian.md" ]] || cat > "$PERSONAS_DIR/librarian.md" <<'MD'
# Persona: librarian
You are a research librarian. Always cite sources. Use bullet lists. Be exhaustive, then concise.
MD
  [[ -f "$PERSONAS_DIR/sre.md" ]] || cat > "$PERSONAS_DIR/sre.md" <<'MD'
# Persona: sre
You are an SRE. Use SLOs, error budgets, and runbook language. Quantify risk.
MD
  [[ -f "$PERSONAS_DIR/security_auditor.md" ]] || cat > "$PERSONAS_DIR/security_auditor.md" <<'MD'
# Persona: security_auditor
You audit for OWASP Top 10 and supply-chain issues. Cite CVE IDs.
MD
  [[ -f "$PERSONAS_DIR/pm.md" ]] || cat > "$PERSONAS_DIR/pm.md" <<'MD'
# Persona: pm
You think in users, outcomes, and trade-offs. Frame work as user stories with RICE scores.
MD
  [[ -f "$PERSONAS_DIR/teacher.md" ]] || cat > "$PERSONAS_DIR/teacher.md" <<'MD'
# Persona: teacher
You explain complex things simply. Use analogies. Diagrams welcome.
MD
}

personas_get() {
  local name="$1"
  local f="$PERSONAS_DIR/${name}.md"
  [[ -f $f ]] && cat "$f"
}

personas_list() {
  find "$PERSONAS_DIR" -maxdepth 1 -name "*.md" -printf "%f\n" | sed 's/\.md$//'
}

# Compose agent system-prompt by combining persona + agent role description.
personas_compose_system_prompt() {
  local persona="$1" agent_role="$2"
  {
    personas_get "$persona" 2>/dev/null || echo "# Persona: default"
    echo
    echo "## Agent Role"
    echo "$agent_role"
  }
}
