#!/usr/bin/env bash
###############################################################################
# install.sh — install orchestrator_skill onto the system
#
# Usage:
#   ./install.sh                       # install into $HOME/.local/share/orchestrator_skill
#   ./install.sh --prefix /opt/orch    # install into a custom directory
#   ./install.sh --with-examples       # also install examples
###############################################################################
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${HOME}/.local/share/orchestrator_skill"
WITH_EXAMPLES=0
WITH_MCP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --with-examples) WITH_EXAMPLES=1; shift ;;
    --with-mcp)      WITH_MCP=1; shift ;;
    *) shift ;;
  esac
done

mkdir -p "$PREFIX"
# Copy the entire skill (sans tests and workspace stuff)
rsync -a --delete \
  --exclude='tests/' \
  --exclude='.github/' \
  --exclude='examples/' \
  --exclude='web/' \
  "$SCRIPT_DIR/" "$PREFIX/"
chmod +x "$PREFIX/skill.sh" "$PREFIX/install.sh" 2>/dev/null || true

if (( WITH_EXAMPLES )); then
  rsync -a --delete "$SCRIPT_DIR/examples/" "$PREFIX/examples/"
fi

# Make `skill` available on PATH
if command -v install >/dev/null; then
  install -m 0755 "$PREFIX/skill.sh" "${PREFIX}/skill" 2>/dev/null || true
fi

cat <<EOF
orchestrator_skill installed at: $PREFIX

To use:
  bash $PREFIX/skill.sh --help
or add $PREFIX to PATH and use:
  skill --help
EOF

if (( WITH_MCP )); then
  pip install --quiet mcp 2>/dev/null || true
  echo "To run the MCP server:  python3 $PREFIX/scripts/mcp_server.py"
fi

# shellcheck disable=SC2317
exit 0
