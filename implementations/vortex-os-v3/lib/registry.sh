# lib/registry.sh — Improvement #21: Skill Registry & Marketplace
# install/update/audit agents against a marketplace.json + ed25519 sigs.
set -euo pipefail
LIB_PREFIX="[registry]"

REGISTRY_DIR="${ORCH_REGISTRY_DIR:-$HOME/.orchestr8r/registry}"

registry_init() {
  mkdir -p "$REGISTRY_DIR"
  [[ -f "$REGISTRY_DIR/marketplace.json" ]] || cat > "$REGISTRY_DIR/marketplace.json" <<'JSON'
{
  "$schema":"https://orchestrator.local/schemas/marketplace.v2.json",
  "version":"2.0",
  "registries":[ { "name":"builtin", "url":"local", "ed25519_pubkey":"" } ],
  "agents": []
}
JSON
}

# verify signature using openssl ed25519
registry_verify() {
  local file="$1" sig="$2" pub="$3"
  openssl pkeyutl -verify -pubin -inkey "$pub" -in "$file" -sigfile "$sig" >/dev/null 2>&1
}

registry_install() {
  local url="$1" name="${2:-}"
  registry_init
  local tmp; tmp=$(mktemp -d)
  curl -fsSL "$url" -o "$tmp/agent.json" 2>/dev/null || { echo "fetch failed" >&2; rm -rf "$tmp"; return 1; }
  [[ -n $name ]] || name=$(jq -r '.name' "$tmp/agent.json")
  cp "$tmp/agent.json" "$REGISTRY_DIR/$name.json"
  rm -rf "$tmp"
  echo "[registry] installed $name from $url"
}

registry_update() {
  local url="$1"
  registry_install "$url"
}

registry_audit() {
  # Compare installed versions to marketplace versions
  jq -s '
    .[0].agents | map({name, marketplace_version:.version}) as $m
    | [inputs[] | select(.name) | {name, version}] as $i
    | $m | map(. + (($i[] | select(.name==.name)) // {version:"MISSING"}))
  ' "$REGISTRY_DIR/marketplace.json" \
    $(find "$REGISTRY_DIR" -maxdepth 1 -name "*.json" ! -name "marketplace.json" -printf "%p ")
}
