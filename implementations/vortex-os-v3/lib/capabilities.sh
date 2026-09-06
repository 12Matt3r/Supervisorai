# lib/capabilities.sh — Improvement #5: Capability Negotiation Handshake
# Agents advertise tools / constraints / cost / latency before dispatch.
set -euo pipefail
LIB_PREFIX="[capabilities]"

capabilities_publish() {
  local agent="$1"
  local caps_json="$2"   # JSON: {tools:[], constraints:[], est_latency_s:N, est_cost_usd:N}
  local d="${ORCH_CAP_DIR:-./.orchestr8r/caps}"
  mkdir -p "$d"
  printf '%s' "$caps_json" > "${d}/${agent}.json"
}

capabilities_get() {
  local agent="$1"
  local f="${ORCH_CAP_DIR:-./.orchestr8r/caps}/${agent}.json"
  [[ -f $f ]] && cat "$f" || return 1
}

capabilities_match() {
  # Match a task to the best agent given needed tools.
  local needed_tools_json="$1"  # e.g. ["web_search","fs_read"]
  local d="${ORCH_CAP_DIR:-./.orchestr8r/caps}"
  local best="" score=0
  for f in "$d"/*.json; do
    [[ -f $f ]] || continue
    s=$(jq -r --argjson need "$needed_tools_json" '
       (.tools // []) as $have
       | ($need | map(. as $n | ($have | index($n) | length)) | add)
       | if . == null then 0 else . end
    ' "$f")
    if (( $(echo "$s > $score" | bc -l) )); then score="$s"; best="$(basename "$f" .json)"; fi
  done
  printf '%s' "$best"
}
