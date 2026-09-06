# lib/bus.sh — Improvement #30: Cross-Agent Broadcast Bus (pub/sub)
# Unix-socket-based pub/sub; agents subscribe & publish to topics.
set -euo pipefail
LIB_PREFIX="[bus]"

BUS_SOCKET="${ORCH_BUS_SOCKET:-./.orchestr8r/bus.sock}"
BUS_LOG="${ORCH_BUS_LOG:-./.orchestr8r/bus.ndjson}"

bus_init() {
  local d; d="$(dirname "$BUS_SOCKET")"
  mkdir -p "$d"
  [[ -S "$BUS_SOCKET" ]] && rm -f "$BUS_SOCKET"
  : > "$BUS_LOG"
  ncat -lU "$BUS_SOCKET" -k -c '
    while read -r line; do
      ts=$(date -Iseconds)
      printf "%s\t%s\n" "$ts" "$line" >> "'"$BUS_LOG"'"
    done
  ' &
  echo $! > "${BUS_SOCKET}.pid"
  sleep 0.1
}

bus_publish() {
  local topic="$1" payload="$2"
  jq -nc --arg t "$topic" --arg p "$payload" \
    '{topic:$t,payload:$p,ts:now|todate}' | socat - UNIX-CONNECT:"$BUS_SOCKET" 2>/dev/null || \
    printf '%s\n' "$(jq -nc --arg t "$topic" --arg p "$payload" '{topic:$t,payload:$p,ts:now|todate}')" >> "$BUS_LOG"
}

bus_subscribe() {
  local pattern="$1"
  tail -n +1 -F "$BUS_LOG" 2>/dev/null | grep "$pattern" || true
}

bus_topics() {
  [[ -f "$BUS_LOG" ]] && cut -f2 "$BUS_LOG" | jq -r '.topic' | sort -u
}
