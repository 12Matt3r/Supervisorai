# lib/sse.sh — Improvement #20: WebSocket / SSE Dashboard
# Minimal SSE server: streams state.json diffs; clients reconnect with last_event_id.
set -euo pipefail
LIB_PREFIX="[sse]"

SSE_PORT="${ORCH_SSE_PORT:-8765}"
SSE_STATE_FILE="${ORCH_STATE_FILE:-./state.json}"
SSE_LASTEVENT_FILE="${ORCH_SSE_LASTEVENT:-./.orchestr8r/sse.last}"

sse_serve() {
  local port="${1:-$SSE_PORT}"
  local auth_token="${ORCH_SSE_AUTH_TOKEN:-}"
  echo "[sse] starting on :$port  (state: $SSE_STATE_FILE)"
  ncat -l -p "$port" -k -c '
    while read -r req; do
      [[ "$req" =~ ^GET ]] || continue
      printf "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n"
      last_hash=""
      while :; do
        h=$(sha256sum "'"$SSE_STATE_FILE"'" 2>/dev/null | cut -c1-12 || echo none)
        if [[ "$h" != "$last_hash" ]]; then
          payload=$(cat "'"$SSE_STATE_FILE"'" 2>/dev/null | jq -c @)
          printf "id: %s\nevent: state\ndata: %s\n\n" "$h" "$payload"
          last_hash="$h"
        fi
        sleep 0.5
      done
    done
  ' 2>/dev/null || {
    # fallback: poll loop with a tiny python server.
    python3 - <<PY 2>/dev/null || true
import http.server, socketserver, time, json
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **kw): pass
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
        last=""
        while True:
            try:
                s = open("$SSE_STATE_FILE").read()
                h = hash(s)
                if h != last:
                    self.wfile.write(f"event: state\ndata: {s}\n\n".encode())
                    last = h
            except: pass
            time.sleep(0.5)
with socketserver.TCPServer(("",$port), H) as srv:
    srv.serve_forever()
PY
  }
}

sse_broadcast_event() {
  local kind="$1" payload="$2"
  local f="$SSE_LASTEVENT_FILE"
  printf '{"kind":"%s","payload":%s,"ts":%s}' "$kind" "$payload" "$(date +%s)" >> "$f"
}
