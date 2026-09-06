#!/bin/bash
# Real MiniMax-M3 adapter (GMI Cloud). Reads the prompt on stdin, emits the
# llm_adapter envelope: {"ok":true,"adapter":"minimax_m3","tokens_out":N,"text":"..."}.
# Falls back to ok:false so the adapter chain can try the next adapter.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/lib/minimax.sh" 2>/dev/null || true
prompt="$(cat)"
if declare -F minimax_chat >/dev/null && minimax_is_configured; then
  text="$(minimax_chat "$prompt" 2048 2>/dev/null)" || text=""
  if [[ -n "$text" ]]; then
    jq -cn --arg t "$text" --argjson tok "$(( ${#text} / 4 ))" \
      '{ok:true, adapter:"minimax_m3", tokens_in:0, tokens_out:$tok, cost_usd:0, confidence:0.9, text:$t}'
    exit 0
  fi
fi
echo '{"ok":false,"adapter":"minimax_m3","reason":"unconfigured_or_error"}'
