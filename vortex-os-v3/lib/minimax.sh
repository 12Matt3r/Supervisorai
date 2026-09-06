#!/bin/bash
# ----------------------------------------------------------------------------
# VORTEX-OS Module: Native MiniMax-M3 Bridge (GMI Cloud)
# ----------------------------------------------------------------------------
# Replaces the hardcoded stub responses with real MiniMax-M3 calls.
# Endpoint : https://api.gmi-serving.com/v1/messages  (Anthropic Messages format)
# Auth     : x-api-key: $GMI_API_KEY
# Model    : $SUPERVISOR_MODEL (default MiniMaxAI/MiniMax-M3)
#
# Env: GMI_API_KEY, GMI_BASE_URL, SUPERVISOR_MODEL. If GMI_API_KEY is unset the
# bridge reports "unconfigured" so callers can fall back to a deterministic stub
# (the demo never hard-crashes).
# ----------------------------------------------------------------------------

GMI_BASE_URL="${GMI_BASE_URL:-https://api.gmi-serving.com/v1}"
SUPERVISOR_MODEL="${SUPERVISOR_MODEL:-MiniMaxAI/MiniMax-M3}"

minimax_is_configured() {
  [[ -n "${GMI_API_KEY:-}" ]]
}

# minimax_chat <prompt> [max_tokens] [system]
# Echoes the model's text output on stdout; returns non-zero on any failure.
minimax_chat() {
  local prompt="$1"
  local max_tokens="${2:-2048}"
  local system="${3:-You are a precise worker agent inside the VORTEX-OS orchestrator. Return exactly what the task asks for, complete and self-contained.}"
  minimax_is_configured || return 3

  local payload
  payload=$(jq -n --arg m "$SUPERVISOR_MODEL" --arg s "$system" --arg p "$prompt" \
    --argjson mt "$max_tokens" \
    '{model:$m, max_tokens:$mt, system:$s, messages:[{role:"user", content:$p}]}')

  local body
  body=$(curl -sS -m 120 --request POST \
    --url "${GMI_BASE_URL}/messages" \
    -H 'Content-Type: application/json' \
    -H "x-api-key: ${GMI_API_KEY}" \
    --data "$payload" 2>/dev/null) || return 4

  # Extract concatenated text blocks from the Anthropic-format response.
  local text
  text=$(echo "$body" | jq -r '[.content[]? | select(.type=="text") | .text] | join("")' 2>/dev/null)
  if [[ -z "$text" || "$text" == "null" ]]; then
    # Surface API errors to the audit log for debugging, then fail.
    echo "$body" | jq -r '.error.message? // empty' >&2 2>/dev/null
    return 5
  fi
  printf '%s' "$text"
}

# minimax_json <prompt> [max_tokens] — like minimax_chat but strips ```json fences
# and prints only the JSON object/array (best-effort).
minimax_json() {
  local raw
  raw=$(minimax_chat "$1" "${2:-2048}" "You are the VORTEX-OS planning core. Respond with a SINGLE valid JSON object and no prose outside it.") || return $?
  # Strip code fences if present.
  raw=$(printf '%s' "$raw" | sed -e 's/^```json//' -e 's/^```//' -e 's/```$//')
  # Trim to first { .. last } if there is surrounding text.
  echo "$raw" | jq -c '.' 2>/dev/null || printf '%s' "$raw"
}
