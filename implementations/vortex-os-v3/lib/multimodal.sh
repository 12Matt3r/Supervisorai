# lib/multimodal.sh — Improvement #12: Multi-Modal Agent Kinds
# Adds kinds: image, audio, video. Definition schema for multimodal agents.
set -euo pipefail
LIB_PREFIX="[multimodal]"

MULTIMODAL_KINDS=("image" "audio" "video" "vision_understand")

multimodal_supported() { printf '%s\n' "${MULTIMODAL_KINDS[@]}"; }

# Build an input envelope from a media url + prompt.
multimodal_envelope_input() {
  local media_url="$1" prompt="$2" kind="${3:-image}"
  jq -nc --arg u "$media_url" --arg p "$prompt" --arg k "$kind" \
    '{kind:$k, media_url:$u, prompt:$p}'
}

# Emit a multimodal result envelope that downstream tasks can ingest.
multimodal_envelope_output() {
  local task_id="$1" ok="$2" media_url="$3"
  jq -nc --arg t "$task_id" --argjson ok "$ok" --arg m "$media_url" \
    '{task:$t, ok:$ok, media_url:$m, ended_at:(now|todate)}'
}
