# lib/llm_adapter.sh — Improvement #8: Adapter Pattern (one agent, many LLMs)
# Adapter chain with fallback (e.g. opus -> sonnet -> haiku).
set -euo pipefail
LIB_PREFIX="[llm_adapter]"

ADAPTERS_DIR="${ORCH_ADAPTERS_DIR:-./adapters}"

llm_adapter_init() {
  mkdir -p "$ADAPTERS_DIR"
  # minimax_m3 is a REAL adapter (adapters/minimax_m3.sh, shipped with the skill)
  # that calls MiniMax-M3 on GMI Cloud. It is listed first so chains prefer it,
  # and the `[[ -f ]] ||` guard never overwrites it with an echo stub.
  for a in minimax_m3 claude_3_5_sonnet gpt_4o claude_3_opus gpt_4o_mini llama_3_1_70b groq_mixtral ollama_llama3; do
    [[ -f "$ADAPTERS_DIR/$a.sh" ]] || cat > "$ADAPTERS_DIR/$a.sh" <<EOF
#!/bin/bash
# Adapter stub: $a
echo '{"ok":true,"adapter":"$a","tokens_in":1,"tokens_out":1,"cost_usd":0.001,"confidence":0.9,"text":"echoed $a"}'
EOF
    chmod +x "$ADAPTERS_DIR/$a.sh"
  done
}

# Resolve a chain like "opus -> sonnet -> haiku" into ["opus","sonnet","haiku"].
llm_adapter_parse_chain() {
  local chain="$1"
  echo "$chain" | tr '>' ',' | tr -d ' ' | tr ',' '\n' | sed '/^$/d'
}

llm_adapter_invoke() {
  local chain="$1" prompt="$2"
  local n="${3:-0}"
  while IFS= read -r adapter; do
    [[ -z $adapter ]] && continue
    local s="$ADAPTERS_DIR/$adapter.sh"
    if [[ -f $s ]]; then
      out=$(printf '%s' "$prompt" | "$s" 2>/dev/null || true)
      if [[ -n $out ]] && jq -e '.ok' <<<"$out" >/dev/null 2>&1; then
        jq --arg a "$adapter" --argjson n "$n" '. + {adapter:$a, attempt:$n}' <<<"$out"
        return 0
      fi
    fi
    n=$((n+1))
  done < <(llm_adapter_parse_chain "$chain")
  return 1
}

# Promote: pick the cheapest adapter that hits a quality target.
llm_adapter_pick_for_budget() {
  local budget_usd="$1"
  if   (( $(echo "$budget_usd < 0.01" | bc -l) )); then echo "ollama_llama3"
  elif (( $(echo "$budget_usd < 0.05" | bc -l) )); then echo "claude_3_5_sonnet"
  else                                                  echo "claude_3_opus"
  fi
}
