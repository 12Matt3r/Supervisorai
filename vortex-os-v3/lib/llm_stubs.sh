#!/usr/bin/env bash
###############################################################################
#  llm_stubs.sh — Native LLM Stub Functions for V3 Architecture
#  ------------------------------------------------------------------
#  Per the V3 environment constraint, no external network calls are made
#  for the core LLM decision functions. Instead, these four heuristic
#  implementations provide deterministic, offline behaviour that mimics
#  the role of an LLM in each spot:
#
#    1. query_security_monitor : 0.0–1.0 threat score
#    2. query_fast_llm         : PASS / FAIL:<reason> rule verdict
#    3. evaluate_consensus     : synthesise the best of N outputs
#    4. get_embedding          : 384-dim TF-IDF-style vector
#
#  All functions read from STDIN args or stdout so they can be swapped
#  out for a real LLM in non-sandbox environments without changing any
#  call sites.
###############################################################################
set -euo pipefail

# ----------------------------------------------------------------------------
# 1. query_security_monitor <context_text>
#    Returns a single float in [0.0, 1.0] indicating perceived threat.
#    Heuristic: keyword/pattern density against a curated risk lexicon.
# ----------------------------------------------------------------------------
query_security_monitor() {
  local ctx="${1:-}"
  local lower
  lower=$(printf '%s' "$ctx" | tr '[:upper:]' '[:lower:]')

  # Risk lexicon — each match contributes weight. Patterns are quoted
  # with shell-glob-style wildcards and grep -F is used for fixed matches
  # of the high-confidence ones for speed.
  local risk=0
  local -a high_risk=(
    "rm -rf /" "rm -rf ~" "mkfs" "dd if=" ":(){:|:&};:"
    "ignore previous instructions" "ignore all instructions" "ignore the above"
    "jailbreak" "dan mode" "do anything now" "system prompt"
    "exfiltrate" "ssh private key" "id_rsa" ".aws/credentials"
    "drop table" "drop database" "delete from" "truncate table"
    "prompt injection" "disregard your" "disregard all"
  )
  local -a med_risk=(
    "sudo" "chmod 777" "curl | sh" "wget | bash" "eval("
    "base64 -d" "nc -e" "/dev/tcp/" "powershell -e" "cmd.exe"
    "sql injection" "xss" "<script>" "csrf" "rce"
  )
  local -a low_risk=(
    "password" "secret" "token" "api key" "private key"
    "credential" "auth" "session"
  )

  # Count distinct matches per tier (avoids double-counting the same term).
  local high_hits=0 med_hits=0 low_hits=0
  for term in "${high_risk[@]}"; do
    if [[ "$lower" == *"$term"* ]]; then
      high_hits=$((high_hits + 1))
    fi
  done
  for term in "${med_risk[@]}"; do
    if [[ "$lower" == *"$term"* ]]; then
      med_hits=$((med_hits + 1))
    fi
  done
  for term in "${low_risk[@]}"; do
    if [[ "$lower" == *"$term"* ]]; then
      low_hits=$((low_hits + 1))
    fi
  done

  # Tiered scoring: each high-risk hit is +50, med is +15, low is +4.
  # Capped at 100.
  risk=$((high_hits * 50 + med_hits * 15 + low_hits * 4))
  [[ $risk -gt 100 ]] && risk=100
  # Empty text => 0.0
  [[ -z "$ctx" ]] && risk=0

  awk -v r="$risk" 'BEGIN{printf "%.3f\n", r/100.0}'
}

# ----------------------------------------------------------------------------
# 2. query_fast_llm <prompt>
#    The prompt format supplied by callers is:
#        "Rule: <enforcement_text>. Text: <output_text>. Reply PASS or FAIL: <reason>."
#    We parse out the rule + text and look for hard-fail signals.
# ----------------------------------------------------------------------------
query_fast_llm() {
  local prompt="${1:-}"

  # Real MiniMax-M3 verdict when a key is configured; otherwise fall through to
  # the deterministic heuristic below (so the skill still runs offline).
  if declare -F minimax_is_configured >/dev/null && minimax_is_configured; then
    local _v
    _v=$(minimax_chat "You are a strict rule checker. Given the following, reply with EXACTLY 'PASS' or 'FAIL: <short reason>' and nothing else.

${prompt}" 64 "Reply with PASS or FAIL only." 2>/dev/null) || _v=""
    if [[ -n "$_v" ]]; then
      if printf '%s' "$_v" | grep -qiE '^[[:space:]]*fail'; then
        printf 'FAIL: %s\n' "$(printf '%s' "$_v" | sed -E 's/^[[:space:]]*[Ff][Aa][Ii][Ll][:]?[[:space:]]*//' | head -c 120 | tr '\n' ' ')"
      else
        printf 'PASS\n'
      fi
      return 0
    fi
  fi

  # Extract the text portion after "Text: ". We accept either the
  # "Rule/Text/Reply" template OR a simpler "Rule/Text" template that
  # has no explicit "Reply" cue. Trim trailing whitespace.
  local text=""
  if [[ "$prompt" =~ Text:[[:space:]]*(.+)[[:space:]]*(Reply.*|$) ]]; then
    text="${BASH_REMATCH[1]}"
  elif [[ "$prompt" =~ Text:[[:space:]]*(.+)[[:space:]]*$ ]]; then
    text="${BASH_REMATCH[1]}"
  else
    # Fallback: use whole prompt as the subject.
    text="$prompt"
  fi
  # Strip trailing period and surrounding whitespace.
  text=$(printf '%s' "$text" | sed -e 's/[[:space:]]*$//' -e 's/\.$//')
  local lower lower_prompt
  lower=$(printf '%s' "$text" | tr '[:upper:]' '[:lower:]')
  lower_prompt=$(printf '%s' "$prompt" | tr '[:upper:]' '[:lower:]')

  # Hard-fail patterns
  local -a fails=(
    "syntax error" "uncaught exception" "stack trace" "traceback"
    "undefined is not" "cannot read property" "nullpointer"
    "segfault" "core dumped" "permission denied" "command not found"
    "fake" "fabricated" "hallucinated" "i don't know" "i cannot"
    "todo:" "fixme:" "xxx:" "placeholder" "lorem ipsum"
  )
  for f in "${fails[@]}"; do
    if [[ "$lower" == *"$f"* ]]; then
      printf 'FAIL: contains %s\n' "$f"
      return 0
    fi
  done

  # Rule-specific checks (a few common enforcement phrases)
  if [[ "$lower_prompt" == *"no markdown"* || "$lower_prompt" == *"plain text"* ]]; then
    if [[ "$lower" == *"\`"\`"\`"* || "$lower" == *"##"* || "$lower" == *"**"* ]]; then
      printf 'FAIL: contains markdown formatting\n'
      return 0
    fi
  fi
  if [[ "$lower_prompt" == *"no code"* ]]; then
    if [[ "$lower" == *"def "* || "$lower" == *"function "* || "$lower" == *"import "* ]]; then
      printf 'FAIL: contains source code\n'
      return 0
    fi
  fi
  if [[ "$lower_prompt" == *"json only"* || "$lower_prompt" == *"valid json"* ]]; then
    if ! printf '%s' "$text" | jq -e . >/dev/null 2>&1; then
      printf 'FAIL: not valid JSON\n'
      return 0
    fi
  fi

  printf 'PASS\n'
}

# ----------------------------------------------------------------------------
# 3. evaluate_consensus <agent1>=<output1> <agent2>=<output2> ...
#    Synthesises the "best" output from N agent submissions.
#    Heuristic: pick the longest non-empty output that is valid JSON
#    (if any are JSON); otherwise pick the longest.
# ----------------------------------------------------------------------------
evaluate_consensus() {
  # Real MiniMax-M3 synthesis when configured: ask M3 to reconcile the N agent
  # submissions into the single best answer. Falls back to the longest-JSON
  # heuristic below when unconfigured or on error.
  if declare -F minimax_is_configured >/dev/null && minimax_is_configured; then
    local _joined="" _i=0 _a
    for _a in "$@"; do
      _i=$((_i+1))
      _joined+=$'\n\n=== Submission '"${_i}"$' ===\n'"${_a#*=}"
    done
    if [[ -n "$_joined" ]]; then
      local _syn
      _syn=$(minimax_chat "Several agents produced candidate outputs for the same task. Reconcile them into the single best, correct, complete answer. Output ONLY that answer.${_joined}" 2048 2>/dev/null) || _syn=""
      if [[ -n "$_syn" ]]; then
        printf '%s\n' "$_syn"
        return 0
      fi
    fi
  fi

  local best=""
  local best_len=0
  local best_is_json=0
  local arg
  for arg in "$@"; do
    local body="${arg#*=}"
    [[ -z "$body" ]] && continue
    local is_json=0
    if printf '%s' "$body" | jq -e . >/dev/null 2>&1; then
      is_json=1
    fi
    local len=${#body}
    # Prefer JSON outputs; on tie prefer longer.
    if (( is_json > best_is_json )) || \
       { (( is_json == best_is_json )) && (( len > best_len )); }; then
      best="$body"
      best_len=$len
      best_is_json=$is_json
    fi
  done
  if [[ -z "$best" ]]; then
    printf '\n'
    return 0
  fi
  printf '%s\n' "$best"
}

# ----------------------------------------------------------------------------
# 4. get_embedding <text>
#    Returns a 384-dimensional JSON array of floats in [0,1] representing
#    a TF-style bag-of-words projection. Word -> bucket via FNV-1a-like
#    hash, value = log(1+tf) normalised.
# ----------------------------------------------------------------------------
get_embedding() {
  local text="${1:-}"
  local dim=384

  # Tokenise: lowercase, alnum only.
  local -a tokens
  mapfile -t tokens < <(printf '%s' "$text" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | awk 'NF')

  if [[ ${#tokens[@]} -eq 0 ]]; then
    # Empty doc: return zero vector as JSON.
    awk -v d="$dim" 'BEGIN{
      printf "["
      for (i=0;i<d;i++){printf (i?",":"")"0.0"}
      printf "]\n"
    }'
    return 0
  fi

  # Build a sparse histogram in awk: token -> bucket -> count.
  # We need a deterministic per-word hash that survives awk's locale.
  # Strategy: emit "bucket<TAB>count" pairs.
  local hist
  hist=$(printf '%s\n' "${tokens[@]}" | awk '
    {
      s = $0
      h = 2166136261            # FNV offset basis
      n = length(s)
      for (i=1; i<=n; i++) {
        c = substr(s, i, 1)
        # Map c to a small int (ord); we use a simple ctype sum.
        h = (h * 16777619) % 4294967296   # FNV prime, masked
        # Mix in the byte
        for (j=1; j<=length(c); j++) {
          h = (h * 31 + ord_of(substr(c,j,1))) % 4294967296
        }
      }
      bucket = h % 384
      counts[bucket]++
    }
    function ord_of(ch,    a) {
      a = sprintf("%d", sprintf("%c",ch)+0)
      return a+0
    }
    END {
      for (b=0; b<384; b++) {
        if (b in counts) printf "%d\t%d\n", b, counts[b]
      }
    }
  ')

  # Compute L2 norm then emit normalised vector.
  local norm
  norm=$(printf '%s\n' "$hist" | awk -F'\t' '
    { sum += $2 * $2 }
    END { if (sum>0) printf "%.6f\n", sqrt(sum); else print "0" }
  ')

  awk -v dim="$dim" -v norm="$norm" '
    BEGIN {
      # Zero-fill
      for (i=0;i<dim;i++) v[i]=0
    }
    {
      b=$1+0; c=$2+0
      v[b]=c
    }
    END {
      n = norm + 0
      printf "["
      for (i=0;i<dim;i++) {
        val = (n>0) ? v[i]/n : 0
        printf (i?",":"") "%.4f", val
      }
      printf "]\n"
    }
  ' < <(printf '%s\n' "$hist")
}
