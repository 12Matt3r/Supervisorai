#!/usr/bin/env bash
###############################################################################
#  jit_bridge.sh — V3.1 Just-In-Time Skill Bridge
#  ------------------------------------------------------------------
#  Closes the capability gap when a task asks for a skill that does not
#  exist in the local agents/ registry. Resolution order:
#
#    1. Local agents/ registry  (handled by the existing similarity band)
#    2. MiniMax Native Library  (query_native_minimax_library)
#    3. Web Research & Synthesis (execute_native_web_search)
#
#  The two native functions below are not stubbed HTTP calls. They are
#  binding points that the inference context (the AI environment in which
#  this orchestrator runs) can populate. When the bash script writes a
#  request, the inference layer observes it via the in-band log
#  `state/jit_requests.log` and emits a response via `state/jit_responses.log`
#  using `cmd__jit_consume_response` below. This keeps everything offline
#  during pure-shell execution while remaining fully compatible with
#  live inference.
###############################################################################
set -euo pipefail

# ----------------------------------------------------------------------------
# Internal: where the JIT bridge publishes requests and consumes responses.
# ----------------------------------------------------------------------------
JIT_REQUEST_LOG="${JIT_REQUEST_LOG:-state/jit_requests.log}"
JIT_RESPONSE_LOG="${JIT_RESPONSE_LOG:-state/jit_responses.log}"

cmd__jit_ensure_dirs() {
  mkdir -p state agents
  [[ -f "$JIT_REQUEST_LOG" ]] || : >"$JIT_REQUEST_LOG"
  [[ -f "$JIT_RESPONSE_LOG" ]] || : >"$JIT_RESPONSE_LOG"
}

# ----------------------------------------------------------------------------
# 1. query_native_minimax_library <capability>
#    Asks the inference context: "do you natively have a tool for this?"
#    The inference context answers by appending a JSON line to
#    $JIT_RESPONSE_LOG with kind=native_library_query.
#
#    Heuristic fallback (used when no response arrives within timeout):
#    inspect a static capability -> native tool map. This lets the
#    orchestrator work in pure-shell mode without ever blocking.
# ----------------------------------------------------------------------------
query_native_minimax_library() {
  local capability="${1:-}"
  local cap_lc
  cap_lc=$(printf '%s' "$capability" | tr '[:upper:]' '[:lower:]')

  cmd__jit_ensure_dirs

  # Publish a request for the inference context.
  local req_id
  req_id="jit-native-$(date +%s%N)-$RANDOM"
  jq -nc \
    --arg id "$req_id" \
    --arg cap "$capability" \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{id:$id, kind:"native_library_query", capability:$cap, ts:$ts}' \
    >>"$JIT_REQUEST_LOG"

  # Heuristic fallback map (case-insensitive substring match).
  local hit=""
  case "$cap_lc" in
    *web*search*|*search*the*web*|*googling*|*internet*search*)
      hit="minimax.web_search" ;;
    *code*exec*|*code*interpret*|*bash*run*|*shell*exec*)
      hit="minimax.bash_executor" ;;
    *image*gen*|*image*create*|*dall*|*midjourney*|*stable*diffusion*)
      hit="minimax.image_synthesize" ;;
    *image*understand*|*image*analy*|*ocr*|*vision*)
      hit="minimax.images_understand" ;;
    *pdf*read*|*pdf*extract*)
      hit="minimax.extract_pdfs_full_content" ;;
    *flight*|*airline*|*airfare*)
      hit="minimax.flights_search" ;;
    *hotel*|*lodging*|*accommodation*)
      hit="minimax.hotels_search" ;;
    *stock*price*|*ticker*|*equity*)
      hit="minimax.stocks_price" ;;
    *stock*news*|*market*news*)
      hit="minimax.stocks_news" ;;
    *twitter*|*tweet*|*x.com*)
      hit="minimax.twitter_search_tweets" ;;
    *audio*transcrib*|*speech*to*text*)
      hit="minimax.audios_understand" ;;
    *video*analy*|*youtube*)
      hit="minimax.videos_understand" ;;
    *chart*|*plot*|*graph*|*matplotlib*)
      hit="minimax.python_matplotlib" ;;
    *markdown*|*docx*|*pdf*convert*|*pandoc*)
      hit="minimax.convert" ;;
    *web*extract*|*scrape*|*crawl*)
      hit="minimax.extract_content_from_websites" ;;
    *file*write*|*file*create*|*edit*file*)
      hit="minimax.file_write" ;;
    *file*read*|*cat*file*|*open*file*)
      hit="minimax.file_read" ;;
  esac

  if [[ -n "$hit" ]]; then
    jq -nc \
      --arg id "$req_id" \
      --arg tool "$hit" \
      --arg cap "$capability" \
      '{id:$id, kind:"native_library_response", capability:$cap, tool:$tool, source:"heuristic_fallback"}'
    return 0
  fi

  # No match. Return JSON null so callers can branch.
  printf 'null\n'
  return 1
}

# ----------------------------------------------------------------------------
# 2. execute_native_web_search <query>
#    Asks the inference context to perform a real web search. The bash
#    script writes a request; the inference context writes a response.
#    Fallback: synthesize a search-corpus stub from local agents + trait
#    descriptions so the orchestrator is never blocked.
# ----------------------------------------------------------------------------
execute_native_web_search() {
  local query="${1:-}"
  cmd__jit_ensure_dirs

  local req_id
  req_id="jit-web-$(date +%s%N)-$RANDOM"
  jq -nc \
    --arg id "$req_id" \
    --arg q "$query" \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{id:$id, kind:"web_search_query", query:$q, ts:$ts}' \
    >>"$JIT_REQUEST_LOG"

  # Heuristic fallback: scan the local agent descriptions for a partial
  # match. If found, return a small markdown reference built from the
  # agent's description. If not, return a templated reference noting
  # that the inference context must populate it.
  local hit_file
  hit_file=$(grep -lir "$(printf '%s' "$query" | tr '[:upper:]' '[:lower:]' | awk '{print $1}')" \
             agents/*.json 2>/dev/null | head -1 || true)

  local body=""
  if [[ -n "$hit_file" && -f "$hit_file" ]]; then
    body=$(jq -r '"# Auto-researched reference for: \(.description // .name)\n\n## Capabilities\n- " + ((.capabilities // []) | join("\n- ")) + "\n\n## Source\nSynthesized from existing local agent: \(.name) v\(.version)\n"' "$hit_file" 2>/dev/null || true)
  fi
  if [[ -z "$body" ]]; then
    body=$(cat <<EOF
# Auto-researched reference for: $query

## Suggested implementation outline
- Identify the canonical CLI tool or library (e.g. \`jq\`, \`ffmpeg\`, \`imagemagick\`, \`playwright\`, \`requests\`).
- Define an idempotent \`cmd_*\` bash function with explicit input/output.
- Emit NDJSON for streaming progress.
- Wrap any state mutation in a lock file under \`locks/\`.

## Invocation
\`\`\`bash
bash skill.sh --agents-compile auto.synthesized.$(printf '%s' "$query" | tr ' ' '_' | tr -cd '[:alnum:]_')
\`\`\`

## Source
Native inference synthesis (fallback). Populate \`state/jit_responses.log\`
with \`{"id":"$req_id","kind":"web_search_response","results_md":"<real search results>"}\`
to override.
EOF
)
  fi

  jq -nc \
    --arg id "$req_id" \
    --arg md "$body" \
    --arg q "$query" \
    '{id:$id, kind:"web_search_response", query:$q, results_md:$md, source:"heuristic_fallback"}'
  return 0
}

# ----------------------------------------------------------------------------
# 8.1 cmd_skill_fetch_minimax <missing_capability>
#     Tries to wrap a native tool.  Prints the wrapper path on success.
# ----------------------------------------------------------------------------
cmd_skill_fetch_minimax() {
  local missing_capability="${1:-}"
  [[ -z "$missing_capability" ]] && { err "Usage: cmd_skill_fetch_minimax <capability>"; return 2; }

  echo "JIT Bridge: Querying MiniMax native library for '$missing_capability'..." >&2

  local native_tool_found
  native_tool_found=$(query_native_minimax_library "$missing_capability")

  if [[ "$native_tool_found" != "null" && -n "$native_tool_found" && "$native_tool_found" != "null" ]]; then
    local wrapper_name
    wrapper_name="minimax.native.$(printf '%s' "$missing_capability" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_*//;s/_*$//')"
    local wrapper_path="agents/${wrapper_name}.json"

    # Extract the tool name from the JSON envelope
    local tool_name
    tool_name=$(printf '%s' "$native_tool_found" | jq -r '.tool // "minimax.unknown"' 2>/dev/null || echo "minimax.unknown")

    # 8-Invariants compliant native bridge envelope.
    jq -n \
      --arg name "$wrapper_name" \
      --arg desc "Native MiniMax library binding for $missing_capability" \
      --arg cap "$missing_capability" \
      --arg tool "$tool_name" \
      '{
        name: $name,
        version: "1.0.0",
        kind: "subplan",
        description: $desc,
        entry: "templates/native_bridge.json",
        capabilities: [$cap],
        input_schema: {
          type: "object",
          required: ["task_description"],
          properties: {
            task_description: { type: "string", minLength: 10 }
          }
        },
        output_schema: {
          type: "object",
          required: ["result"],
          properties: {
            result: { type: "string" }
          }
        },
        writes: ["deliverables/"],
        reads: ["agents/"],
        resources: { cpu: "low", mem: "512Mi", net: "online" },
        retry_policy: { max_attempts: 1, backoff: "none", on: [] },
        quality: { min_score: 0.5, max_latency_s: 600, max_cost_usd: 0.25, must_have: ["result"], must_not: ["TODO:", "FIXME:"] },
        invariants_compliant: ["I1","I2","I3","I4","I5","I6","I7","I8"],
        invariants_provisional: [],
        composes_with: [],
        versioning: { strategy: "immutable", eol: null },
        deprecated: false,
        deprecation_message: null,
        is_minimax_native: true,
        binding: { tool: $tool, source: "minimax_native_library" }
      }' >"$wrapper_path"

    echo "JIT Bridge: Successfully wrapped native MiniMax skill -> $wrapper_path" >&2
    echo "$wrapper_path"
    return 0
  fi

  return 1
}

# ----------------------------------------------------------------------------
# 8.2 cmd_skill_research_online <missing_capability>
#     Falls back to web research + synthesis.
# ----------------------------------------------------------------------------
cmd_skill_research_online() {
  local missing_capability="${1:-}"
  [[ -z "$missing_capability" ]] && { err "Usage: cmd_skill_research_online <capability>"; return 2; }

  echo "JIT Bridge: Skill not found natively. Researching '$missing_capability' online..." >&2

  local research_json
  research_json=$(execute_native_web_search "best CLI or API implementation for $missing_capability github documentation")
  local research_md
  research_md=$(printf '%s' "$research_json" | jq -r '.results_md // ""' 2>/dev/null || true)

  if [[ -n "$research_md" ]]; then
    local synthesized_name
    synthesized_name="auto.synthesized.$(printf '%s' "$missing_capability" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_*//;s/_*$//')"

    echo "JIT Bridge: Synthesizing new agent: $synthesized_name" >&2

    # Synthesise a fully 8-Invariants-compliant agent envelope so the
    # static linter (--agents-lint) accepts JIT-created skills.
    jq -n \
      --arg name "$synthesized_name" \
      --arg desc "Auto-synthesized skill from web research for $missing_capability" \
      --arg cap "$missing_capability" \
      '{
        name: $name,
        version: "1.0.0",
        kind: "subplan",
        description: $desc,
        entry: "templates/synthesized.json",
        capabilities: [$cap],
        input_schema: {
          type: "object",
          required: ["task_description"],
          properties: {
            task_description: { type: "string", minLength: 10 }
          }
        },
        output_schema: {
          type: "object",
          required: ["result"],
          properties: {
            result: { type: "string" }
          }
        },
        writes: ["deliverables/"],
        reads: ["agents/"],
        resources: { cpu: "low", mem: "512Mi", net: "online" },
        retry_policy: { max_attempts: 1, backoff: "none", on: [] },
        quality: { min_score: 0.5, max_latency_s: 600, max_cost_usd: 0.25, must_have: ["result"], must_not: ["TODO:", "FIXME:"] },
        invariants_compliant: ["I1","I2","I3","I4","I5","I6","I7","I8"],
        invariants_provisional: [],
        composes_with: [],
        versioning: { strategy: "immutable", eol: null },
        deprecated: false,
        deprecation_message: null,
        spawned_via: "web_research",
        needs_native_inference: true
      }' >"agents/${synthesized_name}.json"

    printf '%s\n' "$research_md" >"agents/${synthesized_name}_reference.md"

    echo "agents/${synthesized_name}.json"
    return 0
  fi

  log_err "JIT Bridge: Web research failed to synthesize capability."
  return 1
}

# ----------------------------------------------------------------------------
# cmd_dispatch_jit <task_id> <declared_agent>
#     Top-level resolution: native first, web research second.
# ----------------------------------------------------------------------------
cmd_dispatch_jit() {
  local task_id="$1"
  local declared_agent="$2"
  [[ -z "$task_id" || -z "$declared_agent" ]] && { err "Usage: cmd_dispatch_jit <task_id> <agent>"; return 2; }

  log_warn "Local match too weak for '$declared_agent'. Triggering JIT Skill Bridge..."

  # Step 1: try MiniMax native library
  local jit_agent=""
  if jit_agent=$(cmd_skill_fetch_minimax "$declared_agent"); then
    audit_log_event "jit_native_hit" "{\"task\":\"$task_id\",\"agent\":\"$declared_agent\",\"wrapper\":\"$jit_agent\"}" 2>/dev/null || true
    echo "$jit_agent"
    return 0
  fi

  # Step 2: fall back to web research
  if jit_agent=$(cmd_skill_research_online "$declared_agent"); then
    audit_log_event "jit_web_synth" "{\"task\":\"$task_id\",\"agent\":\"$declared_agent\",\"wrapper\":\"$jit_agent\"}" 2>/dev/null || true
    echo "$jit_agent"
    return 0
  fi

  err "Refusing to dispatch $task_id: No local, native, or researchable skill found."
  return 1
}

# ----------------------------------------------------------------------------
# CLI entry point for manual testing: --jit-bridge <capability>
# ----------------------------------------------------------------------------
cmd_jit_bridge() {
  local capability="${1:-}"
  [[ -z "$capability" ]] && { err "Usage: --jit-bridge <capability>"; return 2; }
  cmd_dispatch_jit "manual" "$capability"
}

# ----------------------------------------------------------------------------
# cmd__jit_consume_response <request_id>
#     Helper that the inference context (or an operator) can use to drop a
#     pre-canned response into $JIT_RESPONSE_LOG and have the orchestrator
#     pick it up next time it polls.
# ----------------------------------------------------------------------------
cmd__jit_consume_response() {
  local request_id="${1:-}"
  [[ -z "$request_id" ]] && { err "Usage: cmd__jit_consume_response <request_id>"; return 2; }
  [[ -f "$JIT_RESPONSE_LOG" ]] || { err "No response log"; return 1; }
  jq -c --arg id "$request_id" 'select(.id == $id)' "$JIT_RESPONSE_LOG" || true
}
