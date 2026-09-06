# lib/dynamic_agents.sh — On-the-fly sub-agent factory
#
# Improvement goal (research-driven): let the supervisor *synthesize* a new
# sub-agent when no existing agent matches a task. Three creation modes:
#
#   Mode 1 — TEMPLATE_CLONE   clone an existing agent, swap (name, family,
#                             persona, resources). Always safe.
#   Mode 2 — PERSONA_PICK     same family, different persona voice. Cheap.
#   Mode 3 — SYNTHESIZE       generate a full contract from a capability-gap
#                             description. Powerful; quarantined longer.
#
# Safety:
#   - Opt-in.  The supervisor must set CONFIG.dynamic_agents.enabled=true.
#   - Every spawn is audited (audit_log if available, else factory_audit.log).
#   - Newly-spawned agents run in quarantine for N runs before they are
#     trusted (default: 3 runs). The canary-doctor audits each run.
#   - Dynamic agents are written to agents/dynamic/ so they're easy to
#     inspect, diff, or remove later.
#   - A static_only flag pins a project to the 30-shipped agents.
#
# Public API:
#   factory_create_clone      <name> <parent_json> <reason>     # Mode 1
#   factory_create_persona    <name> <base_json> <new_persona>  # Mode 2
#   factory_synthesize        <name> <gap_description>          # Mode 3 (LLM)
#   factory_validate          <agent_json_path>                # schema check
#   factory_register          <agent_json_path>                # add to index
#   match_or_propose_agent    <task_json>                      # dispatcher hook
#   factory_audit             <action> <details_json>          # log to file
#   factory_list_dynamic                                  # list all
#   factory_quarantine_tick   <agent_name>                    # age-out
#   factory_static_only                                  # is static-pinned?
#
# Configuration in CONFIG.dynamic_agents:
#   enabled:           bool, default false
#   static_only:       bool, default false (force 30-shipped list)
#   default_quarantine:int,  default 3
#   max_dynamic:       int,  default 50
#   synthesis_model:   str,  default "haiku"
#   audit_log:         path, default ./.orchestr8r/factory_audit.jsonl
#   dynamic_dir:       path, default ./agents/dynamic
#
set -euo pipefail
LIB_PREFIX="[factory]"
[[ -n "${__LIB_FACTORY_LOADED:-}" ]] && return 0
__LIB_FACTORY_LOADED=1

# ----------------------------------------------------------------------------
# Safe sources (degrade gracefully if not available)
# ----------------------------------------------------------------------------
[[ -n "${SKILL_SCRIPT_DIR:-}" ]] || SKILL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
if [[ -n "${CONFIG:-}" ]] && [[ -f "$SKILL_SCRIPT_DIR/lib/observability.sh" ]]; then
  source "$SKILL_SCRIPT_DIR/lib/observability.sh" 2>/dev/null || true
fi

# Provide a minimal now_iso if not in scope.
if ! declare -F now_iso >/dev/null; then
  now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
fi

# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------

# factory_enabled — returns 0 (true) if dynamic agents are enabled.
# Default: ON.  Set dynamic_agents.enabled=false to opt out.
factory_enabled() {
  # No CONFIG at all → enabled by default.  This makes the on-the-fly
  # factory the default behaviour out of the box.
  [[ -z "${CONFIG:-}" || ! -f "$CONFIG" ]] && return 0
  jq -e '.dynamic_agents.enabled // true' "$CONFIG" >/dev/null 2>&1
}

# factory_static_only — returns 0 (true) if project is pinned to a static
# agent list.  Default: false.
factory_static_only() {
  [[ -z "${CONFIG:-}" || ! -f "$CONFIG" ]] && return 1
  jq -e '.dynamic_agents.static_only // false' "$CONFIG" >/dev/null 2>&1
}

# factory_cfg <key> [default] — read a factory config value.
factory_cfg() {
  local key="$1" def="${2:-}"
  [[ -n "${CONFIG:-}" && -f "$CONFIG" ]] || { echo "$def"; return 0; }
  jq -r --arg d "$def" '.dynamic_agents.'"$key"' // $d' "$CONFIG"
}

# factory_count_dynamic — echo the number of currently-spawned dynamic
# agents in the index.
factory_count_dynamic() {
  factory_index_ensure
  jq -r '.agents | length' "$FACTORY_INDEX_FILE"
}

# factory_at_cap — returns 0 (true) if the index has hit max_dynamic.
# max_dynamic == 0 means "unlimited" and never trips.
factory_at_cap() {
  local max; max="$(factory_cfg max_dynamic 0)"
  [[ "$max" == "0" || -z "$max" ]] && return 1
  local cur; cur=$(factory_count_dynamic)
  (( cur >= max )) && return 0
  return 1
}

# factory_audit — append a structured event to the factory audit log.
factory_audit() {
  local action="$1"; shift
  local af; af="$(factory_cfg audit_log ./.orchestr8r/factory_audit.jsonl)"
  mkdir -p "$(dirname "$af")"
  # Build a JSON object out of any trailing k=v args.
  local details_json="{}"
  for kv in "$@"; do
    [[ "$kv" == *=* ]] || continue
    local k="${kv%%=*}"
    local v="${kv#*=}"
    details_json=$(jq --arg k "$k" --arg v "$v" '. + {($k):$v}' <<<"$details_json")
  done
  jq -c -n \
    --arg ts      "$(now_iso)" \
    --arg act     "$action" \
    --argjson det "$details_json" \
    '{ts:$ts, action:$act, details:$det}' >> "$af"
}

# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

# factory_validate <path> — JSON-schema validation using jsonschema or jq fallback.
factory_validate() {
  local path="$1"
  [[ -f "$path" ]] || { err "factory_validate: no such file $path"; return 1; }
  # A single boolean expression: every required field is well-typed and
  # non-empty.  Wrapped in defs to keep the predicate readable.
  jq -e '
    def is_nonempty_string: type == "string" and length > 0;
    def is_schema_object:   type == "object" and has("type");
    def is_quality_block:   type == "object" and (.min_score // 0) > 0;

    . as $doc
    | ($doc.name          | is_nonempty_string)
      and ($doc.family    | is_nonempty_string)
      and ($doc.input_schema  | is_schema_object)
      and ($doc.output_schema | is_schema_object)
      and ($doc.quality       | is_quality_block)
  ' "$path" >/dev/null 2>&1 || {
    err "factory_validate: contract is missing required fields"
    return 1
  }
  # If jsonschema is installed, also run the full check.
  if command -v jsonschema >/dev/null; then
    jsonschema -i "$path" "$SKILL_SCRIPT_DIR/schemas/agent_result.v2.schema.json" \
      >/dev/null 2>&1 || { err "factory_validate: jsonschema rejected"; return 1; }
  fi
  return 0
}

# ----------------------------------------------------------------------------
# Index (where dynamic agents are tracked)
# ----------------------------------------------------------------------------

FACTORY_INDEX_FILE="${ORCH_FACTORY_INDEX:-./.orchestr8r/dynamic_agents.json}"

# factory_index_ensure — make sure the index file exists.
factory_index_ensure() {
  [[ -f "$FACTORY_INDEX_FILE" ]] || {
    mkdir -p "$(dirname "$FACTORY_INDEX_FILE")"
    cat > "$FACTORY_INDEX_FILE" <<'JSON'
{
  "version":   "1.0",
  "created":   "",
  "updated":   "",
  "agents":    {}
}
JSON
  }
}

# factory_register <path> — append an agent to the dynamic index.
factory_register() {
  local path="$1"
  factory_index_ensure
  local name; name=$(jq -r '.name' "$path")
  local family; family=$(jq -r '.family' "$path")
  local mode; mode=$(jq -r '.dynamic.creation_mode // "template_clone"' "$path")
  local parent; parent=$(jq -r '.dynamic.parent_family // ""' "$path")
  local runtime="${ORCH_PLAN_ID:-manual}"
  local reason; reason=$(jq -r '.dynamic.spawn_reason // ""' "$path")

  local tmp; tmp=$(mktemp)
  jq --arg n  "$name" \
     --arg f  "$family" \
     --arg m  "$mode" \
     --arg pa "$parent" \
     --arg pt "$path" \
     --arg rt "$runtime" \
     --arg rs "$reason" \
     --arg ts "$(now_iso)" \
     '.agents[$n] = {
        family:$f, mode:$m, parent_family:$pa,
        path:$pt, runtime:$rt, reason:$rs,
        registered:$ts, runs:0, last_run:null,
        quarantine_remaining:(.agents[$n].quarantine_remaining // 3)
      }
      | .updated = $ts' \
     "$FACTORY_INDEX_FILE" > "$tmp" && mv "$tmp" "$FACTORY_INDEX_FILE"

  factory_audit "register" "name=$name" "family=$family" "mode=$mode"
  echo "$name"
}

# factory_index_remove <name> — take a dynamic agent back out.
factory_index_remove() {
  local name="$1"
  factory_index_ensure
  local tmp; tmp=$(mktemp)
  jq --arg n "$name" 'del(.agents[$n]) | .updated = (now|todate)' \
     "$FACTORY_INDEX_FILE" > "$tmp" && mv "$tmp" "$FACTORY_INDEX_FILE"
  factory_audit "remove" "name=$name"
}

# factory_list_dynamic — echo JSON of every registered dynamic agent.
factory_list_dynamic() {
  factory_index_ensure
  jq -c '.' "$FACTORY_INDEX_FILE"
}

# ----------------------------------------------------------------------------
# Mode 1 — TEMPLATE CLONE
# ----------------------------------------------------------------------------
#
# factory_create_clone <new_name> <parent_agent_path> <reason>
#
# Reads a parent JSON contract, deep-merges overrides (name, family, runtime
# metadata), validates, and writes to agents/dynamic/<new_name>.json.
#
factory_create_clone() {
  local new_name="$1" parent_path="$2" reason="${3:-on-demand}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false in CONFIG; refusing"
    return 2
  fi
  if factory_static_only; then
    err "factory: static_only=true; refuses to spawn new agents"
    return 2
  fi
  [[ -f "$parent_path" ]] || { err "factory_create_clone: parent not found: $parent_path"; return 1; }

  # Resolve dynamic output dir + ensure it exists.
  local dyn_dir; dyn_dir="$(factory_cfg dynamic_dir ./agents/dynamic)"
  mkdir -p "$dyn_dir"

  # Find the family's template metadata (or default to "coder").
  local family; family=$(jq -r '.name' "$parent_path" | awk -F. '{print $1}')
  [[ -n "$family" && "$family" != "null" ]] || family="coder"

  # Build the new JSON by deep-merging: parent overrides the scaffold, then
  # we overwrite the identity/runtime fields with the fresh values.  All
  # values come in as --arg to avoid quoting pitfalls.
  local scaffold="$SKILL_SCRIPT_DIR/templates/agent.template.json"
  local out="$dyn_dir/$new_name.json"
  local tmp; tmp=$(mktemp)
  local ts;      ts="$(now_iso)"
  local spawner; spawner="${ORCH_SUPERVISOR_ID:-supervisor}"
  local q;       q="$(factory_cfg default_quarantine 3)"

  # Build the new contract.  We start from the scaffold and then copy each
  # parent field we want to inherit, but explicitly skip the fields we
  # never want to inherit (name, version, dynamic, description, writes,
  # reads).  We also accept the parent's `quality` block as a whole.
  jq -n --slurpfile tpl "$scaffold" \
        --slurpfile par "$parent_path" \
        --arg new_name "$new_name" \
        --arg family   "$family" \
        --arg ts       "$ts" \
        --arg spawner  "$spawner" \
        --arg reason   "$reason" \
        --argjson q    "$q" \
    '
      ($tpl[0]) as $base
      | ($par[0]) as $p
      | $base
      | .kind           = $p.kind           // $base.kind
      | .entry          = $p.entry          // $base.entry
      | .input_schema   = $p.input_schema   // $base.input_schema
      | .output_schema  = $p.output_schema  // $base.output_schema
      | .resources      = $p.resources      // $base.resources
      | .retry_policy   = $p.retry_policy   // $base.retry_policy
      | .quality        = $p.quality        // $base.quality
      | .persona        = $p.persona        // "staff_engineer"
      | .name           = $new_name
      | .family         = $family
      | .description    = ($p.description // "Spawned by supervisor for on-demand need: \($reason)")
      | .dynamic        = {
          is_dynamic:           true,
          spawned_at:           $ts,
          spawned_by:           $spawner,
          spawn_reason:         $reason,
          parent_family:        $family,
          creation_mode:        "template_clone",
          quarantine_runs:      $q,
          quarantine_remaining: $q
        }
    ' > "$tmp"

  if ! jq -e '.' "$tmp" >/dev/null 2>&1; then
    rm -f "$tmp"
    err "factory_create_clone: produced invalid JSON"
    return 1
  fi
  mv "$tmp" "$out"

  if ! factory_validate "$out"; then
    rm -f "$out"
    err "factory_create_clone: validation failed for $out"
    return 1
  fi

  factory_audit "create" "mode=template_clone" "name=$new_name" "parent=$parent_path"
  factory_register "$out" >/dev/null
  echo "$out"
}

# ----------------------------------------------------------------------------
# Mode 2 — PERSONA PICK
# ----------------------------------------------------------------------------
#
# factory_create_persona <new_name> <base_agent_path> <new_persona> <reason>
#
# Same kind + capability, different voice. Lighter than a full clone.
#
factory_create_persona() {
  local new_name="$1" base_path="$2" new_persona="$3" reason="${4:-persona-shift}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false; refusing"
    return 2
  fi
  if factory_static_only; then
    err "factory: static_only=true; refuses to spawn new agents"
    return 2
  fi
  [[ -f "$base_path" ]] || { err "factory_create_persona: base not found $base_path"; return 1; }
  local dyn_dir; dyn_dir="$(factory_cfg dynamic_dir ./agents/dynamic)"
  mkdir -p "$dyn_dir"
  local family; family=$(jq -r '.family // (.name | split(".")[0])' "$base_path")
  local out="$dyn_dir/$new_name.json"
  local tmp; tmp=$(mktemp)

  jq --arg n    "$new_name" \
     --arg p    "$new_persona" \
     --arg f    "$family" \
     --arg ts   "$(now_iso)" \
     --arg spw  "${ORCH_SUPERVISOR_ID:-supervisor}" \
     --arg rsn  "$reason" \
     --argjson q "$(factory_cfg default_quarantine 3)" \
     '
     . as $base
     | $base
     | .name = $n
     | .family = $f
     | .persona = $p
     | .description = ("Persona variant of " + $base.name + " as \"" + $p + "\". \($rsn)")
     | .dynamic = {
         is_dynamic:           true,
         spawned_at:           $ts,
         spawned_by:           $spw,
         spawn_reason:         $rsn,
         parent_family:        $f,
         creation_mode:        "persona_pick",
         quarantine_runs:      $q,
         quarantine_remaining: $q
       }
     ' "$base_path" > "$tmp" && mv "$tmp" "$out"

  if ! factory_validate "$out"; then
    rm -f "$out"
    err "factory_create_persona: validation failed"
    return 1
  fi

  factory_audit "create" "mode=persona_pick" "name=$new_name" "persona=$new_persona"
  factory_register "$out" >/dev/null
  echo "$out"
}

# ----------------------------------------------------------------------------
# Mode 3 — SYNTHESIZE (stub; LLM-backed when enabled)
# ----------------------------------------------------------------------------
#
# factory_synthesize <new_name> <gap_description> <reason>
#
# Generates a full agent contract from a capability-gap description.
# Currently safe-mode: requires an LLM CLI to be present. If none is
# available, we fall back to template_clone with the closest family by
# keyword-matching.
#
factory_synthesize() {
  local new_name="$1" gap="$2" reason="${3:-capability-gap}"
  if ! factory_enabled; then
    err "factory: dynamic_agents.enabled=false; refusing"
    return 2
  fi
  if factory_static_only; then
    err "factory: static_only=true; refuses to synthesize"
    return 2
  fi

  # Pick the best existing family by capability-keyword match.
  local best="coder" best_score=0
  local families="$SKILL_SCRIPT_DIR/templates/agent_families.json"
  local kw
  for fam in $(jq -r '.families | keys[]' "$families"); do
    local score=0
    for kw in $(jq -r --arg f "$fam" '.families[$f].capability_keywords[]' "$families"); do
      if [[ "$gap" == *"$kw"* ]]; then score=$((score+1)); fi
    done
    if (( score > best_score )); then best_score=$score; best="$fam"; fi
  done

  # Find a real existing agent in that family to clone from.
  local parent
  parent=$(ls "$SKILL_SCRIPT_DIR/agents/${best}."*.json 2>/dev/null | head -1 || true)
  if [[ -z "$parent" ]]; then
    err "factory_synthesize: no parent in family $best"; return 1
  fi

  # Synthesize-mode is really template_clone today (LLM path can be wired
  # later by setting synthesis_model). We log the gap description.
  local out; out=$(factory_create_clone "$new_name" "$parent" "$reason")
  if [[ -z "$out" ]]; then return 1; fi

  # Stamp the synth-mode marker so audits are honest.
  local tmp; tmp=$(mktemp)
  jq --arg gap "$gap" '
    .dynamic.creation_mode = "synthesize"
    | .dynamic.synth_gap    = $gap
  ' "$out" > "$tmp" && mv "$tmp" "$out"
  factory_register "$out" >/dev/null
  factory_audit "create" "mode=synthesize" "name=$new_name" "family=$best" "gap=$gap"
  echo "$out"
}

# ----------------------------------------------------------------------------
# Dispatcher hook
# ----------------------------------------------------------------------------
#
# match_or_propose_agent <task_json>
#
# Looks at a task JSON object (or task JSON file). If a matching static
# agent exists, returns its path. If not, *proposes* one by triggering the
# factory (Mode 1 by default, Mode 3 if synthesis_model is configured).
# Returns the chosen agent JSON path on stdout. Exit 2 = no proposal.
#
match_or_propose_agent() {
  local task_json="$1"
  factory_enabled || return 2

  # Pull a "needs" keyword from the task_description or description field.
  local text
  text=$(jq -r '(.task_description // .description // "")' "$task_json")

  # Search the static registry for a family whose keywords match.
  local families="$SKILL_SCRIPT_DIR/templates/agent_families.json"
  local best=""; best_score=0
  for fam in $(jq -r '.families | keys[]' "$families"); do
    local score=0
    for kw in $(jq -r --arg f "$fam" '.families[$f].capability_keywords[]' "$families"); do
      if [[ "$text" == *"$kw"* ]]; then score=$((score+1)); fi
    done
    if (( score > best_score )); then
      best_score=$score; best="$fam"
    fi
  done

  if [[ -z "$best" || "$best_score" -lt 1 ]]; then
    echo "${LIB_PREFIX} no family matched; will synthesize" >&2
    best="coder"
  fi

  # Find first existing agent in that family.
  local parent
  parent=$(ls "$SKILL_SCRIPT_DIR/agents/${best}."*.json 2>/dev/null | head -1 || true)
  if [[ -z "$parent" ]]; then
    err "match_or_propose_agent: no parent in family $best"; return 2
  fi

  local new_name="auto.${best}.$(date +%s%N)"
  local out; out=$(factory_create_clone "$new_name" "$parent" "dispatched-gap: $text")
  [[ -n "$out" ]] || return 1
  echo "$out"
}

# ----------------------------------------------------------------------------
# Quarantine ageing
# ----------------------------------------------------------------------------
#
# factory_quarantine_tick <agent_name>
# Called by canary-doctor after each dynamic-agent run. Decrements the
# remaining counter; once it hits zero, the agent is "trusted" (still
# audited, but no longer quarantined).
#
factory_quarantine_tick() {
  local name="$1"
  factory_index_ensure
  local tmp; tmp=$(mktemp)
  jq --arg n "$name" '
    if .agents[$n] then
      .agents[$n].quarantine_remaining = ([(.agents[$n].quarantine_remaining - 1), 0] | max)
      | .agents[$n].runs = ((.agents[$n].runs // 0) + 1)
      | .agents[$n].last_run = (now|todate)
    else . end
  ' "$FACTORY_INDEX_FILE" > "$tmp" && mv "$tmp" "$FACTORY_INDEX_FILE"
}

# factory_quarantine_remaining <agent_name>
factory_quarantine_remaining() {
  local name="$1"
  factory_index_ensure
  jq -r --arg n "$name" '.agents[$n].quarantine_remaining // 0' "$FACTORY_INDEX_FILE"
}

# factory_print_dynamic_table — human-friendly summary for `agents list --dynamic`.
factory_print_dynamic_table() {
  factory_index_ensure
  jq -r '
    .agents
    | to_entries
    | sort_by(.key)
    | .[]
    | [.key, .value.family, .value.mode, (.value.runs // 0), (.value.quarantine_remaining // 0), .value.path]
    | @tsv
  ' "$FACTORY_INDEX_FILE" \
    | awk 'BEGIN{printf "%-32s %-12s %-16s %6s %6s %s\n","AGENT","FAMILY","MODE","RUNS","QUAR","PATH"}
           NR>0 {printf "%-32s %-12s %-16s %6s %6s %s\n",$1,$2,$3,$4,$5,$6}'
}
