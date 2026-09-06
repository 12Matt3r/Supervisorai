#!/usr/bin/env bats
# tests/observability.bats
# Tests for observability commands.
setup() {
  export TMPDIR="$(mktemp -d)"
  export SKILL_CONFIG="$TMPDIR/config.json"
  cat > "$SKILL_CONFIG" <<JSON
{
  "project":{"name":"o","version":"1.0.0"},
  "paths":{"plan_file":"$TMPDIR/plan.json","state_file":"$TMPDIR/state.json","audit_file":"$TMPDIR/audit.jsonl"},
  "agents":{"research":{"primary":"deep_research_tasks","supports_parallel":true,"best_for":["r"]}},
  "passes":{
    "pass_1_task_analysis_decomposition":{"enabled":true},
    "pass_2_agent_selection_dispatch":{"enabled":true},
    "pass_3_result_aggregation_validation":{"enabled":true},
    "pass_4_logging_sealing":{"enabled":true,"sealed_marker":"[D]"}
  },
  "context_recovery":{"enabled":true},
  "delegation":{"enabled":true},
  "dynamic_routing":{"enabled":false}
}
JSON
  bash "$BATS_TEST_DIRNAME/../skill.sh" --config "$SKILL_CONFIG" --plan >/dev/null
}
teardown() { rm -rf "$TMPDIR"; }
SKILL="$BATS_TEST_DIRNAME/../skill.sh"

@test "--cost-report produces output" {
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file <(echo '{"id":"r1","description":"x","agent_role":"research","depends_on":[],"task_cost_usd":0.123,"task_tokens_used":4567}') >/dev/null
  run bash "$SKILL" --config "$SKILL_CONFIG" --cost-report
  [ "$status" -eq 0 ]
  [[ "$output" =~ "Total cost" ]]
}

@test "--audit emits events" {
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file <(echo '{"id":"r1","description":"x","agent_role":"research","depends_on":[]}') >/dev/null
  bash "$SKILL" --config "$SKILL_CONFIG" --dispatch-ready >/dev/null
  bash "$SKILL" --config "$SKILL_CONFIG" --mark-done r1 ./out/x.md >/dev/null
  run bash "$SKILL" --config "$SKILL_CONFIG" --audit
  [ "$status" -eq 0 ]
  [[ "$output" =~ "dispatch" ]]
  [[ "$output" =~ "mark_done" ]]
}

@test "--trace emits OTel-like JSON" {
  run bash "$SKILL" --config "$SKILL_CONFIG" --trace r1
  [ "$status" -eq 0 ]
  [[ "$output" =~ "trace_id" ]]
}
