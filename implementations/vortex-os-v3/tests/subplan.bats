#!/usr/bin/env bats
# tests/subplan.bats
# Hierarchical orchestration smoke test.
setup() {
  export TMPDIR="$(mktemp -d)"
  export SKILL_CONFIG="$TMPDIR/config.json"
  cat > "$SKILL_CONFIG" <<JSON
{
  "project":{"name":"s","version":"1.0.0"},
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

@test "sub-plan can be listed" {
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file <(echo '{"id":"r1","description":"x","agent_role":"research","depends_on":[]}') >/dev/null
  run bash "$SKILL" --config "$SKILL_CONFIG" --list-subplans
  [ "$status" -eq 0 ]
}
