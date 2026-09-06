#!/usr/bin/env bats
# tests/validate.bats
# Smoke tests for plan validation.
setup() {
  export TMPDIR="$(mktemp -d)"
  export SKILL_CONFIG="$TMPDIR/config.json"
  cat > "$SKILL_CONFIG" <<JSON
{
  "project":{"name":"test","version":"1.0.0"},
  "paths":{"plan_file":"$TMPDIR/plan.json","state_file":"$TMPDIR/state.json","audit_file":"$TMPDIR/audit.jsonl"},
  "agents":{
     "research":{"primary":"deep_research_tasks","supports_parallel":true,"best_for":["r"]},
     "documentation":{"primary":"report_writer_agent","supports_parallel":false,"best_for":["d"]}
   },
  "agent_aliases":{"research":"deep_research_tasks","documentation":"report_writer_agent"},
  "passes":{
    "pass_1_task_analysis_decomposition":{"enabled":true},
    "pass_2_agent_selection_dispatch":{"enabled":true,"default_handoff":"bidirectional"},
    "pass_3_result_aggregation_validation":{"enabled":true},
    "pass_4_logging_sealing":{"enabled":true,"sealed_marker":"[DONE]"}
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

@test "plan can be validated when empty" {
  run bash "$SKILL" --config "$SKILL_CONFIG" --validate
  [ "$status" -eq 0 ]
}

@test "add-task via --add-task-file succeeds" {
  cat > "$TMPDIR/t1.json" <<JSON
{"id":"r1","description":"research X","agent_role":"research","depends_on":[],"output_path":"./out/r1.md"}
JSON
  run bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file "$TMPDIR/t1.json"
  [ "$status" -eq 0 ]
}

@test "invalid agent role is rejected" {
  cat > "$TMPDIR/t1.json" <<JSON
{"id":"r1","description":"x","agent_role":"unknown"}
JSON
  run bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file "$TMPDIR/t1.json"
  run bash "$SKILL" --config "$SKILL_CONFIG" --validate
  [ "$status" -ne 0 ]
}

@test "cycle in depends_on is rejected" {
  cat > "$TMPDIR/a.json" <<JSON
{"id":"a","description":"x","agent_role":"research","depends_on":["b"]}
JSON
  cat > "$TMPDIR/b.json" <<JSON
{"id":"b","description":"y","agent_role":"research","depends_on":["a"]}
JSON
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file "$TMPDIR/a.json"
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file "$TMPDIR/b.json"
  run bash "$SKILL" --config "$SKILL_CONFIG" --validate
  [ "$status" -ne 0 ]
}

@test "seal refuses when tasks are incomplete (without --force)" {
  cat > "$TMPDIR/a.json" <<JSON
{"id":"a","description":"x","agent_role":"research","depends_on":[]}
JSON
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file "$TMPDIR/a.json"
  run bash "$SKILL" --config "$SKILL_CONFIG" --seal
  [ "$status" -ne 0 ]
}

@test "seal succeeds after a skip" {
  cat > "$TMPDIR/a.json" <<JSON
{"id":"a","description":"x","agent_role":"research","depends_on":[]}
JSON
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file "$TMPDIR/a.json"
  bash "$SKILL" --config "$SKILL_CONFIG" --skip a "test" >/dev/null
  run bash "$SKILL" --config "$SKILL_CONFIG" --seal
  [ "$status" -eq 0 ]
}

@test "--list-templates lists templates" {
  run bash "$SKILL" --config "$SKILL_CONFIG" --list-templates
  [ "$status" -eq 0 ]
  [[ "$output" =~ competitor_landscape ]]
}

@test "--topo and --critical-path emit ordering" {
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file <(echo '{"id":"a","description":"x","agent_role":"research","depends_on":[]}') || true
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file <(echo '{"id":"b","description":"y","agent_role":"research","depends_on":["a"]}') || true
  run bash "$SKILL" --config "$SKILL_CONFIG" --topo
  [ "$status" -eq 0 ]
  [[ "$output" =~ "a" ]]
  run bash "$SKILL" --config "$SKILL_CONFIG" --critical-path
  [ "$status" -eq 0 ]
  [[ "$output" =~ "a" ]]
  [[ "$output" =~ "b" ]]
}

@test "--export then --import roundtrip" {
  bash "$SKILL" --config "$SKILL_CONFIG" --add-task-file <(echo '{"id":"a","description":"x","agent_role":"research","depends_on":[]}') || true
  bash "$SKILL" --config "$SKILL_CONFIG" --export "$TMPDIR/bundle.json"
  [ -s "$TMPDIR/bundle.json" ]
  run bash "$SKILL" --config "$SKILL_CONFIG" --import "$TMPDIR/bundle.json"
  [ "$status" -eq 0 ]
}
