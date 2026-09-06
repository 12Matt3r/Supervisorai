# Migration Guide — 1.x → 2.0

This guide covers the breaking changes between v1.0.0 and v2.0.0, and
gives a step-by-step recipe for upgrading an existing project.

---

## TL;DR

1. Install v2 alongside v1 (don't delete v1 yet).
2. Run `python3 scripts/migrate_plan.py OLD_PLAN.json NEW_PLAN.json` to
   convert the old plan.json to v2.
3. Update your config.json — see *"Config schema changes"*.
4. Re-run `skill --validate` against your v2 config and plan.
5. Re-run your orchestration loop. v2 is wire-compatible with v1
   commands; the changes are additive.

---

## Breaking changes

### A. `--add-task` is now a thin wrapper

In v1, `skill --add-task '{ ... }'` relied on shell quoting; in v2
large descriptions with apostrophes broke. Use `--add-task-file <path>`
instead — same semantics, no quoting issues.

```bash
# v1 — brittle
skill --add-task '{"id":"t1","description":"O'Reilly book"}'   # BAD

# v2
cat > /tmp/t.json <<JSON
{"id":"t1","description":"O'Reilly book","agent_role":"research"}
JSON
skill --add-task-file /tmp/t.json
```

### B. `--mark-done <id>` no longer seeds `attempts`

In v1, `mark-done` set the attempts counter. v2 doesn't change
`attempts`. If you were relying on `attempts == 1` after a single
`mark-done`, that still works.

### C. `--seal` is now strict by default

v1 sealed even with failing tasks. v2 refuses:

```bash
# v1 silently sealed anyway
skill --seal   # OK

# v2 refuses unless all tasks complete/skipped
skill --seal
# ERROR: cannot seal: 1 task(s) not complete/skipped

# Workaround (not recommended): force-seal
skill --force --seal
```

If you have a v1 plan that was sealed with `--force`-equivalent
behaviour, run `--unseal` first to flip the `sealed` flag, fix the
statuses, then `--seal` again.

### D. Plan schema: `notes` is now a string (not list)

v1 stored `notes` as a free-form string. v2 keeps it as a string but
documents the convention: separate semantically distinct updates with
`|` (e.g. `attempt 1 failed|attempt 2 succeeded`).

### E. New `sealed` field on plan.json

v1 used `state.sealed_marker` only. v2 adds `plan.sealed: bool` so
probing without reading state is cheap. The `state.sealed_marker` is
still emitted for backward compatibility.

---

## Config schema changes

The `config.schema.json` and `config.example.json` now include:

* `paths.audit_file` — required for the audit log.
* `agent_aliases` — required if you renamed agent roles.
* `passes.pass_2_agent_selection_dispatch.default_handoff` — new.
* `passes.pass_3_result_aggregation_validation.validate_cmd_default`.
* `passes.pass_5_notification` — new (default disabled).
* `quality` — new (defaults to `validate_min_score: 0.7`).
* `notifications` — new (configured channels).
* `dynamic_routing` — new (off by default).

A v1 config is forward-compatible: extra fields are ignored. But if
your v1 config had no `audit_file` path, v2 will quietly skip the
audit log; add a path to enable it.

---

## New commands you'll likely want to use

```bash
# v2-only commands
skill --import-template competitor_landscape --template-var INDUSTRY=AI
skill --spawn-subplan <parent_task_id> "<sub-goal>"
skill --quality-gate <task_id>           # run a per-task quality check
skill --cost-report                      # totals across all tasks
skill --audit                            # view the audit log
skill --watch                            # live state changes
skill --dot | dot -Tpng > plan.png       # visualise the DAG
skill --agents list                      # browse 30 built-in sub-agents
```

> v2.0.1 ships with **30 production-ready sub-agents** under `agents/`.
> After upgrading, you can immediately point any task's `agent_role`
> (or `agent`) at one of them — see
> [`agents/AGENTS_CATALOG.md`](agents/AGENTS_CATALOG.md) for the
> *Question → agent* decision tree.

---

## Converting an existing plan

If you wrote a v1 plan.json, run:

```bash
python3 scripts/migrate_plan.py old_plan.json new_plan.json
```

The script:

* adds the `sealed: false` field if missing,
* normalizes `notes` into a string,
* tags every task with a default `handoff: "bidirectional"`,
* keeps everything else intact.

---

## Validation before/after upgrade

```bash
# Validate the old plan against the old schema
old_skill/skill.sh --validate --config old_config.json

# Run the migration
python3 scripts/migrate_plan.py old_plan.json new_plan.json

# Validate the new plan against the new schema
new_skill/skill.sh --validate --config new_config.json
```

If `new_skill/skill.sh --validate` exits 0, you're good.
