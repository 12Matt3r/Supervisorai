# WORKFLOW.md — The 4+1-Pass Orchestration Workflow (v2)

> **A project-agnostic walk-through of the workflow codified in `skill.sh`.**

```
        ┌──────────────────────────┐
        │  User's high-level goal  │
        └────────────┬─────────────┘
                     ▼
   ┌─────────────────────────────────────┐
P1 │  Pass 1:  Task Analysis             │
   │  & Decomposition                    │
   └──────────────┬──────────────────────┘
                  ▼  plan.json
   ┌─────────────────────────────────────┐
P2 │  Pass 2:  Agent Selection           │
   │  & Dispatch (parallel where possible)│
   └──────────────┬──────────────────────┘
                  ▼  state.json
   ┌─────────────────────────────────────┐
P3 │  Pass 3:  Result Aggregation        │
   │  + Quality Gates                    │
   └──────────────┬──────────────────────┘
                  ▼  outputs verified
   ┌─────────────────────────────────────┐
P4 │  Pass 4:  Logging & Sealing         │
   └──────────────┬──────────────────────┘
                  ▼
   ┌─────────────────────────────────────┐
P5 │  Pass 5:  Notification (opt-in)     │
   └──────────────┬──────────────────────┘
                  ▼
        ┌──────────────────────────┐
        │  Sealed plan             │
        └──────────────────────────┘
```

---

## Pass 1 — Task Analysis & Decomposition

**Goal.** Convert a high-level goal into a flat or hierarchical
list of **atomic tasks**, each with a clear description, target
agent role, explicit dependency list, and (optionally) a retry policy
and a quality gate.

**Inputs.** Goal text, `config.json`, current `plan.json`.

**Outputs.** Updated `plan.json` containing every task the
orchestrator believes is needed.

**How it works.**

1. The orchestrator reads the goal.
2. It enumerates the natural milestones; for each, writes a task with
   `id`, `description`, `agent_role`, `depends_on`, `input`,
   `output_path`, and any of: `handoff`, `retry_policy`,
   `validation`, `expected_duration_minutes`.
3. It calls `--validate` to catch unknown roles, missing deps, and
   cycles.
4. If `dynamic_routing.enabled = true`, `--auto-plan "<goal>"`
   delegates decomposition to an LLM. Otherwise the parent agent
   declares tasks explicitly.

**Why it matters.** Pass 1 turns the user's intent into a
machine-checkable contract. Once on disk, the rest is bookkeeping.

**Configurable:**
`passes.pass_1_task_analysis_decomposition.{max_subtasks,
require_dependencies, validate_against_agents}`.

---

## Pass 2 — Agent Selection & Dispatch

**Goal.** Send every ready task to its assigned agent role in
parallel, respecting the dependency graph and each role's
`max_parallel` cap.

**Inputs.** `plan.json`, `config.json`, `state.json`, `agent_load.json`
(if used).

**Outputs.** Updated `plan.json` (tasks flipped to `dispatched`) and
`state.json` (with `current_executing_task`).

**How it works.**

1. The orchestrator builds the dependency graph.
2. It identifies all tasks with status `pending` (or `ready`) and no
   unmet dependencies.
3. It groups them by `agent_role` and applies the `max_parallel` cap.
4. It also reads `agent_load.json` to skip agents whose reported load
   is over `passes.pass_2_agent_selection_dispatch.max_concurrent %
   threshold`.
5. Each dispatched task gets `status = dispatched`, `started_at`,
   `attempts += 1`, and a default `handoff` mode.
6. A JSONL audit record is appended.
7. *(opt-in)* If `retry_policy` is set and the task transitions to
   `failed` later, the orchestrator auto-decides whether to
   re-dispatch (exponential backoff, jitter) up to `max_attempts`.

**Handoff modes (per task):**

* `one_way` — dispatch only; ignore output. Useful for escalations.
* `bidirectional` — wait for output, write back to parent task (default).
* `fan_in` — collect outputs from many siblings into one task.

**Configurable:**
`passes.pass_2_agent_selection_dispatch.{max_concurrent,
retry_on_failure, max_retries, default_handoff, fallback_agent}`.

---

## Pass 3 — Result Aggregation & Validation

**Goal.** Confirm completeness *and quality*.

**Inputs.** `plan.json`, the filesystem (output paths), each task's
`validation` block (optional), `state.json`.

**Outputs.** Pass/fail per task, an aggregate summary, a list of any
tasks to be re-dispatched.

**How it works.**

1. The orchestrator walks every `complete` task in the plan and
   verifies the `output_path` exists and is non-empty.
2. If a task has a `validation` block, the orchestrator runs the
   declared gates:
   * `validate_cmd` — `subprocess.run(...)`; exit 0 is required.
   * `validate_tool` — invokes an external tool that emits
     `{"score":0..1,"reason":"..."}`; pass requires `score >=
     validate_min_score`.
   * `validate_rubric` — free-form description usable by an LLM judge.
3. Optionally, run `--judge <id> <score> <reason>` to apply a manual
   verdict (e.g. a parent LLM judge).
4. Tasks that fail quality flip back to `failed`. Tasks that pass
   become `complete`.
5. The orchestrator counts by status (`--stats`) and prints the
   summary (`--aggregate`).

**Why it matters.** Pass 3 is the gate that prevents silent failures
from reaching the user. v2 adds three layers: file existence,
shell-level validation, and an LLM judge.

**Configurable:**
`passes.pass_3_result_aggregation_validation.{require_all_tasks_complete,
validate_output_paths_exist, validate_cmd_default}` and
`quality.{default_validate_min_score, default_judge_model}`.

---

## Pass 4 — Logging & Sealing

**Goal.** Make the orchestration auditable, portable, and
**terminal**.

**Inputs.** Final `plan.json`, `state.json`.

**Outputs.** Sealed plan (`plan.sealed=true`, `plan.sealed_at`).
A `sealed_marker` in `state.json`. An audit-log entry.

**How it works.**

1. The orchestrator refuses to seal a plan unless all tasks are
   `complete` or `skipped` (unless `--force` is given).
2. It writes `sealed_at` and `sealed_marker` to both `plan.json` and
   `state.json`.
3. It snapshots `completed_tasks` and `failed_tasks` and appends a
   new entry to `sealed_unsealed_history`.
4. It appends a `seal` event to `audit.jsonl`.
5. After sealing, `--reset` and `--dispatch-ready` refuse without
   `--unseal`.

**Why it matters.** Sealing is the contract that lets multiple agents
collaborate without stepping on each other. After a seal, anyone
re-reading the project knows "this is done."

**Configurable:**
`passes.pass_4_logging_sealing.{sealed_marker, log_format}`.

---

## Pass 5 — Notification (opt-in)

**Goal.** Tell humans or downstream systems that a plan finished.

**Inputs.** The seal event, the `notifications.channels` config block.

**Outputs.** Messages to log/file/webhook/mail/Slack per channel.

**How it works.** Configured channels receive a message formatted
with the optional `message_format` template (default:
`Plan {plan_id} {event}: {goal}`). Channels:

* `log` — print to stdout.
* `file` — append to a file (`path`).
* `webhook` — HTTP POST (`url`, `method`, `content_type`).
* `mail` — invoke a `command`, feed the body to stdin.
* `slack` — POST JSON to a Slack-compatible webhook.

**Why it matters.** Human / system handoff. Without Pass 5, "sealed"
is invisible to anyone watching from the outside.

---

## Hierarchical Sub-Plans (sub-orchestration)

A task can optionally spawn a sub-plan:

```bash
skill --spawn-subplan <parent_task_id> "<sub-goal>"
```

The orchestrator creates `sub_<id>.plan.json` in the project root,
records the sub-plan id on the parent task (`sub_plan_id`), and marks
the parent as `pending` waiting for the sub-plan. When the sub-plan
seals, the parent task flips to `complete`. This is the
supervisor-of-supervisors primitive.

Use it for goals that are themselves mini-projects (a "build the TODO
app" task is really a 10-task project).

---

## Context Recovery

The orchestrator treats context windows as ephemeral. The durable
record is `state.json` plus the audit log. Every state-changing
command writes to both. `--recover` prints the last checkpoint.

```bash
bash skill.sh --recover         # What was I doing?
bash skill.sh --ready           # What's next?
bash skill.sh --next-task       # Get one task as JSON
# ... dispatch it ...
bash skill.sh --mark-done ID ./out/file.md
bash skill.sh --checkpoint
```

The parent never has to remember the plan in its own context; it
reads the file and re-orients.

---

## The Loop (Full)

```bash
while bash skill.sh --ready | grep -q "Ready tasks:"; do
  bash skill.sh --dispatch-ready
  # ... parent does the work for each dispatched task ...
  bash skill.sh --aggregate
  bash skill.sh --checkpoint
done
bash skill.sh --cost-report
bash skill.sh --audit
bash skill.sh --seal
```

---

## Why these passes?

* **Pass 1** is about *intent*: what does the user really want?
* **Pass 2** is about *execution*: who does what, in what order?
* **Pass 3** is about *truth*: did the work *actually* happen *well*?
* **Pass 4** is about *contract*: can we trust this plan is done?
* **Pass 5** is about *communication*: did anyone outside notice?

If you need a sixth pass — say, *Pass 6: Cost Reconciliation* that
emits a usage report — add it under `passes.pass_6_*` in the config,
wire a `cmd_*` function in `lib/commands.sh`, and a flag in
`skill.sh`.

---

## Selecting a Sub-Agent (30 built-in, drop-in usable)

A `plan.json` task names its sub-agent via `agent_role` / `agent`. The
orchestrator resolves the name through this precedence:

1. **Project-local** — `$ORCH_PROJECT/agents/*.json`
2. **Built-in catalog** — `skill_v2/agents/*.json` (**30 ships**;
   catalog at `agents/AGENTS_CATALOG.md`)
3. **User-wide** — `$ORCH_HOME/agents/*.json`
4. **System-wide** — `/etc/orchestrator/agents/*.json`

The catalog covers every realistic orchestration need out of the box:

| Family            | Built-in agents                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------- |
| Coding & Build    | `coder.python`, `coder.typescript`, `coder.rust`, `coder.sql_migrations`, `coder.terraform` |
| Review & QA       | `reviewer.code`, `reviewer.architecture`, `reviewer.dependencies`, `tester.mutation`, `qa.flakehunter` |
| Security          | `security.sast`, `security.deps`, `security.secrets`, `security.threatmodel`                 |
| DevOps & Ops      | `ops.deploy.k8s`, `ops.deploy.serverless`, `ops.incident_responder`, `ops.cost_optimizer`    |
| Data              | `data.profiler`, `data.cleaner`, `data.synthesizer`, `data.lineage`                          |
| Research          | `researcher.web`, `researcher.codebase`, `researcher.academic`                               |
| Writing & Comms   | `writer.docs`, `writer.commit`, `writer.slack_digest`                                        |
| Product & Support | `pm.prioritizer`, `support.triager`                                                          |

> Each agent ships with full input/output JSON Schemas, a
> strictly-declared `writes[]`/`reads[]`, honest `resources`, a bounded
> `retry_policy`, and multi-gate `quality` (min_score, max_latency_s,
> max_cost_usd, must_have, must_not).
>
> **Browse the full catalog** with the *"Question → reach for agent Y"*
> decision tree at [`agents/AGENTS_CATALOG.md`](agents/AGENTS_CATALOG.md).
