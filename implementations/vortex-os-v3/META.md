# META.md — orchestrator_skill v2

> **Skill metadata & contract for the agent ecosystem.**

---

## Identity

| Field        | Value                                                          |
| ------------ | -------------------------------------------------------------- |
| **Name**     | `orchestrator_skill`                                           |
| **Version**  | `2.0.0`                                                        |
| **Kind**     | Reusable multi-agent orchestration layer (Bash + JSON)        |
| **License**  | MIT — see `LICENSE`                                            |
| **Origin**   | Distilled from the MiniMax multi-agent editorial pipeline     |
| **Audience** | Any user or agent that needs to coordinate 5+ sub-agents      |

---

## Purpose

A project-agnostic coordination layer for **decomposing a high-level
goal into a plan**, **dispatching each plan item to the right
sub-agent**, **aggregating the results**, **applying quality gates**,
and **sealing the plan**. v2 adds **hierarchical sub-plans**,
**retry policies**, **cost/token tracking**, **observability**, and
**Pass-5 notifications** over v1.

---

## Inputs

1. A high-level goal (free-form text from the user).
2. A JSON config (`config.json`) describing the agent roster, paths,
   passes, delegation rules, and quality/notification settings.
3. A plan file (`plan.json`) that the orchestrator writes and updates.
4. *(Optional)* Templates, sub-plans, memory snippets, audit log.

## Outputs

For each orchestrated project:

1. A `plan.json` listing every task with deps, statuses, retries, and
   per-task validation settings.
2. A `state.json` for context recovery, with `sealed_unsealed_history`.
3. The actual deliverables produced by sub-agents.
4. A `sealed: true` flag plus a `[STATUS: ORCHESTRATION COMPLETE.]`
   marker in `state.json`.
5. A JSONL `audit.jsonl` log of every state transition.
6. Sub-plan files (`sub_*.plan.json`) when hierarchical orchestration
   is used.

---

## Interface

The skill is `skill.sh` (a thin CLI dispatcher); the heavy lifting
lives in `lib/*.sh`. Every command supports `--config <path>` and the
output is JSON when `--json` is passed.

| Command | Description |
|---|---|
| `--help`, `--version`, `--verbose`, `--force`, `--json` | Generic toggles |
| `--config <path>` | Override active config |
| `--init <dir>` | Bootstrap a new project |
| `--plan`, `--plan-json`, `--status` | Inspect plan / state |
| `--add-task <json>`, `--add-task-file <path>` | Append a task |
| `--import-template <name>` / `--list-templates` | Apply a template (use `--template-var k=v`) |
| `--ready`, `--next-task`, `--dispatch-ready` | Choose & run next batch |
| `--mark-done`, `--mark-failed`, `--escalate`, `--skip`, `--reset` | Update statuses |
| `--validate`, `--aggregate`, `--seal`, `--unseal` | Plan-level operations |
| `--checkpoint`, `--recover` | Context recovery |
| `--topo`, `--critical-path`, `--stats`, `--dot`, `--dag`, `--diff` | Analysis |
| `--doctor` | Environment checks |
| `--quality-gate <id>`, `--judge` | Apply a quality gate |
| `--watch`, `--audit`, `--cost-report`, `--trace` | Observability |
| `--spawn-subplan`, `--list-subplans` | Hierarchy |
| `--export <file>`, `--import <file>`, `--replay` | Portability |
| `--memory-set/get/list/delete/snippet` | Memory store |
| `--auto-plan "<goal>"` | LLM-based plan expansion |
| `--notify` | Fire configured notification channels |
| `--agent-load`, `--record-load` | Resource management |

---

## Behavioural Contract (v2)

The skill guarantees the following, **provided the user does not edit
the plan or state file outside the skill**:

1. After `--add-task`, the plan passes `--validate` (or the
   orchestrator prints a clear error).
2. After `--dispatch-ready`, every task in the ready set has
   `status = dispatched` and a fresh `started_at`. The `attempts`
   counter is incremented.
3. After `--mark-done <id> <output>`, the task has `status = complete`,
   `output_path` set, `completed_at` set, and is appended to
   `state.completed_tasks`.
4. After `--seal`, the plan has `sealed: true`; `--reset`, `--skip`,
   and `--dispatch-ready` refuse to operate on a sealed plan without
   `--unseal`.
5. After `--unseal`, the `sealed_unsealed_history` array gets a new
   entry with `at`, `by`, `action: unseal`.
6. Every state-changing command writes a JSONL record to
   `audit.jsonl` (if `paths.audit_file` is configured).
7. Concurrent writes are protected by `flock` (or a `mkdir`-based lock
   fallback) per file.
8. The state file is the durable record. Context windows can be
   discarded at any time; `--recover` prints the last checkpoint.

---

## Pass Inventory

| Pass | Name | Role |
|---|---|---|
| 1 | Task Analysis & Decomposition | Convert a goal into a dependency graph. |
| 2 | Agent Selection & Dispatch | Route each task to the right agent in parallel. |
| 3 | Result Aggregation & Validation | Verify outputs and apply quality gates. |
| 4 | Logging & Sealing | Audit + state + sealed contract. |
| 5 | Notification | Optional; configured channels. |

---

## Delegation Model (v2)

```
User → Parent Agent (orchestrator_skill) → Sub-Agents
         ↳ Sub-Plan (hierarchical)
              → Sub-Sub-Agents
```

* The parent holds the user's goal and drives the state machine.
* Sub-agents are stateless specialists and run off the task they were
  handed (with the handoff mode determining how control flows back).
* Sub-plans are themselves first-class plans, enabling true
  recursive decomposition.
* The plan and state are persisted on disk; they are the only
  authoritative record.

---

## Files in v2

| File | Purpose |
|---|---|
| `skill.sh` | The dispatcher (`bash skill.sh --help`) |
| `lib/core.sh`, `lib/locking.sh`, `lib/schema.sh`, `lib/graph.sh` | Foundation |
| `lib/plan_state.sh` | Locked plan/state CRUD |
| `lib/dispatch.sh` (via commands.sh) | Dispatch + mark-done/failed |
| `lib/retry.sh`, `lib/quality.sh` | Execution policy |
| `lib/observability.sh` | Audit log, cost, traces, watch |
| `lib/subplan.sh`, `lib/templates.sh` | Hierarchy & templates |
| `lib/export_import.sh`, `lib/memory.sh` | Knowledge across runs |
| `lib/notification.sh`, `lib/aliases.sh` | Pass-5 + decoupling |
| `lib/commands.sh` | All `cmd_*` implementations |
| `schemas/*.json` | Config / plan / state schemas |
| `templates/*.json` | Reusable plan templates |
| `agents/*.json` | **30 built-in sub-agents** (coding, review, QA, security, devops, data, research, writing, product/support) |
| `agents/AGENTS_CATALOG.md` | Index of all 30 sub-agents with the *"Question → agent"* decision tree |
| `examples/*/run.sh` | Runnable demos |
| `web/dashboard.html` | Live status dashboard |
| `scripts/mcp_server.py` | MCP server (stdio + HTTP) |
| `scripts/migrate_plan.py` | v1→v2 plan converter |
| `tests/*.bats` | Bats test suite |
| `install.sh`, `Makefile`, `LICENSE` | Ops |

## Extending the Skill (v2)

1. **New agent role.** Add `agents.<role_name>` and an entry in
   `agent_aliases`. The orchestrator picks it up automatically.
2. **New pass.** Add a new section to `passes.*` in config and a new
   `cmd_*` function in `lib/commands.sh`. Wire a flag into
   `skill.sh`.
3. **New template.** Drop a JSON file in `templates/`.
4. **New notification channel.** Add an entry to `notifications.channels`
   in config.

## Failure Modes & Recovery

| Symptom | Recovery |
|---|---|
| A task is dispatched but never completes | `--mark-failed`; optional auto-retry via `retry_policy` |
| Plan has a cycle | `--validate` refuses; redesign dependencies |
| Context window lost | `--recover` re-reads `state.json`; `--ready` resumes |
| Sub-agent returns wrong format | `--mark-failed` and `--reset`, or pass `--judge` verdict |
| Two dispatches racing | lock primitives prevent corruption; restart parent |
| Plan sealed in error | `--unseal` (audited) |
| Audit log growing too big | rotate via `mv audit.jsonl{,.old}` (planned: `--compact-audit`) |

## Versioning

* **2.0.2** — `IMPROVEMENTS_30.md`: 30 concrete next-step improvements
  (3 tiers: Sub-Agent / Supervisor / Joint), each with pseudocode &
  impact estimate. Plans v2.2 / v3.0.
* **2.0.1** — Shipped **30 built-in sub-agent definitions** under
  `agents/` + `agents/AGENTS_CATALOG.md` index. No code changes — pure
  declarative extension.
* **2.0.0** — Hierarchical plans, quality gates, retry, observability,
  templates, web, MCP.
* **1.0.0** — Initial release. 4-pass workflow, JSON config, plan and
  state files, command surface, 100 use cases.
