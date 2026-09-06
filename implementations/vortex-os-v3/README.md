# orchestrator_skill — Universal Sub-Agent Orchestration (v2.0)

A project-agnostic, configurable **orchestration layer** for delegating
work to a pool of specialised sub-agents. v2 adds hierarchical
sub-plans, quality gates, retry policies, cost/token tracking,
templates, observability, plan export/import, a web dashboard, an MCP
server, and a hard seal contract. Still pure Bash + Python — no
framework dependencies.

* **Plan-as-artifact.** The plan and state are first-class JSON files
  you can version, diff, and review.
* **Hierarchical.** Spawn sub-plans (supervisor-of-supervisors) inside
  any task.
* **Hard seal.** Sealed plans refuse re-dispatch without `--unseal`.
* **Deterministic.** Same goal + same config = same plan. Same plan +
  same agents = same outputs.

---

## Quick Start (60 seconds)

```bash
# 1. Install / bootstrap
./install.sh                           # or: cp -r . my_proj/
skill --init ./my_proj                 # writes config.json, plan.json, state.json

# 2. Apply a template or hand-write tasks
skill --import-template competitor_landscape --template-var INDUSTRY="AI"
# or:
skill --add-task-file t.json

# 3. Inspect & dispatch
skill --ready
skill --validate
skill --dispatch-ready

# 4. Mark work done
skill --mark-done r1 ./out/r1.md

# 5. Observe live
skill --watch            # in another terminal
skill --cost-report

# 6. Close out
skill --seal
```

---

## What's new in 2.0

| Feature | Command | What it does |
|---|---|---|
| Hierarchical sub-plans | `--spawn-subplan`, `--list-subplans` | Spawn nested plans from any task |
| Handoff modes (per task) | `handoff: one_way \| bidirectional \| fan_in` | Three routing modes from the LibreChat #8372 discussion |
| Quality gate | `--quality-gate <id>` | Run `validate_cmd` / `validate_tool` / LLM judge |
| Retry with backoff | per-task `retry_policy` | Exponential backoff + jitter |
| Cost & token tracking | `--cost-report` | Aggregated `task_cost_usd` per task |
| Observability | `--audit`, `--watch`, `--trace` | JSONL audit log, live state, OTel-compatible traces |
| Plan templates | `--list-templates`, `--import-template` | 4 templates bundled; add your own |
| Export/import/replay | `--export`, `--import`, `--replay` | Share plans between projects |
| Memory store | `--memory-set/get/list/delete/snippet` | Cross-plan shared knowledge |
| Dynamic LLM routing | `--auto-plan "<goal>"` | Opt-in: expand a goal into tasks via an LLM |
| Pass-5 notifications | `--notify` + config channels | log, file, webhook, mail, slack |
| Resource load awareness | `--record-load`, `--agent-load` | Sub-agents report CPU/memory |
| Web dashboard | `web/dashboard.html` | Polls `state.json` for live visualisation |
| MCP server | `scripts/mcp_server.py` | Exposes everything as JSON-RPC tools |
| Hard seal | `--seal` / `--unseal` | Sealed plans refuse re-dispatch |
| Doctor | `--doctor` | Pre-flight env checks |
| Cycle-safe writes | automatic | `flock`-based locking on plan.json/state.json |

See `CHANGELOG.md` for the full list and `MIGRATION.md` for the v1 → v2
recipe.

---

## What's Next (Roadmap)

A deep-research pass surfaced **30 concrete next-step improvements**
organised into Sub-Agent, Supervisor, and Joint tiers — including
prompt-cache awareness, learning routers, microVM sandboxes, OTel
GenAI tracing, MCP shared tool registries, and an agent-bus pub/sub.
See [`IMPROVEMENTS_30.md`](IMPROVEMENTS_30.md).

---

## Built-in Sub-Agent Catalog

The orchestrator ships with **30 production-ready sub-agents** in
`agents/` — drop them into any plan without writing new code:

```
Coding & Build   →  coder.python, coder.typescript, coder.rust,
                    coder.sql_migrations, coder.terraform
Review & QA      →  reviewer.code, reviewer.architecture,
                    reviewer.dependencies, tester.mutation,
                    qa.flakehunter
Security         →  security.sast, security.deps, security.secrets,
                    security.threatmodel
DevOps & Ops     →  ops.deploy.k8s, ops.deploy.serverless,
                    ops.incident_responder, ops.cost_optimizer
Data             →  data.profiler, data.cleaner, data.synthesizer,
                    data.lineage
Research         →  researcher.web, researcher.codebase,
                    researcher.academic
Writing & Comms  →  writer.docs, writer.commit, writer.slack_digest
Product & Support →  pm.prioritizer, support.triager
```

Each agent declares its `input_schema`, `output_schema`,
`writes[]`/`reads[]`, `resources`, `retry_policy`, and `quality` gates
in a JSON file the orchestrator validates against `schemas/*`.

Browse the full catalog with the *"Question → reach for agent Y"*
decision tree at  <agents/AGENTS_CATALOG.md>.

> **Discovery precedence:** project-local `agents/` → built-in catalog
> (`skill_v2/agents/`) → user-wide `$ORCH_HOME/agents/` → system-wide
> `/etc/orchestrator/agents/`. Same name in multiple sources → the
> higher-precedence wins, with a warning logged to `audit.jsonl`.

---

## Discovery, Linting & Composition

The orchestrator knows about agents from four sources (highest precedence
first):

1. **Project-local** — `./agents/*.json`
2. **User-wide** — `~/.orchestrator/agents/*.json`
3. **System-wide** — `/etc/orchestrator/agents/*.json`
4. **Built-in** — `skill_v2/agents/*.json` (the 30 shipped agents)

### Discover what's loaded

```bash
# Pretty table (active agents only)
skill --agents-discover

# Include deprecated agents
skill --agents-discover --include-deprecated

# Raw JSON for programmatic use
skill --agents-discover --json | jq '.[] | .name'
```

### Inspect a single agent

```bash
skill --agents-inspect coder.python
skill --agents-validate ./my-custom-agent.json
```

### Lint against the 8 invariants

```bash
# Lint everything
skill --agents-lint --all

# Lint one agent by file or name
skill --agents-lint ./agents/coder.python.json
skill --agents-lint coder.python
```

The 8 invariants (`agents/INVARIANTS.md`):

| # | Invariant | Failure severity |
|---|---|---|
| I1 | Idempotence | High |
| I2 | Resource Honesty | Medium |
| I3 | Write Containment | **Critical** |
| I4 | Read Containment | **Critical** |
| I5 | Sealed Envelope | High |
| I6 | Retry Honesty | Medium |
| I7 | Secret Hygiene | **Critical** |
| I8 | Metric Truthfulness | Low |

### View the composition graph

```bash
# ASCII
skill --agents-graph

# Graphviz (render to PNG)
skill --agents-graph --format dot | dot -Tpng > /tmp/agents.png
```

### Match-or-Propose (auto-dispatch)

When a task references a role with no static agent, the orchestrator runs
**match-or-propose** using weighted Jaccard similarity:

| Similarity | Weight breakdown | Action |
|---|---|---|
| < 0.15 | schema 0.50 + desc 0.25 + caps 0.25 | Reuse silently |
| 0.15 – 0.40 | (same) | Reuse + log `near_match` event |
| 0.40 – 0.70 | (same) | Require human `--approve-spawn` (unless `auto_spawn: true`) |
| > 0.70 | (same) | Refuse dispatch |

Dynamic agents start with `quarantine_remaining: 3`. The canary-doctor
ages them out one tick at a time. When the counter hits zero, the
**lint-or-orphan gate** runs `skill --agents-lint` on the dynamic agent:

- **Lint passes** → promoted to `quarantine_status: "trusted"`
- **Lint fails** → moved to `agents/orphaned/` and audit-logged

Enable auto-aging with `dynamic_agents.auto_tick_quarantine: true` in config.

### Trace a past run

```bash
skill --agents-trace <run_id>
```

Outputs every audit-log entry for that run, chronologically.

### Deprecation workflow

Mark an agent for retirement without breaking existing plans:

```json
{
  "name": "legacy.coder.python2",
  "deprecated": true,
  "deprecation_message": "Replaced by coder.python v2+. See docs/migrations/coder-python-v2.md"
}
```

The agent is hidden from `--agents-discover` by default. Use
`--include-deprecated` to see it; existing plans keep working until the
agent's `eol` date passes.

---

## Dynamic Agent Factory (on-the-fly sub-agent creation)

**The supervisor can create new sub-agents at runtime when no static
agent matches a task.** It is enabled by default in v2.0.3+ — no extra
config required. Opt out with `dynamic_agents.enabled: false`.

### Three creation modes

| Mode | Command | When to use |
|---|---|---|
| **Clone** | `--agents-factory-create <name> <parent> [reason]` | You need a near-copy of an existing agent with a different name |
| **Persona** | `--agents-factory-persona <name> <base> <persona> [reason]` | Same capability, different voice / guard |
| **Synthesize** | `--agents-factory-synthesize <name> <gap> [reason]` | No agent matches at all; auto-pick family by keyword |

### Inspect / manage

```bash
skill --agents-factory-list        # show all spawned agents
skill --agents-factory-show <name> # show index entry
skill --agents-factory-tick <name> # age out one quarantine run
skill --agents-factory-remove <name>
```

### Automatic (no CLI needed)

When you run `skill --dispatch-ready`, the supervisor walks the pending
tasks and, for every role that has **no static match**, auto-spawns a
dynamic agent via the factory before dispatching. Every dynamic agent
is quarantined for `default_quarantine` runs (default 3) and is
auto-ticked each time it gets dispatched — after that, it is "trusted".

### Safety rails

- **Off by default** opt-in is still available: set
  `dynamic_agents.enabled: false` in `config.json`.
- **Static-only mode**: `dynamic_agents.static_only: true` blocks all
  spawns.
- **Cap**: `dynamic_agents.max_dynamic` limits total dynamic agents.
  `0` = unlimited (the default).
- **Audit**: every spawn lands in `<config_dir>/factory_audit.jsonl`.
- **Quarantine**: new agents run for `default_quarantine` (default 3)
  dispatches before they are trusted. The canary-doctor ticks the
  counter automatically.
- **Path safety**: the CLI accepts either a bare name (resolved
  against `agents/`) or an absolute path; no path traversal.

### Example

```bash
# Plan a task that needs a "Kubernetes deployer" — no static agent
# matches, so the supervisor spawns one on the fly:
skill --add-task '{"id":"k1","description":"deploy to k8s",
                   "agent_role":"ops.deploy.k8s"}'
skill --dispatch-ready
#   -> Auto-spawned dynamic agent: auto.ops.deploy.k8s.<ts>
#   -> Dispatching 1 task(s).
#   -> k1 (role=ops.deploy.k8s primary=auto.ops.deploy.k8s.<ts>)
```

---

## The 4 (+1) Passes

| # | Pass | Role |
|---|---|---|
| 1 | Task Analysis & Decomposition | Break the goal into atomic tasks with dependencies. |
| 2 | Agent Selection & Dispatch | Map each task to the right agent; dispatch in parallel. |
| 3 | Result Aggregation & Validation | Verify completeness, quality-gates, and output paths. |
| 4 | Logging & Sealing | Write audit + state, mark the plan complete. |
| 5 | Notification *(opt-in)* | Email / Slack / webhook on seal. |

Pass 5 is configurable in `config.json → passes.pass_5_notification`.
It's off by default.

---

## Hierarchical Orchestration

```bash
# A "build app" task is itself a project. Spawn a sub-plan for it:
skill --spawn-subplan build_app "Sub-project: build the TODO app"

# The parent task flips to "pending" with a sub_plan_id pointer
# until the sub-plan is sealed, at which point the parent flips to "complete".
```

Sub-plans live in the project's own folder as `sub_<id>.plan.json`.
`--list-subplans` walks the tree.

---

## Quality Gates

```json
{
  "id": "build_site",
  "agent_role": "build_static_site",
  "depends_on": ["d_site"],
  "validation": {
    "validate_cmd": "pylint --score . > /dev/null && exit 0 || exit 1",
    "validate_tool": "scripts/judge_site.py",
    "validate_rubric": "Clean design tokens; deployed; lighthouse > 90",
    "validate_min_score": 0.7
  }
}
```

Run `skill --quality-gate build_site` to execute the gates; the task
either flips to `complete` (pass) or `failed` (fail).

---

## Observability

```bash
# Append-only JSONL audit log (every dispatch, mark_done, mark_failed, seal).
skill --audit

# Live state polling
skill --watch         # pretty
skill --watch --json  # raw JSON

# OTel-compatible trace stub per task
skill --trace r1
```

A static dashboard lives at `web/dashboard.html`. Serve it with any HTTP
server:

```bash
cd <project> && python3 -m http.server 8000
# open http://localhost:8000/../web/dashboard.html?endpoint=/path/to/state.json
```

---

## The MCP Server

Expose every command to MCP-aware clients:

```bash
# stdio transport (works with Claude Desktop, mcp-cli, etc.)
python3 scripts/mcp_server.py --config config.json

# or HTTP (for testing)
python3 scripts/mcp_server.py --http --port 8765
```

Register `scripts/mcp_server.py` as the `command` for an MCP server
called `orchestrator`. The server auto-registers the following tools:
`plan.show`, `plan.status`, `plan.add_task`, `plan.dispatch_ready`,
`plan.mark_done`, `plan.mark_failed`, `plan.seal`, `plan.export`.

---

## Files in v2

| File | Purpose |
|---|---|
| `skill.sh` | Thin CLI dispatcher |
| `lib/*.sh` | 14 modular libraries |
| `schemas/*.json` | JSON schemas for config, plan, state |
| `templates/*.json` | 4 reusable plan templates |
| `examples/*/run.sh` | Runnable end-to-end demos |
| `agents/*.json` | **30 built-in sub-agent definitions** (auto-discovered) |
| `agents/AGENTS_CATALOG.md` | Full sub-agent index & decision tree |
| `web/dashboard.html` | Static live-state dashboard |
| `scripts/mcp_server.py` | MCP server (stdio + HTTP) |
| `scripts/migrate_plan.py` | v1→v2 plan converter |
| `tests/*.bats` | Bats smoke tests |
| `install.sh` | One-shot install |
| `Makefile` | `make test` / `make lint` / `make install` |
| `LICENSE`, `CHANGELOG.md`, `MIGRATION.md` | The usual docs |
| `WORKFLOW.md` | Detailed walk-through of the 5 passes |
| `AGENTS.md` | Sub-agent contract (input/output envelope, schemas, retry) |
| `agents/AGENTS_CATALOG.md` | Index of all **30 built-in sub-agents** |
| `META.md` | Skill metadata & delegation contract |
| `REVOLUTION.md` | Why this skill is a paradigm shift |
| `100_USE_CASES.md` | 100 distinct orchestration scenarios |
| `config.example.json` | Generic starter config |
| `config.minimax.json` | Reference config tuned for MiniMax's roster |

---

## License

MIT — see `LICENSE`.
