# Changelog

All notable changes to orchestrator_skill are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.3] — 2026-07-03

### Added
- **Dynamic Sub-Agent Factory** — the supervisor can now *synthesize* a
  new sub-agent on the fly when no existing agent matches a task. Opt-in
  via `dynamic_agents.enabled: true` in `config.json` (off by default
  for safety). Three creation modes are supported:
  - `factory_create_clone`   — clone an existing agent and rename it.
  - `factory_create_persona` — same agent, different persona voice.
  - `factory_synthesize`     — generate a contract from a capability-gap
    description (keyword-matches an existing family and clones its
    base; LLM-backed synthesis is wired in but currently falls back to
    clone when no synthesis model is configured).
  Spawned agents run in **quarantine** for `default_quarantine` runs
  (default 3) before they are trusted. A `match_or_propose_agent`
  dispatcher hook lets the supervisor automatically propose a new
  agent when no static agent matches. Every spawn is written to
  `factory_audit.jsonl` and tracked in `dynamic_agents.json`.
- **Factory CLI** — seven new subcommands on the top-level `skill.sh`
  dispatcher:
  - `--agents-factory-create <name> <parent> [reason]`
  - `--agents-factory-persona <name> <base> <persona> [reason]`
  - `--agents-factory-synthesize <name> <gap> [reason]`
  - `--agents-factory-list`
  - `--agents-factory-show <name>`
  - `--agents-factory-tick <name>`   (age out one quarantine run)
  - `--agents-factory-remove <name>` (drop the agent and its file)
  All seven accept a bare agent name (`coder.python`) and resolve it
  against the static `agents/` directory; pass an absolute path to
  use a non-default parent.
- New files:
  - `lib/dynamic_agents.sh`        — the factory module.
  - `templates/agent.template.json` — generic agent contract scaffold.
  - `templates/agent_families.json` — family metadata for keyword
    matching + persona catalogue.
  - `agents/dynamic/`              — where spawned agents live.
- `config.schema.json` and `config.example.json` now accept a
  `dynamic_agents` block with `enabled`, `static_only`,
  `default_quarantine`, `max_dynamic`, `synthesis_model`,
  `synthesis_provider`, `audit_log`, `dynamic_dir`, and
  `require_approval` keys.

### Changed
- `lib/dynamic_agents.sh` is auto-loaded by the skill in any project
  that has the `dynamic_agents` block in its `config.json` (off by
  default → no behaviour change for existing projects).
- `skill.sh --help` now documents the seven factory subcommands under
  `DYNAMIC AGENT FACTORY`.

## [2.0.2] — 2026-07-02

### Added
- **`IMPROVEMENTS_30.md`** — a deep-research synthesis of 30 concrete
  next-step improvements across 3 tiers:
  - **Sub-Agent (1–12):** prompt-cache awareness, 5-tier memory,
    streaming chunks, MCP shared tool registry, capability negotiation,
    persona pins, per-agent eval suites, LLM adapter pattern,
    cooperative cancellation, OTel GenAI spans, per-secret approval,
    multi-modal kinds.
  - **Supervisor (13–24):** learning router, soft-skills matrix,
    backpressure, dead-letter queue, cross-plan memory carryover,
    cost governor, microVM sandbox, WebSocket/SSE dashboard, skill
    registry, conditional routing, predicted-cost pre-flight,
    mid-plan replanning.
  - **Joint (25–30):** shadow routing, tool-call budget, mid-run
    compaction, periodic canary, confidence-weighted vote, cross-agent
    pub/sub bus.
  Includes impact magnitudes and an implementation recommendation.
  Sources cited: NVIDIA, AWS Strands, Atlan, OTel GenAI WG, Confident AI,
  Skilldex, Claude Marketplace, Portkey, and others.

### Changed
- **`README.md`**, **`AGENTS.md`**, and **`REVOLUTION.md`** now reference
  `IMPROVEMENTS_30.md` for the next-stage roadmap.

## [2.0.1] — 2026-07-02

### Added
- **30 production-ready sub-agent definitions** under `agents/`, spanning
  9 families: coding, review, QA, security, devops, data, research,
  writing, product/support. Drop-in usable — the orchestrator auto-discovers
  any `*.json` in the directory.
- **`agents/AGENTS_CATALOG.md`** — full index with the *"Question →
  reach for agent Y"* decision tree for all 30 sub-agents.
- **Family breakdown**:
  - **Coding & Build (5)** — `coder.python`, `coder.typescript`,
    `coder.rust`, `coder.sql_migrations`, `coder.terraform`
  - **Review & QA (5)** — `reviewer.code`, `reviewer.architecture`,
    `reviewer.dependencies`, `tester.mutation`, `qa.flakehunter`
  - **Security (4)** — `security.sast`, `security.deps`,
    `security.secrets`, `security.threatmodel`
  - **DevOps & Ops (4)** — `ops.deploy.k8s`, `ops.deploy.serverless`,
    `ops.incident_responder`, `ops.cost_optimizer`
  - **Data (4)** — `data.profiler`, `data.cleaner`, `data.synthesizer`,
    `data.lineage`
  - **Research & Knowledge (3)** — `researcher.web`,
    `researcher.codebase`, `researcher.academic`
  - **Writing & Comms (3)** — `writer.docs`, `writer.commit`,
    `writer.slack_digest`
  - **Product & Support (2)** — `pm.prioritizer`, `support.triager`
- Each agent ships with full `input_schema`/`output_schema`,
  `writes[]`/`reads[]`, honest `resources`, bounded `retry_policy`,
  and multi-gate `quality` (min_score, max_latency_s, max_cost_usd,
  must_have, must_not).

### Changed
- **`AGENTS.md`** cross-references the catalog and reflects that the
  built-in roster is now 30, not 15.
- **`META.md`** and **`README.md`** updated to point to
  `agents/AGENTS_CATALOG.md` for the full sub-agent index.

## [2.0.0] — 2026-07

### Added
- **Hierarchical orchestration** (`--spawn-subplan`, `--list-subplans`,
  `parent_plan_id`, `sub_plan_id`). Supervisor-of-supervisors pattern, in
  response to langgraph-supervisor-py and oap-agent-supervisor.
- **Three handoff modes** (`one_way`, `bidirectional`, `fan_in`) per task.
  Inspired by the LibreChat #8372 design discussion.
- **Quality gates**: per-task `validate_cmd`, `validate_tool`,
  `validate_rubric`, `validate_min_score`. Inspired by 12Matt3r/Supervisorai.
- **Retry policy** per task: exponential backoff, jitter, error-class filter.
- **Cost and token tracking**: `task_cost_usd`, `task_tokens_used`,
  `--cost-report` command.
- **Observability**: append-only `audit.jsonl` log, `--watch` for live state,
  OTel-compatible `--trace` stubs, `--audit` to inspect.
- **Plan templates** (4 ships in `templates/`) + `--import-template` /
  `--list-templates`.
- **Export / Import / Replay**: share plans between projects;
  `--replay` refreshes a sealed plan.
- **Memory store**: `--memory-set/get/list/delete/snippet` for cross-plan
  knowledge.
- **Dynamic routing (`--auto-plan`)**: opt-in LLM-driven goal → plan
  expansion (via OpenAI-compatible API).
- **Pass-5 notifications** (`--notify` and configurable channels:
  log, file, webhook, mail, slack).
- **Resource load tracking**: `--agent-load`, `--record-load`, picks the
  lowest-loaded agent when dispatching.
- **Sub-plan reuse**: walk a tree of parent + sub-plans.
- **DOT and ASCII DAG** (`--dot`, `--dag`) for visualising dependencies.
- **Stats and topological analysis** (`--stats`, `--topo`,
  `--critical-path`).
- **Doctor** (`--doctor`): pre-flight environment checks.
- **Plan / task schemas** (`schemas/plan.schema.json`, validation on every
  write).
- **Web dashboard** (`web/dashboard.html`) polling `state.json` for
  live progress visualisation.
- **MCP server** (`scripts/mcp_server.py`) exposing all major commands
  as JSON-RPC tools.
- **Tests** (4 bats files): validation, concurrency, sub-plans,
  observability.
- **CI** (`.github/workflows/ci.yml`): shellcheck + bats + smoke MCP.
- **`install.sh` and `Makefile`** for one-command setup.
- **`agent_aliases`** config block: decouple orchestration roles from
  MiniMax-specific agent names.

### Changed
- **`skill.sh` is now a thin dispatcher**; logic lives in modular
  `lib/*.sh` files.
- The seal contract is now *enforced*: `--seal` flips
  `sealed: true`; `--reset`, `--skip`, and `--dispatch-ready` on a
  sealed plan refuse without `--unseal` first.
- **`--add-task-file`** is the safe way to inject task JSON (escaping-safe).
- **`--validate`** refuses to seal plans with incomplete tasks unless
  `--force` is given.

### Removed
- The implicit `set -euo pipefail` reliance on global state is gone; the
  lock primitives are explicit.

## [1.0.0] — initial

- Single-pass shell orchestrator with a 4-pass workflow, JSON config
  + plan.json + state.json, no formal schema, no quality gate, no
  hierarchical support.
