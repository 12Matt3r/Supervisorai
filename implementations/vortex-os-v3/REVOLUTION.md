# The Orchestrator Revolution — v1 → v2

> *Why v2.0 isn't a release. It's a re-think.*

---

## TL;DR

Orchestrator Skill **v1** was a clever shell script that "ran a plan."
Orchestrator Skill **v2** is a **declarative, hierarchical, quality-gated,
multi-agent operating system** for the shell.

|                       | v1                                  | v2                                                         |
| --------------------- | ----------------------------------- | ---------------------------------------------------------- |
| State                 | JSON files                          | JSON files + JSON-Schema validation + sealed logs          |
| Topology              | Linear sequential                   | DAG with cycles + sub-plans (N-deep)                       |
| Routing               | Static, developer-written           | Template-driven, LLM-assisted, retry-aware                  |
| Observability         | stdout                             | Structured logs + metrics + live web dashboard              |
| Quality               | "Looks fine"                        | Multi-gate (lint + test + schema + must-have)              |
| Concurrency           | Single-threaded                     | File-locked writes; cross-plan safe                         |
| Extensibility         | Edit `skill.sh`                     | Drop a JSON file in `agents/`                                |
| Test Coverage         | "I tried it, it worked"             | `bats` suite + CI                                           |
| Plan-as-Artifact      | Implicit                            | Explicit, sealed, signature-chained                        |
| Error Handling        | Hope + retry flag                   | Retry policy + budget-aware + dead-letter                   |
| Operator UX           | One cryptic command                 | `add-task`, `validate`, `run`, `replay`, `promote`         |

---

## The Story of v1 (2019–2025)

Orchestrator v1 was born from a simple need: *"Please don't make me context-switch
between twelve LLMs to ship a feature."* A small Bash script, a JSON plan, a
4-pass workflow — Task Analysis, Agent Dispatch, Aggregation, Sealing — and
suddenly multi-agent work became **repeatable**.

It worked. A lot.

It also accumulated scars:

- The plan file was edited manually; typos crashed the run.
- The retry flag meant *"try once more, then give up."*
- The Quality Gate was a single `bash` regex.
- Observability meant scrolling the terminal.
- Adding a new agent meant editing `skill.sh`.

The community loved v1 and outgrew it simultaneously.

---

## The Five Axioms That Drove v2

When we sat down to design v2, we wrote down five axioms. Every feature in
v2 must justify itself against at least one of these.

### Axiom I — *Plans are code, not chat.*
A plan file is an **artifact**. It must be validatable, diffable, signable,
and reproducible. v2 introduces JSON-Schema validation, content-addressable
identifiers, and an Ed25519 signature chain on every state mutation.

### Axiom II — *Agents are contracts, not scripts.*
An agent's `kind`, `writes`, `reads`, `retry_policy`, and `quality` are
**declared**, not observed. The orchestrator can therefore **schedule, cap,
and police** every sub-agent without running it — fixing v1's biggest blind
spot.

### Axiom III — *Quality is plural.*
Lint passes. Tests pass. Schema passes. The triple-passage is hardcoded.
v2 makes **gates** composable: a coder agent may gate on `lint + test`,
but a researcher agent gates on `must_have + must_not + max_latency`.

### Axiom IV — *Hierarchy is free.*
A task may be a leaf — or it may be a **sub-plan**, delegating to a nested
orchestration that produces its own DAG. This is the missing primitive
between single-agent tools and full hierarchical agent platforms.

### Axiom V — *The shell is sacred.*
We did not rewrite v2 in Python. We did not abandon `bash`. We
double-down on **shell + jq + GNU coreutils**, because:
- Shell is the **only** language that ships on every Unix-like OS.
- Shell is the **only** language that is itself a tool ecosystem.
- Shell is the **only** language with five decades of battle-tested
  primitives (`flock`, `trap`, `ulimit`, `kill -0`).
v2 **augments** the shell with structured libraries; it does not replace it.

---

## What's New, Feature-by-Feature

### 1. Sub-Plans (Hierarchical Orchestration)
Tasks can now recursively spawn child plans. Depth is unlimited. The parent
plan is **blocked** until the child plan emits its result envelope. Cycles
are detected. Outcomes inherit upward through the hierarchy.

```bash
./skill.sh task add "Research topic X" \
  --agent subplan.research \
  --input '{"topic": "agentic workflows 2026"}'
```

### 2. Retry & Backoff Policies
Every task and every agent has a retry policy. v2 supports `none`,
`linear`, and `exponential` with jitter. Retries are budget-aware — the
plan's `global.limits.max_total_runtime_s` is the absolute cap.

```json
"retry_policy": { "max_attempts": 3, "backoff": "exponential", "on": ["timeout"] }
```

### 3. Multi-Gate Quality
Gates are now **typed and composable**. Built-in gates: `lint`, `test`,
`schema`, `secret_scan`, `size_limit`, `must_have`, `must_not`. Custom
gates are JSON-described in `quality_gates/*.json`.

### 4. Templates
A template is a **pre-validated plan skeleton** with variable
substitution. Examples bundled:
- `templates/coder.json` — single coding task
- `templates/research.json` — multi-step research + synthesis
- `templates/release.json` — lint → test → build → deploy
- `templates/migration.json` — safe database migration workflow

```bash
./skill.sh plan new release --from-template release@v1
./skill.sh plan show
```

### 5. Declarative Sub-Agents
Adding a sub-agent is now:
```bash
./skill.sh agents new my.coder --kind shell --entry 'scripts/coder.sh'
```
Zero edits to `skill.sh`. The v2.0 install ships with **30 production-
ready sub-agent definitions** under `agents/`, spanning 9 families:

| Family             | Built-in agents                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------- |
| Coding & Build     | `coder.python`, `coder.typescript`, `coder.rust`, `coder.sql_migrations`, `coder.terraform`  |
| Review & QA        | `reviewer.code`, `reviewer.architecture`, `reviewer.dependencies`, `tester.mutation`, `qa.flakehunter` |
| Security           | `security.sast`, `security.deps`, `security.secrets`, `security.threatmodel`                 |
| DevOps & Ops       | `ops.deploy.k8s`, `ops.deploy.serverless`, `ops.incident_responder`, `ops.cost_optimizer`    |
| Data               | `data.profiler`, `data.cleaner`, `data.synthesizer`, `data.lineage`                          |
| Research           | `researcher.web`, `researcher.codebase`, `researcher.academic`                               |
| Writing & Comms    | `writer.docs`, `writer.commit`, `writer.slack_digest`                                        |
| Product & Support  | `pm.prioritizer`, `support.triager`                                                          |

Browse the full 30-agent catalog with the *"Question → reach for agent
Y"* decision tree at `agents/AGENTS_CATALOG.md`.

### 6. Live Web Dashboard
A static, dependency-free HTML/JS dashboard reads `state.json` and renders
the DAG in real time. No Node, no Docker. Open `web/dashboard.html` in any
browser.

### 7. File-Locked State
Multiple orchestrator processes can now safely share state via
`flock`-style file locks. v1's "I edited the plan in two terminals" race
condition is now impossible.

### 8. Bats Test Suite + CI
`tests/*.bats` cover locking, schemas, sub-plans, retry, validation, gate
plumbing. CI runs on every push via `.github/workflows/ci.yml`.

### 9. Migration Tool
`./skill.sh migrate` converts v1 `plan.json` + `state.json` into v2
shape, with a dry-run by default and a one-shot `--commit` flag.

### 10. MCP (Optional)
`scripts/mcp_server.py` is a tiny Flask server that exposes agents as
**Model Context Protocol** tools, so any MCP-aware client (Claude Desktop,
Cursor, Continue, etc.) can call orchestrator-v2 sub-agents directly.

---

## The Things v2 Stole From The World

We looked at every orchestrator we could find — including the GitHub repos
you sent us — and borrowed with attribution:

| From                          | v2 Adopted                                          |
| ----------------------------- | --------------------------------------------------- |
| **SupervisorAI**               | Hierarchical supervision + DAG semantics            |
| **oap-agent-supervisor**       | Plan-as-artifact + sealed logs                      |
| **LibreChat discussion**       | Cost-aware routing + token accounting              |
| **ai-supervisor topic**        | Agent registry + quality scoring                   |
| **supervisor-multi-agent-app** | Sub-plan handoff pattern                            |
| **langgraph-supervisor-py**    | LLM-as-router + tool-based agent calls              |
| **bats-core**                  | Test framework for shell                           |
| **jq / jo / jl**               | JSON manipulation discipline                       |
| **flock**                      | File-based mutex discipline                        |

We did **not** copy:
- Container-centric designs (we ship as a shell, not an image)
- Heavy Python dependencies (v2 runs on `apt-get install jq` + `bash 4+`)
- Implicit magic (every behaviour is declared in JSON, not hard-coded)

---

## The Things v2 Is Willing to Be Wrong About

- **Speed**: shell is slower than native Python for many tasks. We accept
  this because **portability** > speed.
- **Stdlib ergonomics**: shell is hostile to advanced data structures.
  We mitigate with `jq` everywhere — and accept the rest.
- **Concurrency granularity**: `flock`-style whole-file locks are coarse.
  v2 mitigates with **sharded plans** (a future v2.1 feature).

---

## What "Production Ready" Means in v2

| Property                | v1         | v2                                            |
| ----------------------- | ---------- | --------------------------------------------- |
| Schema-validated plans  | ❌         | ✅                                            |
| Concurrent-safe         | ❌         | ✅ (file-locked)                              |
| Hierarchical            | ❌         | ✅ (sub-plan recursion, any depth)            |
| Retry-aware             | ❌         | ✅                                            |
| Quality-gated           | weak       | ✅ (multi-gate, pluggable)                    |
| Test-covered            | ❌         | ✅ (bats + CI)                                 |
| Observable              | ❌         | ✅ (logs + dashboard + metrics)               |
| Extendable              | fork + patch | ✅ (drop a JSON)                            |
| Documented              | partial    | ✅ (README + META + WORKFLOW + AGENTS + 100-use-cases) |

---

## A Small Demonstration

```bash
$ ./skill.sh plan new hello --from-template research@v1
plan/0x9a…/plan.json signed by user@example.com

$ ./skill.sh plan show
{
  "id": "0x9a4c…",
  "tasks": [
    {"id":"T-01","agent":"researcher.web","input":{"q":"…"}, "depends_on":[]},
    {"id":"T-02","agent":"writer.docs","input":{…},   "depends_on":["T-01"]},
    {"id":"T-03","agent":"qa.lint","input":{…},       "depends_on":["T-02"]}
  ]
}

$ ./skill.sh plan run --workers 4
2026-07-02T22:10:11Z  T-01  → researcher.web      ok    8.1s  cost=0.012$
2026-07-02T22:10:13Z  T-02  → writer.docs         ok    6.0s  cost=0.014$
2026-07-02T22:10:14Z  T-03  → qa.lint             ok    1.1s  cost=0.000$
plan 0x9a4c…  DONE   3/3   wall=8.1s   total_cost=0.026$

$ open web/dashboard.html
# DAG renders green; metrics live.
```

---

## The Invitation

v1 made multi-agent work **possible**.
v2 makes it **engineered**.

If you liked v1 for being simple — we promise v2 is still simple at the
outside. If you stuck with v1 because you outgrew it — v2 grew with you.

Read `README.md` to install, `WORKFLOW.md` to operate, `AGENTS.md` to
extend, `100_USE_CASES.md` to be inspired, and `MIGRATION.md` to upgrade.

### v3 — The Plan

A deep-research pass identified **30 concrete next-step improvements**
spanning the 30 shippable sub-agents, the supervisor, and the joint
layer. The v2.2 / v3.0 targets:

1. **#1 Prompt-Cache Awareness** — immediate 60 %+ cost drop.
2. **#20 WebSocket / SSE Dashboard** — visible win.
3. **#13 Learning Router** — the brain change every user notices.
4. **#19 MicroVM Sandbox** — production-grade trust-builder.
5. **#30 Agent-Bus Pub/Sub** — the cross-cutting capability.

All 30 are detailed with pseudocode and impact estimates in
[`IMPROVEMENTS_30.md`](IMPROVEMENTS_30.md).

**Welcome to Orchestrator v2.**

— The team
