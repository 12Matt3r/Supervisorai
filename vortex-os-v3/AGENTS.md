# Agent Contracts — Orchestrator Skill v2.0

This document is the **canonical reference** for every sub-agent that can be invoked
by `skill v2`. Every agent is a pure-function-style worker that takes a JSON
task descriptor and returns a sealed result envelope — nothing more, nothing less.

If you are writing a new sub-agent, copy the **Agent Output Contract** section
below verbatim — agents that violate the contract are quarantined by the
Quality Gate (`lib/quality.sh`) and their result is rejected.

---

## 1. Agent Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Sub-Agent Lifecycle                              │
├──────────────────────────────────────────────────────────────────────────┤
│  1. Dispatch    — Orchestrator invokes sub-agent with task descriptor    │
│  2. Idempotent  — Sub-agent produces deterministic output for same input │
│  3. Sealed      — Sub-agent writes result into locked envelope file     │
│  4. Validated   — Quality Gate (`quality.sh`) validates the envelope     │
│  5. Aggregated  — Sub-agent output is merged into plan state            │
│  6. Audit       — Sub-agent call is appended to log/audit.jsonl         │
└──────────────────────────────────────────────────────────────────────────┘
```

Every sub-agent MUST be:
- **Idempotent** — re-running with the same input produces the same output.
- **Resource-tagged** — declares `cpu`, `mem`, `net` requirements.
- **Result-sealed** — never writes outside its declared `writes[]` paths.
- **Log-honest** — never lies in its `metrics`.

---

## 2. Agent Discovery

```bash
# List all registered sub-agents
./skill.sh agents list

# Inspect a single sub-agent
./skill.sh agents inspect builder.coder

# Validate a custom sub-agent definition
./skill.sh agents validate ./my-agent.json
```

Discovery sources (in priority order):
1. `$ORCH_PROJECT/agents/*.json`          ← project-local agents
2. `$ORCH_HOME/agents/*.json`             ← user-wide agents
3. `/etc/orchestrator/agents/*.json`      ← system-wide agents
4. Built-in agents (defined in `lib/agents_builtin.sh`)

---

## 3. Agent Definition Schema

```json
{
  "type": "object",
  "required": ["name", "version", "kind", "entry", "input_schema", "output_schema", "writes", "reads"],
  "properties": {
    "name":         { "type": "string", "pattern": "^[a-z][a-z0-9_.]{2,63}$" },
    "version":      { "type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$" },
    "kind":         { "enum": ["shell", "python", "node", "http", "subplan"] },
    "description":  { "type": "string" },
    "entry":        { "type": "string", "description": "Path or URL relative to skill root, or full URL for http kind." },
    "input_schema": { "type": "object" },
    "output_schema":{ "type": "object" },
    "writes":       { "type": "array", "items": { "type": "string" } },
    "reads":        { "type": "array", "items": { "type": "string" } },
    "resources": {
      "type": "object",
      "properties": {
        "cpu": { "enum": ["low", "med", "high"] },
        "mem": { "type": "string" },
        "net": { "enum": ["offline", "on-demand", "online"] },
        "gpu": { "type": "integer", "default": 0 }
      }
    },
    "retry_policy": {
      "type": "object",
      "properties": {
        "max_attempts": { "type": "integer", "minimum": 0, "default": 0 },
        "backoff":      { "enum": ["none", "linear", "exponential"], "default": "exponential" },
        "on":           { "type": "array", "items": { "enum": ["timeout", "transient_io", "validation_fail", "infra"] }, "default": ["transient_io"] }
      }
    },
    "quality": {
      "type": "object",
      "properties": {
        "min_score":    { "type": "number", "minimum": 0, "maximum": 1, "default": 0.6 },
        "max_latency_s":{ "type": "integer", "minimum": 1, "default": 900 },
        "max_cost_usd": { "type": "number", "minimum": 0, "default": 1.0 },
        "must_have":    { "type": "array", "items": { "type": "string" } },
        "must_not":     { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

---

## 4. The Six Built-In Agent Kinds

### 4.1 `shell` — Executes a script in the project sandbox

```json
{
  "name": "shell.format",
  "kind": "shell",
  "entry": "shfmt -i 2 -w {{file}}",
  "writes": ["{{file}}"],
  "reads":  ["{{file}}"],
  "retry_policy": { "max_attempts": 1 }
}
```

The orchestrator substitutes `{{var}}` placeholders from the task descriptor
before invocation.

### 4.2 `python` — Runs a Python function from a module

```json
{
  "name": "python.static_analysis",
  "kind": "python",
  "entry": "pkg.agents.static_analysis:run",
  "resources": { "cpu": "high", "mem": "2Gi", "net": "offline" }
}
```

### 4.3 `node` — Runs a Node module

```json
{ "name": "node.lint_eslint", "kind": "node", "entry": "agents/eslint.mjs:lint" }
```

### 4.4 `http` — Calls a remote HTTP service (e.g. an LLM endpoint, MCP server)

```json
{
  "name": "http.llm_coder",
  "kind": "http",
  "entry": "http://mcp.local:7474/tools/coder",
  "resources": { "cpu": "low", "net": "online" },
  "retry_policy": { "max_attempts": 3, "backoff": "exponential" }
}
```

### 4.5 `subplan` — Delegates the task to another orchestrator plan

```json
{
  "name": "subplan.research",
  "kind": "subplan",
  "entry": "templates/research.json",
  "writes": ["subplans/research-{{task.id}}/"],
  "retry_policy": { "max_attempts": 0 }
}
```

Sub-plan tasks can be **hierarchically nested** to arbitrary depth. The parent
plan's task is marked `waiting_subplan` until the child finishes.

---

## 5. Standard Agent Roster (built-in)

Every v2.0 install ships with **30 production-ready agents** in
`agents/`, spanning 9 families (coding, review, QA, security, devops,
data, research, writing, product/support). The 15 below are the legacy
core that v2.0 originally introduced; the **full 30-agent catalog** with
input/output schemas, quality gates, and the *"Question → agent"*
decision tree lives in
[`agents/AGENTS_CATALOG.md`](agents/AGENTS_CATALOG.md).

You may override any built-in agent by placing a same-named `*.json`
file in `$ORCH_PROJECT/agents/` — the orchestrator picks up the
project-local copy with a warning logged to `audit.jsonl`.

| Agent                    | Kind   | Purpose                                            |
| ------------------------ | ------ | -------------------------------------------------- |
| `builder.coder`          | subplan| Coder-subagent: implements a coding task           |
| `builder.tester`         | subplan| Tester-subagent: writes + runs tests               |
| `builder.reviewer`       | subplan| Reviewer-subagent: reviews a PR/branch             |
| `planner.decomposer`     | http   | LLM that decomposes a goal into a task DAG         |
| `researcher.web`         | http   | Web search + extract agent                         |
| `researcher.local`       | shell  | Runs grep/ripgrep over local sources               |
| `writer.docs`            | http   | LLM that produces documentation from code           |
| `writer.commit_msg`      | http   | LLM that summarizes git diff into commit message   |
| `qa.lint`                | shell  | Runs linter and reports violations                  |
| `qa.test`                | shell  | Runs test suite, collects results                  |
| `qa.security`            | shell  | Runs bandit/trivy, checks dependencies             |
| `ops.deploy`             | subplan| Deploys the project to staging                     |
| `ops.rollback`           | subplan| Rolls back to last green tag                       |
| `meta.summarizer`        | http   | Summarizes long agent output                       |

---

## 6. Agent Output Contract (THE contract)

Every agent MUST emit this JSON envelope to stdout (or to the file pointed to
by the orchestrator's `--out` flag):

```json
{
  "$schema": "https://orchestrator.local/schemas/agent_result.v2.json",
  "version": "2.0",
  "agent":  { "name": "builder.coder", "version": "1.4.2" },
  "task":   { "id": "T-0017", "attempt": 1 },
  "ok":     true,
  "started_at":  "2026-07-02T22:10:11Z",
  "ended_at":    "2026-07-02T22:10:33Z",
  "duration_ms": 22000,
  "outputs": {
    "files_created":   ["src/foo.py"],
    "files_modified":  ["src/bar.py"],
    "stdout_excerpt":  "...truncated 4KB..."
  },
  "metrics": {
    "tokens_in":  812,
    "tokens_out": 1940,
    "cost_usd":   0.0027,
    "retries":    0,
    "peak_rss_mb":312
  },
  "quality": { "score": 0.92, "passed_gates": ["lint", "test", "schema"] },
  "secrets_referenced": [],
  "errors": []
}
```

If `ok=false`, the orchestrator may invoke the agent's `retry_policy`.

The Quality Gate (`lib/quality.sh`) **rejects** the result if any of the
following are true:

| Check              | Failure condition                                                |
| ------------------ | ---------------------------------------------------------------- |
| `schema`           | Output does not parse against `agent_result.v2.json`             |
| `writes_allowed`   | Agent wrote outside its declared `writes[]` → quarantined        |
| `min_score`        | `quality.score < min_score`                                       |
| `latency_budget`   | `duration_ms > quality.max_latency_s*1000`                        |
| `cost_budget`      | `metrics.cost_usd > quality.max_cost_usd`                         |
| `must_have`        | Any required string missing from `outputs.stdout_excerpt`         |
| `must_not`         | Forbidden string present in `outputs.stdout_excerpt`              |
| `secrets`          | `secrets_referenced` contains secret refs the env lacks approval for |

---

## 7. Authoring a Custom Agent

```bash
./skill.sh agents new my.cool_agent --kind shell --entry 'scripts/cool.sh'
# writes project/agents/my.cool_agent.json

$EDITOR project/agents/my.cool_agent.json
# edit writes, reads, retry_policy, quality

./skill.sh agents inspect my.cool_agent
./skill.sh agents dry-run my.cool_agent --task T-demo
```

The new agent is automatically discovered on next `./skill.sh` invocation.

---

## 8. Hierarchical Sub-Plans

A task of `kind: "subplan"` causes the orchestrator to spawn a **child plan**,
which runs concurrently (or sequentially, depending on its own `mode`) and
emits a result back to the parent.

```text
parent plan:  T-01 ──► T-02 ──► T-03 ──► T-04 ──► done
                              │
                              └─► (subplan: research)
                                    plan: R-01 ──► R-02 ──► R-03 ──► done
```

The parent task is blocked until the child plan emits its result envelope.
Cycles are forbidden and detected by `lib/graph.sh:detect_cycles`.

---

## 9. Retry & Backoff

Every task inherits its agent's `retry_policy`. The orchestrator
(`lib/retry.sh`) implements:

- **none**       — no retry; first failure is final
- **linear**     — sleep N × attempt seconds
- **exponential**— sleep N × 2ⁿ seconds with jitter

Default policy:
```json
{ "max_attempts": 3, "backoff": "exponential", "on": ["transient_io", "timeout"] }
```

The retry loop is bounded by `config.json → global.limits.max_total_runtime_s`.

---

## 10. Versioning & Compatibility

The agent envelope includes `version: "2.0"`. If your agent emits a different
version, the orchestrator will warn and refuse to load unless
`config.agents.allow_version_mismatch=true` is set.

Sub-agents are versioned independently from the skill — an agent version
`2.3.7` may be loaded by `skill` versions `>= 2.0.0`. Compatibility is
declared in the agent's `compatibility` field.

## 10.5 Versioning Policy

When an agent definition changes, three strategies are available:

| Strategy | When to use | Behavior |
|---|---|---|
| **Immutable** (default) | Bug fixes, security patches | Old versions stay; new dispatches use the new version. Existing plans keep working. |
| **Side-by-side** | Breaking changes | Old and new coexist as `name@v1.2.3` and `name@v2.0.0`. Plans pin to a specific version. |
| **Floating tag** | Internal/cached agents only | `latest` always wins. Never use in production. |

Set via the agent's `versioning` field:

```json
{
  "name": "coder.python",
  "version": "1.4.2",
  "versioning": {
    "strategy": "side_by_side",
    "eol": "2027-01-01"
  }
}
```

**Deprecation workflow:**

1. Set `deprecated: true` and provide a `deprecation_message`.
2. The agent is hidden from `--agents-discover` by default.
3. Use `--include-deprecated` to reveal it; existing plans keep working.
4. When the `eol` date passes, the agent should be moved to `agents/orphaned/`.

```json
{
  "name": "legacy.coder.python2",
  "version": "1.0.0",
  "deprecated": true,
  "deprecation_message": "Replaced by coder.python v2+. Migration guide: docs/migrations/coder-python-v2.md"
}
```

---

## 10.6 Composition Graph

Agents can declare `composes_with` to express that they delegate work to or
chain with other agents. The orchestrator renders this as a DAG:

```json
{
  "name": "researcher.web",
  "composes_with": ["writer.docs"]
}
```

Inspect with:

```bash
skill --agents-graph --format ascii
skill --agents-graph --format dot | dot -Tpng > agents.png
```

Use this to:
- Visualize the dependency structure of the 30-agent catalog
- Detect composition cycles (forbidden; detected by `lib/graph.sh`)
- Plan rollouts (changing a high-fan-in agent affects many consumers)

---

## 11. Example: Full `builder.coder` Agent Definition

```json
{
  "name": "builder.coder",
  "version": "1.4.2",
  "kind": "subplan",
  "description": "Implements a coding task end-to-end",
  "entry": "templates/coder.json",
  "input_schema": {
    "type": "object",
    "required": ["task_description", "target_files"],
    "properties": {
      "task_description": { "type": "string" },
      "target_files":     { "type": "array", "items": { "type": "string" } },
      "context":          { "type": "object" }
    }
  },
  "output_schema": {
    "type": "object",
    "required": ["diff", "summary"],
    "properties": {
      "diff":    { "type": "string" },
      "summary": { "type": "string" }
    }
  },
  "writes": ["{{target_files}}", "patch.diff"],
  "reads":  ["{{target_files}}"],
  "resources": { "cpu": "high", "mem": "4Gi", "net": "on-demand", "gpu": 0 },
  "retry_policy": { "max_attempts": 2, "backoff": "exponential", "on": ["transient_io", "timeout"] },
  "quality": {
    "min_score": 0.7,
    "max_latency_s": 1800,
    "max_cost_usd": 1.5,
    "must_have": ["diff", "summary"],
    "must_not":  ["TODO:", "FIXME:"]
  }
}
```

---

## 12. TL;DR for Sub-Agent Authors

1. Pick a **name** that describes what you do, not who you are.
2. Declare **exactly** what you read and what you write.
3. Honor `retry_policy` — don't loop internally; let the orchestrator drive.
4. Always emit the **standard result envelope** or you will be quarantined.
5. Make yourself **idempotent** so the orchestrator can safely retry you.
6. Tag your **resources** truthfully so the orchestrator can schedule fairly.
7. Keep your **quality gates** honest; the orchestrator trusts your score.
8. Never **leak secrets** to logs; declare them in `secrets_referenced`.

If you follow these eight rules, you are a well-behaved orchestrator v2.0
sub-agent.
