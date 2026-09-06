# Sub-Agent Catalog — Orchestrator Skill v2.0

This directory contains **30 production-ready sub-agents** for the
Orchestrator Skill v2.0. Each `*.json` file is a complete, validated agent
definition following the schema in `../AGENTS.md`. Drop them into any
project, run `./skill.sh plan validate <plan>`, and the orchestrator
auto-discovers them on first invocation.

---

## How to Use

```bash
# Validate a single agent
./skill.sh agents validate agents/coder.python.json

# Validate all agents
./skill.sh agents validate --all

# Use in a plan
./skill.sh plan new my-feature --task "Implement foo()" \
  --agent coder.python --input '{...}'

# Discover at runtime
./skill.sh agents list
```

Drop a custom agent into this directory and it's picked up automatically —
no edits to `skill.sh`, no restarts.

---

## The Catalog (30 agents)

### Coding & Build (5)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `coder.python` | subplan | Python 3.12, ruff/black style, pytest tests, must_not=print/TODO |
| `coder.typescript` | subplan | React/Node/Vue/Nest, strict TS, ESLint + Vitest |
| `coder.rust` | subplan | Edition 2021/2024, clippy + cargo test gated, no_unsafe mode |
| `coder.sql_migrations` | subplan | PostgreSQL/MySQL/SQLite reversible migrations, EXPLAIN gating |
| `coder.terraform` | subplan | AWS/GCP/Azure/OpenTofu, tflint + tfsec, cost-delta-aware |

### Review & QA (5)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `reviewer.code` | http | PR-quality comments, complexity report, default/strict/minimal rulesets |
| `reviewer.architecture` | http | ADR-style verdicts, STRIDE-aware, scope at service/system/org |
| `reviewer.dependencies` | subplan | License compatibility (GPL/AGPL/SSPL), abandonment risk |
| `tester.mutation` | shell | Mutmut / Stryker / mull, target_kill_pct, writes survivor tests |
| `qa.flakehunter` | shell | 50-iteration replay, auto-quarantine, common-pattern detection |

### Security (4)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `security.sast` | shell | Semgrep/CodeQL/Bandit, SARIF output, OWASP Top 10 / CWE Top 25 |
| `security.deps` | shell | OSV/NVD/GHSA, prioritized patch plan, auto-open PRs |
| `security.secrets` | shell | Gitleaks/Trufflehog with entropy heuristics, history scan |
| `security.threatmodel` | http | STRIDE/DREAD/LINDDUN, asset focus, risk score |

### DevOps & Ops (4)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `ops.deploy.k8s` | subplan | GitOps/ArgoCD, canary/blue-green, auto-rollback on SLO breach |
| `ops.deploy.serverless` | subplan | AWS Lambda/GCP Functions/Azure, alias traffic shifts |
| `ops.incident_responder` | subplan | Pulls logs/metrics/traces, drafts war-room updates, opens tasks |
| `ops.cost_optimizer` | subplan | Idle resources, oversized DBs, $savings plan, CO2e impact |

### Data (4)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `data.profiler` | shell | Type inference, drift score, PII tagging, baseline comparison |
| `data.cleaner` | shell | Dedup, normalize, impute, redact, great_expectations-style recipe |
| `data.synthesizer` | subplan | FK-aware synthetic rows, drift vs source, PII-safe mode |
| `data.lineage` | http | Column-level, OpenLineage/Datahub output, broken-chain detection |

### Research & Knowledge (3)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `researcher.web` | http | Search + extract + dedup + summarize, domain allowlist |
| `researcher.codebase` | shell | ripgrep + LSIF-aware, semantic search, call-graph snippet |
| `researcher.academic` | http | arXiv / Semantic Scholar / OpenAlex / PubMed, citation graph |

### Writing & Comms (3)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `writer.docs` | http | MDX/MD/RST, voice rules, Mermaid diagrams, reading time |
| `writer.commit` | http | Conventional Commits, commitlint-aware, ticket prefix |
| `writer.slack_digest` | subplan | Daily digest, themes/blockers/decisions/actions |

### Product & Support (2)
| Agent | Kind | Highlights |
| --- | --- | --- |
| `pm.prioritizer` | http | RICE/WSJF/ICE/Kano, capacity-aware sprints, dedupes duplicates |
| `support.triager` | http | Category/severity/team routing, KB hits, SLA-aware auto-reply |

---

## Discovery Precedence

When the orchestrator looks up an agent by name, it searches in this order:

1. `$ORCH_PROJECT/agents/`         ← project-local (your team's custom agents)
2. `skill_v2/agents/`              ← this directory (built-in catalog)
3. `$ORCH_HOME/agents/`            ← user-wide
4. `/etc/orchestrator/agents/`     ← system-wide

Same name in multiple sources → **higher-precedence wins**, with a warning
logged to `audit.jsonl`.

---

## Conventions Held By All 30

Every agent in this catalog obeys these invariants:

1. **`name`** follows `^[a-z][a-z0-9_.]{2,63}$` and uses dotted family.role form.
2. **`version`** follows SemVer `MAJOR.MINOR.PATCH`.
3. **`kind`** is one of `shell`, `python`, `node`, `http`, `subplan`.
4. **`writes`** is non-empty and never overlaps another agent's exclusive areas.
5. **`input_schema` / `output_schema`** are proper JSON Schemas.
6. **`resources`** declares truthful CPU/memory/network requirements.
7. **`retry_policy`** is bounded by `config.json → global.limits.max_total_runtime_s`.
8. **`quality`** declares `min_score`, `max_latency_s`, `max_cost_usd`, `must_have[]`, `must_not[]`.
9. **Idempotent** — same input ⇒ same output, retries are safe.
10. **Secrets-aware** — never logs raw secrets; declares them in `secrets_referenced`.

---

## Adding Your Own Agent

```bash
# Scaffold a new agent definition
./skill.sh agents new my.cool_agent \
  --kind shell \
  --entry 'scripts/cool_runner.sh' \
  --writes 'output/**' \
  --reads 'input/**'

# Validate before using in a plan
./skill.sh agents validate agents/my.cool_agent.json

# Inspect existing built-ins
./skill.sh agents inspect security.deps
```

The new agent is automatically picked up on the next orchestrator run.

---

## When to Reach For Which Sub-Agent

```
┌──────────────────────────────────────────────────────────────────┐
│ Question                     │ Reach for                         │
├──────────────────────────────┼───────────────────────────────────┤
│ "Implement this in Python"   │ coder.python                      │
│ "Review this PR"             │ reviewer.code                     │
│ "Anything secret in here?"   │ security.secrets                  │
│ "Why is prod down?"          │ ops.incident_responder            │
│ "What's the CVE list?"       │ security.deps                     │
│ "Profile this CSV"           │ data.profiler                     │
│ "Synthesize 10k users"       │ data.synthesizer                  │
│ "Search arXiv"               │ researcher.academic               │
│ "Where is X used in repo?"   │ researcher.codebase               │
│ "Triage these tickets"       │ support.triager                   │
│ "Prioritize the backlog"     │ pm.prioritizer                    │
│ "Cut AWS bill 20%"           │ ops.cost_optimizer                │
│ "Find flaky tests"           │ qa.flakehunter                    │
│ "Write a commit message"     │ writer.commit                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Maintenance

To regenerate this README (e.g., when agents are added/removed):

```bash
./skill.sh agents catalog --out agents/AGENTS_CATALOG.md
```

The catalog is **always** derived from the actual `*.json` files — no
manual sync needed.

---

## Status Legend

| Icon | Meaning                                |
| ---- | -------------------------------------- |
| ✅    | Production-tested, wired into a sample |
| 🆕    | Added in this catalog, samples pending |
| ⚠️    | Requires specific infra (e.g. `http` kind needs MCP server running) |

All 30 agents in this directory are ✅ or ⚠️ — no 🆕 partial-impls.
