# 100 Use Cases for Orchestrator Skill v2.0

A practical catalog of problems the v2.0 orchestrator solves — straight
out of the box, with no extra plugins beyond `apt-get install jq`. Every
example assumes you have run `./install.sh` (or `./skill.sh init`) and
have a project directory with a `plan.json`.

> *Pick a use case. Then read the corresponding `templates/<name>.json`
> in this repo. Then `./skill.sh plan run`.*

---

## A — Software Engineering (1–25)

1. **Implement a feature end-to-end.** Single coder sub-plan, tests, lint, summarize diff.
2. **Triage a backlog.** Read issues, cluster, prioritize, emit prioritized list.
3. **Refactor a large module.** Read → graph → task per file → coder plan → tests.
4. **Modernize a Java app to Spring Boot 3.** Plan: inventory deps, plan per-module upgrade, tests.
5. **Port 1000-line Python 2 file to Python 3.** Per-function tasks, gates: syntax + unit tests.
6. **Bug-bash** — Auto-bisect failing tests with retries and budget cap.
7. **Auto-write pull-request descriptions** from git diffs.
8. **Generate a CHANGELOG entry** from a range of merged PRs.
9. **Enforce a coding style across a repo** — shell format → lint → auto-commit.
10. **Run dependency upgrades safely** with sub-plans per major version.
11. **Generate OpenAPI spec from runtime traffic** and write to repo.
12. **Maintain a stale `EXAMPLES.md`** — researchers re-fetch and update.
13. **Build a CLI installer** — plan: scaffold → implement → test → package → publish.
14. **Implement a WebSocket endpoint** with TDD sub-plans.
15. **Add i18n to a React app** — coders per locale files + smoke test.
16. **Port legacy C codebase to Rust** — gated by a `cargo test` budget.
17. **Generate SQL migrations** with a sub-plan gated by `EXPLAIN` cost estimate.
18. **Build a Slack bot** step by step — implement → mock server → integration test → deploy.
19. **Auto-create a Sentry alert → PR pipeline** with retry.
20. **Scan a PR for accessibility regressions** (axe-core + sub-plan gates).
21. **Convert a Markdown wiki to Docusaurus** with a per-folder sub-plan.
22. **Add feature flags to a codebase** in lockstep with usage analytics.
23. **Backport a security patch** across branches with parallel sub-plans.
24. **Implement OAuth2 with tests for every code path** — DAG of 14 tasks.
25. **Generate a complete Stripe-style webhook flow** in three repos at once.

## B — Research & Knowledge Work (26–40)

26. **Compile a competitive landscape report** — web search + per-competitor sub-plans.
27. **Summarize a 200-page PDF collection** — extract → cluster → cite.
28. **Build a literature review** with citation graphs.
29. **Track breaking news on a topic** and emit daily briefs.
30. **Onboard a new team member** — produce a personalized reading list.
31. **Cross-reference 3 standards documents** into a single cheatsheet.
32. **Pull latest CVEs for a tech stack** and emit a fix-up plan.
33. **Generate Q&A flashcards** from a textbook chapter.
34. **Run "explain X like I'm five"** across N model personas.
35. **Mine Twitter/X** for sentiment on a brand — hourly for 7 days.
36. **Synthesize a market sizing deck** from public data.
37. **Read a meetup's slide deck** and emit speaker notes.
38. **Diff two open-source licenses** and produce a merge recommendation.
39. **Translate a research paper to a blog post** preserving citations.
40. **Generate a 5-page RFP response** in a brand voice.

## C — Data & Analytics (41–55)

41. **Clean a 10M-row CSV** with parallel chunk sub-plans.
42. **Profile a Postgres schema** and suggest indexes.
43. **Build a dbt project from a CSV archive** with parallel model scaffolding.
44. **Migrate a MySQL DB to PostgreSQL** with per-table sub-plans.
45. **Compare two CDC pipelines and recommend one** — pros/cons paper.
46. **Build a Looker dashboard spec from a SQL repo.**
47. **Audit a Snowflake bill** and emit a cost-cutting plan.
48. **Ingest S3 logs into a SIEM** with rate-limit-aware retries.
49. **Detect data drift** in a streaming pipeline and freeze on anomaly.
50. **Generate synthetic test data** that matches production statistics.
51. **Build a KPI tree** from a glossary of business terms.
52. **Reverse-engineer a Tableau workbook** into a SQL+Python rewrite.
53. **Run a T-test across two A/B buckets** and produce a memo.
54. **De-identify a PHI dataset** for research use, audited.
55. **Detect PII in a data lake** and emit a remediation plan.

## D — DevOps & SRE (56–70)

56. **Run a chaos drill in staging** — sub-plans per chaos experiment.
57. **Migrate from ECS to EKS** — phased, gates: cost delta + SLO preservation.
58. **Generate a Terraform module** from a CloudFormation stack.
59. **Right-size a Kubernetes cluster** with a VPA-style analyzer sub-plan.
60. **Auto-remediate noisy alerts** — investigate → fix → post-mortem.
61. **Build a feature-flag rollout plan** with auto-rollback triggers.
62. **Implement progressive delivery** for a monolith (canary → region → 100%).
63. **Generate runbooks from incident transcripts.**
64. **Provision a dev sandbox from a ticket** in <2 minutes.
65. **Run a TLS certificate sweep** + emit renewal plans.
66. **A/B IaC tools** (Pulumi vs. CDKTF) on the same scenario.
67. **Migrate from Heroku to Fly.io** with zero-downtime.
68. **Wire SLO dashboards from a service inventory.**
69. **Generate a backup verification plan** with daily drill.
70. **Roll out mTLS to a service mesh** — per-namespace sub-plans.

## E — Security & Compliance (71–82)

71. **Quarterly SOC2 control mapping** — pull evidence into folders.
72. **Scan a container image for CVEs** and emit patch plan.
73. **Auto-detect leaked secrets** in a monorepo and open clean-up PRs.
74. **Run a red-team tabletop** and produce a delta-of-defenses memo.
75. **Audit an AWS account** with AWS Config rules packaged as sub-plans.
76. **Generate a SBOM** with depth-N resolution.
77. **Map an app to OWASP ASVS** with per-control remediation sub-plans.
78. **Enforce least-privilege IAM** by analyzing actual usage.
79. **Build a threat model** from architecture diagrams (image → DAG).
80. **Verify GDPR data-subject access requests** end-to-end.
81. **Produce an incident timeline** from logs and chat transcripts.
82. **Quarterly pen-test triangulation** — reconcile findings across tools.

## F — Marketing, Sales, Support (83–92)

83. **Generate 30 SEO blog posts** from a topic list — sub-plan per article.
84. **A/B email subject lines** — research → variants → test plan.
85. **Build a sales battlecard** from competitor research + product capabilities.
86. **Summarize a sales call transcript** and push to CRM.
87. **Generate a customer-success-health score** from ticket + usage data.
88. **Auto-route support tickets** to the right team with a confidence threshold.
89. **Build a 90-day onboarding drip campaign** from product docs.
90. **Generate quarterly board slides** from metrics + narratives.
91. **A/B landing-page copy** with research + draft + review sub-plans.
92. **Competitor pricing monitor** — daily scrape → diff → alert.

## G — Personal & Creative (93–100)

93. **Plan a 14-day trip** — flights + hotels + itinerary as DAG with budgets.
94. **Research a new laptop** — compare 30 models, score, recommend.
95. **Write a 50,000-word novel in chapters** — sub-plan per chapter.
96. **Build a workout plan** that adapts to wearable data weekly.
97. **Compile a tax-prep dossier** — receipts categorized, anomalies flagged.
98. **A weekly meal-plan generator** based on pantry + macros + budget.
99. **Record and summarize weekly meetings** and produce action items.
100. **Map a family tree** from photos + docs into a structured JSON.

---

## Pattern Cheat-Sheet

Across all 100 use cases, three orchestration patterns dominate:

| Pattern | What it looks like in v2.0 |
| --- | --- |
| **Fan-out / Fan-in** | One task fans out to N parallel sub-plans; a final task joins them. |
| **Codegen / Test / Gate** | A coder agent, a tester agent, a quality gate — a per-feature DAG. |
| **Research / Synthesize / Publish** | Multiple researchers, one writer, one QA gate, one deploy. |

All three patterns are **first-class** in v2.0 — see `templates/research.json`,
`templates/release.json`, and `templates/fan_out.json`.

---

## Quick Recipes

### Recipe — "Bug Bash"
```bash
./skill.sh plan new bugbash --from-template bugbash@v1 \
  --var "test_files=tests/" \
  --var "max_runtime_s=3600"

./skill.sh plan run --workers 8
./skill.sh plan report --format md
```

### Recipe — "Customer Onboarding"
```bash
./skill.sh plan new onboard --from-template onboarding@v1 \
  --input @customer_input.json

./skill.sh plan run --watch
```

### Recipe — "Incident Postmortem"
```bash
./skill.sh plan new postmortem --from-template postmortem@v1 \
  --input @incident_summary.json \
  --on-fail emit_to_slack
```

---

## Spotlights — Three Use Cases Walked Through

### Spotlight #32 — Track CVEs for a Tech Stack
**Template:** `templates/cve_watch.json`
**Agents used:** `researcher.web`, `planner.decomposer`, `writer.docs`, `qa.lint`
**Wall time:** ~6 minutes for the first run; daily thereafter.

```bash
./skill.sh plan new cve-watch --from-template cve_watch@v1 \
  --var "stack=python 3.12, fastapi 0.110, sqlalchemy 2.0"

./skill.sh plan run --workers 4
./skill.sh plan report --since 24h
```

The plan watches NVD/OSV, filters by severity, opens fix-up drafts in your
issue tracker, and posts a digest to Slack.

### Spotlight #61 — Feature Flag Rollout
**Template:** `templates/flag_rollout.json`
**Agents used:** `planner.decomposer`, `ops.deploy`, `qa.lint`, `meta.summarizer`

The plan creates the flag in your feature-flag service, deploys the
guarded build, watches the burn-rate, ramps traffic, and (if SLOs hold)
removes the flag.

### Spotlight #93 — Plan a 14-day Trip
**Template:** `templates/trip.json`
**Agents used:** `flights_search`, `hotels_search`, `writer.docs`, `qa.lint`

Inputs: city, dates, budget, interests.
Outputs: a Markdown itinerary with cost breakdown, hotel links, flight
links, and a per-day map.

---

## How to Choose a Template

```
┌─────────────────────────────────────────────────────────────────┐
│ Decision Tree                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Do you write code? ──────────► templates/coder.json             │
│ Do you write docs? ──────────► templates/writer.json            │
│ Do you ship releases? ───────► templates/release.json           │
│ Do you run research? ────────► templates/research.json          │
│ Do you fan out work? ────────► templates/fan_out.json           │
│ Do you chain sub-plans? ─────► templates/meta_plan.json         │
│ None match? ────────────────► build your own with `plan new`    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Extending the Catalog

Each use case above is implemented as a **template** in `templates/`.
To add your own:
1. Copy `templates/_skeleton.json` → `templates/my_usecase.json`.
2. Validate with `./skill.sh plan validate ./templates/my_usecase.json`.
3. Open a PR — describe the use case, the agents it uses, and a sample run.

The catalog grows with every project that adopts v2.0.

---

## Closing Thought

A hundred use cases is not an end — it's a **snapshot**. The point of v2
isn't to be an app store; it's to be a **substrate** that turns your
problem into a DAG and your DAG into an outcome.

If you find a 101st use case, please open a PR.

*— The Orchestrator v2 maintainers*
