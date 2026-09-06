# 30 Improvements — A Deep Research Pass on v2.0

> *What a hypothetical "deep research skill" surfaced when asked to find
> the next 30 ways to evolve the Orchestrator Skill and its 30 sub-agents.*
>
> Sources synthesised: NVIDIA extreme co-design for cache-aware agents
> ([NVIDIA, 2026](https://developer.nvidia.com/blog/building-for-the-rising-complexity-of-agentic-systems-with-extreme-co-design/)),
> the 5-pattern memory taxonomy from [Atlan, 2026](https://atlan.com/know/agent-memory-architectures/),
> AWS Strands A2A + MCP convention
> ([AWS Dev.to](https://dev.to/aws/dev-track-spotlight-building-scalable-self-orchestrating-ai-workflows-with-a2a-and-mcp-dev415-5bkn)),
> OpenTelemetry GenAI standard ([Digital Applied, 2026](https://www.digitalapplied.com/blog/ai-agent-observability-2026-tracing-monitoring-stack-guide)),
> Confident AI's online-evaluation playbook
> ([Confident AI, 2026](https://www.confident-ai.com/knowledge-base/compare/best-ai-agent-observability-tools-2026)),
> Skilldex-style portable skill registries
> ([arXiv 2604.16911](https://arxiv.org/html/2604.16911v1)),
> and skill-marketplace research
> ([Chris Ayers](https://chris-ayers.com/posts/agent-skills-plugins-marketplace/),
>  [Portkey](https://portkey.ai/blog/skills-registry/),
>  [Claude Code](https://code.claude.com/docs/en/plugin-marketplaces),
>  [Skill Provenance](https://github.com/snapsynapse/skill-provenance)).

The improvements are grouped into **three tiers** — *Sub-Agent* (the 30
shippable agents), *Supervisor* (the orchestrator itself), and *Joint*
(both layers together). Every idea ships with a tiny proof-of-concept
pseudocode block you can lift.

---

## Tier I — Sub-Agent improvements (1–12)

### 1. **Prompt-Cache Awareness** (token-cost ↓ 60–90 %)

Sub-agents of `kind: "http"` should declare a stable
`cache_prefix` (`"system: <agent-name> <version> persona X"`). The
orchestrator prepends this prefix verbatim across turns so downstream
LLM providers hit their cache. Output envelope adds
`metrics.cache_hit_ratio`.

```yaml
input_schema: ...
resources: { net: online }
quality: { cache_aware: true }
# orchestrator runs: invocation → provider returns cached ↻ / fresh tokens
```

### 2. **Five-Tier Memory Plumbing**

Today's agents are amnesiacs. Add five memory tiers per agent:

| Tier           | Lifetime              | Store                           |
| -------------- | --------------------- | ------------------------------- |
| `working`      | one invocation        | stdin/stdout                    |
| `episodic`     | one plan              | `.orchestr8r/memory/episodic.jsonl` |
| `semantic`     | cross-plan, this repo | `.orchestr8r/memory/semantic.md` (RAG-indexed) |
| `procedural`   | global, this user     | `$ORCH_HOME/memory/procedural/` |
| `shared`       | inter-agent bus       | pub/sub topic per agent family  |

Agent definition gains `memory: { tiers: [...], allow_eviction: [...] }`.

### 3. **Streaming Chunk Emission**

Web agents (e.g. `writer.docs`, `support.triager`) should stream
intermediate deltas via NDJSON to `$OUT.chunks.ndjson` so a dashboard
can render progress. Envelope final answer is unchanged. Reduces
*perceived* latency from 30s → 1.5s.

### 4. **Shared Tool Registry (MCP Tools)**

Move "tools" out of agents and into a shared MCP tools manifest
`tools/registry.json`. Each agent declares `tools_required: ["web_search","fs_read"]`,
and the orchestrator fetches tool definitions once from the registry
rather than per-agent.

```json
{
  "tools": {
    "web_search": { "provider": "tavily", "cost_per_call": 0.005 },
    "fs_read":    { "provider": "local", "sandbox": "microvm" }
  }
}
```

### 5. **Capability Negotiation Handshake**

Before dispatch, the orchestrator broadcasts `GET /capabilities` and the
agent replies with `{ tools, constraints, est_latency, est_cost }`. The
supervisor routes on this answer, eliminating "I tried, but I can't".

### 6. **Persona & Voice Pins**

Ship agents with a stable `persona` (the *system prompt* the agent
*belongs* to). Examples shipped: `staff_engineer`, `security_auditor`,
`librarian`, `sre`. Personas are reusable across agents — change one
persona file, update six agents at once.

### 7. **Per-Agent Eval Suite (offline + online)**

Every agent ships with a `evals/` directory:

```text
agents/coder.python/evals/
  ↳ gold_set.json           # 50 known (input, expected_diff_features)
  ↳ oracle_judge.py         # program-of-thought grade
  ↳ online_traces.jsonl     # real captured runs
```

The orchestrator runs the eval suite on every upgrade and reports
`metrics.eval_score`. Online evals record `↻ regen vs latest`.

### 8. **Adapter Pattern (one agent, many LLM providers)**

Replace the implicit "OpenAI" assumption with an `llm_adapter`:

```yaml
adapter: { name: claude_3_5_sonnet, fallback: gpt_4o_mini, retry_chain: [opus, sonnet, haiku] }
```

The orchestrator auto-promotes `haiku` for `must_have[-]: latency`
budgets, `opus` for `must_have[-]: depth`.

### 9. **Cooperative Cancellation Token**

Long-running agents (e.g. `ops.incident_responder`,
`researcher.academic`) accept a `cancel_token`:

```bash
touch /tmp/orch_cancel_$TASK_ID
# agent's loop checks the file on each iteration and exits cleanly
```

The dispatch shell wrapper polls the token every second.

### 10. **OpenTelemetry GenAI Span Emission**

Every agent emits OTel GenAI-standard spans
`agent.invoke → llm.prompt → llm.completion → tool.call* → result`.
The orchestrator ships an exporter to console, OTLP, and
`web/tracing.html`. This makes any OTel-aware debugger work for free.

### 11. **Per-Secret Approval Matrix**

Replace "trust the agent's `secrets_referenced`" with a declarative
`secrets_allowlist`:

```json
{ "secret_id": "github_token", "allowed_agents": ["coder.*", "ops.*"],
  "approver": "ci@acme.com", "rotated_every_days": 30 }
```

The orchestrator refuses to dispatch an agent that needs an
unapproved secret.

### 12. **Multi-Modal Agent Kinds**

Add `kind: "image"`, `kind: "audio"`, `kind: "video"` agents:

- `image.dall_e`, `image.sdxl`, `image.vision_understand`
- `audio.whisper_transcribe`, `audio.elevenlabs_tts`
- `video.transcribe`, `video.shot_classify`

Plans can now generate diagrams from text, narrate a research report,
edit a marketing demo.

---

## Tier II — Supervisor improvements (13–24)

### 13. **Learning Router (per-domain feedback loop)**

The supervisor tracks `agent × task_type → success_rate, mean_cost,
mean_latency`. Dispatch uses an ε-greedy bandit: 90 % best-known, 10 %
exploration. Records `learning.jsonl`; weekly *retraining* updates the
routing table. Threshold-aware: if an agent's success rate dips below
`0.6` it gets *quarantined*.

### 14. **Soft-Skills Matrix Routing**

Routes use a 3-axis capability vector:

```
{ capability: float, confidence: float, cost_per_call: float }
```

LLM-as-judge scores each agent per incoming task on these axes; the
supervisor picks the Pareto-optimal agent. Brings the v1
*developer-written* router into the *data-driven* era.

### 15. **Backpressure / Queue-Depth Awareness**

The supervisor now caps in-flight tasks per `kind` and globally:

```yaml
limits: { coder.*: 4, total: 32, queue_depth_warn: 8 }
```

Backpressure propagates to MCP clients (`mcp_server.py` returns
`-32001` "over capacity").

### 16. **Dead-Letter Queue (DLQ) + Auto-Postmortem**

When a task fails after exhausting its `retry_policy`, it lands in
`dlq/<task_id>/` along with the full input/output envelope. A watcher
agent (`dlq.triage`) auto-drafts a postmortem, emits it as a new task
in the plan, suggests code or plan changes.

### 17. **Cross-Plan Context Carryover**

A child sub-plan can declare `inherit_parent_memory: true` to pull
episodic/semantic memory from its parent. Solves the "I researched
this three plans ago" problem.

### 18. **Cost Governor**

Plan-level `cost_governor: { cap_usd: 5.0, warn_pct: 80, abort_pct: 100 }`.
At 80 % the supervisor prepends a cheap-model tier; at 100 % it
gracefully seals with `status: cost_exceeded` and posts a notification.

### 19. **MicroVM Sandbox Per Dispatch**

`kind: "shell"` agents run inside `firecracker`/`cloud-hypervisor`
microVMs (or `gVisor` containers) so a misbehaving agent cannot leak
across projects. Configurable: `sandbox: microvm | docker | host`.

### 20. **Live WebSocket / SSE Dashboard**

Replace the polling HTML with a WebSocket/SSE channel
`/state-events` that pushes diffs. Adds `auth_token: …`,
`last_event_id: …` for replay-on-reconnect.

### 21. **Skill Registry & Marketplace**

A `~/.orchestr8r/registry/` directory and a public
`marketplace.json` index. Auto-update command:

```bash
./skill.sh agents update              # pulls from configured registry
./skill.sh agents install coder.go    # adds a new agent
./skill.sh agents audit               # checks installed vs registry version, signatures
```

Inspired by Skilldex/Claude/Portkey.

### 22. **Conditional Routing (when-then dispatch)**

Plans can now express routing rules:

```json
{ "if": { "task.x": ">= 0.9", "task.y_is_set": true },
  "dispatch": { "agent": "ops.deploy.k8s", "strategy": "bluegreen" } }
```

This is the v1 → v2 bridge from *static* routing to *reactive* routing.

### 23. **Predicted-Cost Pre-Flight**

Before invoking an LLM-backed agent, the supervisor asks the agent
itself to estimate token-out count and cost — using a cheap LLM
("Haiku") — and offers the user a pre-flight confirmation when the
predicted cost exceeds `human_review_threshold_usd`.

### 24. **Mid-Plan Replanning (branch on failure)**

If a critical task fails, the supervisor branches to a fallback
sub-plan rather than aborting:

```yaml
on_fail:
  branch: fallback.json     # a parallel DAG
  max_wait_s: 60
```

The main plan resumes after the branch returns. Inspired by EITOPS /
SupervisorAI 2025/26 patterns.

---

## Tier III — Joint / Supervisor + Agent improvements (25–30)

### 25. **Side-by-Side Shadow Routing**

Run two candidate agents in parallel; only one result is used, the
other is *shadowed* into a comparison log. Triggers auto-eval: winner
vs shadow → `promotion.jsonl`. A/B ship velocity goes from "monthly" to
"hourly".

### 26. **Tool-Call Budget per Task**

`tool_budget: 20` per task. Exceeding it triggers a *graceful abort*
with `diagnostics.tool_call_history` returned. Stops agent loops
without `read-loop-on-PID-1` debugging.

### 27. **Mid-Run Context Compaction**

For long-running plans, every N tasks the supervisor compresses the
`state.json` working memory, stores a *summary.md*, and replaces raw
context with summary pointers. Inspired by Anthropic's context
engineering patterns.

### 28. **Periodic Canary Agent ("canary-doctor")**

A 31st hidden agent whose only job is to dispatch micro-tasks at
random and assert the orchestrator handles them correctly. Reports
`canary.json` daily. If any assertion fails, the supervisor emits
`status: orchestrator_drift`.

### 29. **Confidence-Weighted Vote**

When multiple sub-agents answer the same question (e.g. two reviewers
of the same PR), the supervisor returns a *weighted-vote* answer: each
agent's `confidence` becomes the vote weight. Output envelope includes
`votes` + `consensus_score`. Solves LLM-judge variance.

### 30. **Cross-Agent Broadcast Bus ("agent-bus")**

A topic-based pub/sub:

```
agent-bus topic=new_severity_vuln
  → security.deps publisher
  → pm.prioritizer subscriber
  → writer.commit_msg subscriber (auto-aggregate CVE PRs)
```

A `lib/bus.sh` exposes `agent_bus_pub`, `agent_bus_sub`. The
*$100 question* for any multi-agent system — *how do they
communicate?* — gets a non-LLM answer.

---

## Bonus (not in the 30, free with the others)

- **A — Time-Travel Debug**: deterministic state snapshot per task;
  `--rewind <task_id>` rewinds the plan to that state and re-runs
  from there.
- **B — Tool-Use Privacy Zone**: each tool call is wrapped in a privacy
  envelope (`{ allow: ["src/**"], redact: [".env"] }`); data leaks are
  syntactically impossible.
- **C — Graceful Degradation Matrix**: per-agent fallback chain
  (`haiku → sonnet → opus → escalate_human`); the supervisor never
  *fails* — it *degrades*.
- **D — Spare-Compute Mode**: when the orchestrator's host is under
  load, non-critical agents (`writer.commit`) defer to a quiet queue.

---

## Estimated Impact Summary

| Area                          | Improves                           | Magnitude   |
| ----------------------------- | ---------------------------------- | ----------- |
| 1, 11, 19, 22                 | Security / isolation               | **High**    |
| 13, 14, 23, 25               | Routing / cost                     | **High**    |
| 6, 7, 8, 17, 29              | Reliability / determinism          | **High**    |
| 2, 27                         | Long-running plans                 | **Med**     |
| 3, 10, 20                    | Developer experience               | **High**    |
| 9, 15, 16, 24               | Robustness under load              | **High**    |
| 4, 5, 21, 30                 | Extensibility / ecosystem          | **Med**     |
| 12, 26, 28                   | New use cases                      | **Med**     |
| Bonus A–D                     | Operations                         | **Med**     |

---

## Implementation Recommendation

Tackle in this order, three weeks per item:

1. **#1** prompt-cache awareness — immediate 60 %+ cost drop.
2. **#20** WebSocket/SSE dashboard — visible win, marketing-worthy.
3. **#13** learning router — the brain change every user notices.
4. **#19** microVM sandbox — production-grade trust-builder.
5. **#30** agent-bus pub/sub — the cross-cutting capability other
   frameworks will borrow.

After these five, v2.2 ships. The remaining 25 ship as v2.3 / v2.4 in
quarterly drops.

---

## Acknowledgements

This pass synthesised public writings from Ram Vegiraju, Richmond
Alake (agentic memory), the OpenTelemetry GenAI working group, the
AWS Strands team, NVIDIA's extreme co-design team, Confident AI,
Braintrust, Agenta, Fiddler, Helicone, Galileo, Maxim AI, MLflow,
the Skilldex authors, the Claude Code marketplace team, and Portkey.
