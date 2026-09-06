# The 8 Invariants — Orchestrator Sub-Agent Contract

Every agent that ships with `orchestrator_skill v2` MUST obey these eight
invariants. The Quality Gate (`lib/quality.sh`) enforces them at runtime.
`skill --agents lint` enforces them statically.

> **Why "8" and not "some number"?** The number is intentionally fixed. Each
> invariant corresponds to a specific failure mode observed in multi-agent
> systems: silent retries, scope creep, schema drift, resource starvation,
> secret leakage, and metric fabrication. Adding a 9th would weaken the
> contract's testability.

---

## I1 — Idempotence

Re-running with identical input produces byte-identical output (excluding
non-deterministic fields like `duration_ms` and timestamps).

**Enforced by:** `--agents lint` (I1 marker), Quality Gate replay test
**Failure mode it prevents:** Cached/stale results from retries corrupting
downstream aggregation.

## I2 — Resource Honesty

Declared `resources.{cpu,mem,net,gpu}` matches actual consumption within ±20%.
Peaks are measured at runtime and compared to declared values.

**Enforced by:** `--agents lint` (I2 marker), `lib/observability.sh` post-run
**Failure mode it prevents:** One agent starving siblings by under-declaring
resource needs.

## I3 — Write Containment

Never writes outside its declared `writes[]`. Violation triggers immediate
quarantine and audit log entry.

**Enforced by:** Quality Gate `writes_allowed` check (`AGENTS.md` §6)
**Failure mode it prevents:** Agents stomping on each other's outputs or
escaping into the host filesystem.

## I4 — Read Containment

Never reads outside its declared `reads[]`. Symmetric to I3.

**Enforced by:** Quality Gate `reads_allowed` check (add to `lib/quality.sh`)
**Failure mode it prevents:** Agents exfiltrating data they weren't
authorized to access.

## I5 — Sealed Envelope

Output strictly conforms to `agent_result.v2.json` schema. Required fields:
`name`, `version`, `kind`, `entry`, `input_schema`, `output_schema`,
`writes`, `reads`. Version mismatches require
`config.agents.allow_version_mismatch=true`.

**Enforced by:** `--agents lint`, schema validator
**Failure mode it prevents:** Silent contract drift breaking aggregation.

## I6 — Retry Honesty

Never loops internally. Lets the orchestrator drive `retry_policy`. An agent
that retries itself bypasses the Quality Gate and audit trail.

**Enforced by:** Runtime timeout vs declared `max_latency_s` ratio check
**Failure mode it prevents:** Agents with hidden retry loops appearing
"successful" while burning quota.

## I7 — Secret Hygiene

No secrets in logs. All referenced secrets declared in `secrets_referenced[]`
with explicit approval codes.

**Enforced by:** Quality Gate `secrets` check
**Failure mode it prevents:** Credential leakage via `stdout_excerpt` or
`metrics` fields.

## I8 — Metric Truthfulness

`metrics.{tokens_in,tokens_out,cost_usd,peak_rss_mb}` are actual measurements,
not estimates. Agents that fabricate metrics corrupt cost reporting and
capacity planning.

**Enforced by:** Cross-check against provider billing API (where available)
**Failure mode it prevents:** Cost budget drift, capacity planning errors.

---

## Declaring Compliance

Every agent JSON should declare compliance:

```json
{
  "name": "coder.python",
  "version": "2.0.0",
  "invariants_compliant": ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]
}
```

Dynamic agents start in **provisional** status:

```json
{
  "invariants_compliant": ["I5"],
  "invariants_provisional": ["I1", "I2", "I3", "I4", "I6", "I7", "I8"]
}
```

Provisional invariants graduate to compliant only after the canary-doctor's
quarantine ticks reach zero AND a successful `--agents lint` run on the
agent.

## Violation Handling

| Violation | Severity | Action |
|---|---|---|
| I1 breach | High | Quarantine, alert, manual review |
| I2 breach | Medium | Quarantine, log, allow one retry |
| I3 breach | **Critical** | Immediate quarantine, halt plan, alert |
| I4 breach | **Critical** | Immediate quarantine, halt plan, alert |
| I5 breach | High | Quarantine, schema migration prompt |
| I6 breach | Medium | Quarantine, expose hidden retry count |
| I7 breach | **Critical** | Immediate quarantine, rotate exposed secret |
| I8 breach | Low | Log, downgrade to "estimated" tag |
