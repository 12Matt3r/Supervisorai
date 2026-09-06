# Configuration & deployment

One page that maps **every** setting SupervisorAI reads, where it lives, what it
affects, and how to go from a dry run to a fully authenticated live deployment.

> **The #1 footgun:** with no key (or a wrong `GMI_BASE_URL`) the client falls
> back to a **deterministic offline mock** so demos never crash. That's great for
> a hackathon and dangerous in production — you can run *silently on the mock*.
> **Run the preflight check below; it fails loudly instead of mocking silently.**

## One command to validate everything

```bash
python scripts/preflight.py            # validates env + JSON configs, pings M3
python scripts/preflight.py --require-live   # exit non-zero unless LIVE M3 (use in CI)
```

It checks your environment, parses every JSON config, and sends **one tiny real
request** to confirm the key authenticates and you're on live MiniMax-M3 — not
the mock. Exit `0` = ready, `1` = misconfigured, `2` = on the mock while
`--require-live`.

## The configuration matrix

There is **one source of truth for the model, endpoint, and key: the environment
(`.env` or exported vars)** — read by *both* the Python engine and the bash
implementations. There is **no** `llm_config.json`, and nothing is duplicated
across files, so there are no precedence conflicts to reason about.

| Setting | Where it's set | Affects | Default | Notes |
|---|---|---|---|---|
| `GMI_API_KEY` | `.env` / env var | Python engine **+** bash skills | *(none)* | **Unset ⇒ offline mock.** Never commit it. |
| `GMI_BASE_URL` | `.env` / env var | both | `https://api.gmi-serving.com/v1` | Wrong value can silently mock — preflight catches it. |
| `SUPERVISOR_MODEL` | `.env` / env var | both | `MiniMaxAI/MiniMax-M3` | |
| `SUPERVISOR_TOKEN_BUDGET` | `.env` / env var | Python engine | *unset (no cap)* | Soft cap on cumulative tokens per run. Env-only — not duplicated in any JSON. |
| `SUPERVISOR_DATA_DIR` | env var | Python engine | `supervisor_data` | Audit logs, pending approvals, weight history. |
| `LOG_LEVEL` | env var | Python engine (server) | `INFO` | |
| `ENABLE_REAL_TIME_ALERTS` | env var | Python engine (server) | `false` | Gates the reporting/alert system. |
| `config/weights.json` | file (engine reads **and writes**) | Python engine — Expectimax weights | repo defaults | Persisted & version-controlled. See [FEEDBACK_LOOP.md](FEEDBACK_LOOP.md). |
| `config.json` | file | Python engine — alerts/reporting | provided | Email / Slack / webhook alert settings. |
| `implementations/vortex-os-v3/config.minimax.json` | file | **bash skill v3 only** | provided | Agent roster / passes for the CLI implementation. |

**Which layer does a setting touch?** The `GMI_*` / `SUPERVISOR_*` env vars drive
both the Python engine and the bash skills identically. The JSON files are
engine-specific: `config/weights.json` + `config.json` belong to the Python
engine; `config.minimax.json` belongs only to the `vortex-os-v3` CLI skill.

## Production-readiness checklist

Go from the offline mock to a fully authenticated live deployment:

- [ ] **1. Key** — `cp .env.example .env`, then set a real `GMI_API_KEY`.
- [ ] **2. Preflight** — `python scripts/preflight.py` → expect **`READY — live MiniMax-M3`**. (Fix anything red before continuing.)
- [ ] **3. Not-on-mock guarantee** — wire `python scripts/preflight.py --require-live` into CI / your container entrypoint so a misconfig fails the deploy instead of silently mocking.
- [ ] **4. Budget** *(optional)* — set `SUPERVISOR_TOKEN_BUDGET` to cap spend per run.
- [ ] **5. Smoke test** — `python demo.py` completes end to end with a live trace.
- [ ] **6. Tests** — `PYTHONPATH=src pytest tests/ -q` → **46 passing**.
- [ ] **7. Server** — `PYTHONPATH=src python src/server/main.py` starts the FastMCP server; call `run_master_goal` / `hitl_status`.
- [ ] **8. Learned weights** — if you've trained the Expectimax weights, back up `config/weights.json` (see [FEEDBACK_LOOP.md](FEEDBACK_LOOP.md) for persistence & versioning).

## Am I on the mock or live?

- **Preflight** states it outright (`live MiniMax-M3` vs `offline mock`).
- **In a run trace**, the demo prints `Model backend: MiniMax-M3 (live via GMI Cloud)` or `deterministic mock (no GMI_API_KEY set)`.
- **In code**, `LLMClient(...).is_configured` is `True` only when a usable key is present, and every mock response carries a `"mock": true` field.
