# SupervisorAI — Demo Runbook (60–90s submission capture)

**MiniMax Week × GMI Cloud Hackathon — Track 1: Reasoning**

Everything below runs against **real MiniMax-M3 on GMI Cloud**. The tagged live
trace *is* the demo — plan, delegate, audit, catch drift, self-heal, gate a
high-stakes step for human approval, and finish with verified, on-disk results.

## 0. One-time setup (off camera)

```bash
export GMI_API_KEY="<your GMI serving key>"     # or put it in .env
pip install -r requirements.txt
```

## Scene 1 — the supervisory loop (~40s, the money shot)

```bash
python demo.py
```

Let the tagged trace scroll. Point the viewer at the tags as they appear:

- `[SUPERVISOR - M3]` — M3 decomposes the goal into a validated task DAG and
  assigns a **specialist** to each task (`analyst.strategic`, `coder.python`,
  `reviewer.code`, `writer.docs`).
- `[DISPATCH]` — each specialist worker runs on M3.
- `[AUDIT]` — M3 puts on its supervisor hat and verifies every output against
  the plan + continuity rules (watch the confidence scores).
- `[DRIFT DETECTED]` / `[RECOVERY]` — when an output fails audit, it is
  re-prompted with a concrete fix hint (the self-correction showcase always
  fires one).
- `[HITL]` — high-stakes tasks **halt for human approval** before anything is
  finalized, then resume once approved.
- `[SYNTHESIS]` — once every dependency passes, M3 synthesizes the verified PR
  summary. Ends on `Goal COMPLETED`.

## Scene 2 — the results are real, not narrated (~20s)

```bash
ls -la deliverables/            # analysis.md, discount.py, test_discount.py, review.md, pr_summary.md
```

Prove the generated fix + tests actually work together:

```bash
mkdir -p /tmp/show/billing && cp deliverables/discount.py /tmp/show/billing/
touch /tmp/show/billing/__init__.py
cp deliverables/test_discount.py /tmp/show/
cd /tmp/show && PYTHONPATH=. python -m pytest test_discount.py -q   # 16 passed
```

## Scene 3 — same engine, as a studio CLI (~20s, optional)

```bash
cd vortex-os
export GMI_API_KEY="$GMI_API_KEY"
./skill.sh --dispatch-master objective.md      # 4-tier trace, writes deliverables/
./skill.sh --hitl-status                        # the Deep-Sleep gate
./skill.sh --audit-trail                        # append-only decision log
```

## Suggested post

> SupervisorAI for #MiniMaxWeek Track 1 (Reasoning): an agent that holds a plan,
> delegates to specialists, and **fact-checks itself** — powered by MiniMax-M3 on
> @GMICloud. It catches its own drift, halts high-stakes steps for me, and ships
> verified code (its generated tests pass against its generated fix). @MiniMax__AI

## Notes for a clean capture

- A full sanitized reference trace is in `docs/sample_live_run.txt`.
- The whole loop is ~30–60s of real M3 calls; if you want it snappier for the
  cut, `max_corrections=1` in `demo.py` reduces retry rounds.
- No key on camera: keep it in `.env` (gitignored) or an environment variable.
