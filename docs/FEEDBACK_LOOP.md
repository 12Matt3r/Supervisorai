# The feedback loop, demystified

The supervisor's Expectimax agent can **learn from human corrections**. This page
turns that from a black box into a glass box: exactly where the learned policy is
stored, whether it survives a restart, how it's version-controlled, and the
concrete arithmetic of a single override.

## Lifecycle of a correction

1. **Override** — an operator disagrees with a decision and corrects it (the
   `submit_feedback` MCP tool). The correction is appended to
   `supervisor_data/feedback.json`.
2. **Retrain** — the `run_training` MCP tool runs `FeedbackTrainer`
   (`src/supervisor_agent/feedback_trainer.py`): it reads `feedback.json` and the
   current weights and computes new ones.
3. **Persist** — `SupervisorCore.update_weights()` applies them in-memory **and
   writes them to `config/weights.json`**, then appends a timestamped entry to
   `supervisor_data/weight_history.jsonl`.
4. **Survive restart** — on the next start, `_load_weights()` reads
   `config/weights.json`, so **the learned behavior persists across restarts**.
5. **See the drift** — `python scripts/plot_weight_drift.py` renders
   `weight_history.jsonl` as a line chart of how each weight has moved over time.

## Persistence, at a glance

| Question | Answer |
|---|---|
| Where are learned weights stored? | **`config/weights.json`** (plain JSON). |
| Do they survive a server restart? | **Yes** — they're read from that file at startup. |
| How are they version-controlled? | It's a file in the repo — commit `config/weights.json` to snapshot a policy; diff it to review changes. |
| How do I back up / share a trained agent? | Copy `config/weights.json` (and, for the audit trail, `supervisor_data/weight_history.jsonl`). |
| How do I reset to defaults? | Restore the default `config/weights.json` (or delete it — `_load_weights()` falls back to built-in defaults). |
| Can I watch it drift? | `scripts/plot_weight_drift.py` → a PNG line chart per weight. |

## The exact math of one override

Training uses a small **perceptron-style** rule with a learning rate of
**`0.01`** (`DEFAULT_LEARNING_RATE`). For each correction, for each weight *k*:

```
adjustment[k] = 0.01 × (heuristic_correct[k] − heuristic_incorrect[k])
W_new[k]      = W_old[k] + adjustment[k]
```

then all weights are **normalized to sum to 1** and **clamped to ≥ 0**.

Each heuristic is in `[0, 1]`, so a single override moves a weight by **at most
`0.01`** (when the correct and incorrect actions differ maximally on that
heuristic). Concretely: **to shift a weight by ~0.05 you need about five
consistent overrides** in the same direction. The learning is deliberately slow
so one stray click can't swing the policy.

> **Worked example.** You keep approving outputs the agent wanted to `CORRECT`,
> and those outputs score high on `quality_score` but the agent had over-weighted
> `inv_error_count`. Each such override nudges `quality_score` up and
> `inv_error_count` down by up to `0.01` before renormalization. After ~5 similar
> overrides the balance has moved ~0.05 — a visible, intentional shift, not a
> silent jump.

### A note the critique got *almost* right

The hard gates **0.4** and **0.9** (quality below 0.4 forces `CORRECT`/`ESCALATE`;
0.9+ offers only `ALLOW`/`WARN`) are **fixed in code** — they are *not* what your
overrides move. Overrides retrain the **utility weights** that rank the available
actions. So feedback shapes *which action wins among those the gate allows*, and
the drift chart shows exactly that.

## The bias risk — and how it's made visible

Learning from a single operator can **overfit to that person's preferences**
rather than objective quality. SupervisorAI doesn't hide this — it surfaces it:

- **`weight_history.jsonl` + the drift chart** make accumulating bias *visible*.
  If one weight is marching steadily in one direction, that's your bias on screen.
- **`feedback.json` is auditable** — you can review exactly which corrections
  shaped the policy.
- **Reset is one file** — restore the default `config/weights.json` to zero it out.

The slow learning rate, the persisted+versioned weights file, and the visible
drift log together turn "the AI mysteriously learned something" into an auditable,
reversible, chartable record.
