#!/usr/bin/env python3
"""
Plot the drift of the Expectimax weights over time — the "glass box" for the
feedback loop. Every time the supervisor's weights are retrained from human
feedback, `SupervisorCore.update_weights()` appends an entry to
`<data_dir>/weight_history.jsonl`. This script turns that log into a line chart
so an operator can literally see how their corrections are shifting the policy
(and spot when they're overfitting to their own bias).

    python scripts/plot_weight_drift.py                       # uses supervisor_data/
    python scripts/plot_weight_drift.py --data-dir path --out drift.png

Weights are shown as **normalized fractions** (each entry divided by its sum),
which is how they actually weigh into a decision.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path


def load_history(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec.get("weights"), dict):
                rows.append(rec)
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="supervisor_data")
    ap.add_argument("--out", default=None, help="output PNG (default: <data-dir>/weight_drift.png)")
    args = ap.parse_args()

    hist_path = Path(args.data_dir) / "weight_history.jsonl"
    rows = load_history(hist_path)
    if not rows:
        print(f"No weight history at {hist_path}.")
        print("It is written each time the supervisor retrains from feedback "
              "(SupervisorCore.update_weights / the run_training MCP tool).")
        return 1

    keys = sorted({k for r in rows for k in r["weights"]})
    # Normalize each entry to fractions (how the weights actually weigh a decision).
    series = {k: [] for k in keys}
    for r in rows:
        w = r["weights"]
        total = sum(v for v in w.values() if isinstance(v, (int, float))) or 1.0
        for k in keys:
            series[k].append(float(w.get(k, 0.0)) / total)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available ({e}). Raw normalized series:")
        for k in keys:
            print(f"  {k}: {[round(v, 3) for v in series[k]]}")
        return 0

    x = list(range(1, len(rows) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in keys:
        ax.plot(x, series[k], marker="o", linewidth=2, label=k)
    ax.set_title("Expectimax weight drift (normalized) — learned from human feedback")
    ax.set_xlabel("training round")
    ax.set_ylabel("weight (fraction of total)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    out = Path(args.out) if args.out else Path(args.data_dir) / "weight_drift.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}  ({len(rows)} training rounds, {len(keys)} weights).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
