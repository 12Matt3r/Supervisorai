# Command-line implementations (experimental)

These are **downstream, command-line clients** of the SupervisorAI engine — not
the core product. The engine is the Python package under [`../src/`](../src),
served via the FastMCP server. Use these for quick local experiments and
terminal-driven runs; use the **Python engine** for anything production-facing.

| Skill | What it is |
|---|---|
| [`vortex-os/`](vortex-os/) | Creative studio flavor — 4-tier chain of command, HITL gate, continuity, on-disk deliverables, driven from `./skill.sh`. |
| [`vortex-os-v3/`](vortex-os-v3/) | Software-engineering flavor — a 30+ agent roster with consensus voting, cost governance, and quality gates. |

Both call **MiniMax-M3 on GMI Cloud** via their `lib/minimax.sh` bridge, and fall
back to a deterministic offline stub when no `GMI_API_KEY` is set.

> **Which should I use?** **Python engine (`src/`) for enterprise / production**
> deployments — typed, tested, async, MCP-served. **Bash skills (here) for local
> experiments** and terminal demos.
