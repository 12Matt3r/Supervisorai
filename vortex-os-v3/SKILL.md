# Universal Multi-Agent Orchestrator v3.1 — JIT-Native Edition

## Overview

A production-grade, pure-Bash sub-agent orchestration skill for planning, dispatching, supervising, and verifying parallel AI agents over a unified 8-Invariant envelope.

## Quick Start

```bash
# Run the master acceptance test suite (6 phases, all green)
bash test_v3_master.sh

# List all available agents
bash skill.sh --agents-discover

# Static-lint every agent against the 8 Invariants
bash skill.sh --agents-lint --all

# Compile a prompt inheritance tree
bash skill.sh --agents-compile <agent_name>

# Dispatch a task through the V3 pipeline
bash skill.sh --dispatch-v3 <task_id> <agent_role>

# Trigger the JIT skill bridge manually
bash skill.sh --jit-bridge "<capability>"

# Inspect JIT request/response logs
bash skill.sh --jit-status
```

## Key Features

- **8-Invariant envelope** — every agent JSON must declare I1–I8 (resources, writes, reads, schema, quality, composability, versioning, deprecation).
- **Just-In-Time Skill Bridge** — three-tier resolution: local agents → native tool library (heuristic map to ~18 MiniMax-native tools) → web-research synthesis that emits a fully-compliant agent JSON.
- **Adversarial & Continuity Gates** — pre-dispatch injection detection plus post-dispatch rule enforcement (no-markdown, no-code, JSON-only).
- **Async Deep Sleep** — agents can `cmd_task_yield` to persist scratchpad to `memory/suspended/` and exit 202 without crashing the master loop.
- **Prompt Inheritance Trees** — `--agents-compile` merges trait files (aesthetic, persona) into compiled prompts.
- **Static Lint + Master Test Suite** — `test_v3_master.sh` mathematically proves every gate, loop, and stub works.

## File Map

| Path | Purpose |
|---|---|
| `skill.sh` | Main entry point and CLI dispatcher |
| `skill.json` | Skill manifest (used by MiniMax skill library) |
| `config.minimax.json` | Live configuration |
| `config.example.json` | Annotated reference configuration |
| `lib/` | 40+ bash library modules |
| `lib/v3.sh` | V3 architecture: hydration, adversarial, consensus, continuity, scaffold |
| `lib/jit_bridge.sh` | Just-in-time skill acquisition (native + web research) |
| `lib/llm_stubs.sh` | Native LLM heuristic stubs (security, fast LLM, embedding, consensus) |
| `lib/async.sh` | Async deep-sleep yielding (exit 202) |
| `lib/commands.sh` | V2 command implementations |
| `agents/` | 30+ static agent JSON envelopes |
| `templates/` | Prompt templates |
| `schemas/` | JSON schemas |
| `tests/` | bats unit tests |
| `test_v3_master.sh` | 6-phase acceptance test suite |
| `install.sh` | One-shot installer |

## Test Suite (6 Phases)

```
[TEST] Phase 1: Static Architecture & Discovery    [PASS]
[TEST] Phase 2: Native LLM Inference Stubs         [PASS]
[TEST] Phase 3: The Continuity Engine              [PASS]
[TEST] Phase 4: The JIT Skill Bridge               [PASS]
[TEST] Phase 5: Asynchronous Yielding              [PASS]
[TEST] Phase 6: Compile Inheritance Trees          [SKIP]
```

## Author

12matt3r — originally built for MiniMax's Agent Hackathon (Honorable Mention), rebuilt with Claude (Claude Code).

## License

MIT
