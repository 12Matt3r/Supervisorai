# VORTEX-OS

> **The Native Autonomous Studio Command Center**
> *Stop managing tasks. Start commanding a native digital workforce.*

---

## Table of Contents

1. [What Is VORTEX-OS?](#1-what-is-vortex-os)
2. [Why Native MiniMax?](#2-why-native-minimax)
3. [The 4-Tier Chain of Command](#3-the-4-tier-chain-of-command)
4. [The 6 Architecture Pillars](#4-the-6-architecture-pillars)
5. [Quick Start](#5-quick-start)
6. [What You Can Build — Representative Use Cases](#6-what-you-can-build--representative-use-cases)
7. [The Complete Command Reference](#7-the-complete-command-reference)
8. [How To Submit A Master Objective](#8-how-to-submit-a-master-objective)
9. [How HITL Works](#9-how-hitl-works)
10. [How The Continuity Engine Works](#10-how-the-continuity-engine-works)
11. [How The Self-Healing Optimizer Works](#11-how-the-self-healing-optimizer-works)
12. [How To Add Your Own Worker Agent](#12-how-to-add-your-own-worker-agent)
13. [How To Save & Reuse Workflows (Golden Path)](#13-how-to-save--reuse-workflows-golden-path)
14. [The 8 Contract Invariants](#14-the-8-contract-invariants)
15. [File Structure](#15-file-structure)
16. [Troubleshooting & Exit Codes](#16-troubleshooting--exit-codes)

---

## 1. What Is VORTEX-OS?

VORTEX-OS is the **Native Autonomous Studio Command Center** for the MiniMax ecosystem. It is not a chatbot, and it is not a single AI model — it is a **Hierarchical Autonomous Orchestration Engine** that treats AI agency like a high-stakes, closed-loop corporate operation.

You give VORTEX-OS a single natural-language **master objective** — for example:

> *"Build a WebSim visual novel module for When Ocean Meets Sky — Mara on her porch in 1994, with a native audio loop, a prose scene, and an interactive HTML page. No smartphones. The radio inside plays only a weather report."*

VORTEX-OS then **decomposes** that objective across specialist domains, **dispatches** parallel worker agents in isolated sandboxes, **enforces** your creative rules via the Continuity Engine, **self-heals** when workers drift, **routes** audio to the native `minimax-music` engine, **generates** video via Hailuo, **processes** sound with local ffmpeg, **verifies** code in a sandbox, and **halts** for your approval before any high-stakes action.

**Everything routes through MiniMax-native engines — no third-party creative services and no fragile external websites.** (The MiniMax engines use your own MiniMax/GMI credentials; "native" means no *extra* third-party dependencies, not zero authentication.)

---

## 2. Why Native MiniMax?

| Problem | VORTEX-OS Solution |
|---|---|
| Fragile external APIs go down | 100% native MiniMax engines — minimax-music, Hailuo, minimax-image |
| External music sites change their terms | Local ffmpeg + Python audio scripts in sandbox — fully self-contained |
| Hallucinated canon violations | Continuity Engine catches & rewrites violations automatically |
| LLM drift on long projects | 4-tier chain of command isolates context per tier |
| High-stakes actions firing silently | Deep-Sleep HITL halts & pages you for approval |
| Repeated planning cost | Golden Path templates replay successful workflows for free |
| No audit trail | Every decision logged to `memory/audit.jsonl` |
| External video services are expensive | Native Hailuo pipeline generates video entirely in-house |
| Code runs differently in prod vs. dev | Sandbox Verifier checks every artifact before it leaves the swarm |

---

## 3. The 4-Tier Chain of Command

```
T0 — GENERAL MANAGER (The Apex)
       │  • Receives your master objective
       │  • Performs deep decomposition
       │  • Creates the master plan
       ▼
T1 — STORE SUPERVISOR (Domain Strategy)
       │  • Maintains the "Golden Path" project files
       │  • Routes to the right specialist swarm
       │  • Manages resource allocation
       ▼
T2 — SHIFT SUPERVISOR (Tactical QA)
       │  • Spawns workers in isolated memory sandboxes
       │  • Enforces the Continuity Engine (universe canon)
       │  • Runs sandbox verification on generated code
       │  • Holds the HITL checkpoint for high-stakes actions
       ▼
T3 — THE CREW (Specialized Workers)
       ├─── writer.docs          (prose, dialogue, narrative)
       ├─── media.native         (minimax-music audio, Hailuo video)
       ├─── coder.typescript     (HTML/CSS/JS, WebSim UIs, VibeOS)
       ├─── coder.python         (ffmpeg scripts, data pipelines, DSP)
       ├─── researcher.web       (web research, citations)
       ├─── analyst.strategic    (strategy, planning, risk)
       └─── designer.brand       (visual identity, continuity)
```

---

## 4. The 6 Architecture Pillars

1. **Hierarchical Decomposition** — 4-tier chain of command isolates context per tier, eliminates context rot.
2. **Continuity Engine** — Catches canonical violations (wrong character details, anachronisms, lore breaks) and forces rewrites.
3. **Self-Healing Prompt Optimizer** — DSPy-inspired: rewrites failing prompts to permanently eliminate failure modes.
4. **Native Engine Routing** — minimax-music for audio, Hailuo for video, minimax-image for art, local ffmpeg for DSP — no external services.
5. **Sandbox Verifier** — Ephemeral isolation checks generated code before it leaves the swarm.
6. **Human-in-the-Loop Gate (Deep-Sleep)** — High-stakes actions halt, suspend state to disk, page you for explicit approval.

---

## 5. Quick Start

The reasoning core (planning, worker generation, and audits) is served by
**MiniMax-M3 on GMI Cloud** via `lib/minimax.sh`. Set your key first; without it
the pipeline falls back to a deterministic stub so it still runs offline:

```bash
export GMI_API_KEY="<your GMI serving key>"   # or add it to a gitignored .env
# GMI_BASE_URL defaults to https://api.gmi-serving.com/v1
# SUPERVISOR_MODEL defaults to MiniMaxAI/MiniMax-M3
```

```bash
cd skills/VORTEX-OS
chmod +x skill.sh verify.sh
bash verify.sh                  # Expected: ALL VERIFICATION CHECKS PASSED

./skill.sh --agents-discover    # List available agents
./skill.sh --agents-lint --all  # Lint all agents against the 8 invariants

# Submit your first master objective
mkdir -p my_project
cat > my_project/objective.md <<'EOF'
# Project: My First VibeOS Module

Build a WebSim scene: a coastal bedroom at sunset in 1994.

## Required deliverables
- 1 ambient audio loop (≈90 BPM, bedroom mood) via minimax-music
- 1 prose scene (200 words) for the protagonist
- 1 interactive HTML page with a fade-lights button

## Continuity rules
- No smartphones, no internet references
- The radio plays only a weather report
- Mara is 19, left-handed

## High-stakes
- package_websim: requires operator approval before final HTML write
EOF

./skill.sh --dispatch-master my_project/objective.md

# When HITL fires:
./skill.sh --hitl-status
./skill.sh --hitl-approve package_websim

# Inspect results:
ls deliverables/
./skill.sh --audit-trail
```

---

## 6. What You Can Build — Representative Use Cases

VORTEX-OS is a general orchestrator, so this list is **illustrative, not exhaustive**. Each item maps to a single `--dispatch-master` call and is grouped by the specialist swarm that leads the work.

### Narrative & Worldbuilding — `writer.narrative`
- **Visual-novel scene** — a branching dialogue scene with an ambient audio loop and an interactive HTML page
- **Show bible** — a multi-episode bible with enforced character arcs and a continuity audit
- **Lore consistency audit** — scan a long manuscript for canon violations (e.g. a character using an item before they find it)
- **Branching dialogue trees** — author multi-path scripts and export them as structured JSON
- **Anachronism check** — scan a period script to flag out-of-era technology or references

### Interactive Code & WebSim — `coder.javascript` / `coder.typescript`
- **Single-file WebSim game** — a self-contained HTML/JS game loop with sound effects and a save state
- **Procedural generation** — dungeon / maze / level algorithms with a playable prototype
- **Retro UI effects** — CRT / VHS / glitch effects triggered on interaction
- **Reusable VN engine** — a WebSim dialogue system with branching, save/load, and audio cues

### Sound Design & Foley — `coder.python` + local ffmpeg
- **Batch sample chopping** — slice a field recording into 2-second percussive hits
- **VHS / cassette degradation** — lowpass + wow/flutter + 60 Hz hum over clean audio
- **Seamless loop crossfade** — overlap tail and head for a gapless ambient drone
- **Synthetic impulse responses** — generate a reverb IR WAV for a named space
- **Granular reorganization** — chop a sample into 10 ms grains and rearrange them

### Native Media — `media.native` (MiniMax image / Hailuo video)
- **Storyboard-to-video** — break a script into shot-by-shot Hailuo prompts
- **Character reference sheets** — consistent HEX palettes, outfit specs, lighting conditions
- **Batch image prompting** — a set of interlocking prompts for a coherent visual series

### Software Automation — `coder.python` / `coder.typescript`
- **Mock REST API** — a locally hosted Flask mock for UI prototyping
- **Legacy refactor** — lint, format, and modularize a messy repository
- **Schema validation** — check generated JSON manifests against the VORTEX-OS invariants
- **Log analysis** — parse `memory/audit.jsonl` to find the longest-running agent tasks

### Research & Data — `researcher.web` / `analyst.strategic`
- **Literature synthesis** — reduce many documents into a single comparative matrix
- **Lore indexing** — parse unstructured notes into a relational SQLite schema
- **Manual condensation** — turn a long technical manual into a quick-start guide
- **Regex generation** — build and test expressions to clean messy input

### Business & Branding — `writer.docs` / `analyst.strategic`
- **Pitch deck** — a persuasive slide deck plus a one-page leave-behind
- **Content calendar** — a dated posting schedule with captions and hooks
- **Brand style guide** — typography, spacing, and brand-voice constraints
- **Launch campaign** — announcement, teasers, a press release, and a recap

### Meta-Orchestration — VORTEX-OS on itself
- **Agent synthesis** — design, code, and register a new specialist manifest, then lint it against the invariants
- **Adversarial red-team** — a swarm ordered to try to break your continuity rules, to test them
- **Token audit** — analyze `memory/audit.jsonl` for inefficient prompt structures
- **Self-documentation** — generate a user manual from the codebase itself

> The engine is not limited to these examples — any goal that spans writing, code, audio, video, research, and packaging can be a single master objective.

## 7. The Complete Command Reference

### Discovery & Inspection
| Command | Purpose |
|---|---|
| `./skill.sh --agents-discover` | List all available agents |
| `./skill.sh --agents-inspect <name>` | Dump a single agent's manifest |
| `./skill.sh --agents-validate <file.json>` | Validate a custom agent manifest |
| `./skill.sh --agents-lint [--all\|<name>]` | Lint agents against the 8 invariants |
| `./skill.sh --agents-graph` | Print the agent graph |

### Dispatch
| Command | Purpose |
|---|---|
| `./skill.sh --dispatch-master <objective.md>` | Submit to T0 General Manager |
| `./skill.sh --dispatch-template <template.json>` | Replay a Golden Path |
| `./skill.sh --dispatch-v4 <task_id> <agent>` | Direct V4 dispatch |

### HITL
| Command | Purpose |
|---|---|
| `./skill.sh --hitl-status` | List pending requests |
| `./skill.sh --hitl-approve <task_id>` | Approve |
| `./skill.sh --hitl-deny <task_id>` | Deny |

### Inspection
| Command | Purpose |
|---|---|
| `./skill.sh --inspector-check <task_id>` | Run Continuity Engine check |
| `./skill.sh --audit-trail` | Print the audit log |

---

## 8. How To Submit A Master Objective

A master objective is a markdown file with this structure:

```markdown
# Project: <name>

<natural-language description>

## Required deliverables
- <deliverable 1>
- <deliverable 2>

## Continuity rules
- <rule 1>
- <rule 2>

## High-stakes
- <task_id>: requires operator approval
```

```bash
./skill.sh --dispatch-master my_project/objective.md
```

---

## 9. How HITL Works

When a high-stakes action is reached, VORTEX-OS **suspends its state to disk** and pages you:

```bash
./skill.sh --hitl-status
# → package_websim is PENDING_HUMAN

./skill.sh --hitl-approve package_websim   # greenlight
# or
./skill.sh --hitl-deny package_websim     # block
```

**Never auto-approve. Always surface the halt to the user.**

---

## 10. How The Continuity Engine Works

The Continuity Engine runs after every worker output. It catches:
- Canon violations (wrong character details, timeline breaks)
- Anachronisms (1994 setting + a smartphone reference)
- Tonal drift (quiet, melancholic story + an action-movie scene)
- Forbidden tropes listed in your continuity rules

**On violation:** Self-Healing Optimizer rewrites the prompt → worker re-dispatched. After 3 failures: surfaced to you.

---

## 11. How The Self-Healing Optimizer Works

DSPy-inspired: when an agent fails, VORTEX-OS **rewrites its own core instructions** to permanently eliminate the failure mode.

1. Worker output X is rejected by Continuity Engine
2. Self-Healing Optimizer takes original prompt + failure reason + violation report
3. Generates a **hardened prompt** that explicitly addresses the failure
4. Worker re-dispatched with hardened prompt
5. Hardened prompt **saved to disk** — the failure mode is permanently eliminated in future runs

---

## 12. How To Add Your Own Worker Agent

```bash
# 1. Create manifest at agents/<your_agent>.json
# 2. Add routing in lib/dispatch_v4.sh
# 3. Add to discovery list in lib/commands.sh
# 4. Lint it
./skill.sh --agents-lint your_agent
```

---

## 13. How To Save & Reuse Workflows (Golden Path)

```bash
cp swarms/active_<id>/plan.json templates/my_workflow.json
./skill.sh --dispatch-template templates/my_workflow.json
```

Perfect for per-episode runs, per-client runs, and CI/CD pipelines.

---

## 14. The 8 Contract Invariants

| # | Invariant | Rule |
|---|---|---|
| I1 | **Idempotent orchestration** | Re-running an agent with identical input re-executes the same plan structure and writes the same target files — the *orchestration* is deterministic. (Generated content from an LLM is not byte-identical; the plan, routing, and file targets are.) |
| I2 | **Resource Honesty** | Declared resources match actual (±20%) |
| I3 | **Write Containment** | Never writes outside declared `writes[]` |
| I4 | **Read Containment** | Never reads outside declared `reads[]` |
| I5 | **Sealed Envelope** | Output conforms to JSON schema |
| I6 | **Retry Honesty** | Never loops internally |
| I7 | **Secret Hygiene** | No secrets in logs |
| I8 | **Metric Truthfulness** | Metrics are actual, not estimated |

---

## 15. File Structure

```
skills/VORTEX-OS/
├── README.md                              ← ★ The knowledge base (first thing you see)
├── INSTRUCTIONS.md                        ← ★ LLM operator knowledge base
├── SKILL.md                               ← LLM instruction brain
├── _meta.json                             ← Platform registration
├── skill.sh                               ← CLI entry point
├── verify.sh                              ← Post-upload verification
│
├── lib/                                   ← Engine (6 modules)
│   ├── swarm.sh                           ← T1/T2 coordination
│   ├── hitl.sh                            ← Deep-Sleep HITL gate
│   ├── inspector.sh                        ← Continuity Engine + invariants
│   ├── prompt_optimizer.sh               ← Self-Healing Optimizer
│   ├── dispatch_v4.sh                     ← T3 worker dispatch
│   └── commands.sh                        ← Command functions
│
├── agents/                                ← 3 supervisor manifests
│   ├── supervisor.store.json
│   ├── supervisor.shift.json
│   └── inspector.governance.json
│
└── state/  swarms/  tasks/  memory/  deliverables/  ← Runtime (created on first run)
```

---

## 16. Troubleshooting & Exit Codes

| Exit Code | Meaning | Action |
|---|---|---|
| `0` | Success | Read `deliverables/` |
| `2` | Bad input / missing file | Check the file path |
| `42` | Continuity violation unresolved after 3 rewrites | Inspect `state/inspector_interventions.log` |
| `100` | Invariant lint failure | Run `./skill.sh --agents-lint --all` |
| `203` | HITL pending | Run `./skill.sh --hitl-status` |
| `127` | Command not found | Install `jq`, `sqlite3`, or `ffmpeg` |

**Common fix:** `apt install jq sqlite3 ffmpeg`

---

## License

MIT

## Author

12matt3r — originally built for MiniMax's Agent Hackathon (Honorable Mention), rebuilt with Claude (Claude Code).
