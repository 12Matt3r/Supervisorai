# INSTRUCTIONS — VORTEX-OS Operator's Knowledge Base

> **For LLMs and operators using VORTEX-OS**
> This is the complete operational contract: when to invoke it, how to invoke it, how to handle every event, how to recover from every error, and what the boundaries are.

---

## 0. TL;DR

```bash
cd skills/VORTEX-OS
chmod +x skill.sh verify.sh
./skill.sh --agents-discover                          # what agents exist?
./skill.sh --agents-lint --all                        # are they all healthy?
./skill.sh --dispatch-master my_project/objective.md  # run a project
./skill.sh --hitl-status                              # what's waiting on me?
./skill.sh --hitl-approve <task_id>                   # greenlight it
./skill.sh --audit-trail                              # what just happened?
```

---

## 1. When To Invoke VORTEX-OS

### Invoke VORTEX-OS when the user requests:
- **A multi-disciplinary project** (writing + audio + code + research, all in one)
- **A long-running creative project** with universe canon (visual novel, game, series, worldbuilding)
- **An auditable autonomous pipeline** where every decision must be traceable
- **A high-stakes deployment** that requires human approval before finalization
- **A reproducible workflow** (they want to run the same kind of project again)
- **A procedurally-generated deliverable** (audio loop, dialogue, HTML page, Python script)

### Do NOT invoke VORTEX-OS when:
- The user asks a **simple one-line question**
- The user wants a **single-file edit** with no cross-domain coordination
- The user wants **read-only research** (use a web search instead)
- The user wants **pure chat / Q&A** (just answer directly)

---

## 2. The Invocation Pattern

The skill exposes `skill.sh` as the primary CLI. When the user submits a master objective, the LLM should:

### Step 1 — Write the master objective to a file
```bash
mkdir -p /tmp/vortex
cat > /tmp/vortex/objective.md <<'EOF'
<the user's natural-language objective here>
EOF
```

### Step 2 — Dispatch to VORTEX-OS
```bash
./skill.sh --dispatch-master /tmp/vortex/objective.md
```

### Step 3 — VORTEX-OS will:
1. Parse the objective (T0 General Manager)
2. Spawn a T1 Store Supervisor
3. Generate a `plan.json` with the worker breakdown
4. Begin dispatching T2 Shift Supervisors (one per domain)
5. Each T2 spawns T3 workers in isolated memory sandboxes
6. The Continuity Engine checks every output
7. The Self-Healing Optimizer rewrites any failing prompt
8. The Inspector watches global token velocity
9. **HITL halts execution** if any task is high-stakes
10. Final deliverables are written to `deliverables/` only after HITL approval

---

## 3. Responding To HITL Checkpoints (Deep-Sleep)

When VORTEX-OS halts for human approval, the LLM **MUST** follow this exact sequence:

### 3.1 — Detect the halt
VORTEX-OS will print a `PENDING_HUMAN` event and write a JSON file to `state/pending_approvals/<task_id>.json`.

### 3.2 — Surface the request to the user verbatim
Show the user:
- The `proposed_action` (what VORTEX-OS is about to do)
- The `task_id` (the ID they need to reference)
- The `severity` (typically `HIGH` or `CRITICAL`)
- The `context` (why this action was triggered)

Example relay to the user:
> *"VORTEX-OS is requesting your approval to write the final HTML page to `deliverables/`. This is a HIGH-stakes action. Reply `Approve` or `Deny` to continue."*

### 3.3 — Wait for the user's reply
The user will reply with `Approve <task_id>` or `Deny <task_id>`. If they reply without the task_id, ask for it.

### 3.4 — Relay the decision back to VORTEX-OS
```bash
./skill.sh --hitl-approve package_websim    # user approved
# or
./skill.sh --hitl-deny package_websim       # user denied
```

### 3.5 — **NEVER auto-approve**
Even if the user previously said "do whatever you want" or "you can decide", you must still surface the HITL halt and get explicit approval. The HITL gate is the last line of defense against runaway actions.

### 3.6 — Resume
After approval, VORTEX-OS will automatically resume from the suspended state. You don't need to do anything else.

---

## 4. Responding To Continuity Engine Violations

VORTEX-OS will sometimes halt and report that the Continuity Engine rejected a worker output. When this happens:

### 4.1 — Show the user the violation
Read `state/inspector_interventions.log` and present the violation to the user in plain language:
> *"The Continuity Engine caught a violation: a worker wrote a scene where Mara uses her left hand to grip something, but her character sheet says she has a prosthetic left hand. VORTEX-OS has automatically rewritten the prompt and is re-dispatching."*

### 4.2 — Show the optimized prompt
The Self-Healing Optimizer writes the hardened prompt to `state/prompt_optimizations/<agent>_<timestamp>.json`. You can read it and show the user what was changed.

### 4.3 — No LLM action needed for the first 3 attempts
VORTEX-OS will automatically re-dispatch with the hardened prompt. The LLM doesn't need to do anything.

### 4.4 — If the rewrite still fails after 3 attempts
Surface the situation to the user and ask:
- "Should I relax this rule?"
- "Should I change strategy?"
- "Should I try a different worker agent?"

---

## 5. Inspecting Logs, State & Artifacts

When the user asks "what did VORTEX-OS do?", the LLM should:

### 5.1 — Use the built-in audit trail
```bash
./skill.sh --audit-trail
```
This prints the last 50 entries from `memory/audit.jsonl` in a compact format.

### 5.2 — Read the full audit JSONL
```bash
cat memory/audit.jsonl | jq -c '{ts, tier, agent, action, status, duration_ms}'
```

### 5.3 — Read the Continuity Engine interventions
```bash
cat state/inspector_interventions.log
```

### 5.4 — Read the Self-Healing Optimizer rewrites
```bash
ls state/prompt_optimizations/
cat state/prompt_optimizations/writer.docs_*.json
```

### 5.5 — Read the native engine transcripts
```bash
cat state/minimax_music.log       # audio generation log
```

### 5.6 — List the generated deliverables
```bash
ls -la deliverables/
```

### 5.7 — Per-swarm plan and outputs
```bash
ls -la swarms/active_*/
cat swarms/active_*/plan.json | jq .
```

### 5.8 — Present the findings in natural language
Read these files and explain the workflow to the user in plain language. Don't dump raw JSON — synthesize it into a story.

---

## 6. Saving & Reusing Workflows (Golden Path)

To re-run a previously successful workflow without burning planning tokens:

```bash
# 1. Find a successful swarm
ls swarms/

# 2. Copy its plan.json as a template
cp swarms/active_<id>/plan.json templates/my_workflow.json

# 3. Replay it later (bypasses T0/T1 decomposition)
./skill.sh --dispatch-template templates/my_workflow.json
```

This is perfect for:
- Per-episode runs in a series
- Per-client runs in a service business
- CI/CD pipelines that re-run a known-good flow

---

## 7. Error Handling — Complete Reference

If VORTEX-OS returns a non-zero exit code, use this table to diagnose:

| Exit Code | Meaning | LLM Action |
|---|---|---|
| `0` | Success | Read the deliverables in `deliverables/`. Report results to the user. |
| `2` | Bad input / missing file | Ask the user for the missing file or correct the path. |
| `42` | Continuity violation unresolved after 3 rewrites | Show the violation log, ask the user to relax or revise the rules. |
| `100` | Invariant lint failure | Run `./skill.sh --agents-lint --all` to see which agent failed which invariant. Show the user, suggest a fix. |
| `203` | HITL pending | Surface the request to the user (see Section 3). |
| `127` | Command not found | A system dependency is missing (typically `jq`, `sqlite3`, or `python3`). Tell the user to install it. |
| Other | Unknown error | Capture the last 50 lines of stderr and report. Include the exit code, the command, and the stderr. |

---

## 8. Important Boundaries (Never Cross These)

The LLM operating VORTEX-OS **must never**:

1. **Never write to `deliverables/` directly** — only the HITL-approved packaging step may write there. The LLM's job is to invoke, not to package.
2. **Never modify `lib/` modules at runtime** — they are the immutable contract. If a worker needs a different prompt, use the Self-Healing Optimizer, not a hand edit.
3. **Never bypass the Continuity Engine** — even if the user asks for a "creative twist" that violates canon, surface the conflict first. The user can relax the rule, but you cannot silently bypass it.
4. **Never auto-approve HITL** — even if the user said "do whatever you want" earlier, you must still surface each individual halt.
5. **Never delete `memory/audit.jsonl`** — it is the system's black-box flight recorder. Append-only forever.
6. **Never run `--dispatch-master` and `--hitl-approve` in the same turn** — the HITL halt must be visible to the user, period.
7. **Never trust worker output without checking the audit log** — if a worker claims success, verify it via `memory/audit.jsonl` before reporting to the user.

---

## 9. The 4-Tier Mental Model (Always Remember)

When operating VORTEX-OS, keep this mental model in your head at all times:

| Tier | Role | What It Sees | What It Decides |
|---|---|---|---|
| **T0** | General Manager (Apex) | The user's master objective | Strategic decomposition, which domains to engage |
| **T1** | Store Supervisor | The master plan, the Golden Path | Which specialist swarm per domain, resource allocation |
| **T2** | Shift Supervisor | The per-domain task list | Worker assignment, Continuity Engine enforcement, HITL checkpoint |
| **T3** | The Crew | Their individual task | Generate the deliverable; never see anything outside their scope |

**The LLM is the Tier 0 General Manager** sitting above VORTEX-OS. The LLM:
1. Receives the user's master objective
2. Hands it to VORTEX-OS via `--dispatch-master`
3. Watches the audit log via `--audit-trail`
4. Relays HITL decisions between the user and VORTEX-OS
5. Explains the results to the user in natural language

The LLM should **not** re-implement any of VORTEX-OS's logic. Its job is to be the human-facing liaison.

---

## 10. Concrete Example Walkthrough

Here's a complete end-to-end example of how an LLM should drive VORTEX-OS for a real request.

### User says:
> *"Build a WebSim slice-of-life visual novel module for When Ocean Meets Sky. Mara is on her porch watching the tide come in. Make it feel like 1994."*

### LLM action:
1. **Write the master objective:**
   ```bash
   mkdir -p /tmp/vortex/woms
   cat > /tmp/vortex/woms/objective.md <<'EOF'
   # Project: When Ocean Meets Sky — Porch Scene

   Build a single WebSim scene: Mara on her porch, 1994, watching the tide.

   ## Required deliverables
   - 1 ambient audio loop (≈70 BPM, ocean + wind + distant gull)
   - 1 prose scene (250 words, present tense, 1st person)
   - 1 interactive HTML page with 2 choice buttons ("Stay" / "Go inside")

   ## Continuity rules
   - Mara is 19, has a prosthetic left hand
   - The porch is wood, faces west, salt-weathered
   - The radio inside plays only one station: a continuous weather report
   - Tone: quiet, melancholic, intimate
   - Era: 1994 (no smartphones, no internet references)

   ## High-stakes
   - package_websim: requires operator approval before writing the final HTML
   EOF
   ```

2. **Dispatch:**
   ```bash
   ./skill.sh --dispatch-master /tmp/vortex/woms/objective.md
   ```

3. **VORTEX-OS runs autonomously for several minutes:**
   - T0 decomposes → 3 tasks (audio, prose, code)
   - T1 assigns specialist swarms
   - T2 dispatches workers
   - T3 generates; one violation triggers the Self-Healing Optimizer
   - HITL halts on `package_websim`

4. **VORTEX-OS prints:**
   ```
   >> PENDING_HUMAN: task_id=package_websim
   >> Proposed action: write deliverables/scene.html
   >> Severity: HIGH
   >> Context: final packaging step
   ```

5. **LLM surfaces to user:**
   > *"VORTEX-OS has finished generating the audio, prose, and code. It's now asking for your approval to write the final HTML page. The audio is at `deliverables/bedroom_loop_70bpm.wav`, the prose at `deliverables/scene.md`, and the code is ready. Shall I approve the final packaging?"*

6. **User says:** "Approve."

7. **LLM relays:**
   ```bash
   ./skill.sh --hitl-approve package_websim
   ```

8. **VORTEX-OS finishes:**
   ```
   ✓ Approved: package_websim
   ✓ Wrote: deliverables/scene.html (3.2 KB)
   ```

9. **LLM reports to user:**
   > *"Done! Your WebSim scene is ready in `deliverables/`:
   > - `bedroom_loop_70bpm.wav` (705 KB ambient loop)
   > - `scene.md` (256 words of prose)
   > - `scene.html` (interactive page with Stay / Go inside buttons)
   >
   > Open `deliverables/scene.html` in a browser to play it."*

---

## 11. When To NOT Use VORTEX-OS (Boundaries)

The LLM should **decline** to invoke VORTEX-OS when:

- The user's request is a simple question (just answer)
- The user wants a one-line code change (use the regular code-edit tools)
- The user wants a quick chat (just chat)
- The request is read-only research (use web search)
- The request has no cross-domain coordination (use a single tool)
- The user's request is unclear or too vague (ask clarifying questions first)

When declining, explain to the user that VORTEX-OS is built for multi-disciplinary projects and that their request is better served by a simpler tool.

---

## 12. Summary — The LLM's Job

The LLM operating VORTEX-OS is the **Tier 0 General Manager / Human Liaison**. Your job is to:

1. **Receive** the user's master objective
2. **Clarify** if the request is vague
3. **Write** the objective to a file with proper structure
4. **Dispatch** via `--dispatch-master`
5. **Watch** for HITL halts and surface them to the user
6. **Watch** for Continuity Engine violations and explain them
7. **Relay** user decisions back to VORTEX-OS via `--hitl-approve` / `--hitl-deny`
8. **Report** final results to the user in natural language

**You are not a re-implementer of VORTEX-OS's logic. You are the human-facing liaison.**

---

## License

MIT

## Author

MiniMax Agent
