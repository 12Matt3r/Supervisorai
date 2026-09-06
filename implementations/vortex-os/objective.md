# Project: When Ocean Meets Sky — Slice of Life Visual Novel Module

**Objective:** Build a fully packaged, interactive "Slice of Life" visual novel module for *When Ocean Meets Sky*, running entirely on local HTML/JS.

## Phase 1: Narrative & Continuity

Write a two-part branching dialogue script between Kai and Sora.

- **Scene A:** Takes place in The Talk Pit (a simple guest intro area with a park bench and grass). They discuss a strange VHS tape they found.
- **Scene B:** Takes place in Kai's bedroom later that night. Kai drops the tape, and it rolls out of sight.

*(Note for writer.narrative: Adhere strictly to these physical laws. No fiery pits, and nothing emerges from under the bed).*

## Phase 2: Media & Atmosphere

Trigger the native minimax-music engine to generate the background audio track for the bedroom scene. The generation payload must specify a fusion of slushwave and post-neuro vapor, sitting around 85 BPM, designed as a seamless, ethereal loop. Save the resulting native audio artifact to the deliverables folder.

## Phase 3: Code & Procedural Mechanics

Write the HTML, CSS, and modular JavaScript for the local deployment. The interface must feature a dialogue box and procedural retro-futuristic aesthetics (specifically, CRT glitches and VHS degradation effects that trigger randomly when the user clicks to advance the dialogue).

## Required Deliverables

- `deliverables/scene_a.json` — Branching dialogue script for Scene A
- `deliverables/scene_b.json` — Branching dialogue script for Scene B
- `deliverables/bedroom_loop.wav` — Native audio track (85 BPM slushwave/post-neuro vapor loop)
- `deliverables/websim_vn.html` — Complete self-contained HTML/JS/CSS visual novel

## Continuity Rules

- The Talk Pit: outdoor, park bench, grass, natural light, afternoon
- Kai's bedroom: night, window with moonlight, analog electronics aesthetic, VHS tape present
- No smartphones, no internet references, no anachronistic tech
- Kai is the introspective one, Sora is the curious one
- VHS tape is mysterious but not supernatural — it rolls under furniture naturally

## High-Stakes

- `package_websim`: requires HITL operator approval before final HTML write to deliverables
