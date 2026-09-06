# TEMPLATE: Screenplay Scene
**Domain:** Narrative & Worldbuilding | **Complexity:** Medium | **Agents:** writer, media.native

---

## WHAT THIS DOES
Writes a single polished screenplay scene with proper formatting, a mood-board image manifest, and a shot-list JSON for the director or cinematographer.

---

## FILL IN THESE FIELDS

**SCENE TITLE:** `________________________`
**SERIES/FILM TITLE:** `________________________`

**LOCATION:** (INT. or EXT. — where and when)
`_______________________________________________________________________`

**CHARACTERS IN SCENE:** (list all with brief descriptions)
1. `_____________` — `________________________________________________________________`
2. `_____________` — `________________________________________________________________`
3. `_____________` — `________________________________________________________________`

**WHAT HAPPENS:** Describe the scene beat by beat (3-6 beats)
`_______________________________________________________________________`
`_______________________________________________________________________`
`_______________________________________________________________________`
`_______________________________________________________________________`

**TONE:** (e.g. tense, comic relief, romantic, horrifying)
`_______________________________________________________________________`

**CAMERA MOOD:** (e.g. intimate close-ups, wide and isolating, handheld and urgent)
`_______________________________________________________________________`

**ESTABLISHING SHOT:** Describe the opening image
`_______________________________________________________________________`

**SCENE MOOD MUSIC:** (describe or give reference)
`_______________________________________________________________________`

**TARGET PAGE COUNT:** `____` pages (1 page ≈ 1 minute)

**ERA:** (e.g. contemporary, 1970s noir, near-future dystopia)
`_______________________________________________________________________`

**CONTINUITY RULES:** (any established rules for this world)
- `___________________________________________________________________`
- `___________________________________________________________________`

---

## WHAT YOU GET BACK
- `scene.txt` — Properly formatted screenplay (industry standard format)
- `mood_board.json` — Image prompt manifest for visual references
- `shot_list.json` — Shot-by-shot breakdown with camera directions and timestamps

---

## HOW TO RUN
```bash
./skill.sh --dispatch-master templates/05_screenplay_scene.md
```
