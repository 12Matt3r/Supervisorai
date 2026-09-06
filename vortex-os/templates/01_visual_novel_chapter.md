# TEMPLATE: Visual Novel Chapter
**Domain:** Narrative & Worldbuilding | **Complexity:** Medium | **Agents:** writer, media.native, coder.typescript

---

## WHAT THIS DOES
Builds a single scene of a visual novel: prose, ambient audio, and an interactive HTML page with branching choice buttons. All continuity rules are enforced automatically.

---

## FILL IN THESE FIELDS

**PROJECT TITLE:** `________________________`

**SETTING:** Where and when does this scene take place?
`_______________________________________________________________________`

**PROTAGONIST NAME:** `_____________`

**PROTAGONIST DETAILS:** (age, appearance, key trait)
`_______________________________________________________________________`

**SCENE SUMMARY:** What happens in this one scene? (1-3 sentences)
`_______________________________________________________________________`

**CHARACTER 2 NAME:** `_____________` (if any)

**CHARACTER 2 ROLE:** (friend, stranger, antagonist)
`_______________________________________________________________________`

**TONE:** (e.g. quiet and melancholic, tense and mysterious, warm and hopeful)
`_______________________________________________________________________`

**ERA/GENRE:** (e.g. 1990s coastal town, cyberpunk neo-Tokyo, high fantasy)
`_______________________________________________________________________`

**AUDIO MOOD:** (e.g. ocean wind, city rain, empty hallway with flickering lights)
`_______________________________________________________________________`

**NUMBER OF CHOICES:** `____` (typically 2)

**CHOICE A — TEXT:** `_____________________________________________`
**CHOICE A — RESULT:** What happens if the player picks this?
`_______________________________________________________________________`

**CHOICE B — TEXT:** `_____________________________________________`
**CHOICE B — RESULT:** What happens if the player picks this?
`_______________________________________________________________________`

**CONTINUITY RULES:** Any rules about this world that must not be broken?
- `___________________________________________________________________`
- `___________________________________________________________________`
- `___________________________________________________________________`

**HIGH-STAKES:** Which task should require your approval before writing the final HTML?
- Default: `package_websim`

---

## WHAT YOU GET BACK
- `scenes.txt` — Full prose scene (300-500 words)
- `bedroom_loop_[bpm].wav` — Ambient audio loop (native minimax-music)
- `scene.html` — Interactive HTML page with your 2 choice buttons
- `plan.json` — Saved as a Golden Path template for reuse

---

## HOW TO RUN
```bash
./skill.sh --dispatch-master templates/01_visual_novel_chapter.md
```
