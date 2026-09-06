# TEMPLATE: Interactive CYOA Book
**Domain:** Narrative & Worldbuilding | **Complexity:** High | **Agents:** writer, media.native, coder.typescript

---

## WHAT THIS DOES
Creates a full choose-your-own-adventure book: multiple story nodes, branching paths, an item tracker, and an interactive HTML reader with save/load states.

---

## FILL IN THESE FIELDS

**STORY TITLE:** `________________________`
**GENRE:** (e.g. sci-fi mystery, fantasy quest, horror survival)
`_______________________________________________________________________`

**PROTAGONIST NAME:** `_____________`
**PROTAGONIST BACKSTORY:** (1-2 sentences)
`_______________________________________________________________________`

**STARTING LOCATION:** `________________________________________________________________`

**NUMBER OF MAJOR BRANCHES:** `____` (minimum 2)

**ENDING A — TITLE:** `________________________`
**ENDING A — DESCRIPTION:** How does this ending feel?
`_______________________________________________________________________`

**ENDING B — TITLE:** `________________________`
**ENDING B — DESCRIPTION:** How does this ending feel?
`_______________________________________________________________________`

**ENDING C — TITLE:** `________________________` (optional)
**ENDING C — DESCRIPTION:**
`_______________________________________________________________________`

**ITEMS THE PLAYER CAN FIND:** (list 3-5)
1. `___________________________________________________________________`
2. `___________________________________________________________________`
3. `___________________________________________________________________`
4. `___________________________________________________________________`
5. `___________________________________________________________________`

**TONE:** (e.g. darkly comic, suspenseful, whimsical)
`_______________________________________________________________________`

**TARGET WORD COUNT:** `____` words total across all nodes

**AUDIO THEME SONG:** (describe or give reference)
`_______________________________________________________________________`

**CONTINUITY RULES:**
- `___________________________________________________________________`
- `___________________________________________________________________`

---

## WHAT YOU GET BACK
- `cyoa_nodes.json` — All story nodes with branching logic
- `items.json` — Item tracker definitions
- `theme_song.wav` — Procedurally generated theme track
- `reader.html` — Full HTML reader with save/load, item tracker, and chapter selector

---

## HOW TO RUN
```bash
./skill.sh --dispatch-master templates/03_cyoa_book.md
```
