# TEMPLATE: Show Bible Generator
**Domain:** Narrative & Worldbuilding | **Complexity:** High | **Agents:** writer, analyst.strategic

---

## WHAT THIS DOES
Auto-compiles a complete TV/streaming show bible: character arcs, episode loglines, world rules, visual tone, and a series arc plan. Designed for When Ocean Meets Sky, The Book of the Fading Age, or any original series.

---

## FILL IN THESE FIELDS

**SERIES TITLE:** `________________________`
**FORMAT:** (e.g. animated series, live-action drama, limited series, ongoing)
`_______________________________________________________________________`

**EPISODE COUNT:** `____` (e.g. 10-episode season, 26-episode ongoing)

**EPISODE RUNTIME:** `____` minutes per episode

**GENRE:** (e.g. coming-of-age drama, sci-fi mystery, animated comedy)
`_______________________________________________________________________`

**PILOT SUMMARY:** What happens in the first episode? (2-3 sentences)
`_______________________________________________________________________`

**SERIES ARC:** What is the big question or conflict that runs through the entire series?
`_______________________________________________________________________`

**CHARACTER 1 — NAME:** `_____________`
**CHARACTER 1 — ROLE:** (protagonist, antagonist, supporting)
`_______________________________________________________________________`
**CHARACTER 1 — ARC:** What personal change does this character undergo across the season?
`_______________________________________________________________________`

**CHARACTER 2 — NAME:** `_____________`
**CHARACTER 2 — ARC:**
`_______________________________________________________________________`

**CHARACTER 3 — NAME:** `_____________`
**CHARACTER 3 — ARC:**
`_______________________________________________________________________`

**CHARACTER 4 — NAME:** `_____________`
**CHARACTER 4 — ARC:**
`_______________________________________________________________________`

**WORLD RULES:** (3-5 rules that define how this universe works)
1. `___________________________________________________________________`
2. `___________________________________________________________________`
3. `___________________________________________________________________`
4. `___________________________________________________________________`
5. `___________________________________________________________________`

**TONE:** (e.g. quietly devastating, propulsive thriller, warm and observational)
`_______________________________________________________________________`

**VISUAL REFERENCE:** (e.g. Wes Anderson color palette, Blade Runner neon, Tarkovsky stillness)
`_______________________________________________________________________`

**AUDIO IDENTITY:** (e.g. sparse piano, electronic score, period-appropriate folk music)
`_______________________________________________________________________`

**CONTINUITY RULES:** (any established lore that must not be broken)
- `___________________________________________________________________`
- `___________________________________________________________________`

---

## WHAT YOU GET BACK
- `show_bible.md` — Complete show bible (30-50 pages)
- `episode_loglines.json` — All episode titles and one-sentence summaries
- `character_arcs.json` — Full season-long character development plans
- `world_rules.md` — The universe rulebook
- `series_timeline.json` — Episode-by-episode arc tracker

---

## HOW TO RUN
```bash
./skill.sh --dispatch-master templates/08_show_bible_generator.md
```
