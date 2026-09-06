# TEMPLATE: Audio Drama Episode
**Domain:** Narrative & Worldbuilding | **Complexity:** Medium-High | **Agents:** writer, media.native, coder.typescript

---

## WHAT THIS DOES
Produces a complete audio drama: a multi-voice script, sound effects, an ambient music bed, and a playable HTML page with a credits screen.

---

## FILL IN THESE FIELDS

**EPISODE TITLE:** `________________________`
**SERIES NAME:** `________________________`
**EPISODE NUMBER:** `____`

**VOICE 1 — NAME:** `_____________`
**VOICE 1 — DESCRIPTION:** (e.g. weathered detective, 50s, tired but sharp)
`_______________________________________________________________________`

**VOICE 2 — NAME:** `_____________`
**VOICE 2 — DESCRIPTION:** (e.g. nervous witness, 30s, speaks fast)
`_______________________________________________________________________`

**VOICE 3 — NAME:** `_____________` (if needed)
**VOICE 3 — DESCRIPTION:**
`_______________________________________________________________________`

**SETTING:** Where and when does this take place?
`_______________________________________________________________________`

**EPISODE SUMMARY:** What happens in this episode? (2-4 sentences)
`_______________________________________________________________________`

**TONE:** (e.g. noir tension, supernatural dread, heart-pounding action)
`_______________________________________________________________________`

**DURATION:** `____` minutes (target runtime)

**SOUND EFFECTS NEEDED:** (list 3-5 required SFX)
1. `___________________________________________________________________`
2. `___________________________________________________________________`
3. `___________________________________________________________________`
4. `___________________________________________________________________`
5. `___________________________________________________________________`

**MUSIC MOOD:** (e.g. tense strings, bass-heavy drone, upbeat jazz)
`_______________________________________________________________________`

**ERA:** (e.g. 1940s radio drama, modern day, far future)
`_______________________________________________________________________`

**CONTINUITY RULES:** Any established series rules that must not be broken?
- `___________________________________________________________________`
- `___________________________________________________________________`

---

## WHAT YOU GET BACK
- `episode_script.txt` — Full multi-voice script
- `sfx_pack.wav` — All sound effects in one file
- `music_bed.wav` — Ambient music loop
- `player.html` — Playable HTML page with play/pause, credits, and scrubber

---

## HOW TO RUN
```bash
./skill.sh --dispatch-master templates/02_audio_drama_episode.md
```
