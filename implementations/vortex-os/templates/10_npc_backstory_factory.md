# TEMPLATE: NPC Backstory Factory
**Domain:** Narrative & Worldbuilding | **Complexity:** Low | **Agents:** writer

## WHAT THIS DOES
Generates a batch of unique NPCs for tabletop RPGs, visual novels, or game projects — each with a name, secret, motivation, visual description, and personality hooks.

## FILL IN
**SETTING NAME:** ________________________
**SETTING DESCRIPTION:** ________________________
**NUMBER OF NPCs:** ____ (recommended: 5-20)
**NPC CATEGORY:** (check all)
- Townspeople / Common folk
- Merchants / Traders
- Nobility / Authority figures
- Rivals / Antagonists
- Quest givers / Key story characters
- Mysterious strangers
- Creatures / Non-human entities
**MORAL TONE:** ________________________
**TECH LEVEL:** ________________________
**SECRET TYPES:** (check all)
- Hidden relationships (family, romantic, rivalries)
- Dark pasts / criminal histories
- Supernatural secrets
- Economic secrets / debts
- Political affiliations
- Magical cursed knowledge
**CONTINUITY RULES:** ________________________

## WHAT YOU GET BACK
- npcs.json — All NPCs in structured JSON
- npcs.md — Human-readable formatted NPC profiles
- npc_portrait_prompts.json — AI image prompts for each character
- faction_map.json — NPC relationship web

## RUN
```bash
./skill.sh --dispatch-master templates/10_npc_backstory_factory.md
```
