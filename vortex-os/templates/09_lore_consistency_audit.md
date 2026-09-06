# TEMPLATE: Lore Consistency Audit
**Domain:** Narrative & Worldbuilding | **Complexity:** Medium | **Agents:** writer, inspector.governance

## WHAT THIS DOES
Reads your manuscript or world-building doc and audits every scene for continuity violations: wrong character details, timeline breaks, anachronisms, rule violations, and item ownership errors.

## FILL IN
**UNIVERSE NAME:** ________________________
**PROJECT TYPE:** (novel manuscript / visual novel / tabletop RPG / TV series)
**CANON DOC:** Path to your lore/world document ________________________
**MANUSCRIPT:** Path to your content ________________________
**CHARACTER RULES:** (define your canon facts)
1. ________________________
2. ________________________
3. ________________________
4. ________________________
**TIMELINE RULES:**
1. ________________________
2. ________________________
3. ________________________
**FORBIDDEN ELEMENTS:** (things that should NEVER appear)
1. ________________________
2. ________________________
**TONE RULES:**
1. ________________________

## WHAT YOU GET BACK
- lore_audit_report.md — Scene-by-scene violation report
- lore_violations.json — Structured violation data
- lore_fix_suggestions.txt — Per-violation rewrite suggestions

## RUN
```bash
./skill.sh --dispatch-master templates/09_lore_consistency_audit.md
```
