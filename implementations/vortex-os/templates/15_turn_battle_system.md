---
title: "Turn-Based Battle System Designer"
description: "Create a tactical turn-based combat system with abilities, status effects, team management, and AI opponents"
category: "game-development"
subcategory: "combat-systems"
complexity: "high"
estimated_time: "90 minutes"
prerequisites:
  - "Turn-based game mechanics"
  - "State machine implementation"
  - "AI decision making"
  - "Balancing game mechanics"
required_inputs:
  - "Character classes and abilities"
  - "Status effect definitions"
  - "Battle arena layout"
  - "AI behavior patterns"
  - "Victory/defeat conditions"
output_format: "Complete turn-based battle engine"
tags:
  - "turn-based"
  - "tactical-combat"
  - "ability-system"
  - "rpg-combat"
  - "ai-battle"
---

# Master Objective: Turn-Based Battle System Designer

## Project Overview

You are creating a sophisticated turn-based battle system that combines tactical positioning, ability usage, status effects, and strategic AI opponents. The system should feel deep and rewarding while remaining accessible, with clear feedback and meaningful choices in every battle.

## Combat System Architecture

### 1. Turn Order and Initiative

**Turn Structure**
- Initiative-based turn order (speed stat)
- Turn meter or point system
- Group turn options (all allies act, then all enemies)
- Delay mechanics (speed manipulation)
- Turn skip and queue system

**Action Economy**
- Action points per turn (fixed or variable)
- Move points vs. action points
- Free actions vs. costing actions
- Overdrive/limit break mechanics
- Action point recovery

**Turn Flow**
1. Determine turn order
2. Active unit selects action
3. Execute action with animations
4. Apply effects and state changes
5. Check for victory/defeat
6. Next turn begins

### 2. Character Classes and Stats

**Base Character Stats**
- Health Points (HP): Damage capacity
- Mana/Skill Points (MP/SP): Ability resource
- Attack: Physical damage output
- Defense: Physical damage reduction
- Magic Attack: Magical damage output
- Magic Defense: Magical damage reduction
- Speed: Turn order and evasion
- Luck: Critical hits and status chance

**Derived Stats**
- Evasion: Chance to avoid attacks
- Critical Rate: Chance for bonus damage
- Critical Damage: Multiplier for crits
- Accuracy: Chance to hit
- Resistance: Status effect reduction
- Recovery: HP/MP regeneration

**Class System**
- Warrior: High HP, physical attacks, defense
- Mage: High MP, magical attacks, AOE
- Ranger: Balanced, ranged attacks, traps
- Healer: Support, HP restoration, buffs
- Assassin: High crit, single target, stealth
- Tank: Extreme defense, taunt, protection
- Custom classes with unique mechanics

### 3. Ability System

**Ability Types**
- Attack abilities (single target, multi-target)
- Heal abilities (single ally, all allies)
- Buff abilities (stat increases)
- Debuff abilities (stat decreases, status effects)
- Utility abilities (revive, teleport, cleanse)
- Passive abilities (always active)

**Ability Properties**
- Name and description
- Mana/resource cost
- Target type (self, single ally, all allies, single enemy, all enemies)
- Range (melee, ranged, global)
- Element (fire, ice, lightning, dark, light, neutral)
- Hit chance (if variable)
- Critical chance
- Animation/effect type

**Ability Scaling**
- Level scaling (stronger at higher levels)
- Stat scaling (damage based on attack stat)
- Elemental multipliers
- Combo bonuses
- Equipment bonuses

### 4. Status Effects System

**Positive Effects**
- Haste: Increased speed and actions
- Shield: Absorb damage
- Regen: Recover HP over time
- Strengthen: Increased damage
- Barrier: Nullify one attack
- Reflect: Return damage to attacker
- Silence Immunity: Cannot be silenced
- Invincible: Cannot be damaged

**Negative Effects**
- Poison: Damage over time
- Burn: Damage over time, spread
- Freeze: Skip turn, take extra damage when hit
- Sleep: Skip turns until damaged
- Silence: Cannot use abilities
- Blind: Reduced accuracy
- Slow: Reduced speed
- Stun: Cannot act
- Curse: Stat reduction over time
- Bleed: Stackable damage over time

**Effect Mechanics**
- Duration (turns or time)
- Stacking rules (intensity vs. duration refresh)
- Priority system (which effect applies)
- Clear conditions (cure, expire, battle end)
- Resistance based on stats

### 5. Targeting and Positioning

**Battle Arena**
- Grid-based (tactical) or free-form
- Position affects ability range
- Flanking bonuses
- Elevation advantages
- Environmental hazards

**Targeting Rules**
- Must be in range
- Must have line of sight (for some abilities)
- Cannot target self (unless specified)
- Cannot target dead units
- AoE positioning requirements

### 6. AI Opponent System

**AI Decision Making**
- Evaluate all possible actions
- Calculate action value/score
- Consider threat assessment
- Factor in current battle state
- Add randomization for variety

**AI Behaviors**
- Aggressive: Focus on damage
- Defensive: Focus on survival
- Support: Prioritize healing
- Balanced: Mix of approaches
- Tactical: Use terrain and combos
- Boss: Special patterns and phases

**AI Difficulty Levels**
- Easy: Predictable, sub-optimal choices
- Normal: Competent decisions
- Hard: Optimal play, reads player
- Expert: Anticipates player strategies

### 7. Victory and Defeat

**Victory Conditions**
- Reduce all enemies to 0 HP
- Survive for X turns
- Complete objective (protect NPC, break object)
- Collect all objectives
- Defeat specific enemy (boss)

**Defeat Conditions**
- All player units reach 0 HP
- Objective unit reaches 0 HP
- Player unit reaches 0 HP (if not full party)
- Exceed turn limit
- Fail objective condition

**Battle Rewards**
- Experience points
- Gold/currency
- Items dropped
- Story/quest progression
- Score/rating (stars)

## Battle UI Design

### Battle Screen Layout
- Party status on one side
- Enemy display area
- Action selection panel
- Current turn indicator
- Battle log/feed
- Menu access button

### Action Selection
- Attack option
- Abilities menu (categorical)
- Items menu
- Defend/skip turn
- Flee/escape option
- Target selection grid

### Unit Display
- HP and resource bars
- Status effect icons
- Level and class indicator
- Buff/debuff timers
- Animation states

### Feedback Systems
- Damage numbers (color coded)
- Critical hit indicators
- Status effect application
- Miss notifications
- Heal numbers
- Status effect icons

## Combat Flow Implementation

### State Machine

```
BATTLE_STATES:
- INIT: Setup battle, place units
- PLAYER_TURN: Player selects action
- TARGETING: Player selects target
- EXECUTE_ACTION: Perform selected action
- ANIMATION: Play action animation
- APPLY_EFFECTS: Calculate and apply damage/healing
- CHECK_STATUS: Update unit states
- ENEMY_TURN: AI selects and executes action
- CHECK_VICTORY: Test win conditions
- CHECK_DEFEAT: Test lose conditions
- BATTLE_END: Show results, distribute rewards
```

### Damage Calculation

```
Base Damage = Attacker.ATK - Defender.DEF (minimum 1)
Elemental Modifier = Element effectiveness (0.5, 1.0, 1.5, 2.0)
Critical Hit = if (rand < Attacker.CRIT) then 2.0 else 1.0
Random Factor = 0.9 to 1.1
Final Damage = Base × Elemental × Critical × Random × Modifiers
```

### Healing Calculation

```
Base Heal = Spell Power
Overheal Cap = Unit's Max HP (or allow overflow)
Final Heal = min(Base Heal, Max HP - Current HP) × Modifiers
```

## Technical Requirements

### Performance
- Smooth animations at 60fps
- Efficient AI calculations
- Quick state transitions
- Responsive UI during animations
- Memory efficient status effects

### Accessibility
- Color blind friendly damage colors
- Screen reader support
- Keyboard navigation
- Adjustable text size
- Reduced motion option

## Quality Standards

### Balance
- No dominant strategy
- All classes viable
- Clear counters exist
- Progression feels fair
- Difficulty scales appropriately

### Polish
- Satisfying hit feedback
- Clear status information
- Intuitive interface
- Smooth animations
- Audio/visual cues

## Success Criteria

- Working turn-based combat loop
- Multiple character classes
- 10+ unique abilities
- 5+ status effects
- AI opponents with behavior
- Victory/defeat conditions
- Battle UI with all elements
- Save battle progress option
