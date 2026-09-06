---
title: "RPG Item Database Creator"
description: "Design and build a comprehensive RPG item database with crafting, equipment slots, stats, and inventory management"
category: "game-development"
subcategory: "rpg-systems"
complexity: "high"
estimated_time: "75 minutes"
prerequisites:
  - "Object-oriented programming concepts"
  - "Database schema design"
  - "RPG game mechanics understanding"
  - "UI/UX design principles"
required_inputs:
  - "Item categories and types"
  - "Stat system definitions"
  - "Rarity tiers and colors"
  - "Crafting recipes"
  - "Equipment slot configurations"
output_format: "Complete item system with UI"
tags:
  - "rpg"
  - "item-system"
  - "crafting"
  - "inventory-management"
  - "equipment-slots"
---

# Master Objective: RPG Item Database Creator

## Project Overview

You are building a comprehensive RPG item database system that handles item creation, equipment management, crafting, and inventory operations. This system should support complex item interactions, meaningful progression, and intuitive player interaction with a visually appealing interface.

## Item System Architecture

### 1. Item Categories and Types

**Primary Categories**
- Equipment (weapons, armor, accessories)
- Consumables (potions, food, scrolls)
- Materials (crafting ingredients, resources)
- Quest Items (key items, story objects)
- Currency (gold, gems, tokens)
- Misc (trade goods, junk, collectibles)

**Item Subcategories**
- Weapons: Sword, Axe, Bow, Staff, Dagger, Fist
- Armor: Helmet, Chest, Gloves, Boots, Shield, Cloak
- Accessories: Ring, Amulet, Belt, Trinket
- Potions: Health, Mana, Stamina, Buff, Debuff Cleansing
- Materials: Ore, Herb, Cloth, Leather, Bone, Essence

**Item Properties**
- Unique ID and display name
- Category and subcategory
- Description/flavor text
- Base value and sell price
- Stack size (1 for equipment, higher for materials)
- Required level
- Required class/race (if applicable)
- Bind on pickup/equip/account

### 2. Stat System

**Primary Stats**
- Strength (physical damage, carrying capacity)
- Agility (attack speed, dodge chance)
- Intelligence (magic damage, mana pool)
- Vitality (health, defense)
- Luck (drop rates, critical chance)
- Endurance (stamina, resistance)

**Combat Stats**
- Physical Damage (min-max range)
- Magic Damage (min-max range)
- Defense (physical damage reduction)
- Magic Resistance (magic damage reduction)
- Critical Chance (%)
- Critical Damage (multiplier)
- Attack Speed
- Cast Speed
- Life Steal
- Block Chance

**Defensive Stats**
- Health Points (current/max)
- Mana Points (current/max)
- Health Regeneration
- Mana Regeneration
- Dodge Chance
- Parry Chance
- Block Amount
- Tenacity (crowd control reduction)

**Special Stats**
- Movement Speed
- Experience Bonus
- Gold Bonus
- Item Discovery
- Cooldown Reduction
- Crowd Control Duration

### 3. Rarity and Affix System

**Rarity Tiers**
- Common (white): Basic items, no special properties
- Uncommon (green): 1-2 affixes
- Rare (blue): 2-3 affixes, stat bonuses
- Epic (purple): 3-4 affixes, powerful bonuses
- Legendary (orange): 4-5 affixes, unique effects
- Mythic (red): Special tier, unique items only

**Affix Generation**
- Prefix and Suffix system
- Stat modifiers (+X to Y)
- Special effects (on-hit, on-kill, passive)
- Conditional bonuses (vs. type, in location)
- Synergy between affixes
- Power scaling by item level

**Unique Items**
- Named legendary items with lore
- Fixed affixes (not random)
- Set bonus when multiple collected
- Visual distinction and particle effects
- Cannot be disenchanted/sold

### 4. Equipment Slots

**Core Equipment Slots**
- Head (helmet, hood, crown)
- Chest (armor, robe, tunic)
- Hands (gloves, gauntlets, bracers)
- Legs (pants, skirt, greaves)
- Feet (boots, shoes, sandals)
- Main Hand (weapon, shield)
- Off Hand (shield, orb, quiver)
- Ring 1 and Ring 2
- Amulet (necklace, pendant)
- Cape/Cloak slot

**Slot-Specific Rules**
- One item per slot (replacing current)
- Slot-based item filtering
- Dual wielding options
- Two-handed weapon restrictions
- Off-hand restrictions by class

### 5. Crafting System

**Crafting Categories**
- Blacksmithing (weapons, armor)
- Alchemy (potions, poisons)
- Enchanting (gem socketing, enchanting)
- Cooking (food, buffs)
- Woodworking (bows, staffs)
- Leatherworking (armor, bags)
- Inscription (scrolls, cards)

**Recipe Structure**
- Required materials and quantities
- Required crafting skill level
- Crafting station requirements
- Craft time and cooldown
- Success rate (with failure mechanics)
- Byproduct chance

**Crafting Interface**
- Recipe browser with filters
- Material inventory check
- Craft button with confirmation
- Progress bar during crafting
- Result preview
- Craft history log

### 6. Inventory Management

**Inventory Grid**
- Grid-based inventory system
- Item slots with size (1x1, 2x2, 2x3, etc.)
- Sort options (by type, level, rarity)
- Filter and search functionality
- Stack management for materials
- Currency display

**Bag and Storage**
- Main inventory (fixed size)
- Additional bag slots
- Bank/storage system
- Shared storage between characters
- Item sorting automation
- Junk selling filter

**Item Operations**
- Equip/unequip
- Use (consumables)
- Drop (with confirmation)
- Destroy (with confirmation)
- Trade (with player)
- Mail (send to other characters)
- Vendor sell
- Crafting material extraction

## Database Design

### Data Schema

```javascript
Item {
  id: string (unique identifier)
  name: string
  category: ItemCategory
  subcategory: string
  rarity: Rarity
  level: number
  value: number
  stackSize: number
  bindType: BindType
  equipmentSlot?: EquipmentSlot
  baseStats: Stat[]
  affixes: Affix[]
  icon: string (sprite reference)
  description: string
  flavorText: string
  isUnique: boolean
  setName?: string
}

Affix {
  type: 'prefix' | 'suffix'
  name: string
  stats: StatModifier[]
  conditions?: Condition[]
  rarity: Rarity
}

Recipe {
  id: string
  name: string
  category: CraftingCategory
  materials: { itemId: string, quantity: number }[]
  result: { itemId: string, quantity: number, chance?: number }
  skillLevel: number
  station?: string
  craftTime: number (seconds)
}
```

### Persistence
- Save inventory to localStorage
- Export/import character data
- Auto-save on changes
- Backup system
- Version migration for updates

## User Interface Design

### Equipment Panel
- Character silhouette with slot highlights
- Equipped items displayed in slots
- Stat summary when hovering items
- Compare mode (current vs. new item)
- Equipment set management

### Item Tooltip
- Item name with rarity color
- Item level and requirements
- All stats listed clearly
- Affix display with color coding
- Flavor text in italics
- Sell value and item ID

### Inventory Panel
- Grid-based item display
- Drag and drop functionality
- Right-click context menu
- Multi-select for bulk operations
- Item preview on hover

### Crafting Interface
- Category tabs
- Recipe list with requirements
- Material availability indicators
- Craft button with feedback
- Result animation
- Queue system for bulk crafting

## Features and Interactions

### Item Interactions
- Drag and drop between slots
- Right-click quick equip/use
- Double-click to use consumables
- Shift-click to move single items
- Control-click to select multiple
- Alt-click to show item ID

### Filtering and Sorting
- Filter by category, rarity, level, type
- Sort by name, level, value, rarity
- Search by item name
- Recent items list
- Favorite items marking

### Feedback Systems
- Item acquired notification
- Equipment changed notification
- Stat change summary
- Crafting success/failure message
- Level requirement warning
- Soulbound confirmation

## Technical Implementation

### Architecture
- Modular item system
- Event-driven updates
- Reactive UI binding
- Efficient sprite atlas loading
- Cached calculations for stats

### Performance
- Lazy loading of item sprites
- Efficient tooltip rendering
- Debounced search/filter
- Virtual scrolling for large lists
- Optimized stat recalculation

## Success Criteria

- Complete item database with 50+ unique items
- Working equipment system with all slots
- Functional crafting with multiple recipes
- Smooth inventory management
- Clear stat calculations
- Polished tooltip and UI
- Save/load persistence working
