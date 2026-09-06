---
title: "Procedural Dungeon Generator"
description: "Build an algorithm that creates unique, interconnected dungeons with rooms, corridors, enemies, and loot"
category: "game-development"
subcategory: "procedural-generation"
complexity: "high"
estimated_time: "60 minutes"
prerequisites:
  - "Graph theory basics"
  - "Tile-based map representation"
  - "Random number generation"
  - "Pathfinding algorithms"
required_inputs:
  - "Dungeon theme and aesthetic"
  - "Room types and sizes"
  - "Corridor connection rules"
  - "Enemy and loot distribution"
  - "Difficulty scaling parameters"
output_format: "Playable dungeon crawler"
tags:
  - "procedural-generation"
  - "roguelike"
  - "dungeon-crawler"
  - "pathfinding"
  - "game-algorithms"
---

# Master Objective: Procedural Dungeon Generator

## Project Overview

You are creating a procedural dungeon generation system that produces unique, explorable dungeons every time. The generator should create interesting layouts with connected rooms, strategic corridors, distributed enemies, and meaningful loot placement while ensuring all areas are reachable and completable.

## Algorithm Architecture

### 1. Grid and Tile System

**Map Representation**
- 2D grid system for tile-based layout
- Tile types: floor, wall, door, corridor, stairs, trap, water, lava
- Room boundaries and connectivity information
- Fog of war data structure
- Navigation mesh for pathfinding

**Tile Properties**
- Walkability flags
- Transparency for line of sight
- Interaction properties
- Visual variation per tile type
- Damage/effect values for hazards

### 2. Room Generation

**Room Types**
- Starting room (larger, safe)
- Combat rooms (varying sizes)
- Treasure rooms (small, rewarding)
- Puzzle rooms (special mechanics)
- Boss rooms (large, climactic)
- Secret rooms (hidden, valuable)
- Rest rooms (healing, save points)

**Room Templates**
- Predefined room shapes (rectangle, L-shape, T-shape, cross)
- Minimum and maximum dimensions
- Door placement rules
- Feature placement within rooms
- Enemy spawn point positions

**Room Placement Algorithm**
- BSP (Binary Space Partitioning) approach
- Random placement with collision detection
- Room connection graph
- Density control per dungeon level
- Special room placement rules (boss far from start)

### 3. Corridor System

**Corridor Generation**
- Connect rooms using MST (Minimum Spanning Tree) plus extras
- L-shaped and straight corridor options
- Tunnel carving through walls
- Corridor width options
- Dead-end corridor creation for exploration reward

**Connectivity Requirements**
- All rooms must be reachable
- No isolated areas
- Multiple paths between important rooms
- Loop creation for backtracking
- Emergency escape routes (optional)

### 4. Enemy Distribution

**Enemy Spawning Rules**
- Enemy budget per room based on room size
- Enemy type selection by dungeon depth
- Clustering behavior (groups vs. solo)
- Elite enemy placement in key locations
- Boss enemy specific spawn rules

**Difficulty Scaling**
- Enemy count increases with depth
- Enemy strength scales with level
- Elite frequency increases
- New enemy types unlock at depths
- Guaranteed variety within a floor

**Spawn Points**
- Away from room entrances
- Behind cover/obstacles
- Near treasure for risk/reward
- Strategic positions for gameplay
- Respawn rules after combat

### 5. Loot and Reward System

**Loot Tables**
- Item rarity tiers (common, uncommon, rare, epic, legendary)
- Type distribution (weapons, armor, consumables, keys)
- Level-appropriate item scaling
- Theme-specific items per dungeon type
- Guaranteed drops vs. random chance

**Placement Strategy**
- Treasure rooms have guaranteed loot
- Enemy drops on death
- Hidden containers (chests, crates, barrels)
- Secret room bonus loot
- Boss kill rewards
- Exploration discovery bonuses

### 6. Dungeon Themes and Variation

**Theme Elements**
- Visual tile sets per theme (dungeon, cave, castle, temple, sewer)
- Thematic enemy types
- Environmental hazards
- Decorative objects
- Ambient music/sounds per theme

**Procedural Variation**
- Random seed for reproducibility
- Guaranteed seeds for sharing
- Difficulty modifiers
- Size variations
- Room density variations

## Rendering and Visualization

### 1. Visual Display

**Rendering Approach**
- Canvas 2D or WebGL rendering
- Tile-based sprite rendering
- Layer system (floor, objects, entities, effects)
- Camera following player
- Smooth scrolling/tiling

**Visual Style**
- Top-down perspective
- Isometric view option
- Consistent pixel art style
- Clear tile differentiation
- Fog of war visibility

### 2. UI Elements

**Minimap**
- Real-time exploration tracking
- Room shapes and connections
- Current position indicator
- Discovered vs. undiscovered areas
- Toggle visibility option

**HUD Display**
- Player health/status
- Current floor/level
- Inventory access
- Minimap toggle
- Menu access

**Interaction System**
- Click/tap to move
- Keyboard navigation (WASD/arrows)
- Combat targeting system
- Item pickup prompts
- Door/interact prompts

## Player Mechanics

### Movement and Controls
- Grid-based movement
- Smooth animation between tiles
- Collision detection
- Stair climbing to next level
- Teleportation mechanics (keys/powers)

### Combat System
- Turn-based or real-time options
- Attack range and line of sight
- Damage calculation
- Enemy AI behaviors
- Death and respawn system

### Progression
- Character leveling
- Equipment upgrades
- New abilities/powers
- Persistent stats across dungeons
- Score/grade system

## Technical Implementation

### Data Structures

```
Dungeon {
  width: number
  height: number
  tiles: Tile[][]
  rooms: Room[]
  corridors: Corridor[]
  enemies: Enemy[]
  items: Item[]
  seed: string
}

Room {
  id: string
  x, y: number
  width, height: number
  type: RoomType
  doors: Door[]
  features: Feature[]
}

Corridor {
  startRoom: string
  endRoom: string
  path: Point[]
  width: number
}
```

### Algorithms to Implement

1. **BSP Tree Builder**: Divide space recursively
2. **Room Placer**: Position rooms in BSP leaves
3. **MST Connector**: Connect rooms efficiently
4. **A* Pathfinding**: Route corridors around obstacles
5. **Line of Sight**: Calculate visibility
6. **Spawn Point Selector**: Place enemies/items intelligently

### Performance Considerations

- Generate dungeons in chunks if large
- Cache tile sprites
- Efficient rendering (only visible tiles)
- Background generation for next floor
- Memory management for large dungeons

## Quality Standards

### Playability
- Every dungeon must be completable
- All loot must be reachable
- No stuck states or impossible situations
- Clear objectives and progression
- Fair difficulty curve

### Variety
- No two dungeons should feel identical
- Meaningful differences between runs
- Interesting decision-making opportunities
- Multiple viable strategies
- Replayability guarantee

### Polish
- Smooth animations and transitions
- Clear visual feedback
- Intuitive controls
- Helpful UI indicators
- Atmospheric audio

## Success Criteria

- Generates valid dungeon in under 1 second
- All rooms connected properly
- No isolated tiles or rooms
- Balanced enemy and loot distribution
- Visually distinct per theme
- Complete player traversal possible
