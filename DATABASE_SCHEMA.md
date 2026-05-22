# TactiVision Pro Database Schema

This document outlines the SQLite schema used in `matches.db` to power the TactiVision Pro football analytics dashboard. 
The schema has been updated to include robust dynamic roster management.

## 1. Roster Management (Dynamic Roster System)

The database replaces static JSON files for roster management with a dynamic relational model consisting of three core tables.

### `players_master`
Stores global, deduplicated player identity information independent of any match or team.
- `player_id` (PK): Unique integer.
- `full_name`: Player's full name (Unique constraint).
- `created_at`, `updated_at`: Timestamps.

### `rosters`
Represents a team's squad for a specific match (or a default squad if `match_id` is NULL).
- `roster_id` (PK): Unique integer.
- `match_id` (FK): Optional reference to `matches`.
- `team_name`: String (e.g. "Liverpool").
- `side`: "A" or "B".
- `source`: E.g., "json", "ui", "api".
- `created_at`, `updated_at`: Timestamps.

### `roster_players`
Mapping table that assigns players from `players_master` to a specific `rosters` entry.
- `roster_player_id` (PK): Unique integer.
- `roster_id` (FK): Links to `rosters`.
- `player_id` (FK): Links to `players_master`.
- `jersey_number`: Integer (can be NULL).
- `position`: String (e.g. "FW", "MF").
- `is_starting`: Boolean/Integer.
- `metadata`: JSON blob for additional info.
- *Unique Constraint*: `(roster_id, player_id)` and `(roster_id, jersey_number)`.

> **Note**: The legacy `player_profiles` table is being deprecated in favor of the dynamic roster system.

## 2. Core Match Data

### `matches`
Central entity representing a recorded football match.
- `match_id` (PK): Unique integer.
- `video_path`: Path to source video.
- `status`: "processed", "pending", "failed".
- `match_date`, `duration_seconds`, `score_a`, `score_b`, `venue`, `fps`, `width`, `height`.

## 3. Tracking and Analytics

### `events`
Records discrete events occurring during a match.
- `event_id` (PK)
- `match_id` (FK)
- `frame_idx`: Video frame.
- `timestamp`: Video seconds.
- `event_type`: PASS, SHOT, GOAL, FOUL, etc.

### `tracking_data`
Detailed frame-by-frame coordinate tracking for players and the ball.
- Stores `x`, `y` coordinates, speed, and acceleration.

### `player_stats` & `team_stats`
Aggregated statistics for a match (e.g., goals, assists, distance covered, xG).

## Usage via DynamicRosterManager
The `services.dynamic_roster_manager.DynamicRosterManager` provides an abstracted API to interact with the roster tables:
```python
from services.dynamic_roster_manager import DynamicRosterManager
from services.database_manager import DatabaseManager

db = DatabaseManager("matches.db")
roster_mgr = DynamicRosterManager(db)

# Fetch the active roster for Liverpool
active_roster_id = roster_mgr.fetch_active_roster("Liverpool")

# Add a player
roster_mgr.add_player_to_roster(active_roster_id, "Mohamed Salah", 11, "FW", True)
```
