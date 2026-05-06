# Database schema (v2)

This document captures the **canonical SQLite schema** used by the app (see `services/database_schema.py` → `SCHEMA_SQL`).

## Core entities (match, teams, players, tracking, stats)

```mermaid
erDiagram
  MATCHES ||--o{ TEAMS : has
  MATCHES ||--o{ PLAYERS : has
  MATCHES ||--o{ EVENTS : has
  MATCHES ||--o{ TRACKING_DATA : has
  MATCHES ||--o{ BALL_TRACKING : has
  MATCHES ||--o{ HEATMAPS : has
  MATCHES ||--o{ TEAM_STATS : has
  MATCHES ||--o{ PLAYER_STATS : has

  TEAMS ||--o{ PLAYERS : fields
  TEAMS ||--o{ EVENTS : credited_to
  TEAMS ||--o{ HEATMAPS : scoped_to
  TEAMS ||--|| TEAM_STATS : aggregates

  PLAYER_PROFILES ||--o{ PLAYERS : identity_link
  PLAYER_PROFILES ||--o{ EVENTS : actor_profile
  PLAYER_PROFILES ||--o{ TRACKING_DATA : tracked_profile
  PLAYER_PROFILES ||--o{ PLAYER_STATS : aggregates_profile

  PLAYERS ||--o{ EVENTS : actor_instance
  PLAYERS ||--o{ TRACKING_DATA : tracked_instance
  PLAYERS ||--|| PLAYER_STATS : aggregates_instance
  PLAYERS ||--o{ HEATMAPS : scoped_to
  PLAYERS ||--o{ BALL_TRACKING : possession_context

  MATCHES {
    int match_id PK
    text video_path
    text team_a
    text team_b
    datetime match_date
    real duration_seconds
    int score_a
    int score_b
    text venue
    real fps
    int width
    int height
    bool processed
    datetime created_at
  }

  TEAMS {
    int team_id PK
    int match_id FK
    text side "A|B"
    text name
    text color
    text formation
  }

  PLAYERS {
    int player_instance_id PK
    int match_id FK
    int team_id FK
    int profile_id FK "nullable"
    int jersey_number
    text position
  }

  PLAYER_PROFILES {
    int profile_id PK
    text name
    text team_name
    int jersey_number
    blob face_encoding
    text face_image_path
    date date_of_birth
    text nationality
    text position_default
    int height_cm
    int weight_kg
    datetime created_at
    datetime updated_at
  }

  EVENTS {
    int event_id PK
    int match_id FK
    text event_type
    real timestamp
    int player_instance_id FK "nullable"
    int profile_id FK "nullable"
    int team_id FK "nullable"
    real x "0..1"
    real y "0..1"
    text metadata_json
  }

  TRACKING_DATA {
    int tracking_id PK
    int match_id FK
    int player_instance_id FK "nullable"
    int profile_id FK "nullable"
    int frame_number
    real timestamp
    real x_px
    real y_px
    real x_norm
    real y_norm
    real speed_mps
    bool has_possession
    text team_in_possession
  }

  BALL_TRACKING {
    int ball_id PK
    int match_id FK
    int frame_number
    real timestamp
    real x_px
    real y_px
    real x_norm
    real y_norm
    real velocity_x
    real velocity_y
    real velocity_magnitude
    text detection_method
    real confidence
    int possessing_player_id FK "nullable"
    text possessing_team
  }

  HEATMAPS {
    int heatmap_id PK
    int match_id FK
    int player_instance_id FK "nullable"
    int team_id FK "nullable"
    text heatmap_type
    text file_path
    text data_json
    datetime created_at
  }

  PLAYER_STATS {
    int stats_id PK
    int match_id FK
    int player_instance_id FK
    int profile_id FK "nullable"
    real total_distance_m
    real sprint_distance_m
    real high_intensity_distance_m
    real avg_speed_mps
    real max_speed_mps
    int passes_attempted
    int passes_completed
    int shots
    int shots_on_target
    int goals
    int assists
    int tackles
    int interceptions
    int sprints
    int possession_count
    real possession_duration_s
    int touches
    real workload_score
    real minutes_played
    text heatmap_path
  }

  TEAM_STATS {
    int team_stats_id PK
    int match_id FK
    int team_id FK
    real possession_percentage
    int possession_count
    int passes_attempted
    int passes_completed
    real pass_accuracy
    int shots
    int shots_on_target
    int goals
    real total_distance_m
    int sprints
    real defensive_third_possession
    real midfield_possession
    real attacking_third_possession
  }
```

## Analytics add-ons (xG, highlights, tactical, scouting)

```mermaid
erDiagram
  MATCHES ||--o{ XG_DATA : has
  EVENTS ||--o{ XG_DATA : derives_from

  MATCHES ||--o{ HIGHLIGHTS : has
  MATCHES ||--o{ TACTICAL_PATTERNS : has
  MATCHES ||--o{ PASSING_NETWORKS : has
  MATCHES ||--o{ ZONE_CONTROL : has
  MATCHES ||--o{ ADVANCED_ANALYTICS : has

  OPPONENT_PROFILES ||--o{ MATCHES : "referenced_by(team_a/team_b)"

  XG_DATA {
    int xg_id PK
    int match_id FK
    int event_id FK "nullable"
    real timestamp
    int frame
    int shooter_id "not FK today"
    text shooter_team
    real x
    real y
    text shot_type
    text body_part
    text outcome
    real distance_to_goal
    real angle_to_goal
    real velocity_mps
    bool big_chance
    real xg_value
    text metadata_json
  }

  HIGHLIGHTS {
    int highlight_id PK
    int match_id FK
    text event_type
    real timestamp
    int frame
    int importance
    int primary_player_id "not FK today"
    int secondary_player_id "not FK today"
    text team
    text description
    real clip_start
    real clip_end
    real xg_value
    real velocity
    text metadata_json
  }

  TACTICAL_PATTERNS {
    int pattern_id PK
    int match_id FK
    text team
    text pattern_type
    text outcome
    real start_time
    real end_time
    real duration
    text start_zone
    text end_zone
    int pass_count
    real distance_covered
    real xg_generated
    text players_involved_json
    text description
  }

  OPPONENT_PROFILES {
    int profile_id PK
    text team_name "unique"
    text preferred_formation
    text primary_style
    real pressing_intensity
    real defensive_line_height
    real set_piece_threat
    real avg_pass_length
    text zone_preferences_json
    text key_players_json
    int matches_analyzed
    datetime last_updated
  }

  PASSING_NETWORKS {
    int network_id PK
    int match_id FK
    text team
    int player_id "not FK today"
    real degree_centrality
    real betweenness_centrality
    real closeness_centrality
    text connections_json
  }

  ZONE_CONTROL {
    int control_id PK
    int match_id FK
    text zone
    real team_a_control
    real team_b_control
    real contested
    real team_a_time
    real team_b_time
  }

  ADVANCED_ANALYTICS {
    int analytics_id PK
    int match_id FK
    text team
    text metric_type
    real metric_value
    text metadata_json
  }
```

## Key modeling notes (current behavior)

- **Per-match vs cross-match identity**: `players` are per-match instances; `player_profiles` is cross-match identity. A `players.profile_id` link is optional.
- **“Team” is per match**: `teams` is scoped by `match_id`; do not treat it as a club master table.
- **Loose player references in some tables**: `xg_data.shooter_id`, `highlights.primary_player_id`, `passing_networks.player_id` are **not foreign keys** today (they typically reference the tracker’s internal player ids). If we want strong relational integrity, these should migrate to `player_instance_id` and/or `profile_id`.

