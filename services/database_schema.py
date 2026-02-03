"""
Comprehensive Database Schema for Football Analytics

Defines the complete database schema for the football analytics system.
Uses SQLite for simplicity and portability.

Tables:
- matches: Match metadata
- teams: Team information per match
- players: Player profiles with face encoding support
- player_stats: Player statistics across matches
- events: Match events (passes, shots, goals, etc.)
- tracking_data: Frame-by-frame player positions for heatmaps
- ball_tracking: Ball position tracking data
- heatmaps: Heatmap references
- player_profiles: Cross-match player profiles with face recognition
"""

# SQL Schema for creating all tables
SCHEMA_SQL = """
-- ============================================================
-- MATCHES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT NOT NULL,
    team_a TEXT NOT NULL,
    team_b TEXT NOT NULL,
    match_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_seconds REAL DEFAULT 0,
    score_a INTEGER DEFAULT 0,
    score_b INTEGER DEFAULT 0,
    venue TEXT,
    fps REAL DEFAULT 25.0,
    width INTEGER DEFAULT 1280,
    height INTEGER DEFAULT 720,
    processed BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TEAMS TABLE (per match)
-- ============================================================
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    side TEXT NOT NULL, -- 'A' or 'B'
    name TEXT NOT NULL,
    color TEXT,
    formation TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    UNIQUE(match_id, side)
);

-- ============================================================
-- PLAYERS TABLE (per match instance)
-- ============================================================
CREATE TABLE IF NOT EXISTS players (
    player_instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    profile_id INTEGER, -- Links to player_profiles for cross-match tracking
    jersey_number INTEGER,
    position TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES player_profiles(profile_id) ON DELETE SET NULL
);

-- ============================================================
-- PLAYER PROFILES TABLE (cross-match identity)
-- ============================================================
CREATE TABLE IF NOT EXISTS player_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team_name TEXT,
    jersey_number INTEGER,
    face_encoding BLOB, -- Serialized face embedding for recognition
    face_image_path TEXT,
    date_of_birth DATE,
    nationality TEXT,
    position_default TEXT,
    height_cm INTEGER,
    weight_kg INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- PLAYER STATS TABLE (aggregated statistics per match)
-- ============================================================
CREATE TABLE IF NOT EXISTS player_stats (
    stats_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    player_instance_id INTEGER NOT NULL,
    profile_id INTEGER,
    
    -- Distance metrics
    total_distance_m REAL DEFAULT 0,
    sprint_distance_m REAL DEFAULT 0,
    high_intensity_distance_m REAL DEFAULT 0,
    
    -- Speed metrics
    avg_speed_mps REAL DEFAULT 0,
    max_speed_mps REAL DEFAULT 0,
    
    -- Event counts
    passes_attempted INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    sprints INTEGER DEFAULT 0,
    
    -- Possession metrics
    possession_count INTEGER DEFAULT 0,
    possession_duration_s REAL DEFAULT 0,
    touches INTEGER DEFAULT 0,
    
    -- Workload metrics
    workload_score REAL DEFAULT 0,
    minutes_played REAL DEFAULT 0,
    
    -- Heatmap reference
    heatmap_path TEXT,
    
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (player_instance_id) REFERENCES players(player_instance_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES player_profiles(profile_id) ON DELETE SET NULL,
    UNIQUE(match_id, player_instance_id)
);

-- ============================================================
-- EVENTS TABLE (passes, shots, goals, etc.)
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    event_type TEXT NOT NULL, -- 'pass', 'shot', 'goal', 'tackle', 'sprint_start', 'sprint_end', etc.
    timestamp REAL NOT NULL, -- seconds from match start
    player_instance_id INTEGER,
    profile_id INTEGER,
    team_id INTEGER,
    
    -- Position data
    x REAL, -- normalized 0-1
    y REAL, -- normalized 0-1
    
    -- Event metadata as JSON
    metadata TEXT, -- JSON with event-specific data
    
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (player_instance_id) REFERENCES players(player_instance_id) ON DELETE SET NULL,
    FOREIGN KEY (profile_id) REFERENCES player_profiles(profile_id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE SET NULL
);

-- ============================================================
-- TRACKING DATA TABLE (frame-by-frame positions)
-- ============================================================
CREATE TABLE IF NOT EXISTS tracking_data (
    tracking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    player_instance_id INTEGER,
    profile_id INTEGER,
    frame_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    
    -- Position (pixel coordinates)
    x_px REAL NOT NULL,
    y_px REAL NOT NULL,
    
    -- Normalized position (0-1)
    x_norm REAL,
    y_norm REAL,
    
    -- Speed at this frame
    speed_mps REAL,
    
    -- Team possession context
    has_possession BOOLEAN DEFAULT 0,
    team_in_possession TEXT,
    
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (player_instance_id) REFERENCES players(player_instance_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES player_profiles(profile_id) ON DELETE SET NULL
);

-- ============================================================
-- BALL TRACKING TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS ball_tracking (
    ball_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    
    -- Position
    x_px REAL,
    y_px REAL,
    x_norm REAL,
    y_norm REAL,
    
    -- Velocity
    velocity_x REAL,
    velocity_y REAL,
    velocity_magnitude REAL,
    
    -- Detection method
    detection_method TEXT, -- 'yolo', 'color', 'predicted'
    confidence REAL,
    
    -- Possession context
    possessing_player_id INTEGER,
    possessing_team TEXT,
    
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (possessing_player_id) REFERENCES players(player_instance_id) ON DELETE SET NULL
);

-- ============================================================
-- HEATMAPS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS heatmaps (
    heatmap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    player_instance_id INTEGER,
    team_id INTEGER,
    heatmap_type TEXT NOT NULL, -- 'player', 'team', 'global', 'ball'
    file_path TEXT NOT NULL,
    data_json TEXT, -- Optional: store heatmap data as JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (player_instance_id) REFERENCES players(player_instance_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
);

-- ============================================================
-- TEAM STATS TABLE (aggregated team statistics per match)
-- ============================================================
CREATE TABLE IF NOT EXISTS team_stats (
    team_stats_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    
    -- Possession
    possession_percentage REAL DEFAULT 50.0,
    possession_count INTEGER DEFAULT 0,
    
    -- Passing
    passes_attempted INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    pass_accuracy REAL DEFAULT 0,
    
    -- Shooting
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    
    -- Physical
    total_distance_m REAL DEFAULT 0,
    sprints INTEGER DEFAULT 0,
    
    -- Zone possession
    defensive_third_possession REAL DEFAULT 0,
    midfield_possession REAL DEFAULT 0,
    attacking_third_possession REAL DEFAULT 0,
    
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
    UNIQUE(match_id, team_id)
);

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team_a, team_b);
CREATE INDEX IF NOT EXISTS idx_players_match ON players(match_id);
CREATE INDEX IF NOT EXISTS idx_players_profile ON players(profile_id);
CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_tracking_match ON tracking_data(match_id);
CREATE INDEX IF NOT EXISTS idx_tracking_frame ON tracking_data(frame_number);
CREATE INDEX IF NOT EXISTS idx_tracking_player ON tracking_data(player_instance_id);
CREATE INDEX IF NOT EXISTS idx_ball_tracking_match ON ball_tracking(match_id);
CREATE INDEX IF NOT EXISTS idx_ball_tracking_frame ON ball_tracking(frame_number);
CREATE INDEX IF NOT EXISTS idx_stats_match ON player_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_stats_profile ON player_stats(profile_id);
CREATE INDEX IF NOT EXISTS idx_profiles_name ON player_profiles(name);

-- ============================================================
-- XG DATA TABLE (Expected Goals)
-- ============================================================
CREATE TABLE IF NOT EXISTS xg_data (
    xg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    event_id INTEGER,
    timestamp REAL NOT NULL,
    frame INTEGER,
    shooter_id INTEGER,
    shooter_team TEXT,
    x REAL NOT NULL,  -- normalized 0-1
    y REAL NOT NULL,  -- normalized 0-1
    shot_type TEXT,  -- 'open_play', 'set_piece', 'header', 'penalty', 'free_kick', 'corner'
    body_part TEXT,  -- 'left_foot', 'right_foot', 'head', 'other'
    outcome TEXT,  -- 'goal', 'saved', 'blocked', 'off_target', 'post', 'unknown'
    distance_to_goal REAL,
    angle_to_goal REAL,
    velocity_mps REAL,
    big_chance BOOLEAN DEFAULT 0,
    xg_value REAL NOT NULL,
    metadata TEXT,  -- JSON with additional data
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE SET NULL
);

-- ============================================================
-- HIGHLIGHTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS highlights (
    highlight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    frame INTEGER,
    importance INTEGER,  -- 1-5 scale
    primary_player_id INTEGER,
    secondary_player_id INTEGER,
    team TEXT,
    description TEXT,
    clip_start REAL,
    clip_end REAL,
    xg_value REAL,
    velocity REAL,
    metadata TEXT,  -- JSON with additional data
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- ============================================================
-- TACTICAL PATTERNS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS tactical_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    pattern_type TEXT NOT NULL,  -- 'passing_sequence', 'attacking_move', 'defensive_action', etc.
    outcome TEXT,  -- 'successful', 'unsuccessful', 'neutral'
    start_time REAL,
    end_time REAL,
    duration REAL,
    start_zone TEXT,
    end_zone TEXT,
    pass_count INTEGER,
    distance_covered REAL,
    xg_generated REAL,
    players_involved TEXT,  -- JSON array of player IDs
    description TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- ============================================================
-- OPPONENT PROFILES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS opponent_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL UNIQUE,
    preferred_formation TEXT,
    primary_style TEXT,
    pressing_intensity REAL,
    defensive_line_height REAL,
    set_piece_threat REAL,
    avg_pass_length REAL,
    zone_preferences TEXT,  -- JSON
    key_players TEXT,  -- JSON
    matches_analyzed INTEGER DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- PASSING NETWORKS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS passing_networks (
    network_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    degree_centrality REAL,
    betweenness_centrality REAL,
    closeness_centrality REAL,
    connections TEXT,  -- JSON array of connected players
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- ============================================================
-- ZONE CONTROL TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS zone_control (
    control_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    zone TEXT NOT NULL,
    team_a_control REAL,
    team_b_control REAL,
    contested REAL,
    team_a_time REAL,
    team_b_time REAL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- ============================================================
-- ADVANCED ANALYTICS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS advanced_analytics (
    analytics_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value REAL,
    metadata TEXT,  -- JSON with additional data
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- ============================================================
-- INDEXES FOR NEW TABLES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_xg_match ON xg_data(match_id);
CREATE INDEX IF NOT EXISTS idx_xg_shooter ON xg_data(shooter_id);
CREATE INDEX IF NOT EXISTS idx_highlights_match ON highlights(match_id);
CREATE INDEX IF NOT EXISTS idx_highlights_importance ON highlights(importance);
CREATE INDEX IF NOT EXISTS idx_patterns_match ON tactical_patterns(match_id);
CREATE INDEX IF NOT EXISTS idx_patterns_team ON tactical_patterns(team);
CREATE INDEX IF NOT EXISTS idx_opponent_name ON opponent_profiles(team_name);
CREATE INDEX IF NOT EXISTS idx_networks_match ON passing_networks(match_id);
CREATE INDEX IF NOT EXISTS idx_zone_control_match ON zone_control(match_id);
"""

# Event types enum for reference
EVENT_TYPES = [
    'pass',
    'pass_completed',
    'pass_intercepted',
    'shot',
    'shot_on_target',
    'goal',
    'tackle',
    'interception',
    'sprint_start',
    'sprint_end',
    'possession_start',
    'possession_end',
    'zone_entry',
    'zone_exit',
    'ball_recovery',
    'ball_loss',
    'substitution',
    'card_yellow',
    'card_red',
    'offside',
    'corner',
    'throw_in',
    'free_kick',
    'penalty',
    'big_chance',
    'save',
    'highlight',
    'tactical_pattern',
]

# Schema version for migrations
SCHEMA_VERSION = 2
