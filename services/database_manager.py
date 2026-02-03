"""
Comprehensive Database Manager for Football Analytics

Provides CRUD operations for all database tables with support for:
- Match management
- Player profiles with face recognition
- Event tracking
- Statistics aggregation
- Data export (CSV/JSON)
- Historical comparisons
"""

import sqlite3
import json
import csv
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

from services.database_schema import SCHEMA_SQL, EVENT_TYPES

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class DatabaseManager:
    """
    Comprehensive database manager for football analytics.
    
    Handles all database operations including:
    - Schema initialization
    - Match CRUD operations
    - Player profile management with face encoding
    - Event tracking
    - Statistics queries
    - Data export
    """
    
    def __init__(self, db_path: str = "matches.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    # ============================================================
    # INITIALIZATION
    # ============================================================
    
    def initialize_database(self):
        """Initialize database with all tables and indexes."""
        with self._get_connection() as conn:
            # First, handle schema migrations for existing tables
            self._migrate_schema(conn)
            # Then run the full schema (CREATE IF NOT EXISTS is safe)
            conn.executescript(SCHEMA_SQL)
        print(f"✅ Database initialized: {self.db_path}")

    def _migrate_schema(self, conn):
        """Handle schema migrations for existing tables."""
        cursor = conn.cursor()

        # Check if matches table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='matches'")
        if cursor.fetchone():
            # Get existing columns
            cursor.execute("PRAGMA table_info(matches)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Add missing columns with defaults
            migrations = [
                ("match_date", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("duration_seconds", "REAL DEFAULT 0"),
                ("score_a", "INTEGER DEFAULT 0"),
                ("score_b", "INTEGER DEFAULT 0"),
                ("venue", "TEXT"),
                ("fps", "REAL DEFAULT 25.0"),
                ("width", "INTEGER DEFAULT 1280"),
                ("height", "INTEGER DEFAULT 720"),
                ("processed", "BOOLEAN DEFAULT 0"),
            ]

            for col_name, col_def in migrations:
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE matches ADD COLUMN {col_name} {col_def}")
                        print(f"  Added column: matches.{col_name}")
                    except sqlite3.OperationalError:
                        pass  # Column already exists

        
    
    def reset_database(self):
        """Reset database - drops all tables and recreates them."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            # Drop all tables
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
        # Reinitialize
        self.initialize_database()
        print("✅ Database reset complete")
    
    # ============================================================
    # MATCH OPERATIONS
    # ============================================================
    
    def create_match(
        self,
        video_path: str,
        team_a: str,
        team_b: str,
        duration_seconds: float = 0,
        score_a: int = 0,
        score_b: int = 0,
        venue: str = None,
        fps: float = 25.0,
        width: int = 1280,
        height: int = 720
    ) -> int:
        """
        Create a new match record.
        
        Returns:
            match_id: ID of created match
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO matches 
                (video_path, team_a, team_b, duration_seconds, score_a, score_b, venue, fps, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (video_path, team_a, team_b, duration_seconds, score_a, score_b, 
                  venue, fps, width, height))
            match_id = cursor.lastrowid
        print(f"✅ Match created: ID={match_id}, {team_a} vs {team_b}")
        return match_id
    
    def get_match(self, match_id: int) -> Optional[Dict]:
        """Get match by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_matches(self, limit: int = None, offset: int = 0) -> List[Dict]:
        """Get all matches, optionally paginated."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM matches ORDER BY match_date DESC LIMIT ? OFFSET ?"
            cursor.execute(query, (limit or -1, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_matches_by_team(self, team_name: str) -> List[Dict]:
        """Get all matches involving a specific team."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM matches 
                WHERE team_a = ? OR team_b = ?
                ORDER BY match_date DESC
            """, (team_name, team_name))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_matches_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict]:
        """Get matches within a date range."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM matches 
                WHERE match_date BETWEEN ? AND ?
                ORDER BY match_date DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            return [dict(row) for row in cursor.fetchall()]
    
    def update_match_score(self, match_id: int, score_a: int, score_b: int):
        """Update match score."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE matches SET score_a = ?, score_b = ? WHERE match_id = ?
            """, (score_a, score_b, match_id))
    
    def update_match_duration(self, match_id: int, duration_seconds: float):
        """Update match duration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE matches SET duration_seconds = ? WHERE match_id = ?
            """, (duration_seconds, match_id))
    
    def delete_match(self, match_id: int):
        """Delete a match and all associated data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM matches WHERE match_id = ?", (match_id,))
        print(f"✅ Match {match_id} deleted")
    
    # ============================================================
    # TEAM OPERATIONS
    # ============================================================
    
    def create_team(
        self, 
        match_id: int, 
        side: str, 
        name: str, 
        color: str = None,
        formation: str = None
    ) -> int:
        """
        Create a team record for a match.
        
        Returns:
            team_id: ID of created team
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO teams (match_id, side, name, color, formation)
                VALUES (?, ?, ?, ?, ?)
            """, (match_id, side, name, color, formation))
            team_id = cursor.lastrowid
        return team_id
    
    def get_teams_for_match(self, match_id: int) -> Dict[str, Dict]:
        """Get both teams for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams WHERE match_id = ?", (match_id,))
            rows = cursor.fetchall()
            return {row['side']: dict(row) for row in rows}
    
    # ============================================================
    # PLAYER PROFILE OPERATIONS (Cross-match identity)
    # ============================================================
    
    def create_player_profile(
        self,
        name: str,
        team_name: str = None,
        jersey_number: int = None,
        face_encoding: np.ndarray = None,
        face_image_path: str = None,
        date_of_birth: str = None,
        nationality: str = None,
        position_default: str = None,
        height_cm: int = None,
        weight_kg: int = None
    ) -> int:
        """
        Create a player profile for cross-match tracking.
        
        Args:
            name: Player name
            team_name: Default team name
            jersey_number: Jersey number
            face_encoding: Face embedding numpy array for recognition
            face_image_path: Path to face image
            date_of_birth: Date of birth (ISO format)
            nationality: Nationality
            position_default: Default position
            height_cm: Height in cm
            weight_kg: Weight in kg
            
        Returns:
            profile_id: ID of created profile
        """
        # Serialize face encoding
        face_blob = None
        if face_encoding is not None:
            face_blob = pickle.dumps(face_encoding)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO player_profiles 
                (name, team_name, jersey_number, face_encoding, face_image_path,
                 date_of_birth, nationality, position_default, height_cm, weight_kg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, team_name, jersey_number, face_blob, face_image_path,
                  date_of_birth, nationality, position_default, height_cm, weight_kg))
            profile_id = cursor.lastrowid
        print(f"✅ Player profile created: {name} (ID={profile_id})")
        return profile_id
    
    def get_player_profile(self, profile_id: int) -> Optional[Dict]:
        """Get player profile by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM player_profiles WHERE profile_id = ?", (profile_id,))
            row = cursor.fetchone()
            if row:
                profile = dict(row)
                # Deserialize face encoding
                if profile.get('face_encoding'):
                    profile['face_encoding'] = pickle.loads(profile['face_encoding'])
                return profile
            return None
    
    def find_player_by_face(
        self, 
        face_encoding: np.ndarray, 
        threshold: float = 0.6
    ) -> Optional[Dict]:
        """
        Find player by face encoding using cosine similarity.
        
        Args:
            face_encoding: Face embedding to match
            threshold: Similarity threshold (0-1, higher is stricter)
            
        Returns:
            Player profile if match found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM player_profiles 
                WHERE face_encoding IS NOT NULL
            """)
            rows = cursor.fetchall()
        
        best_match = None
        best_similarity = -1
        
        for row in rows:
            stored_encoding = pickle.loads(row['face_encoding'])
            # Cosine similarity
            similarity = np.dot(face_encoding, stored_encoding) / (
                np.linalg.norm(face_encoding) * np.linalg.norm(stored_encoding)
            )
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = dict(row)
                best_match['face_encoding'] = stored_encoding
                best_match['similarity'] = similarity
        
        return best_match
    
    def update_player_profile(self, profile_id: int, **kwargs):
        """Update player profile fields."""
        allowed_fields = [
            'name', 'team_name', 'jersey_number', 'face_encoding',
            'face_image_path', 'date_of_birth', 'nationality',
            'position_default', 'height_cm', 'weight_kg'
        ]
        
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return
        
        # Serialize face encoding if provided
        if 'face_encoding' in updates and updates['face_encoding'] is not None:
            updates['face_encoding'] = pickle.dumps(updates['face_encoding'])
        
        updates['updated_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [profile_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE player_profiles 
                SET {set_clause}
                WHERE profile_id = ?
            """, values)
    
    def get_all_player_profiles(self) -> List[Dict]:
        """Get all player profiles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM player_profiles ORDER BY name")
            rows = cursor.fetchall()
            profiles = []
            for row in rows:
                profile = dict(row)
                if profile.get('face_encoding'):
                    profile['face_encoding'] = pickle.loads(profile['face_encoding'])
                profiles.append(profile)
            return profiles
    
    # ============================================================
    # PLAYER INSTANCE OPERATIONS (Per-match)
    # ============================================================
    
    def create_player_instance(
        self,
        match_id: int,
        team_id: int,
        profile_id: int = None,
        jersey_number: int = None,
        position: str = None
    ) -> int:
        """
        Create a player instance for a specific match.
        
        Returns:
            player_instance_id: ID of created player instance
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO players (match_id, team_id, profile_id, jersey_number, position)
                VALUES (?, ?, ?, ?, ?)
            """, (match_id, team_id, profile_id, jersey_number, position))
            player_id = cursor.lastrowid
        return player_id
    
    def get_players_for_match(self, match_id: int) -> List[Dict]:
        """Get all players for a match with their profile info."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.*, pp.name as profile_name, pp.team_name as profile_team
                FROM players p
                LEFT JOIN player_profiles pp ON p.profile_id = pp.profile_id
                WHERE p.match_id = ?
            """, (match_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_player_instance(self, player_instance_id: int) -> Optional[Dict]:
        """Get player instance by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM players WHERE player_instance_id = ?", (player_instance_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def link_player_to_profile(
        self, 
        player_instance_id: int, 
        profile_id: int
    ):
        """Link a player instance to a profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE players SET profile_id = ? WHERE player_instance_id = ?
            """, (profile_id, player_instance_id))
    
    # ============================================================
    # EVENT OPERATIONS
    # ============================================================
    
    def create_event(
        self,
        match_id: int,
        event_type: str,
        timestamp: float,
        player_instance_id: int = None,
        profile_id: int = None,
        team_id: int = None,
        x: float = None,
        y: float = None,
        metadata: Dict = None
    ) -> int:
        """
        Create an event record.
        
        Args:
            match_id: Match ID
            event_type: Type of event (pass, shot, goal, etc.)
            timestamp: Time in seconds from match start
            player_instance_id: Player who performed the action
            profile_id: Player profile ID
            team_id: Team ID
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            metadata: Additional event data as dict
            
        Returns:
            event_id: ID of created event
        """
        metadata_json = json.dumps(convert_numpy_types(metadata)) if metadata else None

        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events 
                (match_id, event_type, timestamp, player_instance_id, profile_id, team_id, x, y, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (match_id, event_type, timestamp, player_instance_id, profile_id, 
                  team_id, x, y, metadata_json))
            event_id = cursor.lastrowid
        return event_id
    
    def create_events_batch(self, events: List[Dict]):
        """
        Create multiple events in a batch for efficiency.
        
        Args:
            events: List of event dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for event in events:
                metadata_json = json.dumps(convert_numpy_types(metadata)) if metadata else None

                cursor.execute("""
                    INSERT INTO events 
                    (match_id, event_type, timestamp, player_instance_id, profile_id, team_id, x, y, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event['match_id'], event['event_type'], event['timestamp'],
                    event.get('player_instance_id'), event.get('profile_id'),
                    event.get('team_id'), event.get('x'), event.get('y'), metadata_json
                ))
    
    def get_events_for_match(
        self, 
        match_id: int, 
        event_type: str = None,
        start_time: float = None,
        end_time: float = None
    ) -> List[Dict]:
        """Get events for a match with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM events WHERE match_id = ?"
            params = [match_id]
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            events = []
            for row in cursor.fetchall():
                event = dict(row)
                if event.get('metadata'):
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
            return events
    
    def get_events_for_player(
        self, 
        profile_id: int, 
        event_type: str = None,
        match_ids: List[int] = None
    ) -> List[Dict]:
        """Get events for a player across matches."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM events WHERE profile_id = ?"
            params = [profile_id]
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            if match_ids:
                placeholders = ','.join(['?' for _ in match_ids])
                query += f" AND match_id IN ({placeholders})"
                params.extend(match_ids)
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            events = []
            for row in cursor.fetchall():
                event = dict(row)
                if event.get('metadata'):
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
            return events
    
    # ============================================================
    # TRACKING DATA OPERATIONS
    # ============================================================
    
    def add_tracking_frame(
        self,
        match_id: int,
        frame_number: int,
        timestamp: float,
        player_positions: List[Dict],
        ball_position: Dict = None
    ):
        """
        Add tracking data for a frame.
        
        Args:
            match_id: Match ID
            frame_number: Frame number
            timestamp: Time in seconds
            player_positions: List of dicts with player position data
            ball_position: Dict with ball position data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert player tracking data
            for pos in player_positions:
                cursor.execute("""
                    INSERT INTO tracking_data 
                    (match_id, player_instance_id, profile_id, frame_number, timestamp,
                     x_px, y_px, x_norm, y_norm, speed_mps, has_possession, team_in_possession)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id, pos.get('player_instance_id'), pos.get('profile_id'),
                    frame_number, timestamp, pos.get('x_px'), pos.get('y_px'),
                    pos.get('x_norm'), pos.get('y_norm'), pos.get('speed_mps'),
                    pos.get('has_possession', False), pos.get('team_in_possession')
                ))
            
            # Insert ball tracking data
            if ball_position:
                cursor.execute("""
                    INSERT INTO ball_tracking 
                    (match_id, frame_number, timestamp, x_px, y_px, x_norm, y_norm,
                     velocity_x, velocity_y, velocity_magnitude, detection_method,
                     confidence, possessing_player_id, possessing_team)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id, frame_number, timestamp,
                    ball_position.get('x_px'), ball_position.get('y_px'),
                    ball_position.get('x_norm'), ball_position.get('y_norm'),
                    ball_position.get('velocity_x'), ball_position.get('velocity_y'),
                    ball_position.get('velocity_magnitude'),
                    ball_position.get('detection_method'),
                    ball_position.get('confidence'),
                    ball_position.get('possessing_player_id'),
                    ball_position.get('possessing_team')
                ))
    
    def get_tracking_data_for_player(
        self,
        player_instance_id: int,
        start_frame: int = None,
        end_frame: int = None
    ) -> List[Dict]:
        """Get tracking data for a player."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM tracking_data WHERE player_instance_id = ?"
            params = [player_instance_id]
            
            if start_frame:
                query += " AND frame_number >= ?"
                params.append(start_frame)
            if end_frame:
                query += " AND frame_number <= ?"
                params.append(end_frame)
            
            query += " ORDER BY frame_number"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_ball_tracking_for_match(
        self,
        match_id: int,
        start_frame: int = None,
        end_frame: int = None
    ) -> List[Dict]:
        """Get ball tracking data for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM ball_tracking WHERE match_id = ?"
            params = [match_id]
            
            if start_frame:
                query += " AND frame_number >= ?"
                params.append(start_frame)
            if end_frame:
                query += " AND frame_number <= ?"
                params.append(end_frame)
            
            query += " ORDER BY frame_number"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # STATISTICS OPERATIONS
    # ============================================================
    
    def save_player_stats(
        self,
        match_id: int,
        player_instance_id: int,
        profile_id: int = None,
        **stats
    ) -> int:
        """
        Save player statistics for a match.
        
        Args:
            match_id: Match ID
            player_instance_id: Player instance ID
            profile_id: Player profile ID
            **stats: Statistics fields
        """
        allowed_fields = [
            'total_distance_m', 'sprint_distance_m', 'high_intensity_distance_m',
            'avg_speed_mps', 'max_speed_mps', 'passes_attempted', 'passes_completed',
            'shots', 'shots_on_target', 'goals', 'assists', 'tackles', 'interceptions',
            'sprints', 'possession_count', 'possession_duration_s', 'touches',
            'workload_score', 'minutes_played', 'heatmap_path'
        ]
        
        # Filter to allowed fields
        data = {k: v for k, v in stats.items() if k in allowed_fields}
        data['match_id'] = match_id
        data['player_instance_id'] = player_instance_id
        if profile_id:
            data['profile_id'] = profile_id
        
        # Build query
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT OR REPLACE INTO player_stats 
                ({columns})
                VALUES ({placeholders})
            """, list(data.values()))
            stats_id = cursor.lastrowid
        return stats_id
    
    def get_player_stats(
        self, 
        player_instance_id: int,
        match_id: int = None
    ) -> Optional[Dict]:
        """Get player statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if match_id:
                cursor.execute("""
                    SELECT * FROM player_stats 
                    WHERE player_instance_id = ? AND match_id = ?
                """, (player_instance_id, match_id))
            else:
                cursor.execute("""
                    SELECT * FROM player_stats 
                    WHERE player_instance_id = ?
                """, (player_instance_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_player_stats_across_matches(self, profile_id: int) -> List[Dict]:
        """Get player statistics across all matches."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ps.*, m.match_date, m.team_a, m.team_b
                FROM player_stats ps
                JOIN matches m ON ps.match_id = m.match_id
                WHERE ps.profile_id = ?
                ORDER BY m.created_at DESC
            """, (profile_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_team_stats(self, match_id: int, team_id: int) -> Optional[Dict]:
        """Get team statistics for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM team_stats 
                WHERE match_id = ? AND team_id = ?
            """, (match_id, team_id))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_team_stats(
        self,
        match_id: int,
        team_id: int,
        **stats
    ):
        """Save team statistics."""
        allowed_fields = [
            'possession_percentage', 'possession_count', 'passes_attempted',
            'passes_completed', 'pass_accuracy', 'shots', 'shots_on_target',
            'goals', 'total_distance_m', 'sprints', 'defensive_third_possession',
            'midfield_possession', 'attacking_third_possession'
        ]
        
        data = {k: v for k, v in stats.items() if k in allowed_fields}
        data['match_id'] = match_id
        data['team_id'] = team_id
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT OR REPLACE INTO team_stats 
                ({columns})
                VALUES ({placeholders})
            """, list(data.values()))
    
    # ============================================================
    # HEATMAP OPERATIONS
    # ============================================================
    
    def save_heatmap(
        self,
        match_id: int,
        heatmap_type: str,
        file_path: str,
        player_instance_id: int = None,
        team_id: int = None,
        data_json: str = None
    ) -> int:
        """Save heatmap reference."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO heatmaps 
                (match_id, player_instance_id, team_id, heatmap_type, file_path, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (match_id, player_instance_id, team_id, heatmap_type, file_path, data_json))
            return cursor.lastrowid
    
    def get_heatmaps_for_match(
        self, 
        match_id: int,
        heatmap_type: str = None
    ) -> List[Dict]:
        """Get heatmaps for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if heatmap_type:
                cursor.execute("""
                    SELECT * FROM heatmaps 
                    WHERE match_id = ? AND heatmap_type = ?
                """, (match_id, heatmap_type))
            else:
                cursor.execute("""
                    SELECT * FROM heatmaps WHERE match_id = ?
                """, (match_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # ANALYTICS & AGGREGATION
    # ============================================================
    
    def get_player_career_stats(self, profile_id: int) -> Dict:
        """
        Get aggregated career statistics for a player.
        
        Returns:
            Dict with aggregated stats across all matches
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT match_id) as matches_played,
                    SUM(total_distance_m) as total_distance_m,
                    AVG(total_distance_m) as avg_distance_per_match,
                    MAX(max_speed_mps) as max_speed_ever,
                    AVG(avg_speed_mps) as avg_speed_overall,
                    SUM(passes_attempted) as total_passes_attempted,
                    SUM(passes_completed) as total_passes_completed,
                    SUM(shots) as total_shots,
                    SUM(goals) as total_goals,
                    SUM(assists) as total_assists,
                    SUM(sprints) as total_sprints
                FROM player_stats
                WHERE profile_id = ?
            """, (profile_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def get_team_trends(
        self,
        team_name: str,
        num_matches: int = 10
    ) -> List[Dict]:
        """
        Get team performance trends over time.
        
        Args:
            team_name: Name of the team
            num_matches: Number of recent matches to analyze
            
        Returns:
            List of match stats with trends
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    m.match_id,
                    m.match_date,
                    m.team_a,
                    m.team_b,
                    m.score_a,
                    m.score_b,
                    ts.possession_percentage,
                    ts.pass_accuracy,
                    ts.shots,
                    ts.goals,
                    ts.total_distance_m,
                    ts.sprints
                FROM matches m
                JOIN teams t ON m.match_id = t.match_id
                JOIN team_stats ts ON t.team_id = ts.team_id
                WHERE t.name = ?
                ORDER BY m.created_at DESC
                LIMIT ?
            """, (team_name, num_matches))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_player_form(
        self,
        profile_id: int,
        num_matches: int = 5
    ) -> Dict:
        """
        Calculate player form/trend over recent matches.
        
        Args:
            profile_id: Player profile ID
            num_matches: Number of recent matches to analyze
            
        Returns:
            Dict with form metrics and trends
        """
        stats = self.get_player_stats_across_matches(profile_id)
        recent_stats = stats[:num_matches]
        
        if not recent_stats:
            return {'status': 'no_data'}
        
        # Calculate averages
        avg_distance = np.mean([s['total_distance_m'] for s in recent_stats if s['total_distance_m']])
        avg_speed = np.mean([s['avg_speed_mps'] for s in recent_stats if s['avg_speed_mps']])
        avg_pass_accuracy = np.mean([
            (s['passes_completed'] / s['passes_attempted'] * 100) 
            for s in recent_stats 
            if s['passes_attempted'] > 0
        ])
        
        # Calculate trend (comparing first half to second half of period)
        half = len(recent_stats) // 2
        if half > 0:
            first_half = recent_stats[-half:]
            second_half = recent_stats[:half]
            
            first_dist = np.mean([s['total_distance_m'] for s in first_half])
            second_dist = np.mean([s['total_distance_m'] for s in second_half])
            distance_trend = ((second_dist - first_dist) / first_dist * 100) if first_dist > 0 else 0
        else:
            distance_trend = 0
        
        return {
            'matches_analyzed': len(recent_stats),
            'avg_distance_m': avg_distance,
            'avg_speed_mps': avg_speed,
            'avg_pass_accuracy': avg_pass_accuracy,
            'distance_trend_percent': distance_trend,
            'trend_direction': 'improving' if distance_trend > 5 else 'declining' if distance_trend < -5 else 'stable',
            'recent_matches': recent_stats
        }
    
    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> List[Dict]:
        """Get head-to-head match history between two teams."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM matches
                WHERE (team_a = ? AND team_b = ?) OR (team_a = ? AND team_b = ?)
                ORDER BY match_date DESC
                LIMIT ?
            """, (team_a, team_b, team_b, team_a, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # EXPORT OPERATIONS
    # ============================================================
    
    def export_match_to_json(self, match_id: int, output_path: str = None) -> str:
        """
        Export all match data to JSON.
        
        Args:
            match_id: Match ID to export
            output_path: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        match = self.get_match(match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        
        # Gather all related data
        export_data = {
            'match': match,
            'teams': self.get_teams_for_match(match_id),
            'players': self.get_players_for_match(match_id),
            'events': self.get_events_for_match(match_id),
            'player_stats': [],
            'team_stats': [],
            'heatmaps': self.get_heatmaps_for_match(match_id)
        }
        
        # Get player stats
        for player in export_data['players']:
            stats = self.get_player_stats(player['player_instance_id'], match_id)
            if stats:
                export_data['player_stats'].append(stats)
        
        # Get team stats
        for side, team in export_data['teams'].items():
            stats = self.get_team_stats(match_id, team['team_id'])
            if stats:
                export_data['team_stats'].append(stats)
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"match_{match_id}_export_{timestamp}.json"
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Match exported to: {output_path}")
        return output_path
    
    def export_player_stats_to_csv(
        self, 
        profile_id: int, 
        output_path: str = None
    ) -> str:
        """
        Export player statistics across matches to CSV.
        
        Args:
            profile_id: Player profile ID
            output_path: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        stats = self.get_player_stats_across_matches(profile_id)
        if not stats:
            raise ValueError(f"No stats found for player {profile_id}")
        
        profile = self.get_player_profile(profile_id)
        player_name = profile['name'] if profile else f"Player_{profile_id}"
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = player_name.replace(' ', '_')
            output_path = f"{safe_name}_stats_{timestamp}.csv"
        
        # Write to CSV
        with open(output_path, 'w', newline='') as f:
            if stats:
                writer = csv.DictWriter(f, fieldnames=stats[0].keys())
                writer.writeheader()
                writer.writerows(stats)
        
        print(f"✅ Player stats exported to: {output_path}")
        return output_path
    
    def export_all_matches_to_csv(self, output_path: str = "all_matches.csv") -> str:
        """Export all matches to CSV."""
        matches = self.get_all_matches()
        
        with open(output_path, 'w', newline='') as f:
            if matches:
                writer = csv.DictWriter(f, fieldnames=matches[0].keys())
                writer.writeheader()
                writer.writerows(matches)
        
        print(f"✅ All matches exported to: {output_path}")
        return output_path
    
    # ============================================================
    # BATCH OPERATIONS FOR DEMO INTEGRATION
    # ============================================================
    
    def save_match_results(
        self,
        video_path: str,
        team_a: str,
        team_b: str,
        metrics: Dict,
        output_dir: Path
    ) -> int:
        """
        Save complete match results after processing.
        
        This is the main integration point for run_demo.py.
        
        Args:
            video_path: Path to video file
            team_a: Team A name
            team_b: Team B name
            metrics: Metrics dictionary from run_demo.py
            output_dir: Directory containing heatmaps and outputs
            
        Returns:
            match_id: ID of saved match
        """
        # Create match
        match_id = self.create_match(
            video_path=video_path,
            team_a=team_a,
            team_b=team_b,
            duration_seconds=metrics.get('frame', 0) / 25.0,  # Assuming 25fps
            fps=25.0,
            width=1280,
            height=720
        )
        
        # Create teams
        teams = {}
        for side in ['A', 'B']:
            team_name = metrics.get('team_names', {}).get(side, f"Team {side}")
            teams[side] = self.create_team(match_id, side, team_name)
        
        # Create player instances and save stats
        tracks = metrics.get('tracks', [])
        player_map = {}  # Maps track player_id to database player_instance_id
        
        for track in tracks:
            player_id = track.get('player_id')
            team = track.get('team', 'A')
            team_id = teams.get(team)
            
            if team_id:
                # Create player instance
                player_instance_id = self.create_player_instance(
                    match_id=match_id,
                    team_id=team_id,
                    jersey_number=track.get('jersey_number'),
                    position=track.get('position')
                )
                player_map[player_id] = player_instance_id
                
                # Save player stats
                self.save_player_stats(
                    match_id=match_id,
                    player_instance_id=player_instance_id,
                    total_distance_m=track.get('total_distance_m', 0),
                    avg_speed_mps=track.get('avg_speed_mps', 0),
                    max_speed_mps=track.get('max_speed_mps', 0),
                    workload_score=track.get('workload_score', 0),
                    heatmap_path=str(output_dir / f"heatmap_player_{player_id}.png") if output_dir else None
                )
        
        # Save events
        events = []
        
        # Pass events
        for pass_event in metrics.get('pass_events', []):
            events.append({
                'match_id': match_id,
                'event_type': 'pass_completed' if pass_event.get('success') else 'pass',
                'timestamp': pass_event.get('timestamp', 0),
                'player_instance_id': player_map.get(pass_event.get('from_player')),
                'metadata': pass_event
            })
        
        # Shot events
        for shot in metrics.get('shot_events', []):
            events.append({
                'match_id': match_id,
                'event_type': 'shot',
                'timestamp': shot.get('timestamp', 0),
                'player_instance_id': player_map.get(shot.get('shooter_id')),
                'metadata': shot
            })
        
        # Sprint events
        for sprint in metrics.get('sprint_events', []):
            events.append({
                'match_id': match_id,
                'event_type': 'sprint_start',
                'timestamp': sprint.get('start_time', 0),
                'player_instance_id': player_map.get(sprint.get('player_id')),
                'metadata': sprint
            })
        
        if events:
            self.create_events_batch(events)
        
        # Save team stats
        possession = metrics.get('possession', {})
        pass_stats = metrics.get('pass_detection', {})
        shot_stats = metrics.get('shot_detection', {})
        sprint_stats = metrics.get('sprint_detection', {})
        
        for side in ['A', 'B']:
            team_id = teams.get(side)
            if team_id:
                self.save_team_stats(
                    match_id=match_id,
                    team_id=team_id,
                    possession_percentage=possession.get('team_possession_percentage', {}).get(side, 50),
                    passes_attempted=pass_stats.get('team_passes', {}).get(side, {}).get('attempted', 0),
                    passes_completed=pass_stats.get('team_passes', {}).get(side, {}).get('completed', 0),
                    shots=shot_stats.get('team_shots', {}).get(side, 0),
                    sprints=sprint_stats.get('team_sprints', {}).get(side, 0)
                )
        
        # Save heatmaps
        heatmap_types = ['global', 'team_A', 'team_B', 'ball']
        for hm_type in heatmap_types:
            hm_path = output_dir / f"heatmap_{hm_type}.png"
            if hm_path.exists():
                team_id = None
                if hm_type == 'team_A':
                    team_id = teams.get('A')
                elif hm_type == 'team_B':
                    team_id = teams.get('B')
                
                self.save_heatmap(
                    match_id=match_id,
                    heatmap_type=hm_type.replace('_', ''),
                    file_path=str(hm_path),
                    team_id=team_id
                )
        
        print(f"✅ Match results saved to database: Match ID={match_id}")
        return match_id
    
    # ============================================================
    # XG DATA OPERATIONS
    # ============================================================
    
    def save_xg_data(
        self,
        match_id: int,
        xg_events: List[Dict]
    ):
        """
        Save xG data for a match.
        
        Args:
            match_id: Match ID
            xg_events: List of xG event dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for event in xg_events:
                cursor.execute("""
                    INSERT INTO xg_data
                    (match_id, timestamp, frame, shooter_id, shooter_team, x, y,
                     shot_type, body_part, outcome, distance_to_goal, angle_to_goal,
                     velocity_mps, big_chance, xg_value, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    event.get('timestamp', 0),
                    event.get('frame'),
                    event.get('shooter_id'),
                    event.get('shooter_team'),
                    event.get('x', 0),
                    event.get('y', 0),
                    event.get('shot_type', 'open_play'),
                    event.get('body_part', 'right_foot'),
                    event.get('outcome', 'unknown'),
                    event.get('distance_to_goal', 0),
                    event.get('angle_to_goal', 0),
                    event.get('velocity_mps', 0),
                    event.get('big_chance', False),
                    event.get('xg_value', 0),
                    json.dumps(event.get('metadata', {}))
                ))
    
    def get_xg_data_for_match(self, match_id: int) -> List[Dict]:
        """Get all xG data for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM xg_data WHERE match_id = ? ORDER BY timestamp
            """, (match_id,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('metadata'):
                    data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            return results
    
    def get_xg_stats_for_match(self, match_id: int) -> Dict[str, Any]:
        """Get aggregated xG statistics for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total xG by team
            cursor.execute("""
                SELECT shooter_team, COUNT(*) as shots, SUM(xg_value) as total_xg
                FROM xg_data
                WHERE match_id = ?
                GROUP BY shooter_team
            """, (match_id,))
            
            team_stats = {}
            for row in cursor.fetchall():
                team_stats[row['shooter_team']] = {
                    'shots': row['shots'],
                    'total_xg': round(row['total_xg'], 2)
                }
            
            # Goals scored
            cursor.execute("""
                SELECT shooter_team, COUNT(*) as goals
                FROM xg_data
                WHERE match_id = ? AND outcome = 'goal'
                GROUP BY shooter_team
            """, (match_id,))
            
            for row in cursor.fetchall():
                if row['shooter_team'] in team_stats:
                    team_stats[row['shooter_team']]['goals'] = row['goals']
            
            return team_stats
    
    # ============================================================
    # HIGHLIGHTS OPERATIONS
    # ============================================================
    
    def save_highlights(
        self,
        match_id: int,
        highlights: List[Dict]
    ):
        """
        Save highlight events for a match.
        
        Args:
            match_id: Match ID
            highlights: List of highlight event dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for highlight in highlights:
                cursor.execute("""
                    INSERT INTO highlights
                    (match_id, event_type, timestamp, frame, importance,
                     primary_player_id, secondary_player_id, team, description,
                     clip_start, clip_end, xg_value, velocity, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    highlight.get('event_type'),
                    highlight.get('timestamp', 0),
                    highlight.get('frame'),
                    highlight.get('importance', 1),
                    highlight.get('primary_player_id'),
                    highlight.get('secondary_player_id'),
                    highlight.get('team'),
                    highlight.get('description', ''),
                    highlight.get('clip_start'),
                    highlight.get('clip_end'),
                    highlight.get('xg_value', 0),
                    highlight.get('velocity', 0),
                    json.dumps(highlight.get('metadata', {}))
                ))
    
    def get_highlights_for_match(
        self,
        match_id: int,
        min_importance: int = None
    ) -> List[Dict]:
        """Get highlights for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if min_importance:
                cursor.execute("""
                    SELECT * FROM highlights
                    WHERE match_id = ? AND importance >= ?
                    ORDER BY timestamp
                """, (match_id, min_importance))
            else:
                cursor.execute("""
                    SELECT * FROM highlights
                    WHERE match_id = ?
                    ORDER BY timestamp
                """, (match_id,))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('metadata'):
                    data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            return results
    
    def get_top_highlights(self, match_id: int, n: int = 10) -> List[Dict]:
        """Get top N highlights by importance."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM highlights
                WHERE match_id = ?
                ORDER BY importance DESC, timestamp
                LIMIT ?
            """, (match_id, n))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('metadata'):
                    data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            return results
    
    # ============================================================
    # TACTICAL PATTERNS OPERATIONS
    # ============================================================
    
    def save_tactical_patterns(
        self,
        match_id: int,
        patterns: List[Dict]
    ):
        """
        Save tactical patterns for a match.
        
        Args:
            match_id: Match ID
            patterns: List of pattern dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for pattern in patterns:
                cursor.execute("""
                    INSERT INTO tactical_patterns
                    (match_id, team, pattern_type, outcome, start_time, end_time,
                     duration, start_zone, end_zone, pass_count, distance_covered,
                     xg_generated, players_involved, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    pattern.get('team'),
                    pattern.get('pattern_type'),
                    pattern.get('outcome'),
                    pattern.get('start_time'),
                    pattern.get('end_time'),
                    pattern.get('duration'),
                    pattern.get('start_zone'),
                    pattern.get('end_zone'),
                    pattern.get('pass_count', 0),
                    pattern.get('distance_covered', 0),
                    pattern.get('xg_generated', 0),
                    json.dumps(pattern.get('players_involved', [])),
                    pattern.get('description', '')
                ))
    
    def get_tactical_patterns_for_match(
        self,
        match_id: int,
        team: str = None,
        pattern_type: str = None
    ) -> List[Dict]:
        """Get tactical patterns for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM tactical_patterns WHERE match_id = ?"
            params = [match_id]
            
            if team:
                query += " AND team = ?"
                params.append(team)
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            query += " ORDER BY start_time"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('players_involved'):
                    data['players_involved'] = json.loads(data['players_involved'])
                results.append(data)
            return results
    
    # ============================================================
    # OPPONENT PROFILES OPERATIONS
    # ============================================================
    
    def save_opponent_profile(self, profile: Dict) -> int:
        """
        Save or update opponent profile.
        
        Args:
            profile: Opponent profile dictionary
            
        Returns:
            profile_id
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if profile exists
            cursor.execute(
                "SELECT profile_id FROM opponent_profiles WHERE team_name = ?",
                (profile.get('team_name'),)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update
                cursor.execute("""
                    UPDATE opponent_profiles SET
                        preferred_formation = ?,
                        primary_style = ?,
                        pressing_intensity = ?,
                        defensive_line_height = ?,
                        set_piece_threat = ?,
                        avg_pass_length = ?,
                        zone_preferences = ?,
                        key_players = ?,
                        matches_analyzed = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE team_name = ?
                """, (
                    profile.get('preferred_formation'),
                    profile.get('primary_style'),
                    profile.get('pressing_intensity'),
                    profile.get('defensive_line_height'),
                    profile.get('set_piece_threat'),
                    profile.get('avg_pass_length'),
                    json.dumps(profile.get('zone_preferences', {})),
                    json.dumps(profile.get('key_players', {})),
                    profile.get('matches_analyzed', 0),
                    profile.get('team_name')
                ))
                return existing['profile_id']
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO opponent_profiles
                    (team_name, preferred_formation, primary_style, pressing_intensity,
                     defensive_line_height, set_piece_threat, avg_pass_length,
                     zone_preferences, key_players, matches_analyzed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.get('team_name'),
                    profile.get('preferred_formation'),
                    profile.get('primary_style'),
                    profile.get('pressing_intensity'),
                    profile.get('defensive_line_height'),
                    profile.get('set_piece_threat'),
                    profile.get('avg_pass_length'),
                    json.dumps(profile.get('zone_preferences', {})),
                    json.dumps(profile.get('key_players', {})),
                    profile.get('matches_analyzed', 0)
                ))
                return cursor.lastrowid
    
    def get_opponent_profile(self, team_name: str) -> Optional[Dict]:
        """Get opponent profile by team name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM opponent_profiles WHERE team_name = ?",
                (team_name,)
            )
            row = cursor.fetchone()
            if row:
                profile = dict(row)
                if profile.get('zone_preferences'):
                    profile['zone_preferences'] = json.loads(profile['zone_preferences'])
                if profile.get('key_players'):
                    profile['key_players'] = json.loads(profile['key_players'])
                return profile
            return None
    
    def get_all_opponent_profiles(self) -> List[Dict]:
        """Get all opponent profiles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM opponent_profiles ORDER BY team_name")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                profile = dict(row)
                if profile.get('zone_preferences'):
                    profile['zone_preferences'] = json.loads(profile['zone_preferences'])
                if profile.get('key_players'):
                    profile['key_players'] = json.loads(profile['key_players'])
                results.append(profile)
            return results
    
    # ============================================================
    # PASSING NETWORK OPERATIONS
    # ============================================================
    
    def save_passing_network(
        self,
        match_id: int,
        team: str,
        network_data: List[Dict]
    ):
        """
        Save passing network data.
        
        Args:
            match_id: Match ID
            team: Team ('A' or 'B')
            network_data: List of player network data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for player_data in network_data:
                cursor.execute("""
                    INSERT INTO passing_networks
                    (match_id, team, player_id, degree_centrality,
                     betweenness_centrality, closeness_centrality, connections)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    team,
                    player_data.get('player_id'),
                    player_data.get('degree_centrality'),
                    player_data.get('betweenness_centrality'),
                    player_data.get('closeness_centrality'),
                    json.dumps(player_data.get('connections', []))
                ))
    
    def get_passing_network(
        self,
        match_id: int,
        team: str
    ) -> List[Dict]:
        """Get passing network for a team."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM passing_networks
                WHERE match_id = ? AND team = ?
            """, (match_id, team))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('connections'):
                    data['connections'] = json.loads(data['connections'])
                results.append(data)
            return results
    
    # ============================================================
    # ZONE CONTROL OPERATIONS
    # ============================================================
    
    def save_zone_control(
        self,
        match_id: int,
        zone_data: List[Dict]
    ):
        """
        Save zone control data.
        
        Args:
            match_id: Match ID
            zone_data: List of zone control dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for zone in zone_data:
                cursor.execute("""
                    INSERT INTO zone_control
                    (match_id, zone, team_a_control, team_b_control, contested,
                     team_a_time, team_b_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    zone.get('zone'),
                    zone.get('team_a_control', 0),
                    zone.get('team_b_control', 0),
                    zone.get('contested', 0),
                    zone.get('team_a_time', 0),
                    zone.get('team_b_time', 0)
                ))
    
    def get_zone_control_for_match(self, match_id: int) -> List[Dict]:
        """Get zone control data for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM zone_control WHERE match_id = ?
            """, (match_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # ADVANCED ANALYTICS OPERATIONS
    # ============================================================
    
    def save_advanced_analytics(
        self,
        match_id: int,
        team: str,
        metrics: List[Dict]
    ):
        """
        Save advanced analytics metrics.
        
        Args:
            match_id: Match ID
            team: Team ('A' or 'B')
            metrics: List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for metric in metrics:
                cursor.execute("""
                    INSERT INTO advanced_analytics
                    (match_id, team, metric_type, metric_value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    match_id,
                    team,
                    metric.get('type'),
                    metric.get('value'),
                    json.dumps(metric.get('metadata', {}))
                ))
    
    def get_advanced_analytics(
        self,
        match_id: int,
        team: str = None
    ) -> List[Dict]:
        """Get advanced analytics for a match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if team:
                cursor.execute("""
                    SELECT * FROM advanced_analytics
                    WHERE match_id = ? AND team = ?
                """, (match_id, team))
            else:
                cursor.execute("""
                    SELECT * FROM advanced_analytics WHERE match_id = ?
                """, (match_id,))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('metadata'):
                    data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            return results
