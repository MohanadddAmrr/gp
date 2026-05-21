"""
Database Integration Layer for TactiVision

Integrates TactiVision analytics with Ahmed's SQLite database.
Handles insertion of:
- Match data
- Team data
- Player tracking data
- Possession events (with zones, pressure, duration)
- Ball tracking data
- Player statistics
- Heatmap references
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class TactiVisionDB:
    """
    Database integration layer for TactiVision analytics.
    
    Connects to Ahmed's SQLite database and provides methods to:
    - Insert match results
    - Store possession tracking data
    - Store ball tracking data
    - Store player statistics
    - Query match data
    """
    
    def __init__(self, db_path: str = "matches.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file (default: matches.db)
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        print(f"✅ Connected to database: {self.db_path}")
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def commit(self):
        """Commit current transaction."""
        if self.conn:
            self.conn.commit()
    
    # ============================================================
    # MATCH OPERATIONS
    # ============================================================
    
    def insert_match(
        self,
        video_name: str,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        venue: str = "Unknown",
        fps: float = 25.0,
        width: int = 1280,
        height: int = 720
    ) -> int:
        """
        Insert a new match record.
        
        Args:
            video_name: Name of the video file
            team_a_name: Name of Team A
            team_b_name: Name of Team B
            venue: Match venue
            fps: Frames per second
            width: Video width
            height: Video height
            
        Returns:
            match_id: ID of inserted match
        """
        self.cursor.execute("""
            INSERT INTO matches (date, team_a, team_b, venue, fps, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(), team_a_name, team_b_name, venue, fps, width, height))
        
        match_id = self.cursor.lastrowid
        self.commit()
        
        print(f"✅ Match inserted: ID={match_id}, {team_a_name} vs {team_b_name}")
        return match_id
    
    def insert_teams(self, match_id: int, team_a_name: str = "Team A", team_b_name: str = "Team B") -> Dict[str, int]:
        """
        Insert team records for a match.
        
        Args:
            match_id: Match ID
            team_a_name: Name of Team A
            team_b_name: Name of Team B
            
        Returns:
            Dict mapping team side to team_id: {'A': team_a_id, 'B': team_b_id}
        """
        # Insert Team A
        self.cursor.execute("""
            INSERT INTO teams (match_id, side, name)
            VALUES (?, ?, ?)
        """, (match_id, 'A', team_a_name))
        team_a_id = self.cursor.lastrowid
        
        # Insert Team B
        self.cursor.execute("""
            INSERT INTO teams (match_id, side, name)
            VALUES (?, ?, ?)
        """, (match_id, 'B', team_b_name))
        team_b_id = self.cursor.lastrowid
        
        self.commit()
        
        print(f"✅ Teams inserted: Team A (ID={team_a_id}), Team B (ID={team_b_id})")
        return {'A': team_a_id, 'B': team_b_id}
    
    # ============================================================
    # PLAYER OPERATIONS
    # ============================================================
    
    def insert_player(
        self,
        match_id: int,
        team_id: int,
        player_number: int,
        player_name: Optional[str] = None
    ) -> int:
        """
        Insert a player record.
        
        Args:
            match_id: Match ID
            team_id: Team ID
            player_number: Player number (from tracking)
            player_name: Player name (optional)
            
        Returns:
            player_id: ID of inserted player
        """
        if player_name is None:
            player_name = f"Player {player_number}"
        
        self.cursor.execute("""
            INSERT INTO players (match_id, team_id, name, number)
            VALUES (?, ?, ?, ?)
        """, (match_id, team_id, player_name, player_number))
        
        player_id = self.cursor.lastrowid
        self.commit()
        
        return player_id
    
    def insert_players_from_tracks(
        self,
        match_id: int,
        team_ids: Dict[str, int],
        tracks_data: List[Dict]
    ) -> Dict[int, int]:
        """
        Insert all players from tracking data.
        
        Args:
            match_id: Match ID
            team_ids: Dict mapping team side to team_id {'A': id, 'B': id}
            tracks_data: List of player tracking data from metrics.json
            
        Returns:
            Dict mapping player_number to player_id in database
        """
        player_id_map = {}
        
        for track in tracks_data:
            # Handle both 'player_id' and 'id' field names (backwards compatibility)
            player_number = track.get('player_id') or track.get('id')
            if player_number is None:
                print(f"⚠️  Skipping track with missing player_id: {track}")
                continue
            
            team_side = track.get('team')
            if team_side not in team_ids:
                print(f"⚠️  Skipping player {player_number} with invalid team: {team_side}")
                continue
            
            team_id = team_ids[team_side]
            
            player_id = self.insert_player(match_id, team_id, player_number)
            player_id_map[player_number] = player_id
        
        print(f"✅ Inserted {len(player_id_map)} players")
        return player_id_map

    # ============================================================
    # POSSESSION EVENT OPERATIONS
    # ============================================================
    
    def insert_possession_event(
        self,
        match_id: int,
        player_id: int,
        timestamp: float,
        possession_data: Dict[str, Any]
    ) -> int:
        """
        Insert a possession event.
        
        Args:
            match_id: Match ID
            player_id: Player ID (from database)
            timestamp: Event timestamp
            possession_data: Dict with possession details (zone, pressure, duration, etc.)
            
        Returns:
            event_id: ID of inserted event
        """
        metadata = json.dumps(possession_data)
        
        self.cursor.execute("""
            INSERT INTO events (match_id, event_type, timestamp, player_id, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (match_id, 'possession', timestamp, player_id, metadata))
        
        event_id = self.cursor.lastrowid
        return event_id
    
    def insert_possession_history(
        self,
        match_id: int,
        player_id_map: Dict[int, int],
        possession_history: List[Dict]
    ):
        """
        Insert all possession events from possession tracking.
        
        Args:
            match_id: Match ID
            player_id_map: Dict mapping player_number to database player_id
            possession_history: List of possession events from PossessionTracker
        """
        for event in possession_history:
            player_number = event['player_id']
            player_id = player_id_map.get(player_number)
            
            if player_id is None:
                continue  # Skip if player not in database
            
            possession_data = {
                'team': event['team'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration': event['duration'],
                'start_frame': event['start_frame'],
                'end_frame': event['end_frame']
            }
            
            self.insert_possession_event(
                match_id,
                player_id,
                event['start_time'],
                possession_data
            )
        
        self.commit()
        print(f"✅ Inserted {len(possession_history)} possession events")
    
    # ============================================================
    # BALL TRACKING OPERATIONS
    # ============================================================
    
    def insert_ball_tracking_events(
        self,
        match_id: int,
        ball_tracking_data: Dict[str, Any]
    ):
        """
        Insert ball tracking data as events.
        
        Args:
            match_id: Match ID
            ball_tracking_data: Ball tracking data from metrics.json
        """
        # Insert overall ball tracking summary
        summary_metadata = json.dumps({
            'total_detections': ball_tracking_data.get('total_detections', 0),
            'detection_rate': ball_tracking_data.get('detection_rate', 0),
            'avg_velocity': ball_tracking_data.get('avg_velocity_px_s', 0),
            'max_velocity': ball_tracking_data.get('max_velocity_px_s', 0),
            'yolo_detections': ball_tracking_data.get('yolo_detections', 0),
            'color_detections': ball_tracking_data.get('color_detections', 0),
            'predicted_detections': ball_tracking_data.get('predicted_detections', 0)
        })
        
        self.cursor.execute("""
            INSERT INTO events (match_id, event_type, timestamp, metadata)
            VALUES (?, ?, ?, ?)
        """, (match_id, 'ball_tracking_summary', 0.0, summary_metadata))
        
        # Optionally insert individual ball positions (can be large!)
        # Uncomment if you want to store every ball position
        # position_history = ball_tracking_data.get('position_history', [])
        # for pos in position_history:
        #     pos_metadata = json.dumps(pos)
        #     self.cursor.execute("""
        #         INSERT INTO events (match_id, event_type, timestamp, metadata)
        #         VALUES (?, ?, ?, ?)
        #     """, (match_id, 'ball_position', pos['timestamp'], pos_metadata))
        
        self.commit()
        print(f"✅ Inserted ball tracking data")
    
    # ============================================================
    # PLAYER STATISTICS OPERATIONS
    # ============================================================
    
    def insert_player_statistics(
        self,
        match_id: int,
        player_id: int,
        stats: Dict[str, Any]
    ) -> int:
        """
        Insert player statistics.
        
        Args:
            match_id: Match ID
            player_id: Player ID (from database)
            stats: Dict with player statistics
            
        Returns:
            stats_id: ID of inserted statistics record
        """
        stats_json = json.dumps(stats)
        
        self.cursor.execute("""
            INSERT INTO player_statistics (match_id, player_id, stats)
            VALUES (?, ?, ?)
        """, (match_id, player_id, stats_json))
        
        stats_id = self.cursor.lastrowid
        return stats_id
    
    def insert_all_player_statistics(
        self,
        match_id: int,
        player_id_map: Dict[int, int],
        tracks_data: List[Dict],
        possession_player_stats: Dict[str, Dict]
    ):
        """
        Insert statistics for all players.
        
        Combines tracking stats and possession stats.
        
        Args:
            match_id: Match ID
            player_id_map: Dict mapping player_number to database player_id
            tracks_data: Player tracking data from metrics.json
            possession_player_stats: Player possession stats from possession tracker
        """
        for track in tracks_data:
            player_number = track['player_id']
            player_id = player_id_map.get(player_number)
            
            if player_id is None:
                continue
            
            # Combine tracking stats and possession stats
            stats = {
                'team': track['team'],
                'total_distance_m': track['total_distance_m'],
                'avg_speed_mps': track['avg_speed_mps'],
                'max_speed_mps': track['max_speed_mps'],
                'workload_score': track['workload_score'],
                'involvement_index': track['involvement_index']
            }
            
            # Add possession stats if available
            poss_stats = possession_player_stats.get(str(player_number))
            if poss_stats:
                stats['possession'] = poss_stats
            
            self.insert_player_statistics(match_id, player_id, stats)
        
        self.commit()
        print(f"✅ Inserted statistics for {len(tracks_data)} players")
    
    # ============================================================
    # HEATMAP OPERATIONS
    # ============================================================
    
    def insert_heatmap(
        self,
        match_id: int,
        player_id: int,
        heatmap_path: str
    ) -> int:
        """
        Insert heatmap reference.
        
        Args:
            match_id: Match ID
            player_id: Player ID (from database)
            heatmap_path: Path to heatmap image file
            
        Returns:
            heatmap_id: ID of inserted heatmap record
        """
        heatmap_data = json.dumps({
            'file_path': heatmap_path,
            'type': 'player_heatmap'
        })
        
        self.cursor.execute("""
            INSERT INTO heatmaps (match_id, player_id, data)
            VALUES (?, ?, ?)
        """, (match_id, player_id, heatmap_data))
        
        heatmap_id = self.cursor.lastrowid
        return heatmap_id
    
    def insert_all_heatmaps(
        self,
        match_id: int,
        player_id_map: Dict[int, int],
        heatmap_dir: Path
    ):
        """
        Insert heatmap references for all players.
        
        Args:
            match_id: Match ID
            player_id_map: Dict mapping player_number to database player_id
            heatmap_dir: Directory containing heatmap images
        """
        count = 0
        for heatmap_file in heatmap_dir.glob("heatmap_player_*.png"):
            # Extract player number from filename
            try:
                player_number = int(heatmap_file.stem.replace("heatmap_player_", ""))
            except:
                continue
            
            player_id = player_id_map.get(player_number)
            if player_id is None:
                continue
            
            self.insert_heatmap(match_id, player_id, str(heatmap_file))
            count += 1
        
        self.commit()
        print(f"✅ Inserted {count} heatmap references")
    
    # ============================================================
    # QUERY OPERATIONS
    # ============================================================
    
    def get_match_by_id(self, match_id: int) -> Optional[Dict]:
        """Get match data by ID."""
        self.cursor.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        row = self.cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'date': row[1],
                'team_a': row[2],
                'team_b': row[3],
                'venue': row[4],
                'fps': row[5],
                'width': row[6],
                'height': row[7]
            }
        return None
    
    def get_all_matches(self) -> List[Dict]:
        """Get all matches."""
        self.cursor.execute("SELECT * FROM matches ORDER BY date DESC")
        rows = self.cursor.fetchall()
        
        matches = []
        for row in rows:
            matches.append({
                'id': row[0],
                'date': row[1],
                'team_a': row[2],
                'team_b': row[3],
                'venue': row[4],
                'fps': row[5],
                'width': row[6],
                'height': row[7]
            })
        
        return matches
    
    def get_possession_events(self, match_id: int) -> List[Dict]:
        """Get all possession events for a match."""
        self.cursor.execute("""
            SELECT * FROM events 
            WHERE match_id = ? AND event_type = 'possession'
            ORDER BY timestamp
        """, (match_id,))
        
        rows = self.cursor.fetchall()
        events = []
        
        for row in rows:
            events.append({
                'id': row[0],
                'match_id': row[1],
                'event_type': row[2],
                'timestamp': row[3],
                'player_id': row[4],
                'metadata': json.loads(row[6])
            })
        
        return events
