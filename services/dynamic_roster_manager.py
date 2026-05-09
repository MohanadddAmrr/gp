import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DynamicRosterManager:
    """
    Centralized manager for roster operations.
    Abstracts CRUD operations and lookups for the new roster tables.
    """
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_all_rosters(self, team_name: Optional[str] = None, match_id: Optional[int] = None) -> List[Dict]:
        """Fetch all rosters, optionally filtered by team name and match id."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            
            query = "SELECT roster_id, match_id, team_name, side, source, created_at, updated_at FROM rosters"
            params = []
            conditions = []
            
            if team_name:
                conditions.append("team_name = ?")
                params.append(team_name)
                
            if match_id is not None:
                conditions.append("match_id = ?")
                params.append(match_id)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY roster_id DESC"
            
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def ensure_roster(self, team_name: str, side: str, match_id: Optional[int] = None, source: str = 'ui') -> int:
        """Create a new roster if one doesn't exist, or return the most recent active one."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO rosters (match_id, team_name, side, source)
                VALUES (?, ?, ?, ?)
                """,
                (match_id, team_name, side, source)
            )
            return cur.lastrowid

    def get_roster_players(self, roster_id: int) -> List[Dict]:
        """Fetch all players belonging to a specific roster."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 
                    rp.roster_player_id, rp.roster_id, rp.player_id, rp.profile_id, 
                    rp.jersey_number, rp.position, rp.is_starting, rp.metadata,
                    pm.full_name
                FROM roster_players rp
                JOIN players_master pm ON rp.player_id = pm.player_id
                WHERE rp.roster_id = ?
                ORDER BY rp.is_starting DESC, rp.jersey_number ASC
                """,
                (roster_id,)
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def upsert_player_master(self, full_name: str) -> int:
        """Ensure a player exists in players_master and return their ID."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT player_id FROM players_master WHERE full_name = ? LIMIT 1", (full_name,))
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute("INSERT INTO players_master (full_name) VALUES (?)", (full_name,))
            return cur.lastrowid

    def add_player_to_roster(self, roster_id: int, full_name: str, jersey_number: Optional[int], position: Optional[str], is_starting: bool) -> int:
        """Add a player to a roster."""
        player_id = self.upsert_player_master(full_name)
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO roster_players (roster_id, player_id, jersey_number, position, is_starting)
                VALUES (?, ?, ?, ?, ?)
                """,
                (roster_id, player_id, jersey_number, position, int(is_starting))
            )
            return cur.lastrowid

    def update_roster_player(self, roster_player_id: int, full_name: str, jersey_number: Optional[int], position: Optional[str], is_starting: bool):
        """Update an existing player in a roster."""
        player_id = self.upsert_player_master(full_name)
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE roster_players
                SET player_id = ?, jersey_number = ?, position = ?, is_starting = ?
                WHERE roster_player_id = ?
                """,
                (player_id, jersey_number, position, int(is_starting), roster_player_id)
            )

    def remove_player_from_roster(self, roster_player_id: int):
        """Remove a player from a roster."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM roster_players WHERE roster_player_id = ?", (roster_player_id,))

    def quick_update_roster_player(self, roster_player_id: int, jersey_number: Optional[int], position: Optional[str], is_starting: Optional[int]):
        """Quickly update an existing player in a roster without changing their name."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE roster_players
                SET jersey_number = COALESCE(?, jersey_number),
                    position = COALESCE(?, position),
                    is_starting = COALESCE(?, is_starting)
                WHERE roster_player_id = ?
                """,
                (jersey_number, position, is_starting, roster_player_id)
            )

    def fetch_active_roster(self, team_name: str, side: Optional[str] = None, match_id: Optional[int] = None) -> Optional[int]:
        """Fetch the most relevant roster ID."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            if match_id is not None:
                cur.execute(
                    """
                    SELECT roster_id FROM rosters
                    WHERE team_name = ? AND side IS ? AND match_id = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (team_name, side, match_id)
                )
            else:
                cur.execute(
                    """
                    SELECT roster_id FROM rosters
                    WHERE team_name = ? AND side IS ? AND match_id IS NULL
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (team_name, side)
                )
            row = cur.fetchone()
            return row[0] if row else None
