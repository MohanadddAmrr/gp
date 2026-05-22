import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RosterManager:
    """
    Service layer for managing database rosters and migrating from static JSON files.
    """

    def __init__(self, db_manager):
        self.db = db_manager

    def import_from_json(self, json_path) -> Tuple[Optional[int], Optional[int]]:
        """Import a roster JSON file into the database.
        Parse the JSON structure, create teams if they don't exist,
        add all players. Return (home_team_id, away_team_id)."""
        # Handle the existing JSON format in rosters/ folder
        # Detect team names from JSON keys or filename
        # Skip duplicate players (same name + jersey + team)
        json_path = Path(json_path)
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return None, None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse JSON {json_path}: {e}")
            return None, None

        match_name = data.get("match") or json_path.stem
        teams = data.get("teams") or {}

        home_team_id = None
        away_team_id = None

        for side in ("A", "B"):
            team_info = teams.get(side) or {}
            team_name = team_info.get("name")
            if not team_name:
                parts = json_path.stem.split("vs")
                if len(parts) == 2:
                    team_name = parts[0].strip().capitalize() if side == "A" else parts[1].strip().capitalize()
                else:
                    team_name = f"Team {side} ({json_path.stem})"

            color = team_info.get("jersey_color") or ('#FFFFFF' if side == 'A' else '#000000')

            team_id = self.db.add_team(name=team_name, primary_color=color)
            if side == "A":
                home_team_id = team_id
            else:
                away_team_id = team_id

            players = team_info.get("players") or {}
            existing_keys = {
                (p["name"], p.get("jersey_number"))
                for p in self.db.get_players_by_team(team_id)
            }
            for jersey_str, p_value in players.items():
                try:
                    jersey_number = int(jersey_str)
                except (ValueError, TypeError):
                    jersey_number = None

                player_name = "Unknown"
                position = None
                if isinstance(p_value, str):
                    player_name = p_value.strip()
                elif isinstance(p_value, dict):
                    player_name = (p_value.get("name") or p_value.get("full_name") or "Unknown").strip()
                    position = p_value.get("position")

                dedupe_key = (player_name, jersey_number)
                if dedupe_key in existing_keys:
                    continue
                existing_keys.add(dedupe_key)

                self.db.add_player(
                    team_id=team_id,
                    name=player_name,
                    jersey_number=jersey_number,
                    position=position
                )

        if home_team_id and away_team_id:
            self.db.assign_roster_to_match(match_name, home_team_id, away_team_id)
            stem_key = json_path.stem
            if str(stem_key) != str(match_name):
                self.db.assign_roster_to_match(stem_key, home_team_id, away_team_id)

        return home_team_id, away_team_id

    def import_all_json_files(self, rosters_dir='rosters/'):
        """Import all JSON files from the rosters directory."""
        target_dir = Path(rosters_dir)
        if not target_dir.exists():
            logger.warning(f"Rosters directory not found: {rosters_dir}")
            return

        for json_file in sorted(target_dir.glob("*.json")):
            logger.info(f"Importing roster JSON: {json_file.name}")
            home_id, away_id = self.import_from_json(json_file)
            if home_id is None or away_id is None:
                logger.warning(f"Incomplete import for {json_file.name}: home={home_id}, away={away_id}")

    def get_player_lookup(self, match_id):
        """Get {jersey_number: player_name} lookup for a match.
        Returns dict for home team and away team separately."""
        roster = self.db.get_match_roster(match_id)
        if roster is None:
            return None  # Signals: fall back to JSON
        home_lookup = self.db.get_roster_lookup(roster['home_team_id'])
        away_lookup = self.db.get_roster_lookup(roster['away_team_id'])
        return {'home': home_lookup, 'away': away_lookup}

    def create_team_with_players(self, team_name, players_list):
        """Convenience: create a team and add all players at once.
        players_list = [{'name': str, 'number': int, 'position': str}, ...]"""
        team_id = self.db.add_team(name=team_name)
        existing_keys = {
            (p["name"], p.get("jersey_number"))
            for p in self.db.get_players_by_team(team_id)
        }
        for p in players_list:
            name = p.get("name", "Unknown").strip()
            num = p.get("number")
            pos = p.get("position")
            key = (name, num)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            self.db.add_player(team_id, name, num, pos)
        return team_id
