"""
Match Data Sync Service
Owner: Ahmed Khaled (Member E) — Task E2

Syncs competitions, matches, and squad data from football-data.org
into the local SQLite database so the dashboard can display live data
without hitting the API on every page load.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MatchDataSync:
    """
    Syncs data from the football-data.org API into the local database.

    Usage:
        from services.api_connector import APIManager
        from services.database_manager import DatabaseManager
        from services.match_data_sync import MatchDataSync

        api = APIManager()
        api.add_football_data(os.environ["FOOTBALL_DATA_API_KEY"])
        db  = DatabaseManager()
        sync = MatchDataSync(api, db)

        sync.sync_competitions()
        sync.sync_matches("2021")          # Premier League
        sync.sync_team_squad("Arsenal FC")
    """

    # Premier League competition ID on football-data.org
    DEFAULT_COMPETITION = "2021"

    def __init__(self, api, db):
        """
        Args:
            api: APIManager instance (already has 'football_data' connector added).
            db:  DatabaseManager instance.
        """
        self.api = api
        self.db = db
        self._connector = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_connector(self):
        """Return the FootballDataConnector, raising clearly if missing."""
        if self._connector is None:
            if "football_data" not in self.api.connectors:
                raise RuntimeError(
                    "FootballDataConnector not registered. "
                    "Call api.add_football_data(API_KEY) first."
                )
            self._connector = self.api.connectors["football_data"]
        return self._connector

    # ------------------------------------------------------------------
    # Public sync methods
    # ------------------------------------------------------------------

    def sync_competitions(self) -> List[Dict]:
        """
        Fetch all available competitions from football-data.org and return them.

        This method is idempotent — calling it multiple times returns the same
        data without creating duplicate DB records (competitions are cached by
        the connector).

        Returns:
            List of competition dicts: [{id, name, area, ...}, ...]
        """
        connector = self._get_connector()
        competitions = connector.get_competitions()

        if not competitions:
            logger.warning("sync_competitions: API returned no competitions.")
            return []

        logger.info("sync_competitions: fetched %d competitions.", len(competitions))
        return competitions

    def sync_matches(
        self,
        competition_id: str = DEFAULT_COMPETITION,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Fetch matches for a competition and attempt to store them in the DB.

        If a DB write fails for one match (e.g. schema mismatch), the error is
        logged and we continue with the rest — partial failures are tolerated.

        Args:
            competition_id: football-data.org competition ID (default: Premier League).
            date_from:      Optional start date filter.
            date_to:        Optional end date filter.

        Returns:
            List of raw match dicts that were successfully fetched from the API.
        """
        connector = self._get_connector()
        match_objects = connector.get_matches(
            competition_id=competition_id,
            date_from=date_from,
            date_to=date_to,
        )

        if not match_objects:
            logger.warning(
                "sync_matches: no matches returned for competition %s.", competition_id
            )
            return []

        saved = 0
        raw_matches = []

        for m in match_objects:
            # Preserve None scores — None means the match hasn't been played yet
            raw = {
                "match_id": m.match_id,
                "competition": m.competition,
                "season": m.season,
                "match_date": m.match_date.isoformat() if m.match_date else "",
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_score": m.home_score,   # keep None for unplayed matches
                "away_score": m.away_score,   # keep None for unplayed matches
                "source": m.source,
            }
            raw_matches.append(raw)

            # Only persist finished matches (scores known)
            if m.home_score is None:
                continue

            # Try to persist — tolerate failures
            try:
                self.db.create_match(
                    video_path=f"api://{m.source}/{m.match_id}",
                    team_a=m.home_team,
                    team_b=m.away_team,
                    score_a=m.home_score,
                    score_b=m.away_score,
                )
                saved += 1
            except Exception as exc:
                logger.warning(
                    "sync_matches: could not save match %s vs %s — %s",
                    m.home_team,
                    m.away_team,
                    exc,
                )

        logger.info(
            "sync_matches: fetched %d, saved %d for competition %s.",
            len(match_objects),
            saved,
            competition_id,
        )
        return raw_matches

    def sync_team_squad(self, team_name: str) -> List[Dict]:
        """
        Fetch a team's squad from football-data.org and save players to the DB.

        Delegates to FootballDataConnector.fetch_and_save_squad so caching and
        DB writes are handled there.  If DynamicRosterManager is not yet merged,
        we fall back to DatabaseManager.create_player_profile.

        Args:
            team_name: Team name as it appears on football-data.org (e.g. "Arsenal FC").

        Returns:
            List of player dicts: [{name, jersey_number, position}, ...]
        """
        connector = self._get_connector()

        # Prefer the rich save method; fall back gracefully
        if hasattr(connector, "fetch_and_save_squad"):
            squad = connector.fetch_and_save_squad(team_name, self.db)
        else:
            squad = connector.fetch_squad(team_name)
            for player in squad:
                try:
                    self.db.create_player_profile(
                        name=player["name"],
                        team_name=team_name,
                        jersey_number=player.get("jersey_number"),
                        position_default=player.get("position"),
                    )
                except Exception as exc:
                    logger.warning(
                        "sync_team_squad: could not save %s — %s",
                        player.get("name"),
                        exc,
                    )

        logger.info(
            "sync_team_squad: synced %d players for %s.", len(squad), team_name
        )
        return squad

    def get_standings(self, competition_id: str = DEFAULT_COMPETITION) -> List[Dict]:
        """
        Fetch current standings for a competition.

        Args:
            competition_id: football-data.org competition ID.

        Returns:
            List of standing-table dicts as returned by the API.
        """
        connector = self._get_connector()
        standings = connector.get_standings(competition_id)
        logger.info(
            "get_standings: fetched %d standing tables for competition %s.",
            len(standings),
            competition_id,
        )
        return standings

    def get_current_matchday(self, competition_id: str = DEFAULT_COMPETITION) -> int:
        """
        Return the current (or most recently played) matchday number.
        Falls back to 1 if the API doesn't provide it.
        """
        connector = self._get_connector()
        md = connector.get_current_matchday(competition_id)
        return md if md is not None else 1

    def get_matches_by_gameweek(
        self,
        competition_id: str = DEFAULT_COMPETITION,
        matchday: int = 1,
    ) -> List[Dict]:
        """
        Fetch all matches for a specific gameweek, newest-first.

        Args:
            competition_id: football-data.org competition ID.
            matchday:       Gameweek / matchday number (1-38).

        Returns:
            List of raw match dicts for that gameweek.
        """
        connector = self._get_connector()
        match_objects = connector.get_matches(
            competition_id=competition_id,
            matchday=matchday,
        )
        raw = []
        for m in match_objects:
            raw.append({
                "match_id":    m.match_id,
                "competition": m.competition,
                "season":      m.season,
                "match_date":  m.match_date.isoformat() if m.match_date else "",
                "home_team":   m.home_team,
                "away_team":   m.away_team,
                "home_score":  m.home_score,
                "away_score":  m.away_score,
                "source":      m.source,
                "matchday":    m.matchday,
            })
        raw.sort(key=lambda m: m.get("match_date", ""), reverse=True)
        return raw

    def get_recent_matches(
        self,
        competition_id: str = DEFAULT_COMPETITION,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Fetch the most recent FINISHED matches for a competition (newest-first).
        Kept for backward compatibility — dashboard now uses get_matches_by_gameweek().
        """
        raw = self.sync_matches(competition_id=competition_id)
        finished = [m for m in raw if m.get("home_score") is not None]
        finished.sort(key=lambda m: m.get("match_date", ""), reverse=True)
        return finished[:limit]
