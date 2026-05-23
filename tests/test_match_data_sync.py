"""
Tests for services/match_data_sync.py
Owner: Ahmed Khaled (Member E) — Task E2
"""

import pytest
from unittest.mock import MagicMock, patch
from services.match_data_sync import MatchDataSync
from services.api_connector import MatchData
from datetime import datetime


def _make_sync(competitions=None, matches=None, standings=None):
    """Helper: build a MatchDataSync with a fully mocked API + DB."""
    connector = MagicMock()
    connector.get_competitions.return_value = competitions or [
        {"id": 2021, "name": "Premier League"}
    ]
    connector.get_standings.return_value = standings or [
        {"stage": "REGULAR_SEASON", "table": [{"position": 1, "team": {"name": "Arsenal FC"}, "points": 79}]}
    ]
    connector.get_matches.return_value = matches or [
        MatchData(
            match_id="1",
            competition="Premier League",
            season="2024",
            match_date=datetime(2024, 9, 15),
            home_team="Arsenal FC",
            away_team="Man City",
            home_score=2,
            away_score=1,
            source="Football-data.org",
        )
    ]
    connector.fetch_squad.return_value = [
        {"name": "Saka", "jersey_number": 7, "position": "FW"}
    ]
    connector.fetch_and_save_squad.return_value = [
        {"name": "Saka", "jersey_number": 7, "position": "FW"}
    ]

    api = MagicMock()
    api.connectors = {"football_data": connector}

    db = MagicMock()
    db.save_match_result.return_value = None
    db.create_player_profile.return_value = None

    return MatchDataSync(api, db)


# ---------------------------------------------------------------------------
# sync_competitions
# ---------------------------------------------------------------------------

class TestSyncCompetitions:
    def test_returns_list_of_competitions(self):
        sync = _make_sync()
        result = sync.sync_competitions()
        assert isinstance(result, list)
        assert result[0]["name"] == "Premier League"

    def test_sync_competitions_idempotent(self):
        """Calling sync_competitions twice should not crash or duplicate data."""
        sync = _make_sync()
        first = sync.sync_competitions()
        second = sync.sync_competitions()
        assert first == second

    def test_returns_empty_list_when_api_fails(self):
        sync = _make_sync()
        sync._get_connector().get_competitions.return_value = []
        result = sync.sync_competitions()
        assert result == []


# ---------------------------------------------------------------------------
# sync_matches
# ---------------------------------------------------------------------------

class TestSyncMatches:
    def test_returns_raw_match_dicts(self):
        sync = _make_sync()
        matches = sync.sync_matches("2021")
        assert len(matches) == 1
        assert matches[0]["home_team"] == "Arsenal FC"
        assert matches[0]["away_team"] == "Man City"

    def test_match_dict_has_required_keys(self):
        sync = _make_sync()
        m = sync.sync_matches("2021")[0]
        for key in ("match_id", "competition", "home_team", "away_team", "home_score", "away_score"):
            assert key in m, f"Missing key: {key}"

    def test_sync_matches_handles_partial_failure(self):
        """DB write errors on individual matches must not abort the whole sync."""
        sync = _make_sync()
        sync.db.save_match_result.side_effect = Exception("DB write error")
        # Should not raise, just log the warning
        result = sync.sync_matches("2021")
        assert isinstance(result, list)  # still returns data

    def test_returns_empty_list_when_api_fails(self):
        sync = _make_sync()
        sync._get_connector().get_matches.return_value = []
        result = sync.sync_matches("2021")
        assert result == []


# ---------------------------------------------------------------------------
# sync_team_squad
# ---------------------------------------------------------------------------

class TestSyncTeamSquad:
    def test_returns_squad_list(self):
        sync = _make_sync()
        squad = sync.sync_team_squad("Arsenal FC")
        assert len(squad) == 1
        assert squad[0]["name"] == "Saka"

    def test_sync_team_squad_creates_players_via_DynamicRosterManager(self):
        """
        When fetch_and_save_squad is available it should be used
        (DynamicRosterManager integration point).
        """
        sync = _make_sync()
        connector = sync._get_connector()
        squad = sync.sync_team_squad("Arsenal FC")
        connector.fetch_and_save_squad.assert_called_once_with("Arsenal FC", sync.db)
        assert len(squad) == 1

    def test_falls_back_to_fetch_squad_when_save_not_available(self):
        """If fetch_and_save_squad is missing, fall back to fetch_squad."""
        sync = _make_sync()
        connector = sync._get_connector()
        del connector.fetch_and_save_squad  # simulate missing method
        squad = sync.sync_team_squad("Arsenal FC")
        connector.fetch_squad.assert_called_once_with("Arsenal FC")
        assert isinstance(squad, list)


# ---------------------------------------------------------------------------
# get_standings
# ---------------------------------------------------------------------------

class TestGetStandings:
    def test_returns_standings(self):
        sync = _make_sync()
        standings = sync.get_standings("2021")
        assert len(standings) == 1
        table = standings[0]["table"]
        assert table[0]["team"]["name"] == "Arsenal FC"

    def test_returns_empty_when_api_fails(self):
        sync = _make_sync()
        sync._get_connector().get_standings.return_value = []
        result = sync.get_standings("2021")
        assert result == []


# ---------------------------------------------------------------------------
# get_recent_matches
# ---------------------------------------------------------------------------

class TestGetRecentMatches:
    def test_returns_limited_matches(self):
        many = [
            MatchData(
                match_id=str(i),
                competition="Premier League",
                season="2024",
                match_date=datetime(2024, 9, i + 1),
                home_team="Team A",
                away_team="Team B",
                home_score=1,
                away_score=0,
                source="Football-data.org",
            )
            for i in range(15)
        ]
        sync = _make_sync(matches=many)
        result = sync.get_recent_matches("2021", limit=5)
        assert len(result) == 5

    def test_newest_first(self):
        sync = _make_sync()
        results = sync.get_recent_matches("2021", limit=10)
        dates = [r["match_date"] for r in results]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# Error: connector not registered
# ---------------------------------------------------------------------------

class TestMissingConnector:
    def test_raises_runtime_error_when_connector_missing(self):
        api = MagicMock()
        api.connectors = {}   # no football_data key
        db = MagicMock()
        sync = MatchDataSync(api, db)
        with pytest.raises(RuntimeError, match="football_data"):
            sync.sync_competitions()
