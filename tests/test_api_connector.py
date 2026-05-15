"""
Tests for services/api_connector.py
Owner: Ahmed Khaled (Member E) — Task E1
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from services.api_connector import (
    FootballDataConnector,
    APIConfig,
    MatchData,
    APIManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "207f21ffeb5641209ff73113ecedf7c0")


def make_connector() -> FootballDataConnector:
    return FootballDataConnector(API_KEY)


# ---------------------------------------------------------------------------
# Unit tests (no real network calls)
# ---------------------------------------------------------------------------

class TestFootballDataConnectorInit:
    """The connector must be created correctly and hold the right config."""

    def test_base_url_is_v4(self):
        conn = make_connector()
        assert "v4" in conn.config.base_url

    def test_api_key_stored(self):
        conn = make_connector()
        assert conn.config.api_key == API_KEY

    def test_rate_limit_default(self):
        conn = make_connector()
        assert conn.config.rate_limit == 10


class TestAuthHeader:
    """Every request must carry the X-Auth-Token header."""

    def test_x_auth_token_added_to_request(self):
        conn = make_connector()
        with patch.object(conn.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"competitions": []}
            mock_get.return_value = mock_resp

            conn.get_competitions()

            _, kwargs = mock_get.call_args
            headers = kwargs.get("headers", {})
            assert "X-Auth-Token" in headers
            assert headers["X-Auth-Token"] == API_KEY


class TestCaching:
    """Results must be cached so we don't hammer the API."""

    def test_competitions_cached_after_first_call(self):
        conn = make_connector()
        fake_competitions = [{"id": 2021, "name": "Premier League"}]

        with patch.object(conn, "_make_request", return_value={"competitions": fake_competitions}) as mock_req:
            conn.get_competitions()
            conn.get_competitions()  # second call — must use cache
            assert mock_req.call_count == 1  # only called once

    def test_standings_cached_after_first_call(self):
        conn = make_connector()
        fake_standings = [{"stage": "REGULAR_SEASON", "table": []}]

        with patch.object(conn, "_make_request", return_value={"standings": fake_standings}) as mock_req:
            conn.get_standings("2021")
            conn.get_standings("2021")
            assert mock_req.call_count == 1


class TestGetCompetitions:
    """get_competitions() must return a list of dicts with 'id' and 'name'."""

    def test_returns_list_on_success(self):
        conn = make_connector()
        fake = {"competitions": [{"id": 2021, "name": "Premier League"}]}
        with patch.object(conn, "_make_request", return_value=fake):
            result = conn.get_competitions()
        assert isinstance(result, list)
        assert result[0]["name"] == "Premier League"

    def test_returns_empty_list_on_failure(self):
        conn = make_connector()
        with patch.object(conn, "_make_request", return_value=None):
            result = conn.get_competitions()
        assert result == []


class TestGetStandings:
    """get_standings() must return a list of standing tables."""

    def test_returns_standings_table(self):
        conn = make_connector()
        fake = {"standings": [{"stage": "REGULAR_SEASON", "table": [{"position": 1}]}]}
        with patch.object(conn, "_make_request", return_value=fake):
            result = conn.get_standings("2021")
        assert len(result) == 1
        assert result[0]["table"][0]["position"] == 1

    def test_returns_empty_on_api_failure(self):
        conn = make_connector()
        with patch.object(conn, "_make_request", return_value=None):
            result = conn.get_standings("2021")
        assert result == []


class TestGetMatches:
    """get_matches() must parse the API response into MatchData objects."""

    def _fake_match_payload(self):
        return {
            "matches": [
                {
                    "id": 123,
                    "competition": {"name": "Premier League"},
                    "season": {"startDate": "2024-08-01"},
                    "utcDate": "2024-09-15T14:00:00Z",
                    "homeTeam": {"name": "Arsenal FC"},
                    "awayTeam": {"name": "Manchester City FC"},
                    "score": {"fullTime": {"home": 2, "away": 1}},
                }
            ]
        }

    def test_returns_match_data_objects(self):
        conn = make_connector()
        with patch.object(conn, "_make_request", return_value=self._fake_match_payload()):
            matches = conn.get_matches(competition_id="2021")
        assert len(matches) == 1
        assert isinstance(matches[0], MatchData)

    def test_match_fields_parsed_correctly(self):
        conn = make_connector()
        with patch.object(conn, "_make_request", return_value=self._fake_match_payload()):
            matches = conn.get_matches(competition_id="2021")
        m = matches[0]
        assert m.home_team == "Arsenal FC"
        assert m.away_team == "Manchester City FC"
        assert m.home_score == 2
        assert m.away_score == 1
        assert m.source == "Football-data.org"

    def test_returns_empty_on_api_failure(self):
        conn = make_connector()
        with patch.object(conn, "_make_request", return_value=None):
            result = conn.get_matches(competition_id="2021")
        assert result == []


class TestGetTeamSquad:
    """fetch_squad() must return players with name, jersey_number, position."""

    def test_fetch_squad_by_team_id(self):
        conn = make_connector()
        fake_team_data = {
            "name": "Arsenal FC",
            "squad": [
                {"name": "Bukayo Saka", "shirtNumber": 7, "position": "Right Winger"},
                {"name": "Martin Odegaard", "shirtNumber": 8, "position": "Midfielder"},
            ],
        }
        with patch.object(conn, "_make_request", return_value=fake_team_data):
            with patch.object(conn, "_get_disk_cached", return_value=None):
                with patch.object(conn, "_set_disk_cached"):
                    squad = conn.fetch_squad(57)  # Arsenal team ID
        assert len(squad) == 2
        assert squad[0]["name"] == "Bukayo Saka"
        assert squad[0]["jersey_number"] == 7

    def test_fetch_squad_returns_empty_on_no_data(self):
        conn = make_connector()
        with patch.object(conn, "_make_request", return_value={}):
            with patch.object(conn, "_get_disk_cached", return_value=None):
                squad = conn.fetch_squad(99999)
        assert squad == []


class TestAPIManager:
    """APIManager must register and use the football_data connector."""

    def test_add_football_data_connector(self):
        manager = APIManager()
        manager.add_football_data(API_KEY)
        assert "football_data" in manager.connectors

    def test_get_team_squad_returns_player_list(self):
        """Integration-style: APIManager → FootballDataConnector → fetch_squad."""
        manager = APIManager()
        manager.add_football_data(API_KEY)
        connector = manager.connectors["football_data"]

        fake_squad = [{"name": "Saka", "jersey_number": 7, "position": "FW"}]
        with patch.object(connector, "fetch_squad", return_value=fake_squad):
            result = connector.fetch_squad(57)
        assert len(result) == 1
        assert result[0]["name"] == "Saka"


# ---------------------------------------------------------------------------
# Live smoke test (skipped in CI if no key set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("FOOTBALL_DATA_API_KEY"),
    reason="No live API key — set FOOTBALL_DATA_API_KEY to run"
)
class TestLiveAPI:
    """Quick smoke tests against the real football-data.org API."""

    def test_live_get_competitions(self):
        conn = make_connector()
        comps = conn.get_competitions()
        assert len(comps) > 0
        assert "name" in comps[0]
        assert "id" in comps[0]

    def test_live_get_premier_league_standings(self):
        conn = make_connector()
        standings = conn.get_standings("2021")
        assert len(standings) > 0
        table = standings[0].get("table", [])
        assert len(table) > 0
        assert "team" in table[0]
        assert "points" in table[0]
