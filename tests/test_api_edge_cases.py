"""
Final API Edge Case Tests — Day 17
Owner: Ahmed Khaled (Member E)

Tests for unusual / failure situations that the demo might hit:
- API rate limit hit (429 response)
- Malformed API response
- Network timeout
- DB already initialised (idempotent init)
- sync_live_matches script runs without crashing
- enrich_all_from_demo_outputs handles missing folders gracefully
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import requests as req

from services.api_connector import FootballDataConnector, APIManager
from services.match_data_sync import MatchDataSync
from services.data_enrichment import DataEnrichment

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "207f21ffeb5641209ff73113ecedf7c0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sync(matches=None):
    connector = MagicMock()
    connector.get_competitions.return_value = [{"id": 2021, "name": "Premier League"}]
    connector.get_matches.return_value = matches or []
    connector.get_standings.return_value = []
    api = MagicMock()
    api.connectors = {"football_data": connector}
    db = MagicMock()
    db.create_match.return_value = 1
    return MatchDataSync(api, db), connector, db


# ---------------------------------------------------------------------------
# Rate limit (HTTP 429)
# ---------------------------------------------------------------------------

class TestRateLimitHandling:
    def test_returns_none_on_429(self):
        """When the API returns 429, _make_request should return None gracefully."""
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError(
                response=MagicMock(status_code=429)
            )
            mock_get.return_value = mock_resp
            result = conn._make_request("competitions")
        assert result is None

    def test_get_competitions_returns_empty_on_429(self):
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn, "_make_request", return_value=None):
            result = conn.get_competitions()
        assert result == []

    def test_get_standings_returns_empty_on_429(self):
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn, "_make_request", return_value=None):
            result = conn.get_standings("2021")
        assert result == []


# ---------------------------------------------------------------------------
# Malformed / unexpected API responses
# ---------------------------------------------------------------------------

class TestMalformedResponses:
    def test_competitions_missing_key(self):
        """API returns 200 but body has no 'competitions' key."""
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn, "_make_request", return_value={"error": "unexpected"}):
            result = conn.get_competitions()
        assert result == []

    def test_matches_missing_key(self):
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn, "_make_request", return_value={"filters": {}}):
            result = conn.get_matches(competition_id="2021")
        assert result == []

    def test_standings_missing_key(self):
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn, "_make_request", return_value={"season": {}}):
            result = conn.get_standings("2021")
        assert result == []

    def test_match_with_null_score_does_not_crash(self):
        """Some in-progress matches have null scores — must not crash."""
        conn = FootballDataConnector(API_KEY)
        payload = {
            "matches": [
                {
                    "id": 1,
                    "competition": {"name": "PL"},
                    "season": {"startDate": "2024-08-01"},
                    "utcDate": "2024-09-15T14:00:00Z",
                    "homeTeam": {"name": "Arsenal FC"},
                    "awayTeam": {"name": "Chelsea FC"},
                    "score": {"fullTime": {"home": None, "away": None}},
                }
            ]
        }
        with patch.object(conn, "_make_request", return_value=payload):
            result = conn.get_matches(competition_id="2021")
        assert len(result) == 1
        assert result[0].home_score is None

    def test_competition_with_empty_name(self):
        conn = FootballDataConnector(API_KEY)
        payload = {"competitions": [{"id": 9999, "name": ""}]}
        with patch.object(conn, "_make_request", return_value=payload):
            result = conn.get_competitions()
        assert result[0]["name"] == ""   # empty string, not a crash


# ---------------------------------------------------------------------------
# Network timeout
# ---------------------------------------------------------------------------

class TestNetworkTimeout:
    def test_timeout_returns_none(self):
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn.session, "get", side_effect=req.exceptions.Timeout):
            result = conn._make_request("competitions")
        assert result is None

    def test_connection_error_returns_none(self):
        conn = FootballDataConnector(API_KEY)
        with patch.object(conn.session, "get", side_effect=req.exceptions.ConnectionError):
            result = conn._make_request("competitions")
        assert result is None


# ---------------------------------------------------------------------------
# DB initialisation is idempotent
# ---------------------------------------------------------------------------

class TestDBIdempotency:
    def test_initialize_database_twice_does_not_crash(self, tmp_path):
        """Calling initialize_database twice must not raise (IF NOT EXISTS guards it)."""
        from services.database_manager import DatabaseManager
        db_path = str(tmp_path / "test.db")
        db = DatabaseManager(db_path)
        db.initialize_database()
        db.initialize_database()   # second call — must be silent

    def test_create_match_persists_to_db(self, tmp_path):
        from services.database_manager import DatabaseManager
        db = DatabaseManager(str(tmp_path / "test.db"))
        db.initialize_database()
        mid = db.create_match(
            video_path="test://liverpool_vs_city.mp4",
            team_a="Liverpool FC",
            team_b="Manchester City FC",
            score_a=1,
            score_b=2,
        )
        assert isinstance(mid, int)
        assert mid > 0
        match = db.get_match(mid)
        assert match["team_a"] == "Liverpool FC"
        assert match["score_a"] == 1


# ---------------------------------------------------------------------------
# sync_live_matches script smoke test
# ---------------------------------------------------------------------------

class TestSyncScript:
    def test_script_imports_without_error(self):
        """The sync script module must be importable."""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "sync_live_matches",
            Path("scripts/sync_live_matches.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Just loading it (not running main()) must not crash
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "TEST_MATCHES")
        assert len(mod.TEST_MATCHES) == 3

    def test_name_match_helper(self):
        from scripts.sync_live_matches import _name_match
        assert _name_match("Liverpool FC", "Liverpool FC")
        assert _name_match("Liverpool FC", "Liverpool")
        assert _name_match("AFC Bournemouth", "AFC Bournemouth")
        assert not _name_match("Manchester City FC", "Arsenal")


# ---------------------------------------------------------------------------
# enrich_all_from_demo_outputs — graceful handling of missing folders
# ---------------------------------------------------------------------------

class TestEnrichAllGraceful:
    def test_missing_demo_outputs_dir_returns_empty(self):
        sync = MagicMock()
        db = MagicMock()
        enricher = DataEnrichment(sync, db)
        result = enricher.enrich_all_from_demo_outputs(
            demo_outputs_dir="/nonexistent/path/xyz"
        )
        assert result == []

    def test_folder_without_metrics_json_is_skipped(self, tmp_path):
        # Create a folder with no metrics.json
        (tmp_path / "arsenal_vs_fulham").mkdir()
        sync = MagicMock()
        sync.sync_matches.return_value = []
        db = MagicMock()
        enricher = DataEnrichment(sync, db)
        result = enricher.enrich_all_from_demo_outputs(demo_outputs_dir=tmp_path)
        assert result == []   # skipped because no metrics.json

    def test_valid_metrics_json_is_processed(self, tmp_path):
        match_dir = tmp_path / "liverpool_vs_city"
        match_dir.mkdir()
        metrics = {"total_frames": 500, "possession": {"home": 60, "away": 40}}
        (match_dir / "metrics.json").write_text(json.dumps(metrics))

        sync = MagicMock()
        sync.sync_matches.return_value = []
        db = MagicMock()
        db.create_match.return_value = 1
        enricher = DataEnrichment(sync, db)
        result = enricher.enrich_all_from_demo_outputs(demo_outputs_dir=tmp_path)
        assert len(result) == 1
        assert result[0]["video_data"]["total_frames"] == 500
