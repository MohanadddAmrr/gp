"""
Tests for services/video_downloader.py and services/data_enrichment.py
Owner: Ahmed Khaled (Member E) — Task E3
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from services.video_downloader import download
from services.data_enrichment import DataEnrichment


# ===========================================================================
# video_downloader tests
# ===========================================================================

class TestVideoDownloaderValidation:
    def test_raises_for_disallowed_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Extension"):
            download("http://example.com/video.exe", tmp_path)

    def test_raises_for_html_extension(self, tmp_path):
        with pytest.raises(ValueError):
            download("http://example.com/clip.html", tmp_path)

    def test_allowed_extensions_accepted(self, tmp_path):
        """MP4 and MKV should pass validation (actual HTTP call is mocked)."""
        for ext in (".mp4", ".mkv"):
            url = f"http://example.com/clip{ext}"
            with patch("services.video_downloader.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.headers = {"Content-Length": "100"}
                mock_resp.iter_content.return_value = [b"fakevideobytes"]
                mock_resp.raise_for_status.return_value = None
                mock_get.return_value = mock_resp
                result = download(url, tmp_path)
                assert result is not None


class TestVideoDownloaderSizeLimit:
    def test_rejects_oversized_file(self, tmp_path):
        """Files larger than max_size_mb must return None without downloading."""
        url = "http://example.com/huge.mp4"
        with patch("services.video_downloader.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"Content-Length": str(600 * 1024 * 1024)}  # 600 MB
            mock_resp.raise_for_status.return_value = None
            mock_resp.iter_content.return_value = []
            mock_get.return_value = mock_resp
            result = download(url, tmp_path, max_size_mb=500)
        assert result is None


class TestVideoDownloaderResume:
    def test_sends_range_header_when_partial_file_exists(self, tmp_path):
        """If a partial file exists, Range header must be set."""
        url = "http://example.com/clip.mp4"
        partial = tmp_path / "clip.mp4"
        partial.write_bytes(b"x" * 100)  # simulate 100-byte partial download

        with patch("services.video_downloader.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 206  # Partial Content
            mock_resp.headers = {"Content-Length": "200"}
            mock_resp.iter_content.return_value = [b"morebytes"]
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            download(url, tmp_path)
            _, kwargs = mock_get.call_args
            assert "Range" in kwargs.get("headers", {})
            assert kwargs["headers"]["Range"] == "bytes=100-"

    def test_returns_existing_file_on_416(self, tmp_path):
        """HTTP 416 means file is already complete — return the path."""
        url = "http://example.com/clip.mp4"
        full_file = tmp_path / "clip.mp4"
        full_file.write_bytes(b"complete")

        with patch("services.video_downloader.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 416
            mock_get.return_value = mock_resp
            result = download(url, tmp_path)
        assert result == full_file


class TestVideoDownloaderNetworkFailure:
    def test_returns_none_on_connection_error(self, tmp_path):
        import requests as req
        url = "http://example.com/clip.mp4"
        with patch("services.video_downloader.requests.get", side_effect=req.exceptions.ConnectionError):
            result = download(url, tmp_path)
        assert result is None


# ===========================================================================
# data_enrichment tests
# ===========================================================================

def _make_enrichment(api_matches=None):
    """Build a DataEnrichment with mocked sync + db."""
    sync = MagicMock()
    sync.sync_matches.return_value = api_matches or [
        {
            "match_id": "99",
            "competition": "Premier League",
            "season": "2024",
            "match_date": "2024-09-15T14:00:00",
            "home_team": "Arsenal FC",
            "away_team": "Manchester City FC",
            "home_score": 2,
            "away_score": 1,
            "source": "Football-data.org",
        }
    ]
    db = MagicMock()
    return DataEnrichment(sync, db)


class TestEnrichMatch:
    def test_returns_required_keys(self):
        enricher = _make_enrichment()
        result = enricher.enrich_match("Arsenal FC", "Manchester City FC")
        for key in ("api_data", "video_data", "enriched", "enriched_at"):
            assert key in result

    def test_enriched_contains_team_names(self):
        enricher = _make_enrichment()
        result = enricher.enrich_match("Arsenal FC", "Manchester City FC")
        e = result["enriched"]
        assert "Arsenal" in e["home_team"]
        assert "Manchester" in e["away_team"]

    def test_enriched_contains_score(self):
        enricher = _make_enrichment()
        result = enricher.enrich_match("Arsenal FC", "Manchester City FC")
        assert result["enriched"]["home_score"] == 2
        assert result["enriched"]["away_score"] == 1

    def test_video_data_merged_when_metrics_provided(self, tmp_path):
        metrics = {
            "total_frames": 1000,
            "possession": {"home": 55, "away": 45},
            "shots": {"home": 8, "away": 5},
            "xg": {"home": 1.8, "away": 0.9},
        }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        enricher = _make_enrichment()
        result = enricher.enrich_match(
            "Arsenal FC", "Manchester City FC", metrics_path=metrics_file
        )
        assert result["enriched"]["total_frames"] == 1000
        assert result["enriched"]["possession_home"] == 55

    def test_no_crash_when_api_returns_no_match(self):
        enricher = _make_enrichment(api_matches=[])
        result = enricher.enrich_match("Unknown Team A", "Unknown Team B")
        assert result["api_data"] == {}
        assert "enriched" in result


class TestTeamMatching:
    def test_fuzzy_match_strips_fc(self):
        assert DataEnrichment._team_matches("Arsenal FC", "Arsenal")
        assert DataEnrichment._team_matches("Manchester City FC", "Manchester City")

    def test_exact_match(self):
        assert DataEnrichment._team_matches("Liverpool FC", "Liverpool FC")

    def test_no_match_different_teams(self):
        assert not DataEnrichment._team_matches("Arsenal FC", "Chelsea")


class TestGuessFolderName:
    def test_vs_separator(self):
        home, away = DataEnrichment._guess_teams_from_folder("arsenal_vs_city")
        assert home == "Arsenal"
        assert away == "City"

    def test_v_separator(self):
        home, away = DataEnrichment._guess_teams_from_folder("liverpool_v_everton")
        assert home == "Liverpool"
        assert away == "Everton"

    def test_unknown_pattern(self):
        home, away = DataEnrichment._guess_teams_from_folder("randomfoldername")
        assert home == "Unknown"
        assert away == "Unknown"


class TestGetEnrichedSummary:
    def test_summary_format(self):
        enricher = _make_enrichment()
        record = {
            "enriched": {
                "home_team": "Arsenal FC",
                "away_team": "Man City",
                "home_score": 2,
                "away_score": 1,
                "competition": "Premier League",
            }
        }
        summary = enricher.get_enriched_summary(record)
        assert "Arsenal" in summary
        assert "2" in summary
        assert "Premier League" in summary
