"""
Data Enrichment Service
Owner: Ahmed Khaled (Member E) — Task E3

Merges external API data (from football-data.org via MatchDataSync)
with local video-analysis data (from the database / metrics JSON)
to produce an enriched match record ready for the dashboard and PDF report.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DataEnrichment:
    """
    Enriches local video-analysis match records with live API data.

    Typical flow
    ------------
    1. Load local metrics from demo_outputs/<match>/ or from matches.db.
    2. Fetch the corresponding API record (fixtures, standings, squad).
    3. Merge both into one enriched dict and optionally write it back to the DB.

    Usage:
        from services.data_enrichment import DataEnrichment
        from services.match_data_sync import MatchDataSync
        from services.database_manager import DatabaseManager

        enricher = DataEnrichment(sync, db)
        result = enricher.enrich_match("Arsenal FC", "Manchester City FC", "2021")
        print(result)
    """

    def __init__(self, sync, db):
        """
        Args:
            sync: MatchDataSync instance.
            db:   DatabaseManager instance.
        """
        self.sync = sync
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_match(
        self,
        home_team: str,
        away_team: str,
        competition_id: str = "2021",
        metrics_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Build an enriched match record combining API and local video data.

        Args:
            home_team:       Home team name (must match football-data.org spelling).
            away_team:       Away team name.
            competition_id:  football-data.org competition ID (default: Premier League).
            metrics_path:    Optional path to a local metrics.json file.

        Returns:
            Dict with keys:
                - "api_data":    raw match dict from the API (or {})
                - "video_data":  local metrics dict (or {})
                - "enriched":    merged summary dict
                - "enriched_at": ISO timestamp
        """
        api_data = self._find_api_match(home_team, away_team, competition_id)
        video_data = self._load_video_metrics(metrics_path) if metrics_path else {}

        enriched = self._merge(home_team, away_team, api_data, video_data)

        result = {
            "api_data": api_data,
            "video_data": video_data,
            "enriched": enriched,
            "enriched_at": datetime.utcnow().isoformat(),
        }

        # Persist enriched data back to DB if possible
        self._save_enriched(enriched)

        return result

    def enrich_all_from_demo_outputs(
        self,
        demo_outputs_dir: str | Path = "demo/demo_outputs",
        competition_id: str = "2021",
    ) -> List[Dict[str, Any]]:
        """
        Walk demo_outputs/ and enrich every match that has a metrics.json.

        Args:
            demo_outputs_dir: Root folder containing per-match sub-folders.
            competition_id:   Competition to search for API data.

        Returns:
            List of enriched match dicts (one per metrics.json found).
        """
        results = []
        demo_dir = Path(demo_outputs_dir)

        if not demo_dir.exists():
            logger.warning("demo_outputs_dir not found: %s", demo_dir)
            return results

        for match_dir in demo_dir.iterdir():
            if not match_dir.is_dir():
                continue
            metrics_file = match_dir / "metrics.json"
            if not metrics_file.exists():
                continue

            # Try to parse team names from the folder name (e.g. "arsenal_vs_city")
            home_team, away_team = self._guess_teams_from_folder(match_dir.name)
            try:
                result = self.enrich_match(
                    home_team=home_team,
                    away_team=away_team,
                    competition_id=competition_id,
                    metrics_path=metrics_file,
                )
                results.append(result)
                logger.info("Enriched: %s", match_dir.name)
            except Exception as exc:
                logger.error("Failed to enrich %s: %s", match_dir.name, exc)

        logger.info("Enriched %d matches from %s", len(results), demo_outputs_dir)
        return results

    def get_enriched_summary(self, enriched_record: Dict[str, Any]) -> str:
        """Return a human-readable one-line summary of an enriched record."""
        e = enriched_record.get("enriched", {})
        home = e.get("home_team", "?")
        away = e.get("away_team", "?")
        score = f"{e.get('home_score', '?')} – {e.get('away_score', '?')}"
        comp = e.get("competition", "Unknown competition")
        return f"{home} {score} {away} ({comp})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_api_match(
        self, home_team: str, away_team: str, competition_id: str
    ) -> Dict:
        """Look up the API record for this fixture."""
        try:
            matches = self.sync.sync_matches(competition_id=competition_id)
            for m in matches:
                if self._team_matches(m.get("home_team", ""), home_team) and \
                   self._team_matches(m.get("away_team", ""), away_team):
                    return m
        except Exception as exc:
            logger.warning("API lookup failed for %s vs %s: %s", home_team, away_team, exc)
        return {}

    @staticmethod
    def _team_matches(api_name: str, query: str) -> bool:
        """Fuzzy team name match — tolerates 'Arsenal FC' vs 'Arsenal'."""
        api_lower = api_name.lower().strip()
        query_lower = query.lower().strip().replace(" fc", "").replace(" cf", "")
        return query_lower in api_lower or api_lower.startswith(query_lower)

    @staticmethod
    def _load_video_metrics(metrics_path: Path) -> Dict:
        """Load local metrics.json, returning {} on failure."""
        try:
            with open(metrics_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load metrics from %s: %s", metrics_path, exc)
            return {}

    @staticmethod
    def _merge(
        home_team: str,
        away_team: str,
        api_data: Dict,
        video_data: Dict,
    ) -> Dict[str, Any]:
        """Combine API and video data into one flat enriched dict."""
        enriched: Dict[str, Any] = {
            # Identity
            "home_team":   api_data.get("home_team", home_team),
            "away_team":   api_data.get("away_team", away_team),
            "competition": api_data.get("competition", ""),
            "season":      api_data.get("season", ""),
            "match_date":  api_data.get("match_date", ""),
            "home_score":  api_data.get("home_score"),
            "away_score":  api_data.get("away_score"),
            # API-sourced metadata
            "api_match_id": api_data.get("match_id", ""),
            "api_source":   api_data.get("source", "football-data.org"),
            # Video-analysis metrics (top-level keys only to keep it flat)
            "total_frames":        video_data.get("total_frames"),
            "possession_home":     video_data.get("possession", {}).get("home"),
            "possession_away":     video_data.get("possession", {}).get("away"),
            "shots_home":          video_data.get("shots", {}).get("home"),
            "shots_away":          video_data.get("shots", {}).get("away"),
            "xg_home":             video_data.get("xg", {}).get("home"),
            "xg_away":             video_data.get("xg", {}).get("away"),
            "tracking_mota":       video_data.get("tracking", {}).get("mota"),
            "tracking_idf1":       video_data.get("tracking", {}).get("idf1"),
            "id_switches":         video_data.get("tracking", {}).get("id_switches"),
        }
        return enriched

    def _save_enriched(self, enriched: Dict[str, Any]):
        """
        Persist enriched match record to the database (best-effort).

        Uses create_match() — the correct DatabaseManager method.
        If the record already exists or the write fails, we log and continue.
        """
        try:
            home = enriched.get("home_team", "")
            away = enriched.get("away_team", "")
            if not home or not away:
                return
            if hasattr(self.db, "create_match"):
                match_id = self.db.create_match(
                    video_path=f"enriched://{enriched.get('api_match_id', 'unknown')}",
                    team_a=home,
                    team_b=away,
                    score_a=enriched.get("home_score") or 0,
                    score_b=enriched.get("away_score") or 0,
                )
                logger.info(
                    "Saved enriched match to DB: ID=%s (%s vs %s)",
                    match_id, home, away,
                )
        except Exception as exc:
            logger.debug("Could not persist enriched record: %s", exc)

    @staticmethod
    def _guess_teams_from_folder(folder_name: str):
        """
        Try to split a folder name like 'arsenal_vs_city' into ('arsenal', 'city').
        Returns ('Unknown', 'Unknown') if the pattern is not recognised.
        """
        for sep in ("_vs_", " vs ", "-vs-", "_v_", " v "):
            if sep in folder_name.lower():
                parts = folder_name.lower().split(sep, 1)
                return parts[0].replace("_", " ").title(), parts[1].replace("_", " ").title()
        return "Unknown", "Unknown"
