"""
Sync Live Match Data — Day 15 Task
Owner: Ahmed Khaled (Member E)

Pulls live data from football-data.org for all 3 test matches
and saves them into matches.db.

Run with:
    python scripts/sync_live_matches.py

This script is safe to run multiple times (idempotent).
"""

import os
import sys
import json
import logging
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 3 test matches used by the team (matches Mohanad's rosters + ground truth)
# ---------------------------------------------------------------------------
TEST_MATCHES = [
    {
        "home_team": "Liverpool FC",
        "away_team": "Manchester City FC",
        "competition_id": "2021",   # Premier League
        "label": "liverpoolvscity",
    },
    {
        "home_team": "Liverpool FC",
        "away_team": "AFC Bournemouth",
        "competition_id": "2021",
        "label": "liverpoolvsbournemouth",
    },
    {
        "home_team": "Arsenal FC",
        "away_team": "Fulham FC",
        "competition_id": "2021",
        "label": "arsenalvsfulham",
    },
]

OUTPUT_DIR = Path("demo_outputs/live_sync")


def main():
    # --- load API key ---
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        # fall back to .env file
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FOOTBALL_DATA_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        logger.error("FOOTBALL_DATA_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    from services.api_connector import APIManager
    from services.database_manager import DatabaseManager
    from services.match_data_sync import MatchDataSync
    from services.data_enrichment import DataEnrichment

    api = APIManager()
    api.add_football_data(api_key)
    db   = DatabaseManager()
    db.initialize_database()   # create tables if they don't exist yet
    sync = MatchDataSync(api, db)
    enricher = DataEnrichment(sync, db)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_summary = []

    logger.info("=" * 60)
    logger.info("Syncing live data for %d test matches", len(TEST_MATCHES))
    logger.info("=" * 60)

    for match in TEST_MATCHES:
        label      = match["label"]
        home_team  = match["home_team"]
        away_team  = match["away_team"]
        comp_id    = match["competition_id"]

        logger.info("Processing: %s vs %s", home_team, away_team)

        # 1. Pull and sync all matches in this competition (cached after first call)
        raw_matches = sync.sync_matches(competition_id=comp_id)
        logger.info("  Fetched %d matches from Premier League", len(raw_matches))

        # 2. Find this specific fixture in the API data
        found = next(
            (m for m in raw_matches
             if _name_match(m.get("home_team", ""), home_team)
             and _name_match(m.get("away_team", ""), away_team)),
            None,
        )

        if found:
            logger.info(
                "  FOUND: %s %s-%s %s  (date: %s)",
                found["home_team"],
                found.get("home_score", "?"),
                found.get("away_score", "?"),
                found["away_team"],
                found.get("match_date", "")[:10],
            )
        else:
            logger.warning(
                "  NOT FOUND in API — storing placeholder for %s vs %s",
                home_team, away_team,
            )
            found = {
                "home_team":  home_team,
                "away_team":  away_team,
                "competition": "Premier League",
                "season":     "2024",
                "match_date": "",
                "home_score": None,
                "away_score": None,
                "source":     "placeholder",
            }

        # 3. Check if there is a local metrics.json for this match
        metrics_path = None
        demo_dir = Path("demo/demo_outputs")
        for folder in demo_dir.iterdir() if demo_dir.exists() else []:
            if label.split("vs")[0] in folder.name.lower():
                candidate = folder / "metrics.json"
                if candidate.exists():
                    metrics_path = candidate
                    break

        # 4. Enrich and save
        try:
            enriched_record = enricher.enrich_match(
                home_team=home_team,
                away_team=away_team,
                competition_id=comp_id,
                metrics_path=metrics_path,
            )
            logger.info(
                "  Enriched: %s",
                enricher.get_enriched_summary(enriched_record),
            )
        except Exception as exc:
            logger.error("  Enrichment failed: %s", exc)
            enriched_record = {"api_data": found, "video_data": {}, "enriched": found}

        # 5. Save result to JSON for traceability
        out_file = OUTPUT_DIR / f"{label}_live.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(enriched_record, f, indent=2, default=str)
        logger.info("  Saved → %s", out_file)

        results_summary.append({
            "label":     label,
            "home_team": home_team,
            "away_team": away_team,
            "found_in_api": found.get("source") != "placeholder",
            "output_file": str(out_file),
        })

    # 6. Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results_summary:
        status = "✓ LIVE" if r["found_in_api"] else "~ PLACEHOLDER"
        logger.info("  [%s] %s vs %s", status, r["home_team"], r["away_team"])

    # 7. Save summary JSON
    summary_path = OUTPUT_DIR / "sync_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Summary saved → %s", summary_path)
    logger.info("Done. %d/%d matches synced.", len(results_summary), len(TEST_MATCHES))


def _name_match(api_name: str, query: str) -> bool:
    """Tolerant team name matching (handles 'Liverpool FC' vs 'Liverpool')."""
    a = api_name.lower().replace(" fc", "").replace(" afc", "").strip()
    q = query.lower().replace(" fc", "").replace(" afc", "").strip()
    return q in a or a in q


if __name__ == "__main__":
    main()
