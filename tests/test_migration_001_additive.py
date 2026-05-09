import sqlite3
from pathlib import Path
import importlib.util
import sys


def _load_migration_001_module(repo_root: Path):
    migration_path = repo_root / "migrations" / "001_additive.py"
    spec = importlib.util.spec_from_file_location("migration_001_additive", migration_path)
    assert spec and spec.loader, f"Failed to load migration module from {migration_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def test_migration_001_adds_missing_columns_and_preserves_data(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    migration = _load_migration_001_module(repo_root)

    db_path = tmp_path / "dev_old.db"
    conn = sqlite3.connect(db_path)
    try:
        # Simulate a pre-v2 DB: minimal matches table (older schema).
        conn.executescript(
            """
            CREATE TABLE matches (
              match_id INTEGER PRIMARY KEY AUTOINCREMENT,
              video_path TEXT NOT NULL,
              team_a TEXT NOT NULL,
              team_b TEXT NOT NULL
            );
            INSERT INTO matches (video_path, team_a, team_b) VALUES ('vid.mp4', 'A', 'B');
            """
        )
        conn.commit()
    finally:
        conn.close()

    # Run migration.
    res = migration.migrate_001_additive_path(str(db_path))
    assert res.created_or_verified_schema is True
    assert res.added_columns >= 1

    # Verify: core tables exist and columns added.
    conn = sqlite3.connect(db_path)
    try:
        cols = _table_columns(conn, "matches")
        for expected in {
            "match_date",
            "duration_seconds",
            "score_a",
            "score_b",
            "venue",
            "fps",
            "width",
            "height",
            "processed",
            "created_at",
        }:
            assert expected in cols

        # Verify: add-on tables exist (created by SCHEMA_SQL).
        for table in [
            "teams",
            "players",
            "player_profiles",
            "players_master",
            "rosters",
            "roster_players",
            "player_stats",
            "events",
            "tracking_data",
            "ball_tracking",
            "heatmaps",
            "team_stats",
            "xg_data",
            "highlights",
            "tactical_patterns",
            "opponent_profiles",
            "passing_networks",
            "zone_control",
            "advanced_analytics",
        ]:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (table,),
            )
            assert cur.fetchone() is not None, f"Expected table {table} to exist"

        # Verify: existing row preserved.
        cur = conn.cursor()
        cur.execute("SELECT match_id, video_path, team_a, team_b FROM matches")
        rows = cur.fetchall()
        assert rows == [(1, "vid.mp4", "A", "B")]
    finally:
        conn.close()


def test_migration_001_is_idempotent(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    migration = _load_migration_001_module(repo_root)

    db_path = tmp_path / "dev_idempotent.db"
    sqlite3.connect(db_path).close()

    first = migration.migrate_001_additive_path(str(db_path))
    second = migration.migrate_001_additive_path(str(db_path))

    assert first.created_or_verified_schema is True
    assert second.created_or_verified_schema is True
    assert second.added_columns == 0

