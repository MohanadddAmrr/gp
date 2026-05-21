"""
Migration 001 (additive): bring existing SQLite DB up to current schema safely.

Goals:
- Never drop tables/columns (additive only)
- Be idempotent (safe to run multiple times)
- Preserve existing data

This migration:
1) Runs the canonical `SCHEMA_SQL` (CREATE TABLE/INDEX IF NOT EXISTS)
2) Adds missing columns to existing tables via ALTER TABLE ... ADD COLUMN
"""

from __future__ import annotations

import sys
from pathlib import Path
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

try:
    from services.database_schema import SCHEMA_SQL, create_schema
except ModuleNotFoundError:
    # When executed from a subdirectory (e.g. `python scripts/...`),
    # ensure repository root is on sys.path so `services` is importable.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from services.database_schema import SCHEMA_SQL, create_schema


ColumnSpec = Tuple[str, str]  # (name, sqlite_column_definition)


@dataclass(frozen=True)
class MigrationResult:
    created_or_verified_schema: bool
    added_columns: int


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    return cur.fetchone() is not None


def _get_existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    # row format: cid, name, type, notnull, dflt_value, pk
    return {row[1] for row in cur.fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Sequence[ColumnSpec]) -> int:
    """
    Add missing columns to an existing table.
    Returns number of columns added.
    """
    if not _table_exists(conn, table):
        return 0

    existing = _get_existing_columns(conn, table)
    cur = conn.cursor()
    added = 0
    for col_name, col_def in columns:
        if col_name in existing:
            continue
        print(f"DEBUG: Adding missing column {col_name} to {table}")
        # SQLite limitation: ALTER TABLE ... ADD COLUMN only supports constant defaults.
        # If a non-constant default is present (e.g. CURRENT_TIMESTAMP), add the column
        # without that default and backfill existing rows.
        needs_backfill_ts = "DEFAULT CURRENT_TIMESTAMP" in col_def.upper()
        alter_def = col_def
        if needs_backfill_ts:
            alter_def = col_def.replace("DEFAULT CURRENT_TIMESTAMP", "").replace("default current_timestamp", "")
            alter_def = " ".join(alter_def.split())

        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {alter_def}")
        if needs_backfill_ts:
            cur.execute(
                f"UPDATE {table} SET {col_name} = CURRENT_TIMESTAMP WHERE {col_name} IS NULL"
            )
        added += 1
    return added


def migrate_001_additive(conn: sqlite3.Connection) -> MigrationResult:
    # Keep FK behavior consistent for new connections.
    conn.execute("PRAGMA foreign_keys = ON;")

    # 1) Ensure any missing columns exist for older DBs.
    #    Important: Some DBs may already have tables (e.g. `matches`) with fewer columns,
    #    while `SCHEMA_SQL` also creates indexes that reference newer columns (e.g. `match_date`).
    #    We must add missing columns first so index creation does not fail.
    #    Notes:
    #    - SQLite cannot add constraints to existing tables via ALTER TABLE (beyond adding a column).
    #    - We keep defs conservative (nullable where possible) so existing rows remain valid.
    column_migrations: Iterable[Tuple[str, Sequence[ColumnSpec]]] = [
        (
            "matches",
            [
                ("match_date", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("duration_seconds", "REAL DEFAULT 0"),
                ("score_a", "INTEGER DEFAULT 0"),
                ("score_b", "INTEGER DEFAULT 0"),
                ("venue", "TEXT"),
                ("fps", "REAL DEFAULT 25.0"),
                ("width", "INTEGER DEFAULT 1280"),
                ("height", "INTEGER DEFAULT 720"),
                ("processed", "BOOLEAN DEFAULT 0"),
                ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ],
        ),
        (
            "teams",
            [
                ("color", "TEXT"),
                ("formation", "TEXT"),
            ],
        ),
        (
            "players",
            [
                ("profile_id", "INTEGER"),
                ("jersey_number", "INTEGER"),
                ("position", "TEXT"),
            ],
        ),
        (
            "player_profiles",
            [
                ("team_name", "TEXT"),
                ("jersey_number", "INTEGER"),
                ("face_encoding", "BLOB"),
                ("face_image_path", "TEXT"),
                ("date_of_birth", "DATE"),
                ("nationality", "TEXT"),
                ("position_default", "TEXT"),
                ("height_cm", "INTEGER"),
                ("weight_kg", "INTEGER"),
                ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ],
        ),
        (
            "players_master",
            [
                ("profile_id", "INTEGER"),
                ("full_name", "TEXT"),
                ("preferred_name", "TEXT"),
                ("date_of_birth", "DATE"),
                ("nationality", "TEXT"),
                ("position_default", "TEXT"),
                ("height_cm", "INTEGER"),
                ("weight_kg", "INTEGER"),
                ("external_ids", "TEXT"),
                ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ],
        ),
        (
            "rosters",
            [
                ("match_id", "INTEGER"),
                ("team_name", "TEXT"),
                ("side", "TEXT"),
                ("source", "TEXT"),
                ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ],
        ),
        (
            "roster_players",
            [
                ("team_id", "INTEGER"),
                ("name", "TEXT"),
                ("jersey_number", "INTEGER"),
                ("position", "TEXT"),
            ],
        ),
        (
            "player_stats",
            [
                ("profile_id", "INTEGER"),
                ("total_distance_m", "REAL DEFAULT 0"),
                ("sprint_distance_m", "REAL DEFAULT 0"),
                ("high_intensity_distance_m", "REAL DEFAULT 0"),
                ("avg_speed_mps", "REAL DEFAULT 0"),
                ("max_speed_mps", "REAL DEFAULT 0"),
                ("passes_attempted", "INTEGER DEFAULT 0"),
                ("passes_completed", "INTEGER DEFAULT 0"),
                ("shots", "INTEGER DEFAULT 0"),
                ("shots_on_target", "INTEGER DEFAULT 0"),
                ("goals", "INTEGER DEFAULT 0"),
                ("assists", "INTEGER DEFAULT 0"),
                ("tackles", "INTEGER DEFAULT 0"),
                ("interceptions", "INTEGER DEFAULT 0"),
                ("sprints", "INTEGER DEFAULT 0"),
                ("possession_count", "INTEGER DEFAULT 0"),
                ("possession_duration_s", "REAL DEFAULT 0"),
                ("touches", "INTEGER DEFAULT 0"),
                ("workload_score", "REAL DEFAULT 0"),
                ("minutes_played", "REAL DEFAULT 0"),
                ("heatmap_path", "TEXT"),
            ],
        ),
        (
            "events",
            [
                ("player_instance_id", "INTEGER"),
                ("profile_id", "INTEGER"),
                ("team_id", "INTEGER"),
                ("x", "REAL"),
                ("y", "REAL"),
                ("metadata", "TEXT"),
            ],
        ),
        (
            "tracking_data",
            [
                ("player_instance_id", "INTEGER"),
                ("profile_id", "INTEGER"),
                ("x_norm", "REAL"),
                ("y_norm", "REAL"),
                ("speed_mps", "REAL"),
                ("has_possession", "BOOLEAN DEFAULT 0"),
                ("team_in_possession", "TEXT"),
            ],
        ),
        (
            "ball_tracking",
            [
                ("x_px", "REAL"),
                ("y_px", "REAL"),
                ("x_norm", "REAL"),
                ("y_norm", "REAL"),
                ("velocity_x", "REAL"),
                ("velocity_y", "REAL"),
                ("velocity_magnitude", "REAL"),
                ("detection_method", "TEXT"),
                ("confidence", "REAL"),
                ("possessing_player_id", "INTEGER"),
                ("possessing_team", "TEXT"),
            ],
        ),
        (
            "heatmaps",
            [
                ("data_json", "TEXT"),
                ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ],
        ),
        (
            "team_stats",
            [
                ("possession_percentage", "REAL DEFAULT 50.0"),
                ("possession_count", "INTEGER DEFAULT 0"),
                ("passes_attempted", "INTEGER DEFAULT 0"),
                ("passes_completed", "INTEGER DEFAULT 0"),
                ("pass_accuracy", "REAL DEFAULT 0"),
                ("shots", "INTEGER DEFAULT 0"),
                ("shots_on_target", "INTEGER DEFAULT 0"),
                ("goals", "INTEGER DEFAULT 0"),
                ("total_distance_m", "REAL DEFAULT 0"),
                ("sprints", "INTEGER DEFAULT 0"),
                ("defensive_third_possession", "REAL DEFAULT 0"),
                ("midfield_possession", "REAL DEFAULT 0"),
                ("attacking_third_possession", "REAL DEFAULT 0"),
            ],
        ),
        (
            "xg_data",
            [
                ("event_id", "INTEGER"),
                ("frame", "INTEGER"),
                ("shooter_id", "INTEGER"),
                ("shooter_team", "TEXT"),
                ("distance_to_goal", "REAL"),
                ("angle_to_goal", "REAL"),
                ("velocity_mps", "REAL"),
                ("big_chance", "BOOLEAN DEFAULT 0"),
                ("metadata", "TEXT"),
            ],
        ),
        (
            "highlights",
            [
                ("frame", "INTEGER"),
                ("importance", "INTEGER"),
                ("primary_player_id", "INTEGER"),
                ("secondary_player_id", "INTEGER"),
                ("team", "TEXT"),
                ("description", "TEXT"),
                ("clip_start", "REAL"),
                ("clip_end", "REAL"),
                ("xg_value", "REAL"),
                ("velocity", "REAL"),
                ("metadata", "TEXT"),
            ],
        ),
        (
            "tactical_patterns",
            [
                ("outcome", "TEXT"),
                ("start_time", "REAL"),
                ("end_time", "REAL"),
                ("duration", "REAL"),
                ("start_zone", "TEXT"),
                ("end_zone", "TEXT"),
                ("pass_count", "INTEGER"),
                ("distance_covered", "REAL"),
                ("xg_generated", "REAL"),
                ("players_involved", "TEXT"),
                ("description", "TEXT"),
            ],
        ),
        (
            "opponent_profiles",
            [
                ("preferred_formation", "TEXT"),
                ("primary_style", "TEXT"),
                ("pressing_intensity", "REAL"),
                ("defensive_line_height", "REAL"),
                ("set_piece_threat", "REAL"),
                ("avg_pass_length", "REAL"),
                ("zone_preferences", "TEXT"),
                ("key_players", "TEXT"),
                ("matches_analyzed", "INTEGER DEFAULT 0"),
                ("last_updated", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ],
        ),
        (
            "passing_networks",
            [
                ("degree_centrality", "REAL"),
                ("betweenness_centrality", "REAL"),
                ("closeness_centrality", "REAL"),
                ("connections", "TEXT"),
            ],
        ),
        (
            "zone_control",
            [
                ("team_a_control", "REAL"),
                ("team_b_control", "REAL"),
                ("contested", "REAL"),
                ("team_a_time", "REAL"),
                ("team_b_time", "REAL"),
            ],
        ),
        (
            "advanced_analytics",
            [
                ("metric_value", "REAL"),
                ("metadata", "TEXT"),
            ],
        ),
    ]

    added_cols = 0
    for table, cols in column_migrations:
        added_cols += _ensure_columns(conn, table, cols)

    # 2) Ensure tables/indexes exist (idempotent).
    conn.executescript(SCHEMA_SQL)
    create_schema(conn)

    return MigrationResult(created_or_verified_schema=True, added_columns=added_cols)


def migrate_001_additive_path(db_path: str) -> MigrationResult:
    conn = sqlite3.connect(db_path)
    try:
        res = migrate_001_additive(conn)
        conn.commit()
        return res
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run additive Migration 001 on a SQLite database.")
    parser.add_argument("--db", default="matches.db", help="Path to SQLite database file (default: matches.db)")
    args = parser.parse_args()

    result = migrate_001_additive_path(args.db)
    print(f"[migration-001] ok, added_columns={result.added_columns}")

