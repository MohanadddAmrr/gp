"""Tests for C1 relational roster DDL in services.database_schema."""

import sqlite3
from pathlib import Path

import pytest

from services.database_schema import (
    LINEUP_PLAYERS_TABLE,
    LINEUP_TEAMS_TABLE,
    MATCH_LINEUPS_TABLE,
    alter_matches_lineup_columns,
    create_schema,
    lineup_index_names,
)


def test_create_schema_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "c1.db"
    conn = sqlite3.connect(str(db))
    try:
        create_schema(conn)
        create_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (LINEUP_TEAMS_TABLE,),
        )
        assert cur.fetchone() is not None
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (LINEUP_PLAYERS_TABLE,),
        )
        assert cur.fetchone() is not None
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (MATCH_LINEUPS_TABLE,),
        )
        assert cur.fetchone() is not None
    finally:
        conn.close()


def test_alter_matches_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "alt.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                team_a TEXT NOT NULL,
                team_b TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE "{LINEUP_TEAMS_TABLE}" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                short_code TEXT,
                primary_color TEXT,
                secondary_color TEXT
            )
            """
        )
        conn.commit()
        assert alter_matches_lineup_columns(conn) == 2
        assert alter_matches_lineup_columns(conn) == 0
        conn.commit()
    finally:
        conn.close()


def test_indexes_created(tmp_path: Path) -> None:
    db = tmp_path / "idx.db"
    conn = sqlite3.connect(str(db))
    try:
        create_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        names = {row[0] for row in cur.fetchall()}
        for idx in lineup_index_names():
            assert idx in names, f"missing index {idx!r}"
    finally:
        conn.close()
