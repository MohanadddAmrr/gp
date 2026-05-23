"""Tests for migrations/001_dynamic_rosters.py roster JSON backfill."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


def _load_migration_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "migrations" / "001_dynamic_rosters.py"
    name = "migrations.dynamic_rosters_001"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _roster_doc(team_a: str, team_b: str, p_a: dict, p_b: dict) -> dict:
    return {
        "match": f"{team_a} vs {team_b}",
        "teams": {
            "A": {
                "name": team_a,
                "jersey_color": "red",
                "players": {str(k): v for k, v in p_a.items()},
            },
            "B": {
                "name": team_b,
                "jersey_color": "blue",
                "players": {str(k): v for k, v in p_b.items()},
            },
        },
    }


def test_migration_loads_all_rosters(tmp_path: Path) -> None:
    m = _load_migration_module()
    roster_dir = tmp_path / "rosters"
    roster_dir.mkdir()
    (roster_dir / "alphavsbeta.json").write_text(
        json.dumps(
            _roster_doc(
                "Alpha United",
                "Beta City",
                {1: "Keeper A", 2: "Def A"},
                {1: "Keeper B", 3: "Mid B"},
            )
        ),
        encoding="utf-8",
    )
    (roster_dir / "gammavsdelta.json").write_text(
        json.dumps(
            _roster_doc(
                "Gamma Town",
                "Delta FC",
                {10: "Striker G"},
                {11: "Winger D"},
            )
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "mig.db"
    conn = sqlite3.connect(str(db_path))
    try:
        s = m.migrate_conn(conn, roster_dir)
        assert len(s.roster_files) == 2
        cur = conn.cursor()
        cur.execute(f'SELECT COUNT(*) FROM "{m.LINEUP_TEAMS_TABLE}"')
        assert cur.fetchone()[0] == 4
        cur.execute(f'SELECT COUNT(*) FROM "{m.LINEUP_PLAYERS_TABLE}"')
        assert cur.fetchone()[0] == 6
    finally:
        conn.close()


def test_migration_idempotent(tmp_path: Path) -> None:
    m = _load_migration_module()
    roster_dir = tmp_path / "rosters2"
    roster_dir.mkdir()
    (roster_dir / "onevstwo.json").write_text(
        json.dumps(
            _roster_doc(
                "One FC",
                "Two SC",
                {7: "P1"},
                {8: "P2"},
            )
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "idem.db"
    conn = sqlite3.connect(str(db_path))
    try:
        m.migrate_conn(conn, roster_dir)
        cur = conn.cursor()
        cur.execute(f'SELECT COUNT(*) FROM "{m.LINEUP_TEAMS_TABLE}"')
        t1 = cur.fetchone()[0]
        cur.execute(f'SELECT COUNT(*) FROM "{m.LINEUP_PLAYERS_TABLE}"')
        p1 = cur.fetchone()[0]
        m.migrate_conn(conn, roster_dir)
        cur.execute(f'SELECT COUNT(*) FROM "{m.LINEUP_TEAMS_TABLE}"')
        t2 = cur.fetchone()[0]
        cur.execute(f'SELECT COUNT(*) FROM "{m.LINEUP_PLAYERS_TABLE}"')
        p2 = cur.fetchone()[0]
        assert t1 == t2 == 2
        assert p1 == p2 == 2
    finally:
        conn.close()
