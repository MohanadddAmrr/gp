"""Tests for DynamicRosterManager (Task C2)."""

import json
from pathlib import Path

import pytest

from services.dynamic_roster_manager import DynamicRosterManager


def test_create_and_get_team(tmp_path: Path) -> None:
    db = tmp_path / "dr.db"
    m = DynamicRosterManager(db)
    tid = m.create_team("X United", "XUN", "#111111", "#222222")
    assert tid > 0
    t = m.get_team(tid)
    assert t is not None
    assert t.name == "X United"
    assert t.short_code == "XUN"
    teams = m.list_teams()
    assert any(x.id == tid for x in teams)


def test_create_player_requires_team(tmp_path: Path) -> None:
    m = DynamicRosterManager(tmp_path / "dr2.db")
    with pytest.raises(ValueError, match="does not exist"):
        m.create_player(999999, "Ghost", 9, "MID")


def test_list_players_by_team(tmp_path: Path) -> None:
    m = DynamicRosterManager(tmp_path / "dr3.db")
    tid = m.create_team("Y FC", "YFC", "#FFFFFF", "#000000")
    m.create_player(tid, "Alice", 5, "DEF")
    m.create_player(tid, "Bob", 6, "MID")
    allp = m.list_players()
    assert len(allp) == 2
    tp = m.list_players(tid)
    assert len(tp) == 2
    assert {p.name for p in tp} == {"Alice", "Bob"}


def test_lineup_round_trip(tmp_path: Path) -> None:
    dbp = tmp_path / "dr4.db"
    m = DynamicRosterManager(dbp)
    import sqlite3

    conn = sqlite3.connect(str(dbp))
    try:
        conn.execute(
            "INSERT INTO matches (video_path, team_a, team_b) VALUES (?, ?, ?)",
            ("test.mp4", "Y FC", "Z FC"),
        )
        conn.commit()
        cur = conn.cursor()
        cur.execute("SELECT match_id FROM matches ORDER BY match_id DESC LIMIT 1")
        mid = int(cur.fetchone()[0])
    finally:
        conn.close()

    t1 = m.create_team("Y FC", "YFC", "#FFFFFF", "#000000")
    t2 = m.create_team("Z FC", "ZFC", "#0000FF", "#FFFFFF")
    p1 = m.create_player(t1, "P1", 1, "GK")
    p2 = m.create_player(t2, "P2", 2, "GK")
    m.set_lineup(mid, [p1, p2], starters={p1})
    lu = m.get_lineup(mid)
    assert len(lu) == 2
    assert {p.id for p in lu} == {p1, p2}


def test_bulk_import_from_json(tmp_path: Path) -> None:
    jf = tmp_path / "tiny.json"
    jf.write_text(
        json.dumps(
            {
                "match": "A vs B",
                "teams": {
                    "A": {"name": "SideA", "jersey_color": "red", "players": {"1": "One"}},
                    "B": {"name": "SideB", "jersey_color": "blue", "players": {"2": "Two"}},
                },
            }
        ),
        encoding="utf-8",
    )
    m = DynamicRosterManager(tmp_path / "dr5.db")
    out = m.bulk_import_from_json(jf)
    assert out.get("players_touched") == 2
    assert out.get("teams_touched") == 2
    names = {t.name for t in m.list_teams()}
    assert "SideA" in names and "SideB" in names
