"""
Migration 001 — Dynamic roster tables + JSON backfill (Task C1 / 7.3).

- Applies services.database_schema.create_schema (lineup_teams, lineup_players, match_lineups).
- Loads rosters/*.json into lineup_teams / lineup_players (parameterized SQL, idempotent).
- Best-effort: link matches.home_team_id / away_team_id and match_lineups when names align.

Does not delete rosters/*.json (Member F may still reference them).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.database_schema import (  # noqa: E402
    LINEUP_PLAYERS_TABLE,
    LINEUP_TEAMS_TABLE,
    MATCH_LINEUPS_TABLE,
    create_schema,
)


@dataclass
class MigrationSummary:
    teams_insert_attempted: int = 0
    players_insert_attempted: int = 0
    roster_files: List[str] = field(default_factory=list)
    file_to_matches: Dict[str, List[int]] = field(default_factory=dict)


def _short_code(team_name: str) -> str:
    alnum = "".join(c for c in (team_name or "") if c.isalnum())
    return (alnum[:3] or "TM").upper()


def _guess_match_ids(
    cur: sqlite3.Cursor, stem: str, team_a: str, team_b: str
) -> List[int]:
    cur.execute(
        """
        SELECT match_id FROM matches
        WHERE (
            (lower(trim(team_a)) = lower(trim(?)) AND lower(trim(team_b)) = lower(trim(?)))
            OR (lower(trim(team_a)) = lower(trim(?)) AND lower(trim(team_b)) = lower(trim(?)))
        )
        """,
        (team_a, team_b, team_b, team_a),
    )
    rows = [r[0] for r in cur.fetchall()]
    if rows:
        return rows
    pat = f"%{stem.lower()}%"
    cur.execute(
        "SELECT match_id FROM matches WHERE lower(CAST(video_path AS TEXT)) LIKE ?",
        (pat,),
    )
    return [r[0] for r in cur.fetchall()]


def _upsert_team(
    cur: sqlite3.Cursor, name: str, short_code: str, primary: str, secondary: str
) -> int:
    cur.execute(
        f"""
        INSERT OR IGNORE INTO "{LINEUP_TEAMS_TABLE}"
            (name, short_code, primary_color, secondary_color)
        VALUES (?, ?, ?, ?)
        """,
        (name.strip(), short_code, primary, secondary),
    )
    cur.execute(
        f'SELECT id FROM "{LINEUP_TEAMS_TABLE}" WHERE name = ? LIMIT 1',
        (name.strip(),),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Failed to resolve team id for {name!r}")
    return int(row[0])


def _insert_player_ignore(
    cur: sqlite3.Cursor, team_id: int, pname: str, jersey: Optional[int], position: Optional[str]
) -> None:
    cur.execute(
        f"""
        INSERT OR IGNORE INTO "{LINEUP_PLAYERS_TABLE}"
            (team_id, name, jersey_number, position, dob, height_cm, weight_kg, active)
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, 1)
        """,
        (team_id, pname.strip(), jersey, position),
    )


def _link_match_lineups(
    cur: sqlite3.Cursor,
    match_id: int,
    home_team_id: int,
    away_team_id: int,
) -> None:
    cur.execute(
        f"""
        UPDATE matches SET
            home_team_id = CASE WHEN home_team_id IS NULL THEN ? ELSE home_team_id END,
            away_team_id = CASE WHEN away_team_id IS NULL THEN ? ELSE away_team_id END
        WHERE match_id = ?
        """,
        (home_team_id, away_team_id, match_id),
    )
    cur.execute(
        f'SELECT id FROM "{LINEUP_PLAYERS_TABLE}" WHERE team_id = ?',
        (home_team_id,),
    )
    home_pids = [int(r[0]) for r in cur.fetchall()]
    cur.execute(
        f'SELECT id FROM "{LINEUP_PLAYERS_TABLE}" WHERE team_id = ?',
        (away_team_id,),
    )
    away_pids = [int(r[0]) for r in cur.fetchall()]
    for pid in home_pids:
        cur.execute(
            f"""
            INSERT OR IGNORE INTO "{MATCH_LINEUPS_TABLE}"
                (match_id, player_id, position_in_lineup, is_starter)
            VALUES (?, ?, ?, 1)
            """,
            (match_id, pid, "home"),
        )
    for pid in away_pids:
        cur.execute(
            f"""
            INSERT OR IGNORE INTO "{MATCH_LINEUPS_TABLE}"
                (match_id, player_id, position_in_lineup, is_starter)
            VALUES (?, ?, ?, 1)
            """,
            (match_id, pid, "away"),
        )


def migrate_conn(conn: sqlite3.Connection, rosters_dir: Path) -> MigrationSummary:
    summary = MigrationSummary()
    create_schema(conn)
    cur = conn.cursor()

    if not rosters_dir.is_dir():
        conn.commit()
        return summary

    json_files = sorted(rosters_dir.glob("*.json"))
    for jpath in json_files:
        summary.roster_files.append(jpath.name)
        try:
            data: Dict[str, Any] = json.loads(jpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        teams = data.get("teams") or {}
        stem = jpath.stem
        team_ids: Dict[str, int] = {}
        resolved_a = ""
        resolved_b = ""

        for side in ("A", "B"):
            info = teams.get(side) or {}
            tname = (info.get("name") or "").strip()
            if not tname:
                parts = stem.split("vs")
                if len(parts) == 2:
                    tname = (
                        parts[0].strip().title()
                        if side == "A"
                        else parts[1].strip().title()
                    )
                else:
                    tname = f"Team {side} ({stem})"
            if side == "A":
                resolved_a = tname
            else:
                resolved_b = tname
            jc = info.get("jersey_color") or ("#FFFFFF" if side == "A" else "#000000")
            sec = "#000000" if side == "A" else "#FFFFFF"
            sc = _short_code(tname)
            tid = _upsert_team(cur, tname, sc, str(jc), str(sec))
            team_ids[side] = tid
            summary.teams_insert_attempted += 1

            players_map = info.get("players") or {}
            for jkey, payload in players_map.items():
                try:
                    jn = int(str(jkey).strip())
                except ValueError:
                    jn = None
                if isinstance(payload, str):
                    pname, pos = payload.strip(), None
                elif isinstance(payload, dict):
                    pname = (
                        payload.get("name")
                        or payload.get("full_name")
                        or "Unknown"
                    )
                    pname = str(pname).strip()
                    pos = payload.get("position")
                    pos = str(pos).strip() if pos else None
                else:
                    pname = str(payload).strip()
                    pos = None
                _insert_player_ignore(cur, tid, pname, jn, pos)
                summary.players_insert_attempted += 1

        tid_a = team_ids.get("A")
        tid_b = team_ids.get("B")
        if tid_a and tid_b:
            mids = _guess_match_ids(cur, stem, resolved_a, resolved_b)
            summary.file_to_matches[jpath.name] = mids
            for mid in mids[:1]:
                _link_match_lineups(cur, int(mid), tid_a, tid_b)

    conn.commit()
    return summary


def migrate_path(db_path: Path, rosters_dir: Optional[Path] = None) -> MigrationSummary:
    rosters_dir = rosters_dir or (_REPO_ROOT / "rosters")
    conn = sqlite3.connect(str(db_path))
    try:
        return migrate_conn(conn, rosters_dir)
    finally:
        conn.close()


def _print_summary(s: MigrationSummary) -> None:
    print("--- Dynamic roster migration (001) ---")
    print(f"Roster files processed: {len(s.roster_files)}")
    print(f"Team upserts (attempted): {s.teams_insert_attempted}")
    print(f"Player inserts (attempted): {s.players_insert_attempted}")
    print("File → candidate match_ids (best-effort):")
    for fn, mids in sorted(s.file_to_matches.items()):
        print(f"  {fn}: {mids or '(none)'}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="C1 dynamic roster migration")
    p.add_argument("--db", type=Path, default=Path("matches.db"), help="SQLite database path")
    p.add_argument(
        "--rosters",
        type=Path,
        default=None,
        help="Directory of roster JSON files (default: <repo>/rosters)",
    )
    args = p.parse_args(argv)
    db_path = args.db
    if not db_path.is_file():
        print(f"Database not found at {db_path} — creating empty file.", file=sys.stderr)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()
    rosters = args.rosters or (_REPO_ROOT / "rosters")
    s = migrate_path(db_path, rosters)
    _print_summary(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
