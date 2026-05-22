"""
Dynamic roster manager (Task C2) — CRUD over C1 lineup_teams / lineup_players / match_lineups.

Uses sqlite3.Row and an LRU-backed snapshot cache invalidated on writes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.database_schema import (
    LINEUP_PLAYERS_TABLE,
    LINEUP_TEAMS_TABLE,
    MATCH_LINEUPS_TABLE,
    create_schema,
)

logger = logging.getLogger(__name__)

_TT = LINEUP_TEAMS_TABLE
_PP = LINEUP_PLAYERS_TABLE
_ML = MATCH_LINEUPS_TABLE


@dataclass
class Team:
    id: int
    name: str
    short_code: Optional[str]
    primary_color: Optional[str]
    secondary_color: Optional[str]


@dataclass
class Player:
    id: int
    team_id: int
    name: str
    jersey_number: Optional[int]
    position: Optional[str]
    dob: Optional[str]
    height_cm: Optional[int]
    weight_kg: Optional[int]
    active: bool


def _row_team(r: sqlite3.Row) -> Team:
    return Team(
        id=int(r["id"]),
        name=str(r["name"]),
        short_code=r["short_code"],
        primary_color=r["primary_color"],
        secondary_color=r["secondary_color"],
    )


def _row_player(r: sqlite3.Row) -> Player:
    return Player(
        id=int(r["id"]),
        team_id=int(r["team_id"]),
        name=str(r["name"]),
        jersey_number=r["jersey_number"],
        position=r["position"],
        dob=r["dob"],
        height_cm=r["height_cm"],
        weight_kg=r["weight_kg"],
        active=bool(int(r["active"])),
    )


@lru_cache(maxsize=128)
def _snapshot_teams(db_path: str, epoch: int) -> Tuple[Tuple[int, str, Optional[str], Optional[str], Optional[str]], ...]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(f'SELECT id, name, short_code, primary_color, secondary_color FROM "{_TT}" ORDER BY name')
        return tuple(
            (
                int(r["id"]),
                str(r["name"]),
                r["short_code"],
                r["primary_color"],
                r["secondary_color"],
            )
            for r in cur.fetchall()
        )
    finally:
        conn.close()


@lru_cache(maxsize=256)
def _snapshot_team_by_id(db_path: str, epoch: int, team_id: int) -> Optional[Tuple[int, str, Optional[str], Optional[str], Optional[str]]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            f'SELECT id, name, short_code, primary_color, secondary_color FROM "{_TT}" WHERE id = ?',
            (team_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return (
            int(r["id"]),
            str(r["name"]),
            r["short_code"],
            r["primary_color"],
            r["secondary_color"],
        )
    finally:
        conn.close()


@lru_cache(maxsize=256)
def _snapshot_players(db_path: str, epoch: int, team_key: str) -> Tuple[Tuple[int, int, str, Optional[int], Optional[str], Optional[str], Optional[int], Optional[int], int], ...]:
    """team_key: '' for all teams, else str(team_id)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        if team_key:
            tid = int(team_key)
            cur.execute(
                f"""
                SELECT id, team_id, name, jersey_number, position, dob, height_cm, weight_kg, active
                FROM "{_PP}" WHERE team_id = ? ORDER BY jersey_number, name
                """,
                (tid,),
            )
        else:
            cur.execute(
                f"""
                SELECT id, team_id, name, jersey_number, position, dob, height_cm, weight_kg, active
                FROM "{_PP}" ORDER BY team_id, jersey_number, name
                """
            )
        return tuple(
            (
                int(r["id"]),
                int(r["team_id"]),
                str(r["name"]),
                r["jersey_number"],
                r["position"],
                r["dob"],
                r["height_cm"],
                r["weight_kg"],
                int(r["active"]),
            )
            for r in cur.fetchall()
        )
    finally:
        conn.close()


@lru_cache(maxsize=512)
def _snapshot_player_by_id(db_path: str, epoch: int, player_id: int) -> Optional[
    Tuple[int, int, str, Optional[int], Optional[str], Optional[str], Optional[int], Optional[int], int]
]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, team_id, name, jersey_number, position, dob, height_cm, weight_kg, active
            FROM "{_PP}" WHERE id = ?
            """,
            (player_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return (
            int(r["id"]),
            int(r["team_id"]),
            str(r["name"]),
            r["jersey_number"],
            r["position"],
            r["dob"],
            r["height_cm"],
            r["weight_kg"],
            int(r["active"]),
        )
    finally:
        conn.close()


@lru_cache(maxsize=256)
def _snapshot_lineup(db_path: str, epoch: int, match_id: int) -> Tuple[int, ...]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT ml.player_id FROM "{_ML}" ml
            JOIN "{_PP}" lp ON lp.id = ml.player_id
            WHERE ml.match_id = ?
            ORDER BY ml.is_starter DESC, lp.jersey_number, lp.name
            """,
            (match_id,),
        )
        return tuple(int(r["player_id"]) for r in cur.fetchall())
    finally:
        conn.close()


def _clear_roster_caches() -> None:
    _snapshot_teams.cache_clear()
    _snapshot_team_by_id.cache_clear()
    _snapshot_players.cache_clear()
    _snapshot_player_by_id.cache_clear()
    _snapshot_lineup.cache_clear()


class DynamicRosterManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._epoch = 0
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)
            conn.commit()
        finally:
            conn.close()

    def _epoch_key(self) -> int:
        return self._epoch

    def invalidate_cache(self) -> None:
        self._epoch += 1
        _clear_roster_caches()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def list_teams(self) -> List[Team]:
        rows = _snapshot_teams(str(self.db_path), self._epoch_key())
        return [Team(*t) for t in rows]

    def get_team(self, team_id: int) -> Optional[Team]:
        t = _snapshot_team_by_id(str(self.db_path), self._epoch_key(), team_id)
        return Team(*t) if t else None

    def create_team(
        self,
        name: str,
        short_code: Optional[str],
        primary_color: Optional[str],
        secondary_color: Optional[str],
    ) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("name is required")
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT OR IGNORE INTO "{_TT}" (name, short_code, primary_color, secondary_color)
                VALUES (?, ?, ?, ?)
                """,
                (name, short_code, primary_color, secondary_color),
            )
            cur.execute(f'SELECT id FROM "{_TT}" WHERE name = ? LIMIT 1', (name,))
            tid = int(cur.fetchone()["id"])
            conn.commit()
        finally:
            conn.close()
        self.invalidate_cache()
        return tid

    def update_team(self, team_id: int, **fields: Any) -> None:
        allowed = {"name", "short_code", "primary_color", "secondary_color"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        cols = ", ".join(f'"{k}" = ?' for k in updates)
        vals = list(updates.values()) + [team_id]
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f'UPDATE "{_TT}" SET {cols} WHERE id = ?', vals)
            conn.commit()
        finally:
            conn.close()
        self.invalidate_cache()

    def list_players(self, team_id: Optional[int] = None) -> List[Player]:
        key = "" if team_id is None else str(int(team_id))
        rows = _snapshot_players(str(self.db_path), self._epoch_key(), key)
        return [
            Player(
                id=r[0],
                team_id=r[1],
                name=r[2],
                jersey_number=r[3],
                position=r[4],
                dob=r[5],
                height_cm=r[6],
                weight_kg=r[7],
                active=bool(r[8]),
            )
            for r in rows
        ]

    def get_player(self, player_id: int) -> Optional[Player]:
        r = _snapshot_player_by_id(str(self.db_path), self._epoch_key(), player_id)
        if not r:
            return None
        return Player(
            id=r[0],
            team_id=r[1],
            name=r[2],
            jersey_number=r[3],
            position=r[4],
            dob=r[5],
            height_cm=r[6],
            weight_kg=r[7],
            active=bool(r[8]),
        )

    def create_player(
        self,
        team_id: int,
        name: str,
        jersey_number: Optional[int],
        position: Optional[str],
    ) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("name is required")
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f'SELECT id FROM "{_TT}" WHERE id = ?', (team_id,))
            if not cur.fetchone():
                raise ValueError(f"team_id {team_id} does not exist")
            cur.execute(
                f"""
                INSERT OR IGNORE INTO "{_PP}"
                    (team_id, name, jersey_number, position, dob, height_cm, weight_kg, active)
                VALUES (?, ?, ?, ?, NULL, NULL, NULL, 1)
                """,
                (team_id, name, jersey_number, position),
            )
            if jersey_number is not None:
                cur.execute(
                    f'SELECT id FROM "{_PP}" WHERE team_id = ? AND jersey_number = ? LIMIT 1',
                    (team_id, jersey_number),
                )
            else:
                cur.execute(
                    f'SELECT id FROM "{_PP}" WHERE team_id = ? AND name = ? AND jersey_number IS NULL LIMIT 1',
                    (team_id, name),
                )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("Failed to resolve player id after insert")
            pid = int(row["id"])
            conn.commit()
        finally:
            conn.close()
        self.invalidate_cache()
        return pid

    def update_player(self, player_id: int, **fields: Any) -> None:
        allowed = {"name", "jersey_number", "position", "dob", "height_cm", "weight_kg", "active", "team_id"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        cols = ", ".join(f'"{k}" = ?' for k in updates)
        vals = list(updates.values()) + [player_id]
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f'UPDATE "{_PP}" SET {cols} WHERE id = ?', vals)
            conn.commit()
        finally:
            conn.close()
        self.invalidate_cache()

    def get_lineup(self, match_id: int) -> List[Player]:
        pids = _snapshot_lineup(str(self.db_path), self._epoch_key(), match_id)
        out: List[Player] = []
        for pid in pids:
            p = self.get_player(pid)
            if p:
                out.append(p)
        return out

    def set_lineup(self, match_id: int, player_ids: Sequence[int], starters: Set[int]) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f'DELETE FROM "{_ML}" WHERE match_id = ?', (match_id,))
            for pid in player_ids:
                cur.execute(
                    f"""
                    INSERT INTO "{_ML}" (match_id, player_id, position_in_lineup, is_starter)
                    VALUES (?, ?, ?, ?)
                    """,
                    (match_id, int(pid), None, 1 if int(pid) in starters else 0),
                )
            conn.commit()
        finally:
            conn.close()
        self.invalidate_cache()

    def bulk_import_from_json(self, json_path: Path) -> Dict[str, Any]:
        path = Path(json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        teams_block = data.get("teams") or {}
        created_teams = 0
        created_players = 0
        for side in ("A", "B"):
            info = teams_block.get(side) or {}
            tname = (info.get("name") or "").strip() or f"Team {side}"
            jc = info.get("jersey_color") or ("#FFFFFF" if side == "A" else "#000000")
            sec = "#000000" if side == "A" else "#FFFFFF"
            sc = ("".join(c for c in tname if c.isalnum())[:3] or "TM").upper()
            tid = self.create_team(tname, sc, str(jc), str(sec))
            created_teams += 1
            for jkey, payload in (info.get("players") or {}).items():
                try:
                    jn = int(str(jkey).strip())
                except ValueError:
                    jn = None
                if isinstance(payload, str):
                    pname, pos = payload.strip(), None
                elif isinstance(payload, dict):
                    pname = str(
                        payload.get("name")
                        or payload.get("full_name")
                        or "Unknown"
                    ).strip()
                    pos = payload.get("position")
                    pos = str(pos).strip() if pos else None
                else:
                    pname = str(payload).strip()
                    pos = None
                self.create_player(tid, pname, jn, pos)
                created_players += 1
        return {
            "path": str(path),
            "teams_touched": created_teams,
            "players_touched": created_players,
        }

    def roster_identity_bundle_from_stem(self, stem: str) -> Optional[Dict[str, Any]]:
        """
        Build a JSON-shaped roster dict (match + teams A/B) from lineup tables for UI / PlayerIdentity.
        Does not read rosters/*.json.
        """
        stem_l = stem.lower()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT match_id, team_a, team_b, home_team_id, away_team_id, video_path
                FROM matches
                WHERE lower(CAST(video_path AS TEXT)) LIKE ?
                   OR lower(CAST(video_path AS TEXT)) LIKE ?
                   OR CAST(match_id AS TEXT) = ?
                LIMIT 1
                """,
                (f"%{stem_l}%", f"%{stem}%", stem),
            )
            row = cur.fetchone()
            if not row:
                return None
            mid = int(row["match_id"])
            ta = str(row["team_a"] or "").strip()
            tb = str(row["team_b"] or "").strip()
            htid = row["home_team_id"]
            atid = row["away_team_id"]

            def _tid_by_name(nm: str) -> Optional[int]:
                if not nm:
                    return None
                cur.execute(
                    f'SELECT id FROM "{_TT}" WHERE lower(name) = lower(?) LIMIT 1',
                    (nm,),
                )
                r = cur.fetchone()
                return int(r["id"]) if r else None

            home_id = int(htid) if htid is not None else _tid_by_name(ta)
            away_id = int(atid) if atid is not None else _tid_by_name(tb)
            if home_id is None or away_id is None:
                return None

            def _players_map(team_id: int) -> Dict[str, str]:
                cur.execute(
                    f"""
                    SELECT lp.jersey_number, lp.name FROM "{_PP}" lp
                    WHERE EXISTS (
                        SELECT 1 FROM "{_ML}" ml
                        WHERE ml.match_id = ? AND ml.player_id = lp.id
                    )
                    AND lp.team_id = ?
                    ORDER BY lp.jersey_number
                    """,
                    (mid, team_id),
                )
                rows = cur.fetchall()
                if not rows:
                    cur.execute(
                        f"""
                        SELECT jersey_number, name FROM "{_PP}"
                        WHERE team_id = ? ORDER BY jersey_number
                        """,
                        (team_id,),
                    )
                    rows = cur.fetchall()
                out: Dict[str, str] = {}
                for r in rows:
                    jn = r["jersey_number"]
                    if jn is None:
                        continue
                    out[str(int(jn))] = str(r["name"])
                return out

            cur.execute(
                f'SELECT name, primary_color FROM "{_TT}" WHERE id = ?',
                (home_id,),
            )
            hr = cur.fetchone()
            cur.execute(
                f'SELECT name, primary_color FROM "{_TT}" WHERE id = ?',
                (away_id,),
            )
            ar = cur.fetchone()
            if not hr or not ar:
                return None

            return {
                "match": f"{hr['name']} vs {ar['name']}",
                "teams": {
                    "A": {
                        "name": hr["name"],
                        "jersey_color": hr["primary_color"] or "",
                        "players": _players_map(home_id),
                    },
                    "B": {
                        "name": ar["name"],
                        "jersey_color": ar["primary_color"] or "",
                        "players": _players_map(away_id),
                    },
                },
            }
        finally:
            conn.close()
