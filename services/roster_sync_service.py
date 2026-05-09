from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RosterSyncResult:
    roster_id: int
    players_upserted: int
    roster_players_upserted: int


class RosterSyncService:
    """
    Skeleton service to sync roster data into the database.

    Current DB model (see `services/database_schema.py`):
    - `players_master`: long-lived player master records
    - `rosters`: roster header (team + optional match scope)
    - `roster_players`: roster membership rows

    This service is intentionally minimal: it provides stable entrypoints and
    safe idempotent upsert semantics, leaving provider-specific mapping for later.
    """

    def __init__(self, db_manager):
        # `db_manager` is expected to be `services.database_manager.DatabaseManager`.
        self.db_manager = db_manager

    # ---------------------------------------------------------------------
    # Public entrypoints
    # ---------------------------------------------------------------------

    def sync_from_json(
        self,
        roster_path: Path | str,
        *,
        match_id: int | None = None,
        source: str = "json",
    ) -> Dict[str, RosterSyncResult]:
        """
        Sync roster JSON (same format used by `PlayerIdentityManager`).

        Returns a dict keyed by side ('A'/'B') with per-team results.
        """
        roster_path = Path(roster_path)
        data = json.loads(roster_path.read_text(encoding="utf-8"))

        teams = (data.get("teams") or {})
        results: Dict[str, RosterSyncResult] = {}

        for side in ("A", "B"):
            team_info = teams.get(side) or {}
            team_name = team_info.get("name") or f"Team {side}"
            players = team_info.get("players") or {}

            results[side] = self._sync_team_roster(
                team_name=team_name,
                side=side,
                players_by_jersey=players,
                match_id=match_id,
                source=source,
            )

        return results

    def sync_from_provider(
        self,
        provider: str,
        *,
        team_name: str,
        season: str | None = None,
        match_id: int | None = None,
    ) -> Dict[str, Any]:
        """
        Placeholder for API/provider-based roster sync (StatsBomb/Opta/etc.).
        """
        raise NotImplementedError(
            f"Provider sync not implemented yet (provider={provider}, team={team_name}, season={season}, match_id={match_id})"
        )

    # ---------------------------------------------------------------------
    # Internal helpers (idempotent upserts)
    # ---------------------------------------------------------------------

    def _sync_team_roster(
        self,
        *,
        team_name: str,
        side: str | None,
        players_by_jersey: Dict[str, Any],
        match_id: int | None,
        source: str,
    ) -> RosterSyncResult:
        """
        Upsert a roster for a team + optional match scope, then upsert members.
        """
        with self.db_manager._get_connection() as conn:  # noqa: SLF001 (existing project pattern)
            cur = conn.cursor()

            # 1) Upsert roster header (match-scoped rosters can have both A/B).
            roster_id = self._upsert_roster(cur, match_id=match_id, team_name=team_name, side=side, source=source)

            # 2) Upsert players + membership rows.
            players_upserted = 0
            roster_players_upserted = 0

            for jersey_str, payload in players_by_jersey.items():
                jersey_number, player_name, position = self._normalize_json_player(jersey_str, payload)
                player_id = self._upsert_player_master(cur, full_name=player_name)
                players_upserted += 1

                self._upsert_roster_player(
                    cur,
                    roster_id=roster_id,
                    player_id=player_id,
                    profile_id=None,
                    jersey_number=jersey_number,
                    position=position,
                    is_starting=False,
                    metadata=None,
                )
                roster_players_upserted += 1

            return RosterSyncResult(
                roster_id=roster_id,
                players_upserted=players_upserted,
                roster_players_upserted=roster_players_upserted,
            )

    @staticmethod
    def _normalize_json_player(jersey_str: str, payload: Any) -> Tuple[int | None, str, str | None]:
        """
        The roster JSON is usually { "<jersey>": "<name>" } but sometimes can be richer.
        Returns (jersey_number, full_name, position)
        """
        try:
            jersey_number = int(jersey_str)
        except Exception:
            jersey_number = None

        if isinstance(payload, str):
            return jersey_number, payload, None
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("full_name") or payload.get("player_name") or "Unknown"
            position = payload.get("position")
            return jersey_number, str(name), str(position) if position is not None else None

        return jersey_number, str(payload), None

    @staticmethod
    def _upsert_roster(cur, *, match_id: int | None, team_name: str, side: str | None, source: str) -> int:
        # Try to find an existing roster header.
        if match_id is None:
            cur.execute(
                """
                SELECT roster_id FROM rosters
                WHERE match_id IS NULL AND team_name = ? AND (side IS ? OR side IS NULL)
                ORDER BY roster_id DESC
                LIMIT 1
                """,
                (team_name, side),
            )
        else:
            cur.execute(
                """
                SELECT roster_id FROM rosters
                WHERE match_id = ? AND team_name = ? AND (side = ? OR (? IS NULL AND side IS NULL))
                LIMIT 1
                """,
                (match_id, team_name, side, side),
            )
        row = cur.fetchone()
        if row:
            roster_id = int(row[0])
            cur.execute(
                "UPDATE rosters SET source = ?, updated_at = CURRENT_TIMESTAMP WHERE roster_id = ?",
                (source, roster_id),
            )
            return roster_id

        cur.execute(
            """
            INSERT INTO rosters (match_id, team_name, side, source)
            VALUES (?, ?, ?, ?)
            """,
            (match_id, team_name, side, source),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _upsert_player_master(cur, *, full_name: str) -> int:
        # Minimal identity: full_name (can be expanded later with external_ids/profile_id).
        cur.execute("SELECT player_id FROM players_master WHERE full_name = ? LIMIT 1", (full_name,))
        row = cur.fetchone()
        if row:
            player_id = int(row[0])
            cur.execute("UPDATE players_master SET updated_at = CURRENT_TIMESTAMP WHERE player_id = ?", (player_id,))
            return player_id

        cur.execute(
            "INSERT INTO players_master (full_name) VALUES (?)",
            (full_name,),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _upsert_roster_player(
        cur,
        *,
        roster_id: int,
        player_id: int | None,
        profile_id: int | None,
        jersey_number: int | None,
        position: str | None,
        is_starting: bool,
        metadata: Dict[str, Any] | None,
    ) -> int:
        # Rely on `uq_roster_player_identity` unique index (ifnull-based) for idempotency.
        metadata_json = json.dumps(metadata) if metadata else None
        cur.execute(
            """
            INSERT OR IGNORE INTO roster_players
              (roster_id, player_id, profile_id, jersey_number, position, is_starting, metadata)
            VALUES
              (?, ?, ?, ?, ?, ?, ?)
            """,
            (roster_id, player_id, profile_id, jersey_number, position, int(is_starting), metadata_json),
        )
        if cur.lastrowid:
            return int(cur.lastrowid)

        # Update the existing row (best-effort).
        cur.execute(
            """
            SELECT roster_player_id FROM roster_players
            WHERE roster_id = ?
              AND player_id IS ?
              AND profile_id IS ?
              AND jersey_number IS ?
            LIMIT 1
            """,
            (roster_id, player_id, profile_id, jersey_number),
        )
        row = cur.fetchone()
        if not row:
            # Should be unreachable, but keep the service resilient.
            raise RuntimeError("Failed to upsert roster_players row")

        roster_player_id = int(row[0])
        cur.execute(
            """
            UPDATE roster_players
            SET position = COALESCE(?, position),
                is_starting = ?,
                metadata = COALESCE(?, metadata)
            WHERE roster_player_id = ?
            """,
            (position, int(is_starting), metadata_json, roster_player_id),
        )
        return roster_player_id

