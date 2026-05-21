from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from services.api_connector import APIManager
    from services.dynamic_roster_manager import DynamicRosterManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RosterSyncResult:
    roster_id: int
    players_upserted: int
    roster_players_upserted: int


class RosterSyncService:
    """
    Roster sync:
    - **Dynamic mode** (C2): `RosterSyncService(api, roster_manager)` — football-data.org squads
      into `lineup_teams` / `lineup_players` via `DynamicRosterManager`.
    - **Legacy mode**: `RosterSyncService(db_manager=db)` — Sprint blueprint `teams` /
      `roster_players` / `match_rosters` via `DatabaseManager`.
    """

    def __init__(
        self,
        api: Optional["APIManager"] = None,
        roster_manager: Optional["DynamicRosterManager"] = None,
        *,
        db_manager=None,
    ):
        self._dynamic = api is not None and roster_manager is not None
        self.api = api
        self.roster_manager = roster_manager
        self.db_manager = db_manager
        if self._dynamic:
            return
        if db_manager is not None:
            return
        raise TypeError("RosterSyncService requires either (api, roster_manager) or db_manager=...")

    def sync_team_squad(self, team_external_id: int) -> int:
        """
        Pull squad for a football-data.org team id and upsert into dynamic roster tables.

        Uses ``FootballDataConnector.fetch_team_squad`` when the football_data connector
        is registered; otherwise falls back to a **mock** that scans ``rosters/*.json``
        (TODO: replace with Member E-only API once fully wired).
        """
        if not self._dynamic or self.api is None or self.roster_manager is None:
            raise RuntimeError("sync_team_squad requires api + roster_manager constructor mode")

        squad: List[Dict[str, Any]] = []
        fd = self.api.connectors.get("football_data")
        if fd is not None and hasattr(fd, "fetch_team_squad"):
            squad = fd.fetch_team_squad(team_external_id)  # type: ignore[assignment]
        if not squad:
            squad = self._mock_squad_from_local_json(team_external_id)
        if not squad:
            logger.warning("sync_team_squad: no squad rows for external_id=%s", team_external_id)
            return 0

        team_name = f"API Team {team_external_id}"
        if fd is not None:
            try:
                raw = fd._make_request(f"teams/{team_external_id}")  # type: ignore[attr-defined]
                if isinstance(raw, dict) and raw.get("name"):
                    team_name = str(raw["name"])
            except Exception as ex:
                logger.debug("Could not resolve team name from API: %s", ex)

        tid = self.roster_manager.create_team(team_name, str(team_external_id)[:3], "#FFFFFF", "#000000")
        n = 0
        for p in squad:
            name = str(p.get("name") or "").strip()
            if not name:
                continue
            jn = p.get("jersey_number")
            if jn is not None:
                try:
                    jn = int(jn)
                except (TypeError, ValueError):
                    jn = None
            pos = p.get("position")
            pos = str(pos).strip() if pos else None
            self.roster_manager.create_player(tid, name, jn, pos)
            n += 1
        return n

    def _mock_squad_from_local_json(self, team_external_id: int) -> List[Dict[str, Any]]:
        """
        TODO(Member E): remove once ``fetch_team_squad`` is the only path.

        Deterministic mock: even id → side A players of first ``rosters/*.json``,
        odd id → side B.
        """
        root = Path(__file__).resolve().parents[1]
        roster_dir = root / "rosters"
        files = sorted(roster_dir.glob("*.json"))
        if not files:
            return []
        try:
            data = json.loads(files[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        teams = data.get("teams") or {}
        side = "A" if team_external_id % 2 == 0 else "B"
        block = teams.get(side) or {}
        players = block.get("players") or {}
        out: List[Dict[str, Any]] = []
        for jk, payload in players.items():
            try:
                jn = int(str(jk).strip())
            except ValueError:
                jn = None
            if isinstance(payload, str):
                out.append({"name": payload, "jersey_number": jn, "position": None})
            elif isinstance(payload, dict):
                nm = payload.get("name") or payload.get("full_name") or "Unknown"
                out.append(
                    {
                        "name": str(nm),
                        "jersey_number": jn,
                        "position": payload.get("position"),
                    }
                )
        return out

    def sync_from_json(
        self,
        roster_path: Path | str,
        *,
        match_id: int | str | None = None,
        source: str = "json",
    ) -> Dict[str, RosterSyncResult]:
        """Legacy: sync roster JSON into DatabaseManager blueprint tables."""
        if self._dynamic or self.db_manager is None:
            raise RuntimeError("sync_from_json requires legacy db_manager= constructor mode")
        roster_path = Path(roster_path)
        data = json.loads(roster_path.read_text(encoding="utf-8"))

        teams = (data.get("teams") or {})
        results: Dict[str, RosterSyncResult] = {}

        home_tid = None
        away_tid = None

        for side in ("A", "B"):
            team_info = teams.get(side) or {}
            team_name = team_info.get("name") or f"Team {side}"
            color = team_info.get("jersey_color") or (
                "#FFFFFF" if side == "A" else "#000000"
            )
            players = team_info.get("players") or {}

            tid = self.db_manager.add_team(team_name, primary_color=color)
            if side == "A":
                home_tid = tid
            else:
                away_tid = tid

            players_added = 0
            for jersey_str, payload in players.items():
                jersey_number, player_name, position = self._normalize_json_player(
                    jersey_str, payload
                )
                self.db_manager.add_player(tid, player_name, jersey_number, position)
                players_added += 1

            results[side] = RosterSyncResult(
                roster_id=tid,
                players_upserted=players_added,
                roster_players_upserted=players_added,
            )

        match_name = data.get("match") or str(match_id or roster_path.stem)
        if home_tid and away_tid:
            self.db_manager.assign_roster_to_match(match_name, home_tid, away_tid)
            stem_key = roster_path.stem
            if str(stem_key) != str(match_name):
                self.db_manager.assign_roster_to_match(stem_key, home_tid, away_tid)

        return results

    @staticmethod
    def _normalize_json_player(
        jersey_str: str, payload: Any
    ) -> Tuple[Optional[int], str, Optional[str]]:
        try:
            jersey_number = int(jersey_str)
        except Exception:
            jersey_number = None

        if isinstance(payload, str):
            return jersey_number, payload, None
        if isinstance(payload, dict):
            name = (
                payload.get("name")
                or payload.get("full_name")
                or payload.get("player_name")
                or "Unknown"
            )
            position = payload.get("position")
            return jersey_number, str(name), str(position) if position is not None else None

        return jersey_number, str(payload), None
