"""
Sprint Detector Module - Player Sprint Detection

Detects when players sprint (speed > 5.5 m/s) and tracks sprint events.

SPRINT DETECTION LOGIC:
1. Calculate frame-to-frame speed for each player
2. If speed >= SPRINT_THRESHOLD (5.5 m/s) -> player is sprinting
3. Track sprint start/end times
4. Only count sprints lasting >= MIN_SPRINT_DURATION (1.0 second)
5. Track max speed reached during each sprint

SPEED CONTEXT:
- Walking: ~1.5 m/s
- Jogging: ~3-4 m/s
- Running: ~4-5.5 m/s
- Sprinting: >5.5 m/s
- Top sprint: 7-9 m/s (professional level)
"""

from typing import Dict, List, Optional, Tuple, Any


class SprintDetector:
    """Detects and tracks player sprints based on speed thresholds."""

    def __init__(
        self,
        sprint_threshold_mps: float = 5.5,
        high_speed_threshold_mps: float = 7.0,
        min_sprint_duration_sec: float = 1.0,
    ):
        self.sprint_threshold = sprint_threshold_mps
        self.high_speed_threshold = high_speed_threshold_mps
        self.min_sprint_duration = min_sprint_duration_sec

        # Per-player sprint state
        self._active_sprints: Dict[int, Dict] = {}

        # Completed sprint events
        self._sprint_events: List[Dict[str, Any]] = []

    def update(
        self,
        player_id: int,
        speed_mps: float,
        timestamp: float,
        team: str = "",
        position: Optional[Tuple[float, float]] = None,
    ):
        """
        Update sprint detection for a single player at a single frame.

        Args:
            player_id: Player track ID
            speed_mps: Current speed in meters per second
            timestamp: Current time in seconds
            team: Team label ('A' or 'B')
            position: (x, y) pixel position (optional)
        """
        state = self._active_sprints.get(player_id)

        if speed_mps >= self.sprint_threshold:
            if state is None:
                self._active_sprints[player_id] = {
                    "sprint_start": timestamp,
                    "max_speed": speed_mps,
                    "team": team,
                    "start_position": position,
                    "speeds": [speed_mps],
                }
            else:
                state["max_speed"] = max(state["max_speed"], speed_mps)
                state["speeds"].append(speed_mps)
        else:
            if state is not None:
                duration = timestamp - state["sprint_start"]
                if duration >= self.min_sprint_duration:
                    avg_speed = sum(state["speeds"]) / len(state["speeds"])
                    self._sprint_events.append({
                        "player_id": int(player_id),
                        "team": state["team"],
                        "start_time": float(state["sprint_start"]),
                        "end_time": float(timestamp),
                        "duration_sec": float(duration),
                        "max_speed_mps": float(state["max_speed"]),
                        "avg_speed_mps": float(avg_speed),
                        "is_high_intensity": state["max_speed"] >= self.high_speed_threshold,
                    })
                del self._active_sprints[player_id]

    def finalize(self, final_timestamp: float):
        """Close any open sprints at end of processing."""
        for player_id, state in list(self._active_sprints.items()):
            duration = final_timestamp - state["sprint_start"]
            if duration >= self.min_sprint_duration:
                avg_speed = sum(state["speeds"]) / len(state["speeds"])
                self._sprint_events.append({
                    "player_id": int(player_id),
                    "team": state["team"],
                    "start_time": float(state["sprint_start"]),
                    "end_time": float(final_timestamp),
                    "duration_sec": float(duration),
                    "max_speed_mps": float(state["max_speed"]),
                    "avg_speed_mps": float(avg_speed),
                    "is_high_intensity": state["max_speed"] >= self.high_speed_threshold,
                })
        self._active_sprints.clear()

    def get_sprint_events(self) -> List[Dict[str, Any]]:
        """Return all recorded sprint events."""
        return list(self._sprint_events)

    def get_player_sprints(self, player_id: int) -> List[Dict[str, Any]]:
        """Return sprint events for a specific player."""
        return [s for s in self._sprint_events if s["player_id"] == player_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregate sprint statistics."""
        events = self._sprint_events
        total = len(events)

        if total == 0:
            return {
                "total_sprints": 0,
                "team_sprints": {"A": 0, "B": 0},
                "high_intensity_sprints": 0,
                "avg_sprint_duration_sec": 0.0,
                "avg_max_speed_mps": 0.0,
                "top_speed_mps": 0.0,
                "total_sprint_time_sec": 0.0,
                "player_sprint_counts": {},
            }

        team_a = sum(1 for s in events if s["team"] == "A")
        team_b = sum(1 for s in events if s["team"] == "B")
        high_intensity = sum(1 for s in events if s["is_high_intensity"])

        durations = [s["duration_sec"] for s in events]
        max_speeds = [s["max_speed_mps"] for s in events]

        player_counts: Dict[str, int] = {}
        for s in events:
            pid = str(s["player_id"])
            player_counts[pid] = player_counts.get(pid, 0) + 1

        return {
            "total_sprints": total,
            "team_sprints": {"A": team_a, "B": team_b},
            "high_intensity_sprints": high_intensity,
            "avg_sprint_duration_sec": round(sum(durations) / total, 2),
            "avg_max_speed_mps": round(sum(max_speeds) / total, 2),
            "top_speed_mps": round(max(max_speeds), 2),
            "total_sprint_time_sec": round(sum(durations), 2),
            "player_sprint_counts": player_counts,
        }