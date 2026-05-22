"""Match analytics runner (P11).

Owns the full downstream analytics suite — possession, passes, shots, xG,
sprints, highlights — behind a single per-frame interface so it can be driven
by any frame source.

`run_demo.py` computes these analytics inline in its main loop; the chunked,
crash-resumable `chunked_video_processor.py` historically did tracking only.
This runner lets the chunked processor produce the same analytics over a full
match: the modules are instantiated once and accumulate in-process across all
chunks (Tier 1 — analytics are not checkpoint-restored on crash-resume).

Usage:
    runner = MatchAnalyticsRunner(frame_width=640, frame_height=360, fps=25.0)
    for each frame:
        runner.process_frame(frame_idx, timestamp, player_positions, ball_box)
    metrics = runner.finalize(total_frames)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from services.ball_tracker import BallTracker
from services.event_detector import EventDetector
from services.highlights_generator import HighlightsGenerator
from services.possession_tracker import PossessionTracker
from services.sprint_detector import SprintDetector
from services.xg_calculator import BodyPart, ShotType, xGCalculator

# Physically implausible player speeds are tracking artifacts — drop them so a
# single bad detection cannot inflate sprint / speed stats.
MAX_PLAYER_SPEED_MPS = 12.0
# Ball velocity sanity cap for shot detection (m/s).
MAX_BALL_VELOCITY_MPS = 45.0


class MatchAnalyticsRunner:
    """Per-frame analytics orchestrator for full-match processing."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float,
        pitch_length_m: float = 105.0,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps if fps > 0 else 25.0
        self.meter_per_px = pitch_length_m / frame_width if frame_width else 0.0

        # Analytics modules — instantiated once, accumulate across all chunks.
        self.possession_tracker = PossessionTracker()
        self.event_detector = EventDetector()
        self.xg_calculator = xGCalculator(pitch_length_m=pitch_length_m)
        self.sprint_detector = SprintDetector()
        self.highlights_generator = HighlightsGenerator()
        self.ball_tracker = BallTracker()

        # Accumulators.
        self.frames_processed = 0
        self.passes_detected = 0
        self.shots_detected = 0
        self.ball_position_history: list[dict] = []
        self._last_timestamp = 0.0
        # Per-player previous position for speed estimation: pid -> (x, y, t).
        self._prev_player_pos: Dict[int, Tuple[float, float, float]] = {}

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_box: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """Feed one frame of detections through every analytics module.

        Args:
            frame_idx: Absolute frame index (continuous across chunks).
            timestamp: Frame time in seconds.
            player_positions: {player_id: (cx, cy, team)} — team is 'A'/'B'.
            ball_box: Ball bounding box (x1, y1, x2, y2), or None if no ball.
        """
        self.frames_processed += 1
        self._last_timestamp = timestamp

        # --- Per-player speed -> sprint detection -------------------------
        for pid, (cx, cy, team) in player_positions.items():
            speed_mps = 0.0
            prev = self._prev_player_pos.get(pid)
            if prev is not None:
                dt = timestamp - prev[2]
                if dt > 0:
                    dist_px = math.hypot(cx - prev[0], cy - prev[1])
                    speed_mps = (dist_px * self.meter_per_px) / dt
                    if speed_mps > MAX_PLAYER_SPEED_MPS:
                        speed_mps = 0.0
            self._prev_player_pos[pid] = (cx, cy, timestamp)
            self.sprint_detector.update(pid, speed_mps, timestamp, team, (cx, cy))

        # --- Ball tracking (with trajectory prediction for missed frames) -
        tracked, _method = self.ball_tracker.update_with_prediction(
            ball_box, frame_idx, timestamp, self.frame_width, self.frame_height,
            max_missing_frames=5,
        )
        if not tracked:
            return

        ball_pos = self.ball_tracker.get_position()
        if ball_pos is None:
            return

        ball_velocity = self.ball_tracker.get_velocity()  # px/s
        ball_direction = self.ball_tracker.get_direction()
        ball_velocity_mps = min(ball_velocity * self.meter_per_px, MAX_BALL_VELOCITY_MPS)

        self.ball_position_history.append(
            {"frame": frame_idx, "x": ball_pos[0], "y": ball_pos[1],
             "timestamp": timestamp, "velocity": ball_velocity}
        )

        # --- Possession ---------------------------------------------------
        current_possessor: Optional[int] = None
        current_team: Optional[str] = None
        if player_positions:
            poss = self.possession_tracker.detect_possession_with_context(
                ball_pos, player_positions, frame_idx, timestamp,
                self.frame_width, self.frame_height,
            )
            if poss:
                current_possessor = poss["player_id"]
                current_team = poss["team"]

        # --- Pass detection ----------------------------------------------
        pass_event = self.event_detector.detect_pass(
            current_possessor, current_team, player_positions,
            ball_velocity_mps, self.meter_per_px, frame_idx, timestamp,
            self.frame_width,
        )
        if pass_event:
            self.passes_detected += 1

        # --- Shot detection + xG -----------------------------------------
        shot_event = self.event_detector.detect_shot(
            ball_pos, tuple(ball_direction), ball_velocity_mps,
            frame_idx, timestamp, self.frame_width, self.frame_height,
        )
        if shot_event and current_possessor is not None:
            self.shots_detected += 1
            norm_x = ball_pos[0] / self.frame_width if self.frame_width else 0.0
            norm_y = ball_pos[1] / self.frame_height if self.frame_height else 0.0
            self.xg_calculator.add_shot(
                timestamp=timestamp,
                frame=frame_idx,
                shooter_id=current_possessor,
                shooter_team=current_team or "A",
                x=norm_x,
                y=norm_y,
                shot_type=ShotType.OPEN_PLAY,
                body_part=BodyPart.RIGHT_FOOT,
                velocity_mps=ball_velocity_mps,
                attacking_right=(current_team == "A"),
            )
            self.highlights_generator.add_big_chance(
                timestamp=timestamp,
                frame=frame_idx,
                shooter_id=current_possessor,
                team=current_team or "A",
                xg_value=min(ball_velocity_mps / 50.0, 1.0),
                description=f"Shot ({ball_velocity_mps:.0f} m/s)",
            )

    def finalize(self, total_frames: int) -> dict:
        """Close open state and return the merged analytics dict.

        The returned shape mirrors the analytics sections run_demo.py writes
        into metrics.json, so downstream dashboards consume it unchanged.
        """
        self.sprint_detector.finalize(self._last_timestamp)

        try:
            highlights_summary = self.highlights_generator.generate_match_summary()
        except Exception:  # noqa: BLE001 — summary is best-effort, never fatal
            highlights_summary = {}

        return {
            "frames_analyzed": self.frames_processed,
            "possession": self.possession_tracker.get_statistics(),
            "passing": {
                "passes_detected": self.passes_detected,
                "statistics": self.event_detector.get_pass_statistics(),
            },
            "shots": {
                "shots_detected": self.shots_detected,
                "statistics": self.event_detector.get_shot_statistics(),
            },
            "xg": self.xg_calculator.get_statistics(),
            "sprints": self.sprint_detector.get_statistics(),
            "highlights": {
                "total_events": len(self.highlights_generator.events),
                "summary": highlights_summary,
            },
            "ball_analytics": {
                "tracked_positions": len(self.ball_position_history),
                "position_history": self.ball_position_history[-500:],
            },
        }
