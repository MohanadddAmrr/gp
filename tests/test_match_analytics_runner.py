"""Tests for services/match_analytics_runner.py (P11)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.match_analytics_runner import MatchAnalyticsRunner  # noqa: E402

EXPECTED_KEYS = {
    "frames_analyzed",
    "possession",
    "passing",
    "shots",
    "xg",
    "sprints",
    "highlights",
    "ball_analytics",
}


def _run(n_frames: int, with_ball: bool = True) -> dict:
    runner = MatchAnalyticsRunner(frame_width=640, frame_height=360, fps=25.0)
    for fi in range(n_frames):
        t = fi / 25.0
        players = {
            1: (100 + fi, 180, "A"),
            2: (300, 200, "A"),
            3: (400, 150, "B"),
            4: (500, 250, "B"),
        }
        ball = (105 + fi, 175, 115 + fi, 185) if with_ball else None
        runner.process_frame(fi, t, players, ball)
    return runner.finalize(n_frames)


def test_finalize_has_all_sections():
    result = _run(50)
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["frames_analyzed"] == 50


def test_finalize_is_json_serializable():
    result = _run(50)
    # Must not raise — metrics.json is written verbatim.
    json.dumps(result)


def test_empty_match_does_not_crash():
    """A match with no ball detections still finalizes cleanly."""
    result = _run(30, with_ball=False)
    assert result["frames_analyzed"] == 30
    assert result["ball_analytics"]["tracked_positions"] == 0
    json.dumps(result)


def test_zero_frames():
    runner = MatchAnalyticsRunner(frame_width=640, frame_height=360, fps=25.0)
    result = runner.finalize(0)
    assert result["frames_analyzed"] == 0
    json.dumps(result)


def test_ball_history_accumulates():
    result = _run(40)
    # The ball is present every frame; the tracker should log positions.
    assert result["ball_analytics"]["tracked_positions"] > 0


def test_meter_per_px_scaling():
    runner = MatchAnalyticsRunner(frame_width=640, frame_height=360, fps=25.0)
    assert abs(runner.meter_per_px - 105.0 / 640) < 1e-9
