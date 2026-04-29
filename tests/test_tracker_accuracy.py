"""Tests for services.improved_tracker (M2).

M3 (Re-ID) and Seif's services.accuracy_evaluator will extend this file with
real MOTA/IDF1/ID-switch assertions; for now we lock the factory contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.improved_tracker import (  # noqa: E402
    SUPPORTED_ALGORITHMS,
    BotSortTracker,
    ByteTrackTracker,
    Tracker,
    build_tracker,
)


class _StubModel:
    """Stand-in for ultralytics.YOLO. Records the kwargs the tracker passes
    so we can assert the right yaml goes through without spinning a real model.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return [{"frame": kwargs.get("source"), "tracker": kwargs.get("tracker")}]


# --- factory contract -----------------------------------------------------


def test_build_tracker_bytetrack_returns_correct_class() -> None:
    model = _StubModel()
    tracker = build_tracker("bytetrack", model)
    assert isinstance(tracker, ByteTrackTracker)
    assert isinstance(tracker, Tracker)
    assert tracker.tracker_yaml == "bytetrack.yaml"
    assert tracker.name == "bytetrack"


def test_build_tracker_botsort_returns_correct_class() -> None:
    model = _StubModel()
    tracker = build_tracker("botsort", model)
    assert isinstance(tracker, BotSortTracker)
    assert isinstance(tracker, Tracker)
    assert tracker.tracker_yaml == "botsort.yaml"
    assert tracker.name == "botsort"


def test_build_tracker_is_case_and_whitespace_insensitive() -> None:
    """Config files are human-edited; tolerate ' ByteTrack ' and similar."""
    model = _StubModel()
    assert isinstance(build_tracker("  ByteTrack  ", model), ByteTrackTracker)
    assert isinstance(build_tracker("BOTSORT", model), BotSortTracker)


def test_build_tracker_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tracker algorithm"):
        build_tracker("deepsort", _StubModel())


def test_build_tracker_non_string_raises() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        build_tracker(None, _StubModel())  # type: ignore[arg-type]


def test_build_tracker_requires_model() -> None:
    with pytest.raises(ValueError, match="loaded YOLO"):
        build_tracker("bytetrack", None)


def test_supported_algorithms_in_sync_with_registry() -> None:
    """Guard against drift: every value in SUPPORTED_ALGORITHMS must build."""
    for algo in SUPPORTED_ALGORITHMS:
        assert isinstance(build_tracker(algo, _StubModel()), Tracker)


# --- pass-through behaviour ------------------------------------------------


def test_tracker_passes_correct_yaml_to_model() -> None:
    """The whole point of M2: only the yaml differs between the two algorithms."""
    model = _StubModel()
    bt = build_tracker("bytetrack", model)
    bs = build_tracker("botsort", model)

    bt.track(source="frame_a", classes=[0, 32], conf=0.3)
    bs.track(source="frame_b", classes=[0, 32], conf=0.3)

    assert len(model.calls) == 2
    assert model.calls[0]["tracker"] == "bytetrack.yaml"
    assert model.calls[1]["tracker"] == "botsort.yaml"
    # And the rest of the kwargs are passed through verbatim.
    assert model.calls[0]["classes"] == [0, 32]
    assert model.calls[0]["conf"] == 0.3
    assert model.calls[0]["persist"] is True


def test_tracker_returns_what_model_returns() -> None:
    """Pass-through return shape: list[Results]-like structure."""
    model = _StubModel()
    tracker = build_tracker("bytetrack", model)
    out = tracker.track(source="frame_x")
    assert isinstance(out, list)
    assert out[0]["tracker"] == "bytetrack.yaml"
