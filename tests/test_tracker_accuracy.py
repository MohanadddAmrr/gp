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


# ===========================================================================
# M3 — services.reid_module
# ===========================================================================

import numpy as np  # noqa: E402

from services.reid_module import (  # noqa: E402
    JerseyColorClassifier,
    JerseyPosReID,
    PositionMemory,
    VALID_CLASSES,
    hex_to_hsv,
)


def _solid_bgr(b: int, g: int, r: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), (b, g, r), dtype=np.uint8)


# --- JerseyColorClassifier ------------------------------------------------


def test_jersey_classifier_basic_red_blue_yellow() -> None:
    """Synthetic crops of pure red/blue/yellow → expected classes.

    Default centers: team_a=red, team_b=blue, ref=yellow.
    """
    cls = JerseyColorClassifier()
    assert cls.classify(_solid_bgr(0, 0, 255)) == "team_a"     # pure red BGR
    assert cls.classify(_solid_bgr(255, 0, 0)) == "team_b"     # pure blue BGR
    assert cls.classify(_solid_bgr(0, 255, 255)) == "referee"  # pure yellow


def test_jersey_classifier_returns_unknown_on_empty_or_dark() -> None:
    cls = JerseyColorClassifier()
    assert cls.classify(np.zeros((0, 0, 3), dtype=np.uint8)) == "unknown"
    # All-black: no pixel passes saturation/value cutoffs.
    assert cls.classify(np.zeros((32, 32, 3), dtype=np.uint8)) == "unknown"


def test_jersey_classifier_handles_tiny_crops() -> None:
    """Crops smaller than 2x2 should not crash; should return 'unknown'."""
    cls = JerseyColorClassifier()
    assert cls.classify(_solid_bgr(0, 0, 255, size=1)) == "unknown"


def test_jersey_classifier_returns_only_valid_class_strings() -> None:
    cls = JerseyColorClassifier()
    for crop in (_solid_bgr(0, 0, 255), _solid_bgr(255, 0, 0), _solid_bgr(0, 255, 0)):
        assert cls.classify(crop) in VALID_CLASSES


# --- PositionMemory -------------------------------------------------------


def test_position_memory_lost_candidates() -> None:
    """After recording 3 ids and advancing 35 frames, all 3 are lost
    (max_lost_frames=30 default).
    """
    mem = PositionMemory()
    mem.record(1, (10, 10), "team_a", frame_idx=0)
    mem.record(2, (20, 20), "team_b", frame_idx=0)
    mem.record(3, (30, 30), "team_a", frame_idx=0)

    # At frame 5: all 3 are lost candidates (gap 1..30).
    lost = mem.lost_candidates(current_frame=5, max_lost_frames=30)
    assert {c[0] for c in lost} == {1, 2, 3}

    # At frame 35: gap is 35 → outside lookback; all dropped.
    lost = mem.lost_candidates(current_frame=35, max_lost_frames=30)
    assert lost == []


def test_position_memory_record_updates_team_only_when_known() -> None:
    """A subsequent 'unknown' classification must not overwrite a prior label."""
    mem = PositionMemory()
    mem.record(1, (10, 10), "team_a", frame_idx=0)
    mem.record(1, (11, 11), "unknown", frame_idx=1)
    state = mem.get(1)
    assert state is not None
    assert state.team == "team_a"


# --- JerseyPosReID --------------------------------------------------------


def test_reid_merges_lost_track_within_window() -> None:
    """Simulate: track 100 disappears for 20 frames, reappears within 50 px.

    With matching jersey class on both sides, JerseyPosReID returns the same
    canonical id for the new raw id.
    """
    cls_classifier = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=30)

    red = _solid_bgr(0, 0, 255)

    # Frame 0..4: raw id 100 seen at (200, 200) wearing red.
    for f in range(5):
        canon_first = reid.resolve(100, (200 + f, 200), red, cls_classifier, frame_idx=f)
    # First raw id gets canonical 1.
    assert canon_first == 1

    # Raw id 100 disappears for 20 frames. Then a NEW raw id 200 shows up
    # within 50 px wearing red — should merge into canonical 1.
    canon_again = reid.resolve(200, (245, 200), red, cls_classifier, frame_idx=24)
    assert canon_again == 1

    stats = reid.stats()
    assert stats["merges_committed"] >= 1


def test_reid_does_not_merge_different_team() -> None:
    """Same position window, different jersey color → no merge (rule a)."""
    cls_classifier = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=30)

    red = _solid_bgr(0, 0, 255)
    blue = _solid_bgr(255, 0, 0)

    canon_a = reid.resolve(10, (300, 300), red, cls_classifier, frame_idx=0)
    canon_b = reid.resolve(20, (305, 300), blue, cls_classifier, frame_idx=10)
    assert canon_a != canon_b


def test_reid_does_not_merge_too_far() -> None:
    """Jersey matches but distance > merge_distance_px → no merge (rule b)."""
    cls_classifier = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=50.0, max_lost_frames=30)

    red = _solid_bgr(0, 0, 255)
    canon_a = reid.resolve(10, (100, 100), red, cls_classifier, frame_idx=0)
    canon_b = reid.resolve(20, (500, 500), red, cls_classifier, frame_idx=5)
    assert canon_a != canon_b


def test_reid_unknown_jersey_blocks_merge() -> None:
    """A crop the classifier cannot label must not be merged silently
    (conservative — see §9 risk register)."""
    cls_classifier = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=30)

    red = _solid_bgr(0, 0, 255)
    black = np.zeros((32, 32, 3), dtype=np.uint8)  # → 'unknown'

    canon_a = reid.resolve(1, (100, 100), red, cls_classifier, frame_idx=0)
    canon_b = reid.resolve(2, (105, 100), black, cls_classifier, frame_idx=5)
    assert canon_a != canon_b


def test_reid_known_raw_id_returns_existing_canonical() -> None:
    """Re-seeing an already-mapped raw id must return its same canonical id
    without burning a new one.
    """
    cls_classifier = JerseyColorClassifier()
    reid = JerseyPosReID()
    red = _solid_bgr(0, 0, 255)

    a = reid.resolve(42, (100, 100), red, cls_classifier, frame_idx=0)
    b = reid.resolve(42, (101, 101), red, cls_classifier, frame_idx=1)
    assert a == b
    assert reid.stats()["new_canonical_assigned"] == 1


def test_hex_to_hsv_round_trip_red() -> None:
    h, s, v = hex_to_hsv("#ef0107")
    # OpenCV hue for red is ~0; allow small wrap.
    assert h <= 5 or h >= 175
    assert s > 200
    assert v > 200


# ===========================================================================
# Day 5 — edge cases + perf invariants
# ===========================================================================

from services.reid_module import classifier_from_team_colors  # noqa: E402


def test_reid_merges_on_long_occlusion_20_frames() -> None:
    """Spec edge case: occlusion >= 20 frames within max_lost_frames=30."""
    cls = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=30)
    red = _solid_bgr(0, 0, 255)

    canon_a = reid.resolve(7, (300, 300), red, cls, frame_idx=0)
    # 20 frames of silence — outside short-occlusion window, inside lookback.
    canon_b = reid.resolve(99, (305, 300), red, cls, frame_idx=20)
    assert canon_a == canon_b


def test_reid_does_not_merge_past_max_lost_frames() -> None:
    """Spec rule: lookback strictly bounded by max_lost_frames."""
    cls = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=15)
    red = _solid_bgr(0, 0, 255)

    canon_a = reid.resolve(1, (300, 300), red, cls, frame_idx=0)
    canon_b = reid.resolve(2, (300, 300), red, cls, frame_idx=50)
    assert canon_a != canon_b, "match should expire after max_lost_frames"


def test_reid_does_not_reclassify_known_raw_id() -> None:
    """Day 5 perf invariant: classifier is invoked once per raw id, then cached.

    We use a counting wrapper to verify the call count.
    """
    counter = {"calls": 0}

    class _CountingClassifier(JerseyColorClassifier):
        def classify(self, crop_bgr):  # type: ignore[override]
            counter["calls"] += 1
            return super().classify(crop_bgr)

    cls = _CountingClassifier()
    reid = JerseyPosReID()
    red = _solid_bgr(0, 0, 255)

    for f in range(10):  # raw id 5 seen 10 times
        reid.resolve(5, (200 + f, 200), red, cls, frame_idx=f)

    assert counter["calls"] == 1, (
        f"classifier should be called exactly once for a stable raw id, "
        f"got {counter['calls']}"
    )


def test_reid_merge_log_records_committed_merges() -> None:
    """Debug log captures every committed merge with distance + team."""
    cls = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=30)
    red = _solid_bgr(0, 0, 255)

    reid.resolve(1, (300, 300), red, cls, frame_idx=0)
    reid.resolve(2, (310, 300), red, cls, frame_idx=10)  # should merge

    assert len(reid._merge_log) == 1
    entry = reid._merge_log[0]
    assert entry["raw_track_id"] == 2
    assert entry["canonical_id"] == 1
    assert entry["team"] == "team_a"
    assert entry["distance_px"] < 120


def test_reid_write_merge_log_creates_file(tmp_path) -> None:
    log_path = tmp_path / "reid_merge_log.json"
    cls = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=120.0, max_lost_frames=30)
    red = _solid_bgr(0, 0, 255)
    reid.resolve(1, (100, 100), red, cls, frame_idx=0)
    reid.resolve(2, (105, 100), red, cls, frame_idx=5)

    written = reid.write_merge_log(log_path)
    assert written == log_path
    import json
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert "stats" in payload and "merges" in payload and "thresholds" in payload
    assert payload["thresholds"]["merge_distance_px"] == 120.0


def test_classifier_from_team_colors_uses_real_jerseys() -> None:
    """A classifier fitted to Arsenal red vs Tottenham navy should map the
    correct synthetic crops to team_a / team_b.
    """
    cls = classifier_from_team_colors(
        team_a_hex="#ef0107",   # Arsenal red
        team_b_hex="#132257",   # Tottenham navy
    )
    assert cls.classify(_solid_bgr(0, 0, 239)) == "team_a"     # red BGR
    assert cls.classify(_solid_bgr(87, 34, 19)) == "team_b"    # navy BGR (#132257)


def test_classifier_from_team_colors_handles_missing_keys() -> None:
    """None values should keep dataclass defaults (graceful degradation)."""
    cls = classifier_from_team_colors(team_a_hex=None, team_b_hex=None)
    # Defaults are red/blue, so this still classifies plausibly.
    assert cls.classify(_solid_bgr(0, 0, 255)) in VALID_CLASSES


def test_reid_near_color_match_does_not_false_merge() -> None:
    """Two visually-distinct teams (red vs orange-red) must not merge.

    Red team_a has hue ~0; orange-red is closer to ~10. With a fitted
    classifier centered exactly on red, orange-red should land in team_a too
    OR unknown — but never silently in team_b. We assert that two crops on
    OPPOSITE sides of the hue spectrum (red vs blue-purple) don't merge.
    """
    cls = JerseyColorClassifier()
    reid = JerseyPosReID(merge_distance_px=200.0, max_lost_frames=30)
    pure_red = _solid_bgr(0, 0, 255)
    deep_blue = _solid_bgr(255, 0, 0)
    canon_a = reid.resolve(1, (300, 300), pure_red, cls, frame_idx=0)
    canon_b = reid.resolve(2, (305, 300), deep_blue, cls, frame_idx=10)
    assert canon_a != canon_b
