"""
tests/test_ground_truth.py — Task D1 & D3 (Seif)

Tests for:
  - tools/annotation_helper.py  (ground truth JSON format + resume logic)
  - services/ground_truth_manager.py  (list_clips, load, coverage, merge, export)

All tests use tmp_path (pytest fixture) so nothing touches real GT files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ground_truth_manager import GroundTruthManager, GroundTruthSummary


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_gt_payload(
    video: str = "test.mp4",
    fps: float = 25.0,
    frame_size: list = None,
    frames: list = None,
    source: str = "human",
) -> dict:
    """Build a minimal valid GT payload dict."""
    return {
        "video": video,
        "fps": fps,
        "frame_size": frame_size or [640, 360],
        "coordinate_space": "inference_640x360",
        "source": source,
        "frames": frames or [],
    }


def _make_frame(frame_idx: int, n_team_a: int = 3, n_team_b: int = 3,
                include_ball: bool = True) -> dict:
    """Create a synthetic annotated frame with the standard object schema."""
    objects = []
    oid = 1
    for _ in range(n_team_a):
        objects.append({"id": oid, "class": "team_a",
                        "bbox": [10 * oid, 10, 20, 40], "conf": 1.0})
        oid += 1
    for _ in range(n_team_b):
        objects.append({"id": oid, "class": "team_b",
                        "bbox": [10 * oid, 10, 20, 40], "conf": 1.0})
        oid += 1
    if include_ball:
        objects.append({"id": oid, "class": "ball",
                        "bbox": [300, 150, 10, 10], "conf": 1.0})
    return {"frame_idx": frame_idx, "objects": objects}


def _write_gt(tmp_path: Path, stem: str, payload: dict) -> Path:
    """Write a GT JSON file into tmp_path and return the path."""
    p = tmp_path / f"{stem}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Task D1 — Ground truth JSON format round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthFormatRoundTrip:
    """Validate that the annotation format survives a write → read → modify → write cycle."""

    def test_round_trip_preserves_all_top_level_fields(self, tmp_path):
        """Writing and reading a payload returns identical top-level keys and values."""
        payload = _make_gt_payload(
            video="match.mp4",
            fps=25.0,
            frame_size=[640, 360],
            frames=[_make_frame(0), _make_frame(25)],
        )
        p = _write_gt(tmp_path, "match", payload)

        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["video"] == payload["video"]
        assert loaded["fps"] == payload["fps"]
        assert loaded["frame_size"] == payload["frame_size"]
        assert loaded["coordinate_space"] == payload["coordinate_space"]
        assert loaded["source"] == payload["source"]
        assert len(loaded["frames"]) == 2

    def test_round_trip_object_schema(self, tmp_path):
        """Each object in a frame must have id, class, bbox (len 4), conf."""
        frame = _make_frame(0, n_team_a=2, n_team_b=2, include_ball=True)
        payload = _make_gt_payload(frames=[frame])
        p = _write_gt(tmp_path, "schema_check", payload)

        loaded = json.loads(p.read_text(encoding="utf-8"))
        for obj in loaded["frames"][0]["objects"]:
            assert "id" in obj
            assert "class" in obj
            assert "bbox" in obj
            assert len(obj["bbox"]) == 4
            assert "conf" in obj

    def test_modify_and_re_save(self, tmp_path):
        """Load → change a class → save → reload confirms the change persisted."""
        frames = [_make_frame(0)]
        payload = _make_gt_payload(frames=frames)
        p = _write_gt(tmp_path, "modify_test", payload)

        loaded = json.loads(p.read_text(encoding="utf-8"))
        # Change the first object's class to goalkeeper
        loaded["frames"][0]["objects"][0]["class"] = "goalkeeper"
        p.write_text(json.dumps(loaded, indent=2), encoding="utf-8")

        reloaded = json.loads(p.read_text(encoding="utf-8"))
        assert reloaded["frames"][0]["objects"][0]["class"] == "goalkeeper"

    def test_empty_frames_list_is_valid(self, tmp_path):
        """A payload with zero frames is a valid (not-yet-annotated) file."""
        payload = _make_gt_payload(frames=[])
        p = _write_gt(tmp_path, "empty", payload)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["frames"] == []

    def test_frame_idx_is_preserved(self, tmp_path):
        """frame_idx values must survive serialisation without drift."""
        frame_indices = [0, 25, 50, 100, 2500]
        frames = [_make_frame(idx) for idx in frame_indices]
        payload = _make_gt_payload(frames=frames)
        p = _write_gt(tmp_path, "frame_idx", payload)

        loaded = json.loads(p.read_text(encoding="utf-8"))
        saved_indices = [f["frame_idx"] for f in loaded["frames"]]
        assert saved_indices == frame_indices


# ── annotation_helper resume logic ────────────────────────────────────────────

class TestAnnotationHelperResume:
    """
    Test that already-annotated frames are skipped on resume.
    We test the pure data-logic path (no cv2 window) by inspecting the
    payload structure that annotation_helper would produce.
    """

    def test_resume_skips_already_annotated_frames(self, tmp_path):
        """
        If frames 25 and 50 are already in the JSON, a new annotation pass
        over frames [25, 50, 75] should only need to annotate frame 75.
        """
        existing_frames = [_make_frame(25), _make_frame(50)]
        payload = _make_gt_payload(frames=existing_frames)
        p = _write_gt(tmp_path, "resume_test", payload)

        # Simulate what annotation_helper does: load existing, find annotated set.
        loaded = json.loads(p.read_text(encoding="utf-8"))
        already_done = {int(f["frame_idx"]) for f in loaded["frames"]}

        to_visit = [25, 50, 75]
        remaining = [idx for idx in to_visit if idx not in already_done]

        assert remaining == [75], (
            f"Expected only [75] to remain, got {remaining}"
        )

    def test_resume_with_no_existing_file_needs_all_frames(self, tmp_path):
        """If no existing JSON, all sampled frames need annotation."""
        gt_path = tmp_path / "new_clip.json"
        assert not gt_path.exists()

        # Simulate: load existing returns empty frames
        already_done: set[int] = set()
        to_visit = [0, 25, 50, 75]
        remaining = [idx for idx in to_visit if idx not in already_done]
        assert remaining == to_visit

    def test_annotated_frame_indices_are_correct_type(self, tmp_path):
        """frame_idx values must be comparable as integers."""
        frames = [{"frame_idx": "25", "objects": []},   # string (edge case)
                  {"frame_idx": 50, "objects": []}]      # int
        payload = _make_gt_payload(frames=frames)
        p = _write_gt(tmp_path, "type_check", payload)

        loaded = json.loads(p.read_text(encoding="utf-8"))
        # annotation_helper uses int() cast; verify int() works on both
        indices = {int(f["frame_idx"]) for f in loaded["frames"]}
        assert 25 in indices
        assert 50 in indices


# ═══════════════════════════════════════════════════════════════════════════════
# Task D3 — GroundTruthManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthManagerListClips:

    def test_list_clips_empty_dir(self, tmp_path):
        mgr = GroundTruthManager(gt_dir=tmp_path)
        assert mgr.list_clips() == []

    def test_list_clips_after_writing_two(self, tmp_path):
        """Two written GT files → list_clips returns two summaries."""
        p1 = _make_gt_payload("clip_a.mp4", frames=[_make_frame(0), _make_frame(25)])
        p2 = _make_gt_payload("clip_b.mp4", frames=[_make_frame(0)])
        _write_gt(tmp_path, "clip_a", p1)
        _write_gt(tmp_path, "clip_b", p2)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        summaries = mgr.list_clips()

        assert len(summaries) == 2
        stems = {s.stem for s in summaries}
        assert stems == {"clip_a", "clip_b"}

    def test_list_clips_skips_corrupt_json(self, tmp_path):
        """A corrupt JSON file must not crash list_clips; it is silently skipped."""
        good = _make_gt_payload(frames=[_make_frame(0)])
        _write_gt(tmp_path, "good", good)
        (tmp_path / "bad.json").write_text("NOT JSON {{{", encoding="utf-8")

        mgr = GroundTruthManager(gt_dir=tmp_path)
        summaries = mgr.list_clips()
        assert len(summaries) == 1
        assert summaries[0].stem == "good"

    def test_list_clips_skips_non_dict_json(self, tmp_path):
        """A JSON list at the top level is not our format; it must be skipped."""
        (tmp_path / "old_format.json").write_text("[1, 2, 3]", encoding="utf-8")
        mgr = GroundTruthManager(gt_dir=tmp_path)
        assert mgr.list_clips() == []

    def test_summary_fields_are_populated(self, tmp_path):
        """A GroundTruthSummary must carry correct fps, frame_size, counts."""
        frames = [
            _make_frame(0, n_team_a=3, n_team_b=2, include_ball=True),
            _make_frame(25, n_team_a=2, n_team_b=3, include_ball=False),
        ]
        payload = _make_gt_payload("match.mp4", fps=50.0,
                                   frame_size=[640, 360], frames=frames)
        _write_gt(tmp_path, "match", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        s = mgr.list_clips()[0]

        assert isinstance(s, GroundTruthSummary)
        assert s.fps == 50.0
        assert s.frame_size == [640, 360]
        assert s.n_frames_annotated == 2
        assert s.video_filename == "match.mp4"
        # team_a: 3+2=5, team_b: 2+3=5, ball: 1 (only first frame)
        assert s.class_counts.get("team_a", 0) == 5
        assert s.class_counts.get("team_b", 0) == 5
        assert s.class_counts.get("ball", 0) == 1


class TestGroundTruthManagerLoad:

    def test_load_returns_correct_payload(self, tmp_path):
        payload = _make_gt_payload("test.mp4", frames=[_make_frame(0)])
        _write_gt(tmp_path, "test", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        loaded = mgr.load("test")
        assert loaded["video"] == "test.mp4"
        assert len(loaded["frames"]) == 1

    def test_load_raises_on_unknown_stem(self, tmp_path):
        mgr = GroundTruthManager(gt_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            mgr.load("does_not_exist")

    def test_load_accepts_stem_with_or_without_extension(self, tmp_path):
        payload = _make_gt_payload(frames=[_make_frame(0)])
        _write_gt(tmp_path, "myfile", payload)
        mgr = GroundTruthManager(gt_dir=tmp_path)

        # Both forms should work
        a = mgr.load("myfile")
        b = mgr.load("myfile.json")
        assert a["frames"] == b["frames"]


class TestGroundTruthManagerCoveragePercent:

    def test_coverage_percent_basic(self, tmp_path):
        """3 annotated frames out of 100 total = 3.0%."""
        frames = [_make_frame(i) for i in [0, 25, 50]]
        payload = _make_gt_payload(frames=frames)
        _write_gt(tmp_path, "clip", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        cov = mgr.coverage_percent("clip", total_video_frames=100)
        assert cov == 3.0

    def test_coverage_percent_full(self, tmp_path):
        """All frames annotated → 100.0%."""
        frames = [_make_frame(i) for i in range(50)]
        payload = _make_gt_payload(frames=frames)
        _write_gt(tmp_path, "full", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        assert mgr.coverage_percent("full", total_video_frames=50) == 100.0

    def test_coverage_percent_zero_total(self, tmp_path):
        """total_video_frames=0 must return 0.0 (no division by zero)."""
        payload = _make_gt_payload(frames=[_make_frame(0)])
        _write_gt(tmp_path, "zerotest", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        assert mgr.coverage_percent("zerotest", total_video_frames=0) == 0.0

    def test_coverage_percent_negative_total(self, tmp_path):
        """Negative total_video_frames returns 0.0."""
        payload = _make_gt_payload(frames=[_make_frame(0)])
        _write_gt(tmp_path, "neg", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        assert mgr.coverage_percent("neg", total_video_frames=-10) == 0.0


class TestGroundTruthManagerMergeClasses:

    def test_merge_classes(self, tmp_path):
        """Remapping goalkeeper → team_a updates all matching objects."""
        objects = [
            {"id": 1, "class": "goalkeeper", "bbox": [0, 0, 20, 40], "conf": 1.0},
            {"id": 2, "class": "team_b",     "bbox": [50, 0, 20, 40], "conf": 1.0},
        ]
        payload = _make_gt_payload(frames=[{"frame_idx": 0, "objects": objects}])
        _write_gt(tmp_path, "gk_test", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        mgr.merge_classes("gk_test", {"goalkeeper": "team_a"})

        reloaded = mgr.load("gk_test")
        classes = [o["class"] for o in reloaded["frames"][0]["objects"]]
        assert classes == ["team_a", "team_b"]

    def test_merge_classes_invalid_target_raises(self, tmp_path):
        """Mapping to an invalid class should raise ValueError."""
        payload = _make_gt_payload(frames=[_make_frame(0)])
        _write_gt(tmp_path, "invalid_target", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        with pytest.raises(ValueError, match="invalid target class"):
            mgr.merge_classes("invalid_target", {"team_a": "superplayer"})

    def test_merge_classes_unknown_source_is_ignored(self, tmp_path):
        """Objects whose class is not in the mapping are left unchanged."""
        objects = [
            {"id": 1, "class": "team_a", "bbox": [0, 0, 20, 40], "conf": 1.0},
            {"id": 2, "class": "team_b", "bbox": [50, 0, 20, 40], "conf": 1.0},
        ]
        payload = _make_gt_payload(frames=[{"frame_idx": 0, "objects": objects}])
        _write_gt(tmp_path, "unchanged", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        # Mapping only applies to "referee" — none exist, so nothing changes.
        mgr.merge_classes("unchanged", {"referee": "team_a"})

        reloaded = mgr.load("unchanged")
        classes = [o["class"] for o in reloaded["frames"][0]["objects"]]
        assert classes == ["team_a", "team_b"]

    def test_merge_classes_is_idempotent(self, tmp_path):
        """Running the same merge twice yields the same result."""
        objects = [{"id": 1, "class": "goalkeeper", "bbox": [0, 0, 20, 40], "conf": 1.0}]
        payload = _make_gt_payload(frames=[{"frame_idx": 0, "objects": objects}])
        _write_gt(tmp_path, "idempotent_merge", payload)

        mgr = GroundTruthManager(gt_dir=tmp_path)
        mgr.merge_classes("idempotent_merge", {"goalkeeper": "team_a"})
        mgr.merge_classes("idempotent_merge", {"goalkeeper": "team_a"})  # second run

        reloaded = mgr.load("idempotent_merge")
        classes = [o["class"] for o in reloaded["frames"][0]["objects"]]
        assert classes == ["team_a"]


class TestGroundTruthManagerExportCsv:

    def test_export_summary_csv_creates_file(self, tmp_path):
        """export_summary_csv must create a file with the right headers."""
        frames = [_make_frame(i) for i in [0, 25, 50]]
        _write_gt(tmp_path, "clip_a", _make_gt_payload("a.mp4", frames=frames))
        _write_gt(tmp_path, "clip_b", _make_gt_payload("b.mp4", frames=[_make_frame(0)]))

        mgr = GroundTruthManager(gt_dir=tmp_path)
        out = tmp_path / "summary.csv"
        mgr.export_summary_csv(out)

        assert out.exists()
        lines = out.read_text(encoding="utf-8").splitlines()
        # Header + 2 data rows
        assert len(lines) == 3
        header = lines[0]
        assert "stem" in header
        assert "n_frames_annotated" in header
        assert "count_team_a" in header

    def test_export_summary_csv_row_counts(self, tmp_path):
        """CSV n_frames_annotated column must match the actual frame count."""
        import csv, io
        frames_a = [_make_frame(i) for i in [0, 25, 50]]  # 3 frames
        _write_gt(tmp_path, "clip_x", _make_gt_payload(frames=frames_a))

        mgr = GroundTruthManager(gt_dir=tmp_path)
        out = tmp_path / "out.csv"
        mgr.export_summary_csv(out)

        reader = csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))
        rows = list(reader)
        assert len(rows) == 1
        assert int(rows[0]["n_frames_annotated"]) == 3

    def test_export_summary_csv_empty_dir(self, tmp_path):
        """Empty GT dir → CSV with only a header row (no data rows)."""
        import csv, io
        mgr = GroundTruthManager(gt_dir=tmp_path)
        out = tmp_path / "empty.csv"
        mgr.export_summary_csv(out)
        reader = csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))
        assert list(reader) == []
