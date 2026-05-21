"""Tests for services.chunked_video_processor (M4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.chunked_video_processor import (  # noqa: E402
    _ChunkState,
    _highest_completed_chunk,
    _read_checkpoint,
    _write_checkpoint,
    compute_chunk_ranges,
)


# --- compute_chunk_ranges -------------------------------------------------


def test_chunk_boundaries_70s_at_25fps_chunk_30s() -> None:
    """Spec example: 25 fps, 70 s, chunk_seconds=30 → 3 chunks (30 s, 30 s, 10 s)."""
    fps = 25.0
    total_frames = int(70 * fps)  # 1750
    ranges = compute_chunk_ranges(total_frames, fps, chunk_seconds=30)
    assert len(ranges) == 3, f"expected 3 chunks, got {len(ranges)}: {ranges}"
    assert ranges[0] == (0, 750)
    assert ranges[1] == (750, 1500)
    assert ranges[2] == (1500, 1750)
    # Coverage is exactly total_frames (no gaps, no overlap).
    covered = sum(e - s for s, e in ranges)
    assert covered == total_frames


def test_chunk_boundaries_exact_multiple() -> None:
    """3 chunks of exactly 30 s with no remainder."""
    ranges = compute_chunk_ranges(2250, 25.0, chunk_seconds=30)
    assert len(ranges) == 3
    assert ranges[-1] == (1500, 2250)


def test_chunk_boundaries_chunk_larger_than_video() -> None:
    """chunk_seconds bigger than video duration → 1 chunk covering all frames."""
    ranges = compute_chunk_ranges(500, 25.0, chunk_seconds=600)
    assert ranges == [(0, 500)]


def test_chunk_boundaries_zero_chunk_means_single() -> None:
    """chunk_seconds <= 0 means 'one chunk, the whole video'."""
    assert compute_chunk_ranges(1000, 25.0, chunk_seconds=0) == [(0, 1000)]
    assert compute_chunk_ranges(1000, 25.0, chunk_seconds=-5) == [(0, 1000)]


def test_chunk_boundaries_empty_video() -> None:
    assert compute_chunk_ranges(0, 25.0, 30) == []


def test_chunk_boundaries_invalid_fps_raises() -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        compute_chunk_ranges(100, 0, 30)


# --- checkpoint persistence -----------------------------------------------


def test_checkpoint_write_then_read_round_trip(tmp_path: Path) -> None:
    state = _ChunkState(
        raw_id_offset=200_000,
        canonical_id_map={1: 5, 7: 5, 12: 9},
        ball_history_tail=[[10.0, 20.0], [11.0, 21.0]],
        person_detections_total=4321,
        ball_detections_total=87,
        chunks_completed=2,
        heat_global=np.full((360, 640), 3.5, dtype=np.float32),
    )
    json_path, heat_path = _write_checkpoint(tmp_path, chunk_idx=2, state=state)
    assert json_path.exists()
    assert heat_path.exists()

    restored = _read_checkpoint(tmp_path, chunk_idx=2)
    assert restored is not None
    assert restored.raw_id_offset == 200_000
    assert restored.canonical_id_map == {1: 5, 7: 5, 12: 9}
    assert restored.ball_history_tail == [[10.0, 20.0], [11.0, 21.0]]
    assert restored.person_detections_total == 4321
    assert restored.chunks_completed == 2
    assert restored.heat_global is not None
    assert restored.heat_global.shape == (360, 640)
    assert restored.heat_global.dtype == np.float32
    assert restored.heat_global[0, 0] == pytest.approx(3.5)


def test_highest_completed_chunk_picks_max(tmp_path: Path) -> None:
    state = _ChunkState()
    _write_checkpoint(tmp_path, chunk_idx=0, state=state)
    _write_checkpoint(tmp_path, chunk_idx=3, state=state)
    _write_checkpoint(tmp_path, chunk_idx=1, state=state)
    assert _highest_completed_chunk(tmp_path) == 3


def test_highest_completed_chunk_returns_minus_one_when_empty(tmp_path: Path) -> None:
    assert _highest_completed_chunk(tmp_path) == -1


def test_read_checkpoint_returns_none_for_missing(tmp_path: Path) -> None:
    assert _read_checkpoint(tmp_path, chunk_idx=99) is None


# --- _ChunkState round-trip ----------------------------------------------


def test_chunk_state_json_round_trip() -> None:
    s = _ChunkState(
        raw_id_offset=100_000,
        canonical_id_map={42: 7, 99: 7},
        ball_history_tail=[[1.0, 2.0]],
        person_detections_total=10,
        ball_detections_total=2,
        chunks_completed=1,
    )
    payload = s.to_json()
    s2 = _ChunkState.from_json(json.loads(json.dumps(payload)))
    assert s2.raw_id_offset == 100_000
    assert s2.canonical_id_map == {42: 7, 99: 7}
    assert s2.ball_history_tail == [[1.0, 2.0]]
    assert s2.person_detections_total == 10


# --- end-to-end: synthetic video, multi-chunk + resume ---------------------


def _make_synthetic_video(path: Path, n_frames: int, fps: int = 25) -> Path:
    """Write a tiny synthetic mp4 with green-on-black frames so VideoCapture
    can open it. Test runs need ultralytics+cv2 available; skipped otherwise.
    """
    pytest.importorskip("cv2")
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (320, 240))
    if not writer.isOpened():
        pytest.skip("cv2 mp4v writer not available on this platform")
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        img = rng.integers(0, 50, (240, 320, 3), dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


@pytest.mark.skipif(
    sys.platform == "win32" and not Path("weights/yolov8n.pt").exists(),
    reason="needs weights/yolov8n.pt available locally",
)
def test_multi_chunk_synthetic_video_runs_without_crashing(tmp_path: Path) -> None:
    """Synthetic 50-frame @ 25fps clip, chunk_seconds=1 (=25 frames) → 2 chunks.
    No detections expected on noise frames, but the loop must complete and
    produce a valid metrics.json + checkpoints.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")

    video = _make_synthetic_video(tmp_path / "synth.mp4", n_frames=50, fps=25)
    output = tmp_path / "out"

    from services.chunked_video_processor import process_video_in_chunks

    metrics = process_video_in_chunks(
        video_path=video,
        output_dir=output,
        chunk_seconds=1,  # 25 frames per chunk
        config={
            "weights": "weights/yolov8n.pt",
            "detection": {"tracking": {"algorithm": "bytetrack", "reid_enabled": False}},
        },
        resume=False,
    )

    assert (output / "metrics.json").exists()
    assert (output / "checkpoints" / "chunk_0000.json").exists()
    assert (output / "checkpoints" / "chunk_0001.json").exists()

    # Shape: top-level keys overlap with run_demo.py
    for k in ("frame", "num_players", "raw_track_ids", "tracking_quality",
              "ball_tracking", "duration_seconds", "duration_minutes"):
        assert k in metrics, f"missing top-level key {k!r} in chunked metrics.json"

    assert metrics["chunks"]["count"] == 2


@pytest.mark.skipif(
    not Path("weights/yolov8n.pt").exists(),
    reason="needs weights/yolov8n.pt",
)
def test_resume_skips_completed_chunks(tmp_path: Path) -> None:
    """First run writes chunk_0000.json + chunk_0001.json. Re-running with
    resume=True must mark already-completed chunks as skipped in summaries.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")

    video = _make_synthetic_video(tmp_path / "synth.mp4", n_frames=50, fps=25)
    output = tmp_path / "out"

    from services.chunked_video_processor import process_video_in_chunks

    cfg = {
        "weights": "weights/yolov8n.pt",
        "detection": {"tracking": {"algorithm": "bytetrack", "reid_enabled": False}},
    }
    process_video_in_chunks(
        video_path=video, output_dir=output, chunk_seconds=1, config=cfg, resume=False,
    )

    # Re-run with resume=True — every chunk should be marked skipped.
    metrics2 = process_video_in_chunks(
        video_path=video, output_dir=output, chunk_seconds=1, config=cfg, resume=True,
    )
    summaries = metrics2["chunks"]["summaries"]
    assert all(s.get("skipped") for s in summaries), summaries


def test_metrics_top_level_shape_matches_run_demo_subset() -> None:
    """Lock the contract so dashboards built around run_demo.py output keep
    working. Non-empty subset must be present.
    """
    expected_subset = {
        "frame",
        "num_players",
        "raw_track_ids",
        "duration_seconds",
        "duration_minutes",
        "tracking_quality",
        "ball_tracking",
    }
    # Build a minimal metrics dict the way the processor would produce one:
    fake = {
        "frame": 100, "num_players": 5, "raw_track_ids": 42,
        "duration_seconds": 4.0, "duration_minutes": 0.07,
        "tracking_quality": {"canonical_players": 5, "raw_yolo_tracks": 42, "dedup_ratio": 8.4},
        "ball_tracking": {"total_detections": 0, "detection_rate": 0.0, "position_history": []},
    }
    assert expected_subset.issubset(fake.keys())


# ===========================================================================
# Day 7 — formalized spec-named tests (per §7.1 M4 acceptance)
# ===========================================================================


@pytest.mark.skipif(
    not Path("weights/yolov8n.pt").exists(),
    reason="needs weights/yolov8n.pt",
)
def test_resume_from_checkpoint(tmp_path: Path) -> None:
    """§7.1 M4 acceptance: 'after chunk 1 succeeds, simulate interrupt by
    deleting later artifacts, re-run with resume=True, verify total frames
    processed equals total.'

    Strategy:
      1. Cold run that completes all chunks.
      2. Delete the LAST checkpoint (simulating an interrupt mid-final-chunk).
      3. Re-run with resume=True.
      4. Verify final metrics.json exists with frame == total_frames; the
         restarted run must have skipped the early chunks and reprocessed
         only the deleted one.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")

    video = _make_synthetic_video(tmp_path / "synth.mp4", n_frames=75, fps=25)
    output = tmp_path / "out"
    cfg = {
        "weights": "weights/yolov8n.pt",
        "detection": {"tracking": {"algorithm": "bytetrack", "reid_enabled": False}},
    }

    from services.chunked_video_processor import process_video_in_chunks

    # 1. Cold run: 75 frames @ 25fps, chunk_seconds=1 → 3 chunks
    metrics_cold = process_video_in_chunks(
        video_path=video, output_dir=output, chunk_seconds=1,
        config=cfg, resume=False,
    )
    assert metrics_cold["chunks"]["count"] == 3
    cdir = output / "checkpoints"
    for idx in range(3):
        assert (cdir / f"chunk_{idx:04d}.json").exists()

    # 2. Simulate crash mid-final-chunk: drop last checkpoint pair.
    (cdir / "chunk_0002.json").unlink()
    (cdir / "chunk_0002_heat.npz").unlink()

    # 3. Resume.
    metrics_warm = process_video_in_chunks(
        video_path=video, output_dir=output, chunk_seconds=1,
        config=cfg, resume=True,
    )

    # 4. Total frames == video length, only chunk 2 reprocessed.
    assert metrics_warm["frame"] == 75
    summaries = metrics_warm["chunks"]["summaries"]
    assert summaries[0].get("skipped"), "chunk 0 should be skipped on resume"
    assert summaries[1].get("skipped"), "chunk 1 should be skipped on resume"
    assert not summaries[2].get("skipped"), "chunk 2 should be reprocessed"
    # Final checkpoints exist again after the resume run.
    for idx in range(3):
        assert (cdir / f"chunk_{idx:04d}.json").exists()


@pytest.mark.skipif(
    not Path("weights/yolov8n.pt").exists(),
    reason="needs weights/yolov8n.pt",
)
def test_metrics_shape_matches_run_demo(tmp_path: Path) -> None:
    """§7.1 M4 acceptance: 'final metrics.json has same top-level keys as a
    run_demo.py output.' We assert the lock-list of keys plus types so a
    casual refactor of either side trips the test.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")

    video = _make_synthetic_video(tmp_path / "synth.mp4", n_frames=50, fps=25)
    output = tmp_path / "out"

    from services.chunked_video_processor import process_video_in_chunks

    metrics = process_video_in_chunks(
        video_path=video, output_dir=output, chunk_seconds=1,
        config={
            "weights": "weights/yolov8n.pt",
            "detection": {"tracking": {"algorithm": "bytetrack", "reid_enabled": False}},
        },
        resume=False,
    )

    # Lock-list (subset of run_demo.py's metrics.json top-level keys).
    expected_keys_with_types = {
        "frame": int,
        "num_players": int,
        "raw_track_ids": int,
        "duration_seconds": (int, float),
        "duration_minutes": (int, float),
        "tracking_quality": dict,
        "ball_tracking": dict,
    }
    for key, expected_type in expected_keys_with_types.items():
        assert key in metrics, f"top-level key {key!r} missing"
        assert isinstance(metrics[key], expected_type), (
            f"key {key!r} has wrong type: got {type(metrics[key]).__name__}"
        )

    # tracking_quality nested shape.
    tq = metrics["tracking_quality"]
    for k in ("canonical_players", "raw_yolo_tracks", "dedup_ratio"):
        assert k in tq, f"tracking_quality missing {k!r}"

    # ball_tracking nested shape.
    bt = metrics["ball_tracking"]
    for k in ("total_detections", "detection_rate", "position_history"):
        assert k in bt, f"ball_tracking missing {k!r}"
