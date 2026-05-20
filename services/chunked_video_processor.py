"""
Chunked Video Processor (M4) — long-match stability via chunked + checkpointed
inference.

Why this exists
---------------
A 90-minute match cannot be processed in one VideoCapture pass without OOM
on a laptop GPU/RAM. This module runs the existing tracking + Re-ID pipeline
in N-second chunks, persisting per-chunk state to disk so:

1. RAM never holds more than one chunk's worth of frames + heatmaps.
2. A crash mid-run is recoverable: re-running with `resume=True` jumps the
   VideoCapture forward to the highest completed chunk.
3. The merged `metrics.json` shape is identical to `run_demo.py`'s output
   (subset of top-level keys), so downstream dashboards consume it unchanged.

Design notes
------------
- **State carried across chunks**: canonical id map (so a player keeps the
  same canonical id between chunks 1 and 2), ball position history tail
  (for cross-chunk velocity smoothing), and the global heatmap accumulator.
- **Heatmaps are persisted as compressed `.npz`** between chunks — far
  smaller on disk than JSON-encoded arrays.
- **Day 6 v1 scope**: ships chunked + checkpointed loop with the M2/M3
  pipeline (BoT-SORT/ByteTrack + Re-ID). Day 7 lands the full
  resume-from-checkpoint test + Day 9 the 45-min real-clip run.

CLI
---
    python -m services.chunked_video_processor \
        --video PATH \
        --output PATH \
        [--chunk-seconds 600] \
        [--no-resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import psutil  # peak-RAM tracking for long-match acceptance (Day 9)
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False


def _peak_ram_mb() -> float:
    """Process RSS in MB, or 0 when psutil isn't available."""
    if not _HAS_PSUTIL:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)


logger = logging.getLogger(__name__)


# --- Frame-level chunk math -----------------------------------------------


def compute_chunk_ranges(
    total_frames: int, fps: float, chunk_seconds: float
) -> list[tuple[int, int]]:
    """Return [(start_frame, end_frame_exclusive), ...] partitioning the video.

    The last chunk may be shorter than `chunk_seconds`. `chunk_seconds <= 0`
    means single-chunk (whole video). `total_frames <= 0` returns `[]`.

    Examples:
        25 fps, 70 s, chunk_seconds=30 → chunks of 750 frames each:
            (0, 750), (750, 1500), (1500, 1750)
    """
    if total_frames <= 0:
        return []
    if chunk_seconds is None or chunk_seconds <= 0:
        return [(0, total_frames)]
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    chunk_frames = max(1, int(round(chunk_seconds * fps)))
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        ranges.append((start, end))
        start = end
    return ranges


# --- State carried across chunks ------------------------------------------


@dataclass
class _ChunkState:
    """Shared state survives between chunks. Persisted as JSON between chunks
    + heat arrays as np.savez_compressed (kept off the JSON for size).
    """

    raw_id_offset: int = 0  # shift raw track ids by this between chunks
    canonical_id_map: dict[int, int] = field(default_factory=dict)
    ball_history_tail: list[list[float]] = field(default_factory=list)
    person_detections_total: int = 0
    ball_detections_total: int = 0
    ball_second_pass_total: int = 0  # P15 RT-DETR recoveries on YOLO misses
    chunks_completed: int = 0
    # Heat arrays live in numpy, persisted via savez_compressed (NOT in JSON):
    heat_global: Optional[np.ndarray] = None  # (H, W) float32

    def to_json(self) -> dict:
        return {
            "raw_id_offset": int(self.raw_id_offset),
            "canonical_id_map": {str(k): int(v) for k, v in self.canonical_id_map.items()},
            "ball_history_tail": self.ball_history_tail,
            "person_detections_total": int(self.person_detections_total),
            "ball_detections_total": int(self.ball_detections_total),
            "ball_second_pass_total": int(self.ball_second_pass_total),
            "chunks_completed": int(self.chunks_completed),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "_ChunkState":
        s = cls()
        s.raw_id_offset = int(payload.get("raw_id_offset", 0))
        s.canonical_id_map = {int(k): int(v) for k, v in payload.get("canonical_id_map", {}).items()}
        s.ball_history_tail = list(payload.get("ball_history_tail", []))
        s.person_detections_total = int(payload.get("person_detections_total", 0))
        s.ball_detections_total = int(payload.get("ball_detections_total", 0))
        s.ball_second_pass_total = int(payload.get("ball_second_pass_total", 0))
        s.chunks_completed = int(payload.get("chunks_completed", 0))
        return s


# --- Checkpoint persistence -----------------------------------------------


def _checkpoint_paths(output_dir: Path, chunk_idx: int) -> tuple[Path, Path]:
    cdir = output_dir / "checkpoints"
    return cdir / f"chunk_{chunk_idx:04d}.json", cdir / f"chunk_{chunk_idx:04d}_heat.npz"


def _write_checkpoint(
    output_dir: Path, chunk_idx: int, state: _ChunkState
) -> tuple[Path, Path]:
    """Persist state to disk after a successful chunk."""
    json_path, heat_path = _checkpoint_paths(output_dir, chunk_idx)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(state.to_json(), indent=2), encoding="utf-8")
    if state.heat_global is not None:
        np.savez_compressed(heat_path, heat_global=state.heat_global)
    return json_path, heat_path


def _read_checkpoint(output_dir: Path, chunk_idx: int) -> Optional[_ChunkState]:
    json_path, heat_path = _checkpoint_paths(output_dir, chunk_idx)
    if not json_path.exists():
        return None
    state = _ChunkState.from_json(json.loads(json_path.read_text(encoding="utf-8")))
    if heat_path.exists():
        with np.load(heat_path) as data:
            if "heat_global" in data.files:
                state.heat_global = data["heat_global"]
    return state


def _highest_completed_chunk(output_dir: Path) -> int:
    """Largest chunk_idx for which a checkpoint json exists. -1 if none."""
    cdir = output_dir / "checkpoints"
    if not cdir.exists():
        return -1
    best = -1
    for p in cdir.glob("chunk_*.json"):
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx > best:
            best = idx
    return best


# --- Per-chunk processing -------------------------------------------------


def _classify_team(reid_classifier, crop, cx: float, frame_w: int) -> str:
    """Resolve a player's team label ('A'/'B').

    Uses the jersey-colour classifier when a crop is available, falling back
    to a left/right-half heuristic (the same fallback run_demo.py uses).
    """
    if reid_classifier is not None and crop is not None and crop.size:
        jc = reid_classifier.classify(crop)
        if jc == "team_a":
            return "A"
        if jc == "team_b":
            return "B"
    return "A" if cx < frame_w / 2 else "B"


def _process_chunk(
    cap,
    start_frame: int,
    end_frame: int,
    state: _ChunkState,
    *,
    yolo,
    tracker,
    reid_layer,
    reid_classifier,
    track_dedup,
    target_size: tuple[int, int],
    frame_size: tuple[int, int],
    fps: float,
    analytics_runner=None,
    ball_second_pass=None,
) -> dict:
    """Run inference + Re-ID + dedup over [start_frame, end_frame). Updates
    `state` in-place and returns a per-chunk summary dict.

    When `analytics_runner` is given, per-frame player/ball detections are also
    fed through the full analytics suite (possession, passes, shots, xG,
    sprints, highlights) so a long match yields the same analytics as
    run_demo.py.

    When `ball_second_pass` is given (P15), an RT-DETR ball detector is run on
    frames where the primary tracker missed the ball, before the detections
    reach the analytics runner.
    """
    import cv2

    target_w, target_h = target_size
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if state.heat_global is None:
        state.heat_global = np.zeros((target_h, target_w), dtype=np.float32)

    chunk_person_dets = 0
    chunk_ball_dets = 0
    chunk_ball_second_pass = 0
    chunk_canonical_ids: set[int] = set()
    frame_idx = start_frame
    # Seed the second-pass crop window from the carried ball-history tail.
    last_known_ball_pos: Optional[tuple[float, float]] = (
        tuple(state.ball_history_tail[-1]) if state.ball_history_tail else None
    )

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.resize(frame, (target_w, target_h))

        # Per-frame detections assembled for the analytics runner.
        frame_players: dict[int, tuple[float, float, str]] = {}
        frame_ball_box: Optional[tuple[float, float, float, float]] = None

        res_list = tracker.track(
            source=frame, classes=[0, 32], conf=0.3, imgsz=640,
            verbose=False, persist=True,
        )
        if res_list:
            res = res_list[0]
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int) + state.raw_id_offset
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)

                for tid, box, cls_id in zip(ids, xyxy, cls_arr):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    if cls_id == 0:  # person
                        chunk_person_dets += 1
                        eff_tid = int(tid)
                        # A crop is needed for Re-ID and/or team classification.
                        crop = None
                        if (reid_layer is not None and reid_classifier is not None) \
                                or analytics_runner is not None:
                            x1i, y1i = max(int(x1), 0), max(int(y1), 0)
                            x2i, y2i = max(int(x2), x1i + 1), max(int(y2), y1i + 1)
                            crop = frame[y1i:y2i, x1i:x2i]
                        if reid_layer is not None and reid_classifier is not None:
                            eff_tid = reid_layer.resolve(
                                raw_track_id=int(tid),
                                position=(cx, cy),
                                crop_bgr=crop,
                                classifier=reid_classifier,
                                frame_idx=frame_idx,
                            )
                        pid = track_dedup.resolve(eff_tid, (cx, cy), frame_idx)
                        chunk_canonical_ids.add(pid)
                        state.canonical_id_map[int(tid)] = int(pid)

                        ix, iy = int(cx), int(cy)
                        if 0 <= ix < target_w and 0 <= iy < target_h:
                            state.heat_global[iy, ix] += 1

                        if analytics_runner is not None:
                            team = _classify_team(reid_classifier, crop, cx, target_w)
                            frame_players[pid] = (cx, cy, team)
                    elif cls_id == 32:  # ball
                        chunk_ball_dets += 1
                        state.ball_history_tail.append([float(cx), float(cy)])
                        # Keep tail bounded so we don't blow JSON size between chunks
                        if len(state.ball_history_tail) > 30:
                            state.ball_history_tail = state.ball_history_tail[-30:]
                        frame_ball_box = (float(x1), float(y1), float(x2), float(y2))
                track_dedup.end_frame(frame_idx)

        # P15 — RT-DETR ball-only second pass on frames the primary tracker
        # missed, before detections reach the analytics runner.
        if frame_ball_box is None and ball_second_pass is not None:
            second_box = ball_second_pass.detect_best(frame, last_known_ball_pos)
            if second_box is not None:
                frame_ball_box = second_box
                chunk_ball_second_pass += 1
                bcx = (second_box[0] + second_box[2]) / 2.0
                bcy = (second_box[1] + second_box[3]) / 2.0
                state.ball_history_tail.append([bcx, bcy])
                if len(state.ball_history_tail) > 30:
                    state.ball_history_tail = state.ball_history_tail[-30:]

        if frame_ball_box is not None:
            last_known_ball_pos = (
                (frame_ball_box[0] + frame_ball_box[2]) / 2.0,
                (frame_ball_box[1] + frame_ball_box[3]) / 2.0,
            )

        if analytics_runner is not None:
            analytics_runner.process_frame(
                frame_idx, frame_idx / fps, frame_players, frame_ball_box
            )
        frame_idx += 1

    state.person_detections_total += chunk_person_dets
    state.ball_detections_total += chunk_ball_dets
    state.ball_second_pass_total += chunk_ball_second_pass

    return {
        "frames_processed": frame_idx - start_frame,
        "person_detections": chunk_person_dets,
        "ball_detections": chunk_ball_dets,
        "ball_second_pass_detections": chunk_ball_second_pass,
        "canonical_ids_in_chunk": len(chunk_canonical_ids),
    }


# --- Public API -----------------------------------------------------------


def process_video_in_chunks(
    video_path: Path,
    output_dir: Path,
    chunk_seconds: float = 600.0,
    config: Optional[dict] = None,
    resume: bool = True,
    enable_analytics: bool = True,
    enable_ball_second_pass: bool = False,
) -> dict:
    """Process a video in chunks with persistent checkpoints.

    Final output:
        <output_dir>/metrics.json            (run_demo-compatible top-level shape)
        <output_dir>/checkpoints/chunk_NNNN.json + chunk_NNNN_heat.npz

    When `enable_analytics` is True (default), the full analytics suite
    (possession, passes, shots, xG, sprints, highlights) runs across all
    chunks and is merged into metrics.json under "analytics". Analytics
    accumulate in-process and are not checkpoint-restored: a crash-resume
    that skips chunks yields analytics covering only the resumed chunks,
    flagged via "analytics".["coverage"].

    When `enable_ball_second_pass` is True (P15), an RT-DETR ball detector
    recovers the ball on frames the primary tracker missed. Off by default:
    on CPU it is markedly slower, so it is opt-in for the long-match path.

    Returns the merged metrics dict.
    """
    import cv2
    from ultralytics import YOLO

    from services.improved_tracker import build_tracker
    from services.reid_module import (
        JerseyColorClassifier,
        JerseyPosReID,
        classifier_from_team_colors,
    )
    from services.tracking_filters import TrackDeduplicator

    cfg = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (orig_w, orig_h)
    target_size = (640, 360)  # match run_demo.py

    chunks = compute_chunk_ranges(total_frames, fps, chunk_seconds)
    if not chunks:
        cap.release()
        raise RuntimeError(f"Video had no frames: {video_path}")

    logger.info(
        "video=%s fps=%.2f total_frames=%d -> %d chunk(s) of <=%ds",
        video_path.name, fps, total_frames, len(chunks), int(chunk_seconds),
    )

    # Resume: skip chunks whose checkpoint exists.
    start_chunk_idx = 0
    state = _ChunkState()
    if resume:
        last = _highest_completed_chunk(output_dir)
        if last >= 0:
            cp = _read_checkpoint(output_dir, last)
            if cp is not None:
                state = cp
                start_chunk_idx = last + 1
                logger.info("resuming after completed chunk %d", last)

    # Build pipeline (one model + tracker + reid for the whole run; reset
    # tracker state between chunks via raw_id_offset).
    detection_cfg = (cfg.get("detection") or {})
    tracking_cfg = detection_cfg.get("tracking") or {}
    weights = cfg.get("weights", "yolov8n.pt")
    yolo = YOLO(weights)
    algorithm = str(tracking_cfg.get("algorithm", "bytetrack")).lower()
    tracker = build_tracker(algorithm, yolo)

    reid_layer = None
    reid_classifier = None
    if bool(tracking_cfg.get("reid_enabled", True)):
        reid_layer = JerseyPosReID(
            merge_distance_px=float(tracking_cfg.get("reid_merge_distance_px", 120.0)),
            max_lost_frames=int(tracking_cfg.get("reid_max_lost_frames", 30)),
        )
        team_colors = cfg.get("team_colors") or {}
        ta_hex = team_colors.get(cfg.get("team_a")) if cfg.get("team_a") else None
        tb_hex = team_colors.get(cfg.get("team_b")) if cfg.get("team_b") else None
        reid_classifier = (
            classifier_from_team_colors(ta_hex, tb_hex)
            if (ta_hex or tb_hex)
            else JerseyColorClassifier()
        )

    track_dedup = TrackDeduplicator(merge_distance_px=80.0, max_lost_frames=30)

    # P11 — full analytics suite (possession/passes/shots/xG/sprints/highlights).
    analytics_runner = None
    if enable_analytics:
        from services.match_analytics_runner import MatchAnalyticsRunner

        analytics_runner = MatchAnalyticsRunner(
            frame_width=target_size[0],
            frame_height=target_size[1],
            fps=fps,
        )

    # P15 — RT-DETR ball-only second pass (opt-in; slower on CPU).
    ball_second_pass = None
    if enable_ball_second_pass:
        from services.rtdetr_ball_detector import RTDetrBallDetector

        sp_cfg = (cfg.get("ball_tracking") or {}).get("second_pass") or {}
        ball_second_pass = RTDetrBallDetector(
            conf=float(sp_cfg.get("confidence_threshold", 0.25)),
            crop_size=int(sp_cfg.get("crop_size", 640)),
        )
        logger.info("ball second pass: enabled (RT-DETR)")

    chunk_summaries: list[dict] = []
    # Day 9: peak RAM tracking — sampled after each chunk write.
    peak_ram_mb = _peak_ram_mb()

    t0 = time.perf_counter()
    for idx, (s, e) in enumerate(chunks):
        if idx < start_chunk_idx:
            chunk_summaries.append({"chunk": idx, "skipped": True, "frames": e - s})
            continue
        state.raw_id_offset = idx * 100_000  # mirror run_demo's stitching scheme
        summary = _process_chunk(
            cap, s, e, state,
            yolo=yolo, tracker=tracker,
            reid_layer=reid_layer, reid_classifier=reid_classifier,
            track_dedup=track_dedup,
            target_size=target_size, frame_size=frame_size,
            fps=fps, analytics_runner=analytics_runner,
            ball_second_pass=ball_second_pass,
        )
        summary["chunk"] = idx
        summary["start_frame"] = s
        summary["end_frame"] = e
        state.chunks_completed = idx + 1
        _write_checkpoint(output_dir, idx, state)
        chunk_summaries.append(summary)

        # Sample RAM after each chunk; carry the peak.
        chunk_ram = _peak_ram_mb()
        summary["ram_mb"] = round(chunk_ram, 1)
        peak_ram_mb = max(peak_ram_mb, chunk_ram)

        logger.info(
            "chunk %d/%d done frames=%d-%d  person=%d ball=%d canonical=%d ram=%.0fMB",
            idx + 1, len(chunks), s, e,
            summary["person_detections"], summary["ball_detections"],
            summary["canonical_ids_in_chunk"], chunk_ram,
        )

    elapsed = time.perf_counter() - t0
    cap.release()

    # Merge into a metrics.json with the same shape run_demo.py produces
    # (subset of top-level keys; downstream dashboards keep working).
    final_canonicals = len(set(state.canonical_id_map.values()))
    raw_track_count = len(state.canonical_id_map)
    duration_seconds = total_frames / fps if fps > 0 else 0
    metrics = {
        "frame": total_frames,
        "num_players": final_canonicals,
        "raw_track_ids": raw_track_count,
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_seconds / 60.0, 2),
        "tracking_quality": {
            "canonical_players": final_canonicals,
            "raw_yolo_tracks": raw_track_count,
            "dedup_ratio": round(raw_track_count / max(final_canonicals, 1), 1),
        },
        "ball_tracking": {
            "total_detections": int(state.ball_detections_total),
            "detection_rate": (
                state.ball_detections_total / total_frames if total_frames else 0.0
            ),
            "second_pass_recoveries": int(state.ball_second_pass_total),
            "position_history": state.ball_history_tail,
        },
        "video": video_path.name,
        "fps": fps,
        "chunks": {
            "count": len(chunks),
            "chunk_seconds": chunk_seconds,
            "summaries": chunk_summaries,
            "wall_clock_seconds": round(elapsed, 1),
            "peak_ram_mb": round(peak_ram_mb, 1),
        },
        "tracker": getattr(tracker, "name", algorithm),
        "reid_enabled": reid_layer is not None,
    }
    if reid_layer is not None:
        metrics["reid_stats"] = reid_layer.stats()

    # P11 — merge the full analytics suite computed across all chunks.
    if analytics_runner is not None:
        analytics = analytics_runner.finalize(total_frames)
        analytics["coverage"] = (
            "full"
            if start_chunk_idx == 0
            else f"partial (resumed from chunk {start_chunk_idx}; "
            f"earlier chunks not re-analyzed)"
        )
        metrics["analytics"] = analytics

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


# --- CLI ------------------------------------------------------------------


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.chunked_video_processor",
        description="Run tracking + Re-ID over a video in checkpointed chunks.",
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=600.0,
        help="Chunk length in seconds (default: 600 = 10 minutes).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start from chunk 0.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
        help="Tracker algorithm.",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable Jersey + Position Re-ID layer.",
    )
    parser.add_argument(
        "--weights",
        default="weights/yolov8n.pt",
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--no-analytics",
        action="store_true",
        help="Skip the full analytics suite (tracking only).",
    )
    parser.add_argument(
        "--ball-second-pass",
        action="store_true",
        help="Enable the RT-DETR ball-only second pass (P15; slower on CPU).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = {
        "weights": args.weights,
        "detection": {
            "tracking": {
                "algorithm": args.algorithm,
                "reid_enabled": args.reid,
            },
        },
    }
    metrics = process_video_in_chunks(
        video_path=args.video,
        output_dir=args.output,
        chunk_seconds=args.chunk_seconds,
        config=cfg,
        resume=not args.no_resume,
        enable_analytics=not args.no_analytics,
        enable_ball_second_pass=args.ball_second_pass,
    )
    print(
        f"chunks={metrics['chunks']['count']} "
        f"frames={metrics['frame']} "
        f"canonical={metrics['num_players']} "
        f"wall_clock={metrics['chunks']['wall_clock_seconds']}s"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
