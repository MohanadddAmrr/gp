"""
services/ground_truth_manager.py — Task D3 (Seif)

Manages the ground truth annotation files produced by tools/annotation_helper.py.
Provides a clean API used by:
  - services/accuracy_evaluator.py  (to load GT for scoring)
  - services/multi_model_evaluator.py (to compute mAP when GT exists)
  - demo/dashboard_pages/accuracy_report.py (to list clips + coverage)

All GT files live in tests/ground_truth/ and follow the unified track-file schema:
{
  "video": "<filename>",
  "fps": <fps>,
  "frame_size": [w, h],
  "coordinate_space": "inference_640x360",
  "source": "human",
  "frames": [
    {"frame_idx": 25, "objects": [{"id": 1, "class": "team_a",
                                    "bbox": [x,y,w,h], "conf": 1.0}]}
  ]
}

Class vocabulary: team_a | team_b | referee | goalkeeper | ball | ignore
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_GT_DIR = Path(__file__).resolve().parent.parent / "tests" / "ground_truth"
VALID_CLASSES = {"team_a", "team_b", "referee", "goalkeeper", "ball", "ignore"}

# ── data classes ──────────────────────────────────────────────────────────────


@dataclass
class GroundTruthSummary:
    """Lightweight summary of one annotated clip."""
    stem: str                       # video stem, e.g. "arsenalvsfulham_clip"
    video_filename: str             # original video filename
    fps: float
    frame_size: list[int]           # [w, h]
    n_frames_annotated: int
    class_counts: dict[str, int]    # e.g. {"team_a": 340, "ball": 87, ...}
    source: str                     # "human" | "pseudo" | "pipeline"
    gt_path: Path


# ── GroundTruthManager ────────────────────────────────────────────────────────


class GroundTruthManager:
    """
    Manages ground truth JSON files in *gt_dir*.

    Usage
    -----
        mgr = GroundTruthManager()
        for summary in mgr.list_clips():
            print(summary.stem, summary.n_frames_annotated)

        gt = mgr.load("arsenalvsfulham_clip")
        mgr.merge_classes("arsenalvsfulham_clip", {"goalkeeper": "team_a"})
        print(mgr.coverage_percent("arsenalvsfulham_clip", total_video_frames=9000))
        mgr.export_summary_csv(Path("reports/gt_summary.csv"))
    """

    def __init__(self, gt_dir: Path = DEFAULT_GT_DIR) -> None:
        self.gt_dir = Path(gt_dir)
        self.gt_dir.mkdir(parents=True, exist_ok=True)

    # ── discovery ─────────────────────────────────────────────────────────────

    def _gt_paths(self) -> list[Path]:
        """Return all *.json files in gt_dir, sorted by name."""
        return sorted(self.gt_dir.glob("*.json"))

    def list_clips(self) -> list[GroundTruthSummary]:
        """Return a summary for every valid GT file found in gt_dir."""
        summaries: list[GroundTruthSummary] = []
        for path in self._gt_paths():
            try:
                payload = self._read_raw(path)
                summaries.append(self._summarise(path, payload))
            except Exception:
                # Corrupt or wrong-format file — skip silently
                continue
        return summaries

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, video_stem: str) -> dict:
        """
        Load the full annotation payload for *video_stem*.

        Returns the raw dict (same shape as annotation_helper.py output).
        Raises FileNotFoundError when the stem is not found.
        """
        path = self._resolve_path(video_stem)
        return self._read_raw(path)

    # ── class merging ─────────────────────────────────────────────────────────

    def merge_classes(self, video_stem: str, mapping: dict[str, str]) -> None:
        """
        Remap class labels in-place for all objects in the GT file.

        Example
        -------
            # Collapse both goalkeeper variants into their team class:
            mgr.merge_classes("clip", {"goalkeeper_a": "team_a",
                                        "goalkeeper_b": "team_b"})
            # Or simply treat goalkeepers as team_a:
            mgr.merge_classes("clip", {"goalkeeper": "team_a"})

        Invalid target classes (not in VALID_CLASSES) raise ValueError.
        Unknown source classes are silently left unchanged.
        The file is overwritten atomically on success.
        """
        bad_targets = {v for v in mapping.values() if v not in VALID_CLASSES}
        if bad_targets:
            raise ValueError(
                f"merge_classes: invalid target class(es): {bad_targets}. "
                f"Must be one of {VALID_CLASSES}"
            )

        path = self._resolve_path(video_stem)
        payload = self._read_raw(path)

        changed = 0
        for frame in payload.get("frames", []):
            for obj in frame.get("objects", []):
                old_cls = obj.get("class", "")
                if old_cls in mapping:
                    obj["class"] = mapping[old_cls]
                    changed += 1

        self._write_raw(path, payload)
        print(f"[GroundTruthManager] merge_classes: {changed} objects remapped in {path.name}")

    # ── coverage ──────────────────────────────────────────────────────────────

    def coverage_percent(self, video_stem: str, total_video_frames: int) -> float:
        """
        What fraction of the video's frames are annotated?

        Returns a float in [0.0, 100.0].
        If total_video_frames <= 0, returns 0.0.
        """
        if total_video_frames <= 0:
            return 0.0
        payload = self.load(video_stem)
        n_annotated = len(payload.get("frames", []))
        return round(100.0 * n_annotated / total_video_frames, 2)

    # ── export ────────────────────────────────────────────────────────────────

    def export_summary_csv(self, out: Path) -> None:
        """
        Write a CSV with one row per GT clip.

        Columns: stem, video_filename, fps, frame_w, frame_h,
                 n_frames_annotated, source,
                 count_team_a, count_team_b, count_referee,
                 count_goalkeeper, count_ball, count_ignore
        """
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        summaries = self.list_clips()
        class_cols = ["team_a", "team_b", "referee", "goalkeeper", "ball", "ignore"]
        fieldnames = (
            ["stem", "video_filename", "fps", "frame_w", "frame_h",
             "n_frames_annotated", "source"]
            + [f"count_{c}" for c in class_cols]
        )

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries:
                row: dict = {
                    "stem": s.stem,
                    "video_filename": s.video_filename,
                    "fps": s.fps,
                    "frame_w": s.frame_size[0] if len(s.frame_size) > 0 else "",
                    "frame_h": s.frame_size[1] if len(s.frame_size) > 1 else "",
                    "n_frames_annotated": s.n_frames_annotated,
                    "source": s.source,
                }
                for c in class_cols:
                    row[f"count_{c}"] = s.class_counts.get(c, 0)
                writer.writerow(row)

        print(f"[GroundTruthManager] Summary CSV written → {out}  ({len(summaries)} clips)")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_path(self, video_stem: str) -> Path:
        """Find the GT file for *video_stem*, trying with and without .json."""
        candidate = self.gt_dir / f"{video_stem}.json"
        if candidate.exists():
            return candidate
        # Maybe the caller passed a full filename
        candidate2 = self.gt_dir / video_stem
        if candidate2.exists():
            return candidate2
        raise FileNotFoundError(
            f"Ground truth file not found for stem '{video_stem}' "
            f"in {self.gt_dir}. "
            f"Available: {[p.stem for p in self._gt_paths()]}"
        )

    @staticmethod
    def _read_raw(path: Path) -> dict:
        """Load and parse a GT JSON file. Raises on parse error."""
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(
                f"GT file {path.name} must be a JSON object (dict), "
                f"got {type(payload).__name__}."
            )
        return payload

    @staticmethod
    def _write_raw(path: Path, payload: dict) -> None:
        """Atomically overwrite a GT file."""
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _summarise(path: Path, payload: dict) -> GroundTruthSummary:
        """Build a GroundTruthSummary from a loaded payload dict."""
        class_counts: dict[str, int] = {}
        frames = payload.get("frames", [])
        for frame in frames:
            for obj in frame.get("objects", []):
                cls = obj.get("class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

        frame_size = payload.get("frame_size", [0, 0])
        if not isinstance(frame_size, list) or len(frame_size) < 2:
            frame_size = [0, 0]

        return GroundTruthSummary(
            stem=path.stem,
            video_filename=payload.get("video", path.stem),
            fps=float(payload.get("fps", 0.0)),
            frame_size=frame_size,
            n_frames_annotated=len(frames),
            class_counts=class_counts,
            source=payload.get("source", "unknown"),
            gt_path=path,
        )
