"""
Batch Match Processor - Process multiple match videos automatically.

Scans an input directory for video files, processes each one through the
full TactiVision pipeline, stores results in the database, and generates
a batch summary report.

Usage:
    from services.batch_processor import BatchProcessor
    bp = BatchProcessor(config, db_manager)
    result = bp.process_batch("path/to/videos/")
"""

import gc
import json
import time
import re
import torch
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ultralytics import YOLO

from services.database_manager import DatabaseManager


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class MatchResult:
    """Result from processing a single match."""
    match_id: str
    filename: str
    video_path: str
    success: bool
    processing_time_s: float = 0.0
    duration_minutes: float = 0.0
    events_detected: int = 0
    possession_a: float = 0.0
    possession_b: float = 0.0
    shots: int = 0
    passes: int = 0
    sprints: int = 0
    formations_detected: int = 0
    xg_total: float = 0.0
    players_detected: int = 0
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Aggregate result from batch processing."""
    total_processed: int = 0
    total_failed: int = 0
    total_time_s: float = 0.0
    match_results: List[MatchResult] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)


class BatchProcessor:
    """Process multiple match videos in sequence with crash isolation."""

    def __init__(self, config: Optional[Dict] = None, db_manager: Optional[DatabaseManager] = None):
        self.config = config or {}
        self.db_manager = db_manager

        # Extract chunk settings from config for pass-through
        video_cfg = self.config.get("video", {})
        self.chunk_size_minutes = video_cfg.get("chunk_size_minutes", 5.0)
        self.memory_limit_percent = video_cfg.get("memory_limit_percent", 80.0)

    def scan_directory(self, input_dir: str) -> List[Path]:
        """Find all supported video files in directory, sorted by name."""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        videos = []
        for ext in SUPPORTED_EXTENSIONS:
            videos.extend(input_path.glob(f"*{ext}"))
        videos.sort(key=lambda p: p.name.lower())
        return videos

    def _make_match_id(self, filename: str) -> str:
        """Generate a unique match_id from timestamp and filename."""
        slug = re.sub(r"[^a-z0-9]+", "_", filename.lower().rsplit(".", 1)[0]).strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"match_{ts}_{slug}"

    def _find_roster(self, video_path: Path) -> Optional[Path]:
        """Try to find a matching roster JSON for a video."""
        roster_dir = video_path.parent / "rosters"
        if not roster_dir.exists():
            roster_dir = Path("rosters")
        if not roster_dir.exists():
            return None

        stem_lower = video_path.stem.lower()
        for roster_file in roster_dir.glob("*.json"):
            if roster_file.stem.lower() in stem_lower or stem_lower in roster_file.stem.lower():
                return roster_file

        rosters = list(roster_dir.glob("*.json"))
        if len(rosters) == 1:
            return rosters[0]
        return None

    def _clear_memory(self):
        """Force garbage collection and clear GPU cache between matches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_metrics(self, output_dir: Path) -> Dict[str, Any]:
        """Read metrics.json produced by the pipeline for a processed match."""
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def process_batch(
        self,
        input_dir: str,
        roster_path: Optional[str] = None,
        progress_callback=None,
    ) -> BatchResult:
        """
        Process all video files in input_dir through the full pipeline.

        Args:
            input_dir: Directory containing match videos.
            roster_path: Optional explicit roster JSON path (used for all matches).
            progress_callback: Optional callable(current, total, filename) for progress.

        Returns:
            BatchResult with per-match stats and errors.
        """
        videos = self.scan_directory(input_dir)
        if not videos:
            print(f"  No video files found in {input_dir}")
            return BatchResult()

        total = len(videos)
        print(f"\n{'=' * 60}")
        print(f"  BATCH PROCESSING: {total} match{'es' if total != 1 else ''}")
        print(f"{'=' * 60}")

        # Load YOLO model once for all matches
        print("  Loading YOLO model...")
        model = YOLO("yolov8n.pt")

        # Ensure database is ready
        if self.db_manager is not None:
            self.db_manager.initialize_database()

        batch_start = time.time()
        result = BatchResult()

        for idx, video_path in enumerate(videos, 1):
            match_id = self._make_match_id(video_path.name)
            print(f"\n  [{idx}/{total}] Processing: {video_path.name}")
            print(f"           Match ID: {match_id}")

            if progress_callback:
                progress_callback(idx, total, video_path.name)

            match_start = time.time()
            match_result = MatchResult(
                match_id=match_id,
                filename=video_path.name,
                video_path=str(video_path),
                success=False,
            )

            try:
                # Import pipeline here to keep it isolated per match
                from demo.run_demo import process_video as run_processing

                run_processing(
                    video_path,
                    model,
                    db_manager=self.db_manager,
                    chunk_size_minutes=self.chunk_size_minutes,
                    memory_limit_percent=self.memory_limit_percent,
                )

                match_result.success = True
                match_result.processing_time_s = round(time.time() - match_start, 1)

                # Read metrics from the output directory
                from demo.run_demo import OUTPUT_BASE
                out_dir = OUTPUT_BASE / video_path.stem
                metrics = self._extract_metrics(out_dir)

                if metrics:
                    match_result.duration_minutes = metrics.get("duration_minutes", 0)
                    match_result.players_detected = metrics.get("num_players", 0)

                    poss = metrics.get("possession", {}).get("team_possession_percentage", {})
                    match_result.possession_a = round(poss.get("A", 0), 1)
                    match_result.possession_b = round(poss.get("B", 0), 1)

                    match_result.passes = metrics.get("pass_detection", {}).get("total_passes", 0)
                    match_result.shots = metrics.get("shot_detection", {}).get("total_shots", 0)
                    match_result.sprints = metrics.get("sprint_detection", {}).get("total_sprints", 0)

                    xg = metrics.get("xg_analysis", {})
                    match_result.xg_total = round(xg.get("total_xg", 0), 2)

                    tac = metrics.get("tactical_analysis", {})
                    formations = tac.get("current_formations", {})
                    match_result.formations_detected = len(formations) if isinstance(formations, dict) else 0

                    # Count total events
                    match_result.events_detected = (
                        match_result.passes + match_result.shots + match_result.sprints
                    )

                print(f"  Completed in {match_result.processing_time_s}s | "
                      f"Events: {match_result.events_detected} | "
                      f"Possession: {match_result.possession_a}%-{match_result.possession_b}%")

            except Exception as e:
                match_result.processing_time_s = round(time.time() - match_start, 1)
                match_result.error = str(e)
                result.errors.append({"filename": video_path.name, "error": str(e)})
                print(f"  FAILED: {e}")

            result.match_results.append(match_result)
            if match_result.success:
                result.total_processed += 1
            else:
                result.total_failed += 1

            # Memory cleanup between matches
            self._clear_memory()
            print(f"  Memory cleared after match {idx}/{total}")

        result.total_time_s = round(time.time() - batch_start, 1)

        print(f"\n{'=' * 60}")
        print(f"  BATCH COMPLETE: {result.total_processed}/{total} succeeded, "
              f"{result.total_failed} failed in {result.total_time_s}s")
        print(f"{'=' * 60}")

        return result

    def generate_summary(self, result: BatchResult, output_path: str = "batch_summary.json") -> str:
        """
        Save batch_summary.json with per-match stats and aggregates.

        Returns:
            Path to the generated summary file.
        """
        per_match = []
        for mr in result.match_results:
            per_match.append({
                "match_id": mr.match_id,
                "filename": mr.filename,
                "video_path": mr.video_path,
                "success": mr.success,
                "duration_minutes": mr.duration_minutes,
                "processing_time_s": mr.processing_time_s,
                "events_detected": mr.events_detected,
                "possession_split": f"{mr.possession_a}%-{mr.possession_b}%",
                "shots": mr.shots,
                "passes": mr.passes,
                "sprints": mr.sprints,
                "formations_detected": mr.formations_detected,
                "xg_total": mr.xg_total,
                "players_detected": mr.players_detected,
                "error": mr.error,
            })

        successful = [m for m in result.match_results if m.success]
        avg_time = (
            round(sum(m.processing_time_s for m in successful) / len(successful), 1)
            if successful else 0
        )
        total_events = sum(m.events_detected for m in successful)

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_matches_processed": result.total_processed,
            "total_matches_failed": result.total_failed,
            "total_time_s": result.total_time_s,
            "aggregate": {
                "average_processing_time_s": avg_time,
                "total_events_across_all_matches": total_events,
                "total_shots": sum(m.shots for m in successful),
                "total_passes": sum(m.passes for m in successful),
                "total_sprints": sum(m.sprints for m in successful),
                "average_xg": round(
                    sum(m.xg_total for m in successful) / len(successful), 2
                ) if successful else 0,
            },
            "per_match": per_match,
            "errors": result.errors,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Summary saved to: {out}")
        return str(out)
