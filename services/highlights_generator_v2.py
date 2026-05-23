"""
Highlights Generator v2 - Rewritten with weighted importance scoring.

Automatically detects and generates key moments from match metrics.
Scores events by weighted importance and creates highlight clips.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

logger = logging.getLogger(__name__)


@dataclass
class Highlight:
    """Represents a highlight moment."""
    time: float
    type: str
    importance: float
    clip_path: Optional[Path]
    description: str


class HighlightsGenerator:
    """Generates highlights from match metrics with weighted importance scoring."""

    # Event importance weights
    IMPORTANCE_WEIGHTS = {
        'goal': 1.0,
        'big_chance': 0.8,
        'save': 0.7,
        'shot_on_target': 0.6,
        'sprint_high_intensity': 0.5,
        'dribble_success': 0.4,
        'pass_progressive': 0.3,
    }

    # Clip timing
    PRE_EVENT_BUFFER = 4.0  # seconds before event
    POST_EVENT_BUFFER = 4.0  # seconds after event

    def __init__(self, output_dir: Path):
        """
        Initialize highlights generator.

        Args:
            output_dir: Directory to save highlight clips
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.output_dir / 'clips'
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Check if ffmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> bool:
        """Check if ffmpeg is available on system."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("ffmpeg not found. Highlight clips will not be generated.")
            return False

    def generate(self, metrics: Dict, video_path: Path, top_n: int = 12) -> List[Highlight]:
        """
        Generate highlights from metrics and video.

        Args:
            metrics: Match metrics dict from metrics.json
            video_path: Path to video file
            top_n: Number of top highlights to extract (default 12)

        Returns:
            List of Highlight objects
        """
        video_path = Path(video_path)

        # Extract and score events
        events = self._extract_events(metrics)

        if not events:
            logger.info("No events found in metrics")
            return []

        # Score events by importance
        scored_events = self._score_events(events)

        # Select top N non-overlapping events
        selected_events = self._select_non_overlapping(scored_events, top_n)

        # Create highlights (with clips if ffmpeg available)
        highlights = []
        for idx, event in enumerate(selected_events):
            clip_path = None

            if self.ffmpeg_available and video_path.exists():
                clip_path = self._cut_clip(video_path, event, idx)

            highlight = Highlight(
                time=event['time'],
                type=event['type'],
                importance=event['importance'],
                clip_path=clip_path,
                description=event['description']
            )
            highlights.append(highlight)

        return highlights

    def _extract_events(self, metrics: Dict) -> List[Dict[str, Any]]:
        """Extract all scoreable events from metrics."""
        events = []

        # Goals (implicit in shots marked as goals)
        for shot in metrics.get('shot_events', []):
            if shot.get('is_goal'):
                events.append({
                    'time': shot.get('timestamp', 0),
                    'type': 'goal',
                    'importance': self.IMPORTANCE_WEIGHTS['goal'],
                    'shooter': shot.get('shooter_id'),
                    'description': f"Goal by Player {shot.get('shooter_id')}"
                })

        # Big chances (high xG shots, assuming velocity/angle proxy)
        for shot in metrics.get('shot_events', []):
            if not shot.get('is_goal'):
                # Consider high velocity shots as big chances
                velocity = shot.get('velocity_mps', 0)
                angle = abs(shot.get('angle_to_goal_deg', 45))

                if velocity > 15 and angle < 30:
                    events.append({
                        'time': shot.get('timestamp', 0),
                        'type': 'big_chance',
                        'importance': self.IMPORTANCE_WEIGHTS['big_chance'],
                        'shooter': shot.get('shooter_id'),
                        'description': f"Big chance by Player {shot.get('shooter_id')} ({velocity:.1f} m/s)"
                    })
                else:
                    # Regular shot on target
                    events.append({
                        'time': shot.get('timestamp', 0),
                        'type': 'shot_on_target',
                        'importance': self.IMPORTANCE_WEIGHTS['shot_on_target'],
                        'shooter': shot.get('shooter_id'),
                        'description': f"Shot on target by Player {shot.get('shooter_id')}"
                    })

        # High intensity sprints (>= 8 m/s)
        for sprint in metrics.get('sprint_events', []):
            if sprint.get('max_speed_mps', 0) >= 8.0:
                events.append({
                    'time': sprint.get('start_time', 0),
                    'type': 'sprint_high_intensity',
                    'importance': self.IMPORTANCE_WEIGHTS['sprint_high_intensity'],
                    'player': sprint.get('player_id'),
                    'description': f"High intensity sprint: {sprint.get('max_speed_mps', 0):.1f} m/s"
                })

        # Progressive passes (> 30m distance)
        for pass_event in metrics.get('pass_events', []):
            if pass_event.get('distance_m', 0) > 30 and pass_event.get('outcome') == 'complete':
                events.append({
                    'time': pass_event.get('timestamp', 0),
                    'type': 'pass_progressive',
                    'importance': self.IMPORTANCE_WEIGHTS['pass_progressive'],
                    'passer': pass_event.get('passer_id'),
                    'description': f"Progressive pass: {pass_event.get('distance_m', 0):.1f}m"
                })

        return events

    def _score_events(self, events: List[Dict]) -> List[Dict]:
        """Score and sort events by importance."""
        # Events already have importance from _extract_events
        # Sort by importance (descending) then by time
        scored = sorted(events, key=lambda x: (-x['importance'], x['time']))
        return scored

    def _select_non_overlapping(self, events: List[Dict],
                               top_n: int) -> List[Dict]:
        """Select top N non-overlapping events."""
        selected = []
        min_gap = self.PRE_EVENT_BUFFER + self.POST_EVENT_BUFFER

        for event in events:
            if len(selected) >= top_n:
                break

            # Check if event overlaps with already selected
            event_time = event['time']
            overlap = False

            for selected_event in selected:
                selected_time = selected_event['time']
                if abs(event_time - selected_time) < min_gap:
                    overlap = True
                    break

            if not overlap:
                selected.append(event)

        # Sort selected by time
        selected.sort(key=lambda x: x['time'])
        return selected

    def _cut_clip(self, video_path: Path, event: Dict, clip_idx: int) -> Optional[Path]:
        """
        Cut a clip from video using ffmpeg.

        Args:
            video_path: Path to source video
            event: Event dict with timing
            clip_idx: Index for clip filename

        Returns:
            Path to generated clip or None if failed
        """
        if not self.ffmpeg_available:
            return None

        try:
            # Calculate clip boundaries
            start_time = max(0, event['time'] - self.PRE_EVENT_BUFFER)
            end_time = event['time'] + self.POST_EVENT_BUFFER
            duration = end_time - start_time

            # Generate clip filename
            event_type = event['type']
            clip_path = self.clips_dir / f"{clip_idx:02d}_{event_type}.mp4"

            # Check if clip already exists (idempotent)
            if clip_path.exists():
                logger.debug(f"Clip {clip_path} already exists, skipping")
                return clip_path

            # Run ffmpeg to extract clip
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', f'{start_time:.2f}',
                '-t', f'{duration:.2f}',
                '-c:v', 'libx264',
                '-crf', '23',
                '-c:a', 'aac',
                '-y',
                str(clip_path)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0 and clip_path.exists():
                logger.info(f"Generated clip: {clip_path}")
                return clip_path
            else:
                logger.error(f"ffmpeg failed to generate clip: {result.stderr.decode()}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg timeout while generating clip {clip_idx}")
            return None
        except Exception as e:
            logger.error(f"Error generating clip {clip_idx}: {e}")
            return None
