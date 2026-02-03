"""
Tracking Filters Module - Data Quality for Player & Ball Tracking

Fixes common tracking issues:
1. Speed sanity capping - rejects physically impossible speeds
2. Speed smoothing - moving average to reduce frame-to-frame noise
3. Track deduplication - merges fragmented track IDs into real players

WHY THIS IS NEEDED:
- YOLO tracker assigns new IDs when players get occluded and reappear
- This inflates player count (354 "players" vs 22 real ones)
- Frame-to-frame speed is noisy and ID jumps create phantom teleportation speeds
- Color ball detector picks up false positives (white objects that aren't balls)

REAL-WORLD SPEED LIMITS:
- Player max sprint: ~12 m/s (Usain Bolt peak ~12.4 m/s)
- Player max sustained: ~10 m/s
- Ball max velocity: ~45 m/s (powerful shot ~160 km/h)
- Ball max realistic in-play: ~35 m/s
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import deque
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================
MAX_PLAYER_SPEED_MPS = 12.0   # Fastest human sprint ever
MAX_BALL_SPEED_PPS = 5000.0   # Max ball speed in pixels/second (sanity cap)


# ============================================================
# SPEED SMOOTHING
# ============================================================
class SpeedSmoother:
    """
    Applies a moving average to per-player speed readings.
    
    Reduces noise from frame-to-frame jitter so sprint detection
    triggers on real sprints, not tracking artifacts.
    """

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Number of frames to average over (default 5)
        """
        self.window_size = window_size
        self._buffers: Dict[int, deque] = {}  # player_id -> deque of speeds

    def smooth(self, player_id: int, raw_speed_mps: float) -> float:
        """
        Add a raw speed reading and return the smoothed value.

        Args:
            player_id: Player track ID
            raw_speed_mps: Raw frame-to-frame speed in m/s

        Returns:
            Smoothed speed (moving average of last N readings)
        """
        if player_id not in self._buffers:
            self._buffers[player_id] = deque(maxlen=self.window_size)

        self._buffers[player_id].append(raw_speed_mps)
        buf = self._buffers[player_id]
        return sum(buf) / len(buf)

    def reset(self, player_id: int = None):
        """Reset smoothing buffer(s)."""
        if player_id is not None:
            self._buffers.pop(player_id, None)
        else:
            self._buffers.clear()


def cap_player_speed(raw_speed_mps: float, max_speed: float = MAX_PLAYER_SPEED_MPS) -> float:
    """
    Cap a player speed to physically possible values.

    If raw speed exceeds the max, return 0.0 (treat as tracking error).
    We return 0 instead of clamping because a 140 m/s reading means the
    track ID jumped - the player didn't actually move.

    Args:
        raw_speed_mps: Raw calculated speed
        max_speed: Maximum realistic speed (default 12 m/s)

    Returns:
        Speed if realistic, 0.0 if impossibly fast
    """
    if raw_speed_mps > max_speed:
        return 0.0
    return raw_speed_mps


def cap_ball_speed(raw_speed_pps: float, max_speed: float = MAX_BALL_SPEED_PPS) -> float:
    """
    Cap ball speed in pixels/second to reject false detections.

    Args:
        raw_speed_pps: Raw ball speed in pixels per second
        max_speed: Maximum realistic speed in px/s

    Returns:
        Speed if realistic, 0.0 if impossibly fast
    """
    if raw_speed_pps > max_speed:
        return 0.0
    return raw_speed_pps


# ============================================================
# TRACK DEDUPLICATION
# ============================================================
class TrackDeduplicator:
    """
    Merges fragmented YOLO track IDs into canonical player IDs.

    PROBLEM:
    YOLO's tracker assigns new IDs when a player is temporarily occluded
    (behind another player, off-screen, etc.). A single real player can
    accumulate 10+ different track IDs during a match.

    SOLUTION:
    When a new track ID appears, check if its position is very close to where
    a recently-lost track was last seen. If so, merge them under one canonical ID.

    This reduces "354 players" down to a realistic ~20-30.
    """

    def __init__(
        self,
        merge_distance_px: float = 80.0,
        max_lost_frames: int = 30,
    ):
        """
        Args:
            merge_distance_px: Max pixel distance to consider two tracks the same player
            max_lost_frames: How many frames a track can be missing before we stop trying to merge
        """
        self.merge_distance = merge_distance_px
        self.max_lost_frames = max_lost_frames

        # Canonical track state: {canonical_id: {last_pos, last_frame, raw_ids}}
        self._canonical_tracks: Dict[int, Dict] = {}

        # Mapping: raw_track_id -> canonical_id
        self._id_map: Dict[int, int] = {}

        # Track IDs seen this frame (to detect disappearances)
        self._prev_frame_ids: Set[int] = set()
        self._current_frame_ids: Set[int] = set()
        self._next_canonical_id: int = 1

    def resolve(
        self,
        raw_track_id: int,
        position: Tuple[float, float],
        frame_idx: int,
    ) -> int:
        """
        Resolve a raw YOLO track ID to a canonical (deduplicated) player ID.

        Args:
            raw_track_id: The track ID from YOLO
            position: (cx, cy) center position of the player
            frame_idx: Current frame index

        Returns:
            Canonical player ID (stable across occlusions)
        """
        self._current_frame_ids.add(raw_track_id)

        # Already mapped?
        if raw_track_id in self._id_map:
            canon_id = self._id_map[raw_track_id]
            self._canonical_tracks[canon_id]["last_pos"] = position
            self._canonical_tracks[canon_id]["last_frame"] = frame_idx
            return canon_id

        # New raw ID - check if it matches a recently lost canonical track
        best_match = None
        best_dist = float("inf")

        for canon_id, state in self._canonical_tracks.items():
            # Only consider tracks that are currently "lost" (not seen this frame via another raw ID)
            frames_missing = frame_idx - state["last_frame"]
            if frames_missing < 1 or frames_missing > self.max_lost_frames:
                continue

            # Check if any raw ID for this canonical track is still active
            still_active = any(
                rid in self._current_frame_ids and rid != raw_track_id
                for rid in state["raw_ids"]
            )
            if still_active:
                continue

            # Distance check
            dx = position[0] - state["last_pos"][0]
            dy = position[1] - state["last_pos"][1]
            dist = np.hypot(dx, dy)

            if dist < self.merge_distance and dist < best_dist:
                best_dist = dist
                best_match = canon_id

        if best_match is not None:
            # Merge: this raw ID is the same player
            self._id_map[raw_track_id] = best_match
            self._canonical_tracks[best_match]["raw_ids"].add(raw_track_id)
            self._canonical_tracks[best_match]["last_pos"] = position
            self._canonical_tracks[best_match]["last_frame"] = frame_idx
            return best_match

        # Truly new player
        canon_id = self._next_canonical_id
        self._next_canonical_id += 1
        self._id_map[raw_track_id] = canon_id
        self._canonical_tracks[canon_id] = {
            "last_pos": position,
            "last_frame": frame_idx,
            "raw_ids": {raw_track_id},
        }
        return canon_id

    def end_frame(self, frame_idx: int):
        """Call at the end of each frame to update lost-track bookkeeping."""
        self._prev_frame_ids = self._current_frame_ids.copy()
        self._current_frame_ids = set()

        # Prune very old canonical tracks to save memory
        stale = [
            cid for cid, state in self._canonical_tracks.items()
            if frame_idx - state["last_frame"] > self.max_lost_frames * 3
        ]
        for cid in stale:
            del self._canonical_tracks[cid]

    def get_canonical_count(self) -> int:
        """Return number of unique canonical players detected so far."""
        return len(self._canonical_tracks)

    def get_id_map(self) -> Dict[int, int]:
        """Return the raw -> canonical ID mapping."""
        return dict(self._id_map)
