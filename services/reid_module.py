"""
Jersey + Position Re-Identification (M3).

Why this exists
---------------
The doctor's third concern was tracking accuracy. M2 (BoT-SORT) closes the
gap on short-window association; this module closes it on *long* occlusions
where motion alone fails. We refine the existing position-only
`TrackDeduplicator` with a deterministic jersey-color check.

Re-ID rule (per §7.1 M3 of the 21-day plan):
    When a new raw track ID appears, check
      (a) jersey class matches a recently-lost canonical ID from the same team
      (b) Euclidean distance < merge_distance_px to that lost ID's last position
    If BOTH match, the new raw ID is the same player → reuse the canonical ID.

Pipeline position:
    raw YOLO track id ──► JerseyPosReID.resolve(...) ──► canonical_id

Sits *before* `services.tracking_filters.TrackDeduplicator` so the cheap
position-only dedup remains a safety net for anything Re-ID could not match
(e.g. when no crop is available).

Public surface
--------------
- JerseyColorClassifier  HSV-histogram team classifier
- PositionMemory         per-canonical-id last-K positions, team, timestamp
- JerseyPosReID          the merge orchestrator (the thing run_demo.py calls)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# -- Class vocabulary -------------------------------------------------------

JerseyClass = str  # one of: team_a, team_b, referee, goalkeeper, unknown

VALID_CLASSES: tuple[JerseyClass, ...] = (
    "team_a",
    "team_b",
    "referee",
    "goalkeeper",
    "unknown",
)


# ---------------------------------------------------------------------------
# JerseyColorClassifier
# ---------------------------------------------------------------------------


@dataclass
class JerseyColorClassifier:
    """HSV-histogram jersey classifier.

    Each registered class is defined by a representative HSV center (H, S, V).
    `classify(crop_bgr)` finds the dominant hue band of the crop and returns
    the class with the closest center.

    Defaults are sensible for a typical match (red team_a, blue team_b, yellow
    referee, green goalkeeper). Override `team_a_hsv` / `team_b_hsv` per match
    via the constructor — every match should re-fit jersey colors from
    config.yaml team_colors before running.
    """

    team_a_hsv: tuple[int, int, int] = (0, 200, 200)        # red
    team_b_hsv: tuple[int, int, int] = (110, 200, 200)      # blue
    ref_hsv: tuple[int, int, int] = (30, 200, 200)          # yellow
    goalkeeper_hsv: tuple[int, int, int] = (60, 200, 200)   # green

    #: Crop is downscaled to this size before classification (cheap + robust
    #: against partial occlusions / motion blur on small detections).
    target_size: int = 32

    #: Saturation/value cutoffs to ignore near-black/white pixels (pitch lines,
    #: shadows). Helps the dominant-hue computation stay on the jersey itself.
    min_saturation: int = 60
    min_value: int = 60

    def _centers(self) -> dict[JerseyClass, tuple[int, int, int]]:
        return {
            "team_a": self.team_a_hsv,
            "team_b": self.team_b_hsv,
            "referee": self.ref_hsv,
            "goalkeeper": self.goalkeeper_hsv,
        }

    def classify(self, crop_bgr: np.ndarray) -> JerseyClass:
        """Return the closest class. Returns 'unknown' on empty/degenerate
        crops or when no pixel survives the saturation/value cutoff.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown"
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            return "unknown"

        # Lazy cv2 import keeps the module cheap for unit tests that only
        # exercise the dataclass surface.
        import cv2

        h, w = crop_bgr.shape[:2]
        if h < 2 or w < 2:
            return "unknown"

        small = cv2.resize(
            crop_bgr,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        mask = (S >= self.min_saturation) & (V >= self.min_value)
        if not mask.any():
            return "unknown"

        # Dominant hue across surviving pixels.
        hist = np.bincount(H[mask].ravel(), minlength=180)
        dominant_h = int(np.argmax(hist))
        # Use median saturation/value of the masked region as the rep V/S so
        # we compare apples to apples with the registered HSV centers.
        rep_s = int(np.median(S[mask]))
        rep_v = int(np.median(V[mask]))

        best_class: JerseyClass = "unknown"
        best_dist = float("inf")
        for cls, center in self._centers().items():
            d = self._hsv_distance((dominant_h, rep_s, rep_v), center)
            if d < best_dist:
                best_dist = d
                best_class = cls
        return best_class

    @staticmethod
    def _hsv_distance(
        a: tuple[int, int, int], b: tuple[int, int, int]
    ) -> float:
        """Cyclic distance on Hue (0..179, wraps), Euclidean on S/V.

        Hue is given more weight than S/V because S/V vary with lighting,
        while hue is mostly a function of the actual jersey color.
        """
        ha, sa, va = a
        hb, sb, vb = b
        dh = min(abs(ha - hb), 180 - abs(ha - hb))
        ds = abs(sa - sb)
        dv = abs(va - vb)
        return math.sqrt((dh * 3) ** 2 + ds**2 + dv**2)


# ---------------------------------------------------------------------------
# PositionMemory
# ---------------------------------------------------------------------------


@dataclass
class _CanonicalState:
    last_pos: tuple[float, float]
    last_frame: int
    team: JerseyClass
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))


class PositionMemory:
    """Records last-K positions per canonical ID + their team.

    `lost_candidates(current_frame, max_lost_frames)` returns canonical IDs
    that disappeared within the lookback window — those are the ids the
    Re-ID layer is allowed to match a new raw track against.
    """

    def __init__(self, history_k: int = 8) -> None:
        self._states: dict[int, _CanonicalState] = {}
        self._history_k = history_k

    def record(
        self,
        canonical_id: int,
        position: tuple[float, float],
        team: JerseyClass,
        frame_idx: int,
    ) -> None:
        state = self._states.get(canonical_id)
        if state is None:
            state = _CanonicalState(
                last_pos=position,
                last_frame=frame_idx,
                team=team,
                history=deque(maxlen=self._history_k),
            )
            self._states[canonical_id] = state
        else:
            state.last_pos = position
            state.last_frame = frame_idx
            # Only update team if the new classification is more confident
            # (i.e. not 'unknown'); otherwise keep the existing label.
            if team != "unknown":
                state.team = team
        state.history.append(position)

    def lost_candidates(
        self, current_frame: int, max_lost_frames: int = 30
    ) -> list[tuple[int, tuple[float, float], JerseyClass]]:
        """Canonical IDs whose last_frame is in (current_frame - max_lost_frames,
        current_frame). Returns (canon_id, last_pos, team) tuples.
        """
        out: list[tuple[int, tuple[float, float], JerseyClass]] = []
        for canon_id, state in self._states.items():
            gap = current_frame - state.last_frame
            if 1 <= gap <= max_lost_frames:
                out.append((canon_id, state.last_pos, state.team))
        return out

    def get(self, canonical_id: int) -> Optional[_CanonicalState]:
        return self._states.get(canonical_id)

    def clear(self) -> None:
        self._states.clear()

    def __len__(self) -> int:  # pragma: no cover
        return len(self._states)


# ---------------------------------------------------------------------------
# JerseyPosReID
# ---------------------------------------------------------------------------


class JerseyPosReID:
    """Re-Identification layer — assigns a stable canonical ID to each raw
    track ID using jersey color + last-known position.

    Conservative by design: a new raw ID only merges into a lost canonical ID
    when **both** the jersey class matches AND the distance < merge_distance_px.
    Either-only would risk false merges (the §9 risk register top item).
    """

    def __init__(
        self,
        merge_distance_px: float = 120.0,
        max_lost_frames: int = 30,
        debug_log_path: "Optional[Path]" = None,
    ) -> None:
        self.merge_distance_px = merge_distance_px
        self.max_lost_frames = max_lost_frames

        self._memory = PositionMemory()
        self._raw_to_canonical: dict[int, int] = {}
        self._next_canonical_id = 1

        # Day 5: per-raw-id classification cache. The classifier is the most
        # expensive step in the layer; classifying once per raw id (instead of
        # every frame) is a ~10x speedup and equivalent for stable raw ids.
        self._raw_id_team: dict[int, JerseyClass] = {}

        # Day 5: debug log of every committed merge — for threshold tuning
        # and for inclusion in the thesis 5.3 reproducibility appendix.
        self._merge_log: list[dict] = []
        self._debug_log_path = debug_log_path

        # Diagnostic counters.
        self.merges_attempted = 0
        self.merges_committed = 0
        self.new_canonical_assigned = 0

    # ---- public API -------------------------------------------------------

    def resolve(
        self,
        raw_track_id: int,
        position: tuple[float, float],
        crop_bgr: np.ndarray | None,
        classifier: JerseyColorClassifier,
        frame_idx: int,
    ) -> int:
        """Map a raw YOLO track id to a canonical player id.

        Falls back to a fresh canonical id when no lost candidate matches
        on both jersey class and distance.
        """
        # Already mapped → refresh memory with cached team and return. We
        # NEVER reclassify a known raw id: the classifier is the slow path,
        # and a stable raw id is, by definition, the same player wearing the
        # same jersey. (Day 5 perf win.)
        if raw_track_id in self._raw_to_canonical:
            canon = self._raw_to_canonical[raw_track_id]
            cached_team = self._raw_id_team.get(raw_track_id, "unknown")
            self._memory.record(canon, position, cached_team, frame_idx)
            return canon

        # New raw id — classify the crop ONCE and cache the result.
        team = classifier.classify(crop_bgr) if crop_bgr is not None else "unknown"
        self._raw_id_team[raw_track_id] = team

        candidates = self._memory.lost_candidates(
            current_frame=frame_idx, max_lost_frames=self.max_lost_frames
        )
        self.merges_attempted += 1

        best_canon: int | None = None
        best_dist = float("inf")
        for canon_id, last_pos, last_team in candidates:
            # Rule (a): jersey class must match. 'unknown' on either side
            # disqualifies — we only merge when we're confident.
            if team == "unknown" or last_team == "unknown":
                continue
            if team != last_team:
                continue
            # Rule (b): position must be within merge_distance_px.
            d = math.hypot(position[0] - last_pos[0], position[1] - last_pos[1])
            if d < self.merge_distance_px and d < best_dist:
                best_dist = d
                best_canon = canon_id

        if best_canon is not None:
            self._raw_to_canonical[raw_track_id] = best_canon
            self._memory.record(best_canon, position, team, frame_idx)
            self.merges_committed += 1
            self._merge_log.append({
                "frame_idx": frame_idx,
                "raw_track_id": int(raw_track_id),
                "canonical_id": int(best_canon),
                "team": team,
                "distance_px": round(float(best_dist), 2),
                "position": [round(float(position[0]), 1), round(float(position[1]), 1)],
            })
            return best_canon

        # No match → fresh canonical id.
        canon = self._next_canonical_id
        self._next_canonical_id += 1
        self._raw_to_canonical[raw_track_id] = canon
        self._memory.record(canon, position, team, frame_idx)
        self.new_canonical_assigned += 1
        return canon

    # ---- debug log persistence -------------------------------------------

    def write_merge_log(self, path: "Optional[Path]" = None) -> "Path":
        """Persist the merge log to JSON.

        Defaults to the path passed at construction. Returns the actual path
        written so callers can include it in run summaries.
        """
        import json
        from pathlib import Path as _P

        target = _P(path) if path is not None else self._debug_log_path
        if target is None:
            raise ValueError(
                "No debug_log_path supplied. Pass `path=` or set on the constructor."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "stats": self.stats(),
                    "thresholds": {
                        "merge_distance_px": self.merge_distance_px,
                        "max_lost_frames": self.max_lost_frames,
                    },
                    "merges": self._merge_log,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    # ---- diagnostics ------------------------------------------------------

    @property
    def canonical_count(self) -> int:
        """Distinct canonical ids assigned so far."""
        return self._next_canonical_id - 1

    def stats(self) -> dict[str, int]:
        return {
            "raw_ids_seen": len(self._raw_to_canonical),
            "canonical_count": self.canonical_count,
            "merges_attempted": self.merges_attempted,
            "merges_committed": self.merges_committed,
            "new_canonical_assigned": self.new_canonical_assigned,
        }


# ---------------------------------------------------------------------------
# Convenience: build a classifier from config.yaml team_colors
# ---------------------------------------------------------------------------


def hex_to_hsv(hex_color: str) -> tuple[int, int, int]:
    """'#ef0107' → (H, S, V) in OpenCV convention (H ∈ [0, 179])."""
    s = hex_color.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Bad hex color: {hex_color!r}")
    r, g, b = (int(s[i : i + 2], 16) for i in (0, 2, 4))
    import cv2

    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def classifier_from_team_colors(
    team_a_hex: Optional[str],
    team_b_hex: Optional[str],
    referee_hex: Optional[str] = None,
    goalkeeper_hex: Optional[str] = None,
) -> JerseyColorClassifier:
    """Build a JerseyColorClassifier with HSV centers fitted to actual match
    jersey colors (read from config.yaml team_colors per match).

    None values keep the dataclass defaults — useful when only the two team
    colors are known and the referee/goalkeeper kits aren't recorded.
    """
    kwargs: dict = {}
    if team_a_hex:
        kwargs["team_a_hsv"] = hex_to_hsv(team_a_hex)
    if team_b_hex:
        kwargs["team_b_hsv"] = hex_to_hsv(team_b_hex)
    if referee_hex:
        kwargs["ref_hsv"] = hex_to_hsv(referee_hex)
    if goalkeeper_hex:
        kwargs["goalkeeper_hsv"] = hex_to_hsv(goalkeeper_hex)
    return JerseyColorClassifier(**kwargs)
