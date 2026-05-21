"""
Improved Tracker (M2) — pluggable detector tracker behind a config switch.

Why this exists
---------------
The doctor's third concern was tracking accuracy. Today the project pins
ByteTrack implicitly (Ultralytics default). This module introduces a thin
abstraction so we can flip between ByteTrack and BoT-SORT from config.yaml
and measure the difference (mAP-style numbers come from M3 + Seif's
accuracy_evaluator).

Both implementations are thin wrappers around `ultralytics.YOLO.track(...)`
with the right tracker yaml passed through. We do NOT reimplement tracking
ourselves — Ultralytics ships both `bytetrack.yaml` and `botsort.yaml` inside
its package and resolves them by string name.

Usage
-----
    from ultralytics import YOLO
    from services.improved_tracker import build_tracker

    model = YOLO("yolov8n.pt")
    tracker = build_tracker("botsort", model)
    res_list = tracker.track(source=frame, classes=[0, 32], conf=0.3, imgsz=640)

The shape of `track(...)`'s return value is exactly what Ultralytics yields
(`list[Results]`), so existing call-sites consume it unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


SUPPORTED_ALGORITHMS: tuple[str, ...] = ("bytetrack", "botsort")


class Tracker(ABC):
    """Abstract base. Subclasses select which Ultralytics tracker yaml ships."""

    #: Algorithm identifier used in config.yaml (`detection.tracking.algorithm`).
    name: str = ""
    #: Ultralytics tracker yaml resolved by name from inside the package.
    tracker_yaml: str = ""

    def __init__(self, model: Any) -> None:
        if model is None:
            raise ValueError("Tracker requires a loaded YOLO/RTDETR model.")
        self.model = model

    @abstractmethod
    def track(
        self,
        source: Any,
        classes: list[int] | None = None,
        conf: float = 0.3,
        persist: bool = True,
        device: str = "cpu",
        verbose: bool = False,
        imgsz: int = 640,
        **extra: Any,
    ) -> list[Any]:
        """Run tracking on a frame (or iterable of frames).

        Returns the same shape Ultralytics yields: `list[Results]`.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(yaml={self.tracker_yaml!r})"


class _UltralyticsTracker(Tracker):
    """Shared implementation: pass-through to `model.track(...)` with the
    appropriate tracker yaml. Concrete subclasses set `name` + `tracker_yaml`.
    """

    def track(
        self,
        source: Any,
        classes: list[int] | None = None,
        conf: float = 0.3,
        persist: bool = True,
        device: str = "cpu",
        verbose: bool = False,
        imgsz: int = 640,
        **extra: Any,
    ) -> list[Any]:
        return self.model.track(
            source=source,
            persist=persist,
            tracker=self.tracker_yaml,
            classes=classes if classes is not None else [0, 32],
            conf=conf,
            device=device,
            verbose=verbose,
            imgsz=imgsz,
            **extra,
        )


class ByteTrackTracker(_UltralyticsTracker):
    """ByteTrack via Ultralytics (`bytetrack.yaml`).

    This is the current production tracker; keeping the wrapper means our
    A/B comparison is clean: identical surface, only the yaml changes.
    """

    name = "bytetrack"
    tracker_yaml = "bytetrack.yaml"


class BotSortTracker(_UltralyticsTracker):
    """BoT-SORT via Ultralytics (`botsort.yaml`).

    Adds appearance-based Re-ID on top of motion (vs ByteTrack's motion-only).
    Expectation: fewer ID switches on long occlusions; we measure with M3 +
    services.accuracy_evaluator.
    """

    name = "botsort"
    tracker_yaml = "botsort.yaml"


_REGISTRY: dict[str, type[Tracker]] = {
    ByteTrackTracker.name: ByteTrackTracker,
    BotSortTracker.name: BotSortTracker,
}


def build_tracker(algorithm: str, model: Any) -> Tracker:
    """Factory. Returns a Tracker instance for `algorithm`.

    Raises
    ------
    ValueError
        If `algorithm` is not one of `SUPPORTED_ALGORITHMS`. The message
        lists what we do support so config typos fail fast.
    """
    if not isinstance(algorithm, str):
        raise ValueError(
            f"algorithm must be a string, got {type(algorithm).__name__}"
        )
    key = algorithm.strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(SUPPORTED_ALGORITHMS)
        raise ValueError(
            f"Unknown tracker algorithm {algorithm!r}. Supported: {supported}"
        )
    return cls(model)
