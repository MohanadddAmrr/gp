"""
Multi-Model Evaluator (M1)

Benchmarks multiple object detection models on the same video frames so we can
defend the model choice with numbers (FPS, detections/frame, mean confidence,
file size, GPU peak memory, mAP@50 when ground truth is available).

Day 1 status: skeleton. Public surface is fixed; implementation lands on Day 2.

Models in scope:
    yolov8n, yolov8s, yolov8m, yolo11n  -> ultralytics.YOLO
    rtdetr-l                             -> ultralytics.RTDETR

CLI (Day 2):
    python -m services.multi_model_evaluator \
        --videos demo/input_videos/<clip>.mp4 \
        --models yolov8n yolov8s yolov8m yolo11n rtdetr-l \
        --frames 300 \
        --output benchmarks/multi_model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


SUPPORTED_MODELS: dict[str, "ModelSpec"] = {}

Framework = Literal["ultralytics_yolo", "ultralytics_rtdetr"]


@dataclass(frozen=True)
class ModelSpec:
    """Identity of a detection model under evaluation.

    weights_filename is the file ultralytics will look for / auto-download
    (e.g. 'yolov8n.pt'). framework selects the wrapper class.
    """

    name: str
    weights_filename: str
    framework: Framework
    input_size: int = 640


def _register(spec: ModelSpec) -> ModelSpec:
    SUPPORTED_MODELS[spec.name] = spec
    return spec


_register(ModelSpec("yolov8n", "yolov8n.pt", "ultralytics_yolo"))
_register(ModelSpec("yolov8s", "yolov8s.pt", "ultralytics_yolo"))
_register(ModelSpec("yolov8m", "yolov8m.pt", "ultralytics_yolo"))
_register(ModelSpec("yolo11n", "yolo11n.pt", "ultralytics_yolo"))
_register(ModelSpec("rtdetr-l", "rtdetr-l.pt", "ultralytics_rtdetr"))


def resolve_model(name: str) -> ModelSpec:
    """Look up a ModelSpec by name. Raises ValueError on unknown names."""
    try:
        return SUPPORTED_MODELS[name]
    except KeyError:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(
            f"Unknown model {name!r}. Supported: {supported}"
        ) from None


@dataclass
class EvaluationResult:
    """One row in the benchmark CSV (per (model, video) pair)."""

    model: str
    video: str
    frames_evaluated: int = 0
    detections_per_frame_person: float = 0.0
    detections_per_frame_ball: float = 0.0
    mean_confidence: float = 0.0
    fps: float = 0.0
    weights_size_mb: float = 0.0
    gpu_peak_mb: float = 0.0
    map50: float | None = None  # populated when ground truth available


def evaluate_models(
    videos: Iterable[Path],
    models: Iterable[str],
    frames: int = 300,
    output_dir: Path = Path("benchmarks/multi_model"),
    target_classes: tuple[int, ...] = (0, 32),
    imgsz: int = 640,
) -> Path:
    """Run each model over evenly-sampled frames of each video.

    Day 1: skeleton. Day 2 lands the real implementation, which writes:
        <output_dir>/<UTC ts>/results.csv
        <output_dir>/<UTC ts>/results.json
        <output_dir>/<UTC ts>/results.png   (FPS + mean detections bar chart)

    Returns the timestamped output directory.
    """
    raise NotImplementedError(
        "evaluate_models lands on Day 2 (M1 implementation). "
        "Day 1 only ships the public surface so downstream code can import."
    )


def _cli() -> None:  # pragma: no cover - lands on Day 2
    raise NotImplementedError("CLI lands on Day 2.")


if __name__ == "__main__":  # pragma: no cover
    _cli()
