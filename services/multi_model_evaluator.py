"""
Multi-Model Evaluator (M1)

Benchmarks multiple object detection models on the same video frames so we can
defend the model choice with numbers (FPS, detections/frame, mean confidence,
file size, GPU peak memory, mAP@50 when ground truth is available).

Models in scope:
    yolov8n, yolov8s, yolov8m, yolo11n  -> ultralytics.YOLO
    rtdetr-l                             -> ultralytics.RTDETR

CLI:
    python -m services.multi_model_evaluator \
        --videos demo/input_videos/<clip>.mp4 \
        --models yolov8n yolov8s yolov8m yolo11n rtdetr-l \
        --frames 300 \
        --output benchmarks/multi_model
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

logger = logging.getLogger(__name__)


# --- Model registry --------------------------------------------------------

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


# --- Internals -------------------------------------------------------------

WEIGHTS_DIR = Path("weights")


def _resolve_weights_path(spec: ModelSpec) -> Path:
    """Prefer weights/<file> if present, else fall back to bare filename so
    ultralytics auto-downloads it into the CWD. We never invent paths.
    """
    candidate = WEIGHTS_DIR / spec.weights_filename
    if candidate.exists():
        return candidate
    return Path(spec.weights_filename)


def _load_model(spec: ModelSpec):
    """Lazy-import ultralytics so the module is cheap to import in tests."""
    weights = _resolve_weights_path(spec)
    if spec.framework == "ultralytics_yolo":
        from ultralytics import YOLO

        return YOLO(str(weights))
    if spec.framework == "ultralytics_rtdetr":
        from ultralytics import RTDETR

        return RTDETR(str(weights))
    raise ValueError(f"Unsupported framework: {spec.framework}")


def _evenly_sampled_indices(total_frames: int, n: int) -> list[int]:
    """Pick n indices evenly across [0, total_frames). Falls back to all
    available frames when total_frames < n.
    """
    if total_frames <= 0:
        return []
    if n >= total_frames:
        return list(range(total_frames))
    # Evenly spaced; integer rounding is fine for benchmarking.
    step = total_frames / n
    return [min(total_frames - 1, int(i * step)) for i in range(n)]


def _gpu_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # pragma: no cover - never block a CPU run
        pass
    return 0.0


def _reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass


def _benchmark_one(
    spec: ModelSpec,
    video_path: Path,
    frames: int,
    target_classes: tuple[int, ...],
    imgsz: int,
) -> EvaluationResult:
    """Benchmark a single (model, video) pair."""
    import cv2

    result = EvaluationResult(model=spec.name, video=video_path.name)

    weights_path = _resolve_weights_path(spec)
    if weights_path.exists():
        result.weights_size_mb = weights_path.stat().st_size / (1024 * 1024)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    indices = _evenly_sampled_indices(total, frames)
    if not indices:
        cap.release()
        return result

    model = _load_model(spec)
    _reset_gpu_peak()

    person_counts: list[int] = []
    ball_counts: list[int] = []
    confidences: list[float] = []

    person_class, ball_class = (target_classes + (0, 32))[:2]

    t0 = time.perf_counter()
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Match run_demo.py: classes [0,32], conf 0.3, no augment.
        res_list = model.predict(
            source=frame,
            classes=list(target_classes),
            conf=0.3,
            imgsz=imgsz,
            verbose=False,
        )
        if not res_list:
            person_counts.append(0)
            ball_counts.append(0)
            continue
        res = res_list[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.cls is None:
            person_counts.append(0)
            ball_counts.append(0)
            continue

        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None

        person_counts.append(int((cls == person_class).sum()))
        ball_counts.append(int((cls == ball_class).sum()))
        if conf is not None and conf.size:
            confidences.extend(conf.tolist())

    elapsed = time.perf_counter() - t0
    cap.release()

    n_eval = len(person_counts)
    result.frames_evaluated = n_eval
    if n_eval:
        result.detections_per_frame_person = sum(person_counts) / n_eval
        result.detections_per_frame_ball = sum(ball_counts) / n_eval
        result.fps = n_eval / elapsed if elapsed > 0 else 0.0
    if confidences:
        result.mean_confidence = sum(confidences) / len(confidences)
    result.gpu_peak_mb = _gpu_peak_mb()

    return result


# --- Output writers --------------------------------------------------------

CSV_COLUMNS = [
    "model",
    "video",
    "frames_evaluated",
    "detections_per_frame_person",
    "detections_per_frame_ball",
    "mean_confidence",
    "fps",
    "weights_size_mb",
    "gpu_peak_mb",
    "map50",
]


def _write_csv(results: list[EvaluationResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in CSV_COLUMNS})


def _write_json(results: list[EvaluationResult], path: Path) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_chart(results: list[EvaluationResult], path: Path) -> None:
    """Bar chart: FPS (left axis) + mean detections per frame (right axis).
    Aggregates across videos by model.
    """
    if not results:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[EvaluationResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    names = list(by_model)
    fps = [
        sum(r.fps for r in rs) / len(rs) if rs else 0.0 for rs in by_model.values()
    ]
    dets = [
        sum(
            (r.detections_per_frame_person + r.detections_per_frame_ball) for r in rs
        )
        / len(rs)
        if rs
        else 0.0
        for rs in by_model.values()
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    width = 0.4
    ax1.bar([i - width / 2 for i in x], fps, width=width, label="FPS", color="#1f77b4")
    ax1.set_ylabel("FPS")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20)

    ax2 = ax1.twinx()
    ax2.bar(
        [i + width / 2 for i in x],
        dets,
        width=width,
        label="Mean dets/frame",
        color="#ff7f0e",
    )
    ax2.set_ylabel("Mean detections / frame")

    fig.suptitle("Multi-model benchmark — FPS vs mean detections per frame")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Public API ------------------------------------------------------------


def evaluate_models(
    videos: Iterable[Path],
    models: Iterable[str],
    frames: int = 300,
    output_dir: Path = Path("benchmarks/multi_model"),
    target_classes: tuple[int, ...] = (0, 32),
    imgsz: int = 640,
) -> Path:
    """Run each model over evenly-sampled frames of each video.

    Writes:
        <output_dir>/<UTC ts>/results.csv
        <output_dir>/<UTC ts>/results.json
        <output_dir>/<UTC ts>/results.png

    Returns the timestamped output directory.
    """
    videos = [Path(v) for v in videos]
    model_names = list(models)

    # Validate up-front so we fail fast on typos before loading any weights.
    specs = [resolve_model(n) for n in model_names]
    for v in videos:
        if not v.exists():
            raise FileNotFoundError(f"Video not found: {v}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvaluationResult] = []
    for spec in specs:
        for video in videos:
            logger.info("benchmarking %s on %s", spec.name, video.name)
            result = _benchmark_one(
                spec=spec,
                video_path=video,
                frames=frames,
                target_classes=target_classes,
                imgsz=imgsz,
            )
            results.append(result)
            logger.info(
                "  -> %d frames, %.1f FPS, %.2f conf, %.1f person/frame, %.2f ball/frame",
                result.frames_evaluated,
                result.fps,
                result.mean_confidence,
                result.detections_per_frame_person,
                result.detections_per_frame_ball,
            )

    _write_csv(results, run_dir / "results.csv")
    _write_json(results, run_dir / "results.json")
    _write_chart(results, run_dir / "results.png")
    return run_dir


# --- CLI -------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.multi_model_evaluator",
        description="Benchmark detection models on shared video frames.",
    )
    parser.add_argument(
        "--videos", nargs="+", required=True, type=Path, help="Input video paths."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        help=f"Models (default: {' '.join(SUPPORTED_MODELS)}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Frames to sample evenly per video (default: 300).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/multi_model"),
        help="Output root (default: benchmarks/multi_model).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = evaluate_models(
        videos=args.videos,
        models=args.models,
        frames=args.frames,
        output_dir=args.output,
        imgsz=args.imgsz,
    )
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
