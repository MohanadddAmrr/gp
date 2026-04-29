"""Tests for services.multi_model_evaluator (M1)."""

from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

import pytest

# Make project root importable when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.multi_model_evaluator import (  # noqa: E402
    CSV_COLUMNS,
    SUPPORTED_MODELS,
    ModelSpec,
    _evenly_sampled_indices,
    evaluate_models,
    resolve_model,
)


# --- test_modelspec_resolves ----------------------------------------------


def test_resolve_model_known_returns_spec() -> None:
    spec = resolve_model("yolov8n")
    assert isinstance(spec, ModelSpec)
    assert spec.weights_filename == "yolov8n.pt"
    assert spec.framework == "ultralytics_yolo"


def test_resolve_model_all_supported_have_pt_files() -> None:
    for name, spec in SUPPORTED_MODELS.items():
        assert spec.name == name
        assert spec.weights_filename.endswith(".pt")
        assert spec.framework in {"ultralytics_yolo", "ultralytics_rtdetr"}


def test_resolve_model_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("totally-fake-model")


def test_evenly_sampled_indices_basic() -> None:
    assert _evenly_sampled_indices(0, 10) == []
    assert _evenly_sampled_indices(5, 10) == [0, 1, 2, 3, 4]
    out = _evenly_sampled_indices(100, 10)
    assert len(out) == 10
    assert out[0] == 0
    assert out[-1] < 100
    # Strictly increasing
    assert all(b > a for a, b in zip(out, out[1:]))


# --- test_evaluate_models_smoke -------------------------------------------


def _short_clip() -> Path | None:
    """Pick the smallest available real input video for a smoke run."""
    candidates = [
        Path("input_videos/test (2).mp4"),
        Path("input_videos/liverpoolvstottenham.mp4"),
        Path("input_videos/epl_newcastle_vs_utd_2026.mp4"),
        Path("demo/input_videos/test (2).mp4"),
    ]
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return None
    return min(existing, key=lambda p: p.stat().st_size)


@pytest.mark.skipif(
    _short_clip() is None, reason="no input video available for smoke test"
)
@pytest.mark.skipif(
    shutil.which("python") is None and not os.environ.get("PYTHONPATH"),
    reason="needs python to import ultralytics",
)
def test_evaluate_models_smoke(tmp_path: Path) -> None:
    """Run a 5-frame smoke benchmark with yolov8n on the shortest available
    clip. Verify the standard outputs land on disk with the expected shape.
    """
    pytest.importorskip("ultralytics")
    pytest.importorskip("cv2")

    clip = _short_clip()
    assert clip is not None  # guarded by skipif

    out = evaluate_models(
        videos=[clip],
        models=["yolov8n"],
        frames=5,
        output_dir=tmp_path,
        imgsz=320,
    )

    csv_path = out / "results.csv"
    json_path = out / "results.json"
    png_path = out / "results.png"

    assert csv_path.exists(), "results.csv missing"
    assert json_path.exists(), "results.json missing"
    assert png_path.exists(), "results.png missing"

    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"
    row = rows[0]
    for col in CSV_COLUMNS:
        assert col in row, f"missing column {col}"
    assert row["model"] == "yolov8n"
    assert int(row["frames_evaluated"]) >= 1
