"""Tests for services/rtdetr_ball_detector.py (P15).

Covers the crop-window geometry and stats bookkeeping, which is the logic
that does not require loading the RT-DETR weights. The live-inference path
is exercised by scripts/measure_ball_p15.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.rtdetr_ball_detector import RTDetrBallDetector  # noqa: E402


def test_crop_window_centered():
    """A crop away from the edges is exactly crop_size and centered."""
    det = RTDetrBallDetector(crop_size=200)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop, x0, y0 = det._crop_window(frame, (640.0, 360.0))
    assert crop.shape[:2] == (200, 200)
    assert (x0, y0) == (540, 260)  # 640-100, 360-100


def test_crop_window_clamps_at_edges():
    """A crop near a corner is clamped so it stays inside the frame."""
    det = RTDetrBallDetector(crop_size=200)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    crop, x0, y0 = det._crop_window(frame, (5.0, 5.0))
    assert (x0, y0) == (0, 0)
    assert crop.shape[:2] == (200, 200)

    crop, x0, y0 = det._crop_window(frame, (1279.0, 719.0))
    assert (x0, y0) == (1080, 520)  # 1280-200, 720-200
    assert crop.shape[:2] == (200, 200)


def test_crop_window_smaller_than_crop_size():
    """When the frame is smaller than crop_size, the crop is the whole frame."""
    det = RTDetrBallDetector(crop_size=640)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    crop, x0, y0 = det._crop_window(frame, (320.0, 180.0))
    # crop_size is clamped to min(640, 640, 360) = 360
    assert crop.shape[:2] == (360, 360)
    assert (x0, y0) == (140, 0)


def test_stats_start_empty():
    det = RTDetrBallDetector()
    s = det.stats()
    assert s == {"calls": 0, "hits": 0, "hit_rate": 0.0}


def test_lazy_model_not_loaded_on_init():
    """Constructing the detector must not load the heavy RT-DETR weights."""
    det = RTDetrBallDetector()
    assert det._model is None


def test_missing_weights_raises():
    det = RTDetrBallDetector(weights_path=Path("weights/does_not_exist.pt"))
    with pytest.raises(FileNotFoundError):
        det._ensure_model()
