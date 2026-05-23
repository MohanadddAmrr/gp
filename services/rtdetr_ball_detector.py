"""RT-DETR ball-only second-pass detector.

The primary YOLO model in the pipeline detects the ball in only a small
fraction of frames (~2-22% on the benchmark clips). RT-DETR is markedly
stronger on the ball (0.6-1.6 ball detections/frame on the same clips) but
too slow to run full-frame on every frame on CPU (~1 FPS).

This detector is meant as a *conditional second pass*: it is invoked only on
frames where the primary detector missed the ball. When a previous ball
position is known it runs RT-DETR on a crop window around that position,
which is much cheaper than a full-frame pass and keeps the search local.

COCO class 32 = "sports ball".
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SPORTS_BALL_CLASS = 32
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "rtdetr-l.pt"


class RTDetrBallDetector:
    """Crop-based RT-DETR ball detector for use as a second-pass fallback."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        conf: float = 0.25,
        crop_size: int = 640,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self.conf = conf
        self.crop_size = crop_size
        self.imgsz = imgsz
        self.device = device
        self._model = None  # lazy-loaded on first detect
        # Lightweight stats for reporting / measurement.
        self.calls = 0
        self.hits = 0

    def _ensure_model(self) -> None:
        if self._model is None:
            from ultralytics import RTDETR

            if not self.weights_path.exists():
                raise FileNotFoundError(
                    f"RT-DETR weights not found: {self.weights_path}"
                )
            self._model = RTDETR(str(self.weights_path))

    def _infer(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Run RT-DETR on an image, return [(x1,y1,x2,y2,conf)] for the ball class."""
        self._ensure_model()
        res = self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        out: List[Tuple[float, float, float, float, float]] = []
        for box in res.boxes:
            if int(box.cls.item()) != SPORTS_BALL_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(box.conf.item())))
        return out

    def _crop_window(
        self, frame: np.ndarray, center: Tuple[float, float]
    ) -> Tuple[np.ndarray, int, int]:
        """Crop a crop_size square around `center`, clamped to the frame.

        Returns (crop, x_offset, y_offset) so detections can be translated back.
        """
        h, w = frame.shape[:2]
        cs = min(self.crop_size, w, h)
        cx, cy = center
        x0 = int(round(cx - cs / 2))
        y0 = int(round(cy - cs / 2))
        x0 = max(0, min(x0, w - cs))
        y0 = max(0, min(y0, h - cs))
        return frame[y0 : y0 + cs, x0 : x0 + cs], x0, y0

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, float]]:
        """Full-frame ball detection. Returns [(x1,y1,x2,y2,conf)]."""
        return self._infer(frame)

    def detect_best(
        self,
        frame: np.ndarray,
        last_known_pos: Optional[Tuple[float, float]] = None,
        max_jump_px: float = 300.0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Detect the single most likely ball.

        When `last_known_pos` is given, RT-DETR runs on a crop window around
        it (fast, local) and detections farther than `max_jump_px` from that
        position are rejected as likely false positives. With no prior
        position it falls back to a full-frame pass.

        Returns a bounding box (x1, y1, x2, y2) in full-frame pixel
        coordinates, or None.
        """
        self.calls += 1

        if last_known_pos is not None:
            crop, x_off, y_off = self._crop_window(frame, last_known_pos)
            raw = self._infer(crop)
            dets = [
                (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, c)
                for (x1, y1, x2, y2, c) in raw
            ]
            ref_x, ref_y = last_known_pos
            valid = []
            for x1, y1, x2, y2, c in dets:
                bcx = (x1 + x2) / 2.0
                bcy = (y1 + y2) / 2.0
                if np.hypot(bcx - ref_x, bcy - ref_y) <= max_jump_px:
                    valid.append((x1, y1, x2, y2, c))
            if not valid:
                return None
            best = max(valid, key=lambda d: d[4])
            self.hits += 1
            return best[:4]

        dets = self._infer(frame)
        if not dets:
            return None
        best = max(dets, key=lambda d: d[4])
        self.hits += 1
        return best[:4]

    def stats(self) -> dict:
        """Return second-pass usage stats."""
        return {
            "calls": self.calls,
            "hits": self.hits,
            "hit_rate": round(self.hits / self.calls, 4) if self.calls else 0.0,
        }
