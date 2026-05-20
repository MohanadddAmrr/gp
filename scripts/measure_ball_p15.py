"""Measure the P15 RT-DETR ball-only second pass.

Processes a clip with the primary detector (yolov8n, the pipeline default),
then re-runs the RT-DETR second pass on every frame where the primary
detector missed the ball. Reports the ball-detection-rate lift and the
per-frame time cost so the speed/accuracy trade-off is explicit.

The GT ball positions are unannotated, so this measures detection *rate*
(fraction of frames with a ball), the same metric the model report tracked.

Usage:
    python scripts/measure_ball_p15.py --video "input_videos/test (2).mp4" --frames 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from services.rtdetr_ball_detector import RTDetrBallDetector  # noqa: E402

BALL_CLASS = 32
PERSON_CLASS = 0
PROC_SIZE = (640, 360)  # same resize the demo pipeline uses


def yolo_ball_box(res) -> tuple[float, float, float, float] | None:
    """Return the highest-confidence ball box from a YOLO Results object."""
    best = None
    best_conf = -1.0
    for box in res.boxes:
        if int(box.cls.item()) != BALL_CLASS:
            continue
        c = float(box.conf.item())
        if c > best_conf:
            best_conf = c
            best = tuple(box.xyxy[0].tolist())
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--crop-size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", type=str, default=str(REPO / "benchmarks" / "ball_p15_measure.json"))
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = REPO / video
    assert video.exists(), f"missing video: {video}"

    from ultralytics import YOLO

    yolo = YOLO(str(REPO / "weights" / "yolov8n.pt"))
    rtdetr = RTDetrBallDetector(crop_size=args.crop_size, conf=0.25)

    cap = cv2.VideoCapture(str(video))
    n_target = min(args.frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    frames_processed = 0
    yolo_ball_frames = 0
    rtdetr_recovered = 0
    yolo_time = 0.0
    rtdetr_time = 0.0
    last_known_pos: tuple[float, float] | None = None

    print(f"[measure] {video.name}  frames={n_target}  crop={args.crop_size}")
    while frames_processed < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, PROC_SIZE)

        t0 = time.perf_counter()
        res = yolo.predict(frame, imgsz=640, conf=args.conf, classes=[PERSON_CLASS, BALL_CLASS], verbose=False)[0]
        yolo_time += time.perf_counter() - t0

        ball = yolo_ball_box(res)
        if ball is not None:
            yolo_ball_frames += 1
        else:
            # Second pass only on frames the primary detector missed.
            t1 = time.perf_counter()
            ball = rtdetr.detect_best(frame, last_known_pos)
            rtdetr_time += time.perf_counter() - t1
            if ball is not None:
                rtdetr_recovered += 1

        if ball is not None:
            last_known_pos = ((ball[0] + ball[2]) / 2.0, (ball[1] + ball[3]) / 2.0)

        frames_processed += 1
        if frames_processed % 50 == 0:
            print(f"  ...{frames_processed}/{n_target}  yolo_ball={yolo_ball_frames}  recovered={rtdetr_recovered}")
    cap.release()

    yolo_misses = frames_processed - yolo_ball_frames
    combined = yolo_ball_frames + rtdetr_recovered
    result = {
        "video": video.name,
        "frames_processed": frames_processed,
        "yolo_only": {
            "ball_frames": yolo_ball_frames,
            "detection_rate": round(yolo_ball_frames / frames_processed, 4) if frames_processed else 0.0,
            "avg_ms_per_frame": round(1000 * yolo_time / frames_processed, 1) if frames_processed else 0.0,
            "fps": round(frames_processed / yolo_time, 2) if yolo_time else 0.0,
        },
        "with_second_pass": {
            "ball_frames": combined,
            "detection_rate": round(combined / frames_processed, 4) if frames_processed else 0.0,
            "frames_recovered_by_rtdetr": rtdetr_recovered,
            "recovery_rate_on_misses": round(rtdetr_recovered / yolo_misses, 4) if yolo_misses else 0.0,
            "second_pass_invocations": yolo_misses,
            "avg_ms_per_frame": round(1000 * (yolo_time + rtdetr_time) / frames_processed, 1) if frames_processed else 0.0,
            "fps": round(frames_processed / (yolo_time + rtdetr_time), 2) if (yolo_time + rtdetr_time) else 0.0,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    yo = result["yolo_only"]
    sp = result["with_second_pass"]
    print("\n=== P15 BALL SECOND-PASS MEASUREMENT ===")
    print(f"frames: {frames_processed}")
    print(f"YOLO only        : ball in {yo['ball_frames']:4d}  "
          f"rate={yo['detection_rate']*100:5.1f}%   {yo['fps']:.2f} FPS")
    print(f"+ RT-DETR 2nd pass: ball in {sp['ball_frames']:4d}  "
          f"rate={sp['detection_rate']*100:5.1f}%   {sp['fps']:.2f} FPS")
    print(f"recovered {rtdetr_recovered} of {yolo_misses} missed frames "
          f"({sp['recovery_rate_on_misses']*100:.1f}% of misses)")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
