"""
Track-file generator (P13) — runs the detection + tracking + Re-ID pipeline
over a video range and writes the unified track-file schema consumed by
`services.accuracy_evaluator`.

Two uses, one tool:

1. Draft ground truth ("pseudo-labelling"): run on a short clip, then hand the
   `--source pseudo` output to a human to correct (fix boxes, merge split IDs,
   delete false positives). Correcting a draft is far faster than annotating
   from scratch.
2. Predictions: run with whatever pipeline config is under test (`--source
   pipeline`) and score it against the corrected GT with the evaluator.

Bboxes are emitted in the 640x360 inference space — the same space the rest of
the pipeline uses — so GT and predictions need no rescaling.

CLI
---
    python -m scripts.make_tracks \
        --video input_videos/arsenalvsfulham.mp4 \
        --output tests/ground_truth/arsenalvsfulham_clip.json \
        --start-frame 0 --end-frame 3000 --stride 25 \
        --algorithm botsort --reid --source pseudo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TARGET_SIZE = (640, 360)  # (w, h) — matches run_demo.py / chunked processor


def generate_tracks(
    video_path: Path,
    output_path: Path,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    stride: int = 25,
    algorithm: str = "botsort",
    reid: bool = True,
    weights: str = "yolov8n.pt",
    source: str = "pipeline",
) -> dict:
    """Run the pipeline over [start_frame, end_frame) at `stride` and write the
    unified track file. Returns the written payload.
    """
    import cv2
    from ultralytics import YOLO

    from services.improved_tracker import build_tracker
    from services.reid_module import JerseyColorClassifier, JerseyPosReID

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last_frame = total_frames if end_frame is None else min(end_frame, total_frames)
    target_w, target_h = _TARGET_SIZE

    yolo = YOLO(weights)
    tracker = build_tracker(algorithm, yolo)
    classifier = JerseyColorClassifier()
    reid_layer = JerseyPosReID() if reid else None

    frames_out: list[dict] = []
    for frame_idx in range(start_frame, last_frame, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.resize(frame, (target_w, target_h))

        res_list = tracker.track(
            source=frame, classes=[0, 32], conf=0.3, imgsz=640,
            verbose=False, persist=True,
        )
        objects: list[dict] = []
        if res_list:
            res = res_list[0]
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)
                conf_arr = res.boxes.conf.cpu().numpy()

                for raw_id, box, cls_id, conf in zip(ids, xyxy, cls_arr, conf_arr):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    bbox = [round(x1, 1), round(y1, 1),
                            round(x2 - x1, 1), round(y2 - y1, 1)]

                    if cls_id == 0:  # person
                        x1i, y1i = max(int(x1), 0), max(int(y1), 0)
                        x2i, y2i = max(int(x2), x1i + 1), max(int(y2), y1i + 1)
                        crop = frame[y1i:y2i, x1i:x2i]
                        jersey_class = classifier.classify(crop)
                        if jersey_class == "unknown":
                            jersey_class = "ignore"
                        obj_id = int(raw_id)
                        if reid_layer is not None:
                            obj_id = reid_layer.resolve(
                                raw_track_id=int(raw_id),
                                position=(cx, cy),
                                crop_bgr=crop,
                                classifier=classifier,
                                frame_idx=frame_idx,
                            )
                        objects.append({
                            "id": obj_id,
                            "class": jersey_class,
                            "bbox": bbox,
                            "conf": round(float(conf), 4),
                        })
                    elif cls_id == 32:  # ball
                        objects.append({
                            "id": int(raw_id),
                            "class": "ball",
                            "bbox": bbox,
                            "conf": round(float(conf), 4),
                        })
        frames_out.append({"frame_idx": frame_idx, "objects": objects})

    cap.release()

    payload = {
        "video": Path(video_path).name,
        "fps": fps,
        "frame_size": [target_w, target_h],
        "coordinate_space": "inference_640x360",
        "source": source,
        "generator": {
            "algorithm": algorithm,
            "reid": reid,
            "weights": weights,
            "start_frame": start_frame,
            "end_frame": last_frame,
            "stride": stride,
        },
        "frames": frames_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.make_tracks",
        description="Generate a unified track file (draft GT or predictions).",
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Exclusive end frame (default: end of video).",
    )
    parser.add_argument(
        "--stride", type=int, default=25,
        help="Process every Nth frame (default: 25 ~ 1 fps at 25 fps).",
    )
    parser.add_argument(
        "--algorithm", choices=("bytetrack", "botsort"), default="botsort",
    )
    parser.add_argument(
        "--reid", action="store_true",
        help="Enable the Jersey + Position Re-ID layer (canonical IDs).",
    )
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights path.")
    parser.add_argument(
        "--source", choices=("pipeline", "pseudo", "human"), default="pipeline",
        help="Provenance tag written into the file.",
    )
    args = parser.parse_args(argv)

    payload = generate_tracks(
        video_path=args.video,
        output_path=args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        algorithm=args.algorithm,
        reid=args.reid,
        weights=args.weights,
        source=args.source,
    )
    n_objs = sum(len(f["objects"]) for f in payload["frames"])
    print(
        f"wrote {len(payload['frames'])} frames / {n_objs} objects "
        f"-> {args.output}  (source={args.source})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
