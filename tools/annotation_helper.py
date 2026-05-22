"""
tools/annotation_helper.py — Task D1 (Seif)
Keyboard-driven ground truth annotation tool for TactiVision Pro.

Usage
-----
    python tools/annotation_helper.py --video input_videos/match.mp4
    python tools/annotation_helper.py --video input_videos/match.mp4 --stride 25 --start 0 --end 500

Controls (per frame)
--------------------
    LEFT / RIGHT arrow  : previous / next detection in this frame
    T                   : cycle team for selected detection
                          (team_a → team_b → referee → goalkeeper → ball → ignore → team_a …)
    DEL / D             : delete selected detection
    A                   : enter "add" mode — drag mouse to draw a new bbox
    SPACE               : commit frame and advance to next
    S                   : save without advancing
    Q                   : quit and save all

Output
------
    tests/ground_truth/<video_stem>.json
    {
      "video": "<filename>",
      "fps": <fps>,
      "frame_size": [w, h],
      "coordinate_space": "inference_640x360",
      "source": "human",
      "frames": [
        {"frame_idx": 25, "objects": [{"id": 1, "class": "team_a", "bbox": [x,y,w,h], "conf": 1.0}, …]},
        …
      ]
    }

Idempotent resume: if the output JSON already exists, already-annotated frames
are loaded and skipped; only un-annotated frames in the requested range are shown.
Auto-save every 10 committed frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
INFER_W, INFER_H = 640, 360
CLASSES = ["team_a", "team_b", "referee", "goalkeeper", "ball", "ignore"]

CLASS_COLORS = {
    "team_a":     (0,   0,   220),   # red
    "team_b":     (220, 0,   0),     # blue
    "referee":    (0,   220, 220),   # yellow
    "goalkeeper": (0,   165, 255),   # orange
    "ball":       (0,   220, 0),     # green
    "ignore":     (128, 128, 128),   # grey
}
DEFAULT_COLOR = (200, 200, 200)

AUTOSAVE_EVERY = 10          # commit this many frames then auto-save
DEFAULT_STRIDE = 25          # sample 1 frame per second at 25 fps


# ── YOLO pre-fill (optional) ─────────────────────────────────────────────────

def _prefill_with_yolo(frame_bgr: np.ndarray, model) -> list[dict]:
    """Run YOLOv8n and return detections in annotation format."""
    results = model(frame_bgr, imgsz=(INFER_W, INFER_H), verbose=False)[0]
    objects: list[dict] = []
    obj_id = 1
    for box in results.boxes:
        cls_int = int(box.cls[0])
        if cls_int == 0:       # person
            cls_str = "team_a"
        elif cls_int == 32:    # sports ball
            cls_str = "ball"
        else:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        # scale to inference space
        scale_x = INFER_W / frame_bgr.shape[1]
        scale_y = INFER_H / frame_bgr.shape[0]
        bx = x1 * scale_x
        by = y1 * scale_y
        bw = (x2 - x1) * scale_x
        bh = (y2 - y1) * scale_y
        objects.append({"id": obj_id, "class": cls_str,
                         "bbox": [round(bx, 1), round(by, 1),
                                  round(bw, 1), round(bh, 1)],
                         "conf": round(conf, 4)})
        obj_id += 1
    return objects


# ── drawing helpers ──────────────────────────────────────────────────────────

def _draw_frame(display: np.ndarray, objects: list[dict],
                selected: int, mode: str,
                drag_start=None, drag_end=None,
                frame_idx: int = 0, total: int = 0,
                committed: int = 0) -> np.ndarray:
    """Render objects on a copy of *display* and return it."""
    out = display.copy()
    # Scale factor from inference space to display space
    dh, dw = out.shape[:2]
    sx = dw / INFER_W
    sy = dh / INFER_H

    for i, obj in enumerate(objects):
        x, y, w, h = obj["bbox"]
        px, py, pw, ph = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
        color = CLASS_COLORS.get(obj["class"], DEFAULT_COLOR)
        thickness = 3 if i == selected else 1
        cv2.rectangle(out, (px, py), (px + pw, py + ph), color, thickness)
        label = f"[{i}] {obj['class']} (id={obj['id']})"
        cv2.putText(out, label, (px, max(py - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Drag rectangle for "add" mode
    if mode == "add" and drag_start and drag_end:
        cv2.rectangle(out, drag_start, drag_end, (0, 255, 255), 1)

    # HUD
    hud = (f"Frame {frame_idx}  [{committed+1}/{total}]  "
           f"mode={mode}  sel={selected}/{len(objects)-1}  "
           f"objs={len(objects)}")
    cv2.rectangle(out, (0, 0), (dw, 20), (30, 30, 30), -1)
    cv2.putText(out, hud, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    key_hint = "[T]=cycle class  [A]=add  [DEL/D]=delete  [SPACE]=commit  [S]=save  [Q]=quit"
    cv2.rectangle(out, (0, dh - 20), (dw, dh), (30, 30, 30), -1)
    cv2.putText(out, key_hint, (4, dh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return out


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def _load_existing(out_path: Path) -> dict:
    """Load an existing annotation file; return empty skeleton on failure."""
    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"frames": []}


def _save(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def _annotated_frame_indices(payload: dict) -> set[int]:
    return {int(f["frame_idx"]) for f in payload.get("frames", [])}


def _next_id(objects: list[dict]) -> int:
    if not objects:
        return 1
    return max(o["id"] for o in objects) + 1


# ── mouse callback state ─────────────────────────────────────────────────────

class _MouseState:
    def __init__(self):
        self.drawing = False
        self.start: tuple[int, int] | None = None
        self.end:   tuple[int, int] | None = None

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.end = (x, y)


# ── main annotation loop ─────────────────────────────────────────────────────

def annotate(
    video_path: Path,
    out_path: Path,
    stride: int = DEFAULT_STRIDE,
    start_frame: int = 0,
    end_frame: int | None = None,
    use_yolo: bool = True,
) -> dict:
    """Open *video_path* and let the operator annotate sampled frames.

    Returns the final annotation payload dict.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = total_frames - 1

    # Build list of frame indices to annotate
    to_visit = list(range(start_frame, end_frame + 1, stride))
    if not to_visit:
        print("[!] No frames to annotate in the given range.")
        cap.release()
        return {}

    # Load or create output payload
    payload = _load_existing(out_path)
    already_done = _annotated_frame_indices(payload)

    payload.setdefault("video", video_path.name)
    payload["fps"] = fps
    payload["frame_size"] = [orig_w, orig_h]
    payload["coordinate_space"] = "inference_640x360"
    payload["source"] = "human"

    # Build frames index for quick access
    frames_by_idx: dict[int, dict] = {
        int(f["frame_idx"]): f for f in payload.get("frames", [])
    }

    # Load YOLO for pre-fill
    model = None
    if use_yolo:
        try:
            from ultralytics import YOLO as _YOLO
            model = _YOLO("yolov8n.pt")
            print("[*] YOLOv8n loaded for pre-fill.")
        except Exception as e:
            print(f"[!] Could not load YOLO ({e}); will not pre-fill.")

    # Window setup
    WIN = "TactiVision Annotator"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    ms = _MouseState()
    cv2.setMouseCallback(WIN, ms.callback)

    remaining = [idx for idx in to_visit if idx not in already_done]
    print(f"[*] {len(to_visit)} frames sampled; "
          f"{len(already_done)} already annotated; "
          f"{len(remaining)} to go.")

    committed_count = 0
    i = 0                       # index into *remaining*
    mode = "select"             # "select" | "add"
    selected = 0

    while i < len(remaining):
        frame_idx = remaining[i]

        # Seek and read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_orig = cap.read()
        if not ret:
            print(f"[!] Could not read frame {frame_idx}; skipping.")
            i += 1
            continue

        # Resize to inference size for annotation
        frame_infer = cv2.resize(frame_orig, (INFER_W, INFER_H))

        # Pre-fill or load existing objects
        if frame_idx in frames_by_idx:
            objects: list[dict] = list(frames_by_idx[frame_idx]["objects"])
        elif model is not None:
            objects = _prefill_with_yolo(frame_infer, model)
        else:
            objects = []

        selected = min(selected, len(objects) - 1) if objects else 0
        mode = "select"
        ms.start = ms.end = None

        while True:
            drawn = _draw_frame(
                frame_infer, objects, selected, mode,
                drag_start=ms.start, drag_end=ms.end,
                frame_idx=frame_idx,
                total=len(remaining),
                committed=i,
            )
            cv2.imshow(WIN, drawn)
            key = cv2.waitKey(30) & 0xFF

            # ── quit ─────────────────────────────────────────────────────
            if key == ord('q') or key == ord('Q'):
                _save(out_path, _rebuild_payload(payload, frames_by_idx))
                print(f"[*] Saved to {out_path} and quit.")
                cap.release()
                cv2.destroyAllWindows()
                return payload

            # ── save without advancing ────────────────────────────────────
            elif key == ord('s') or key == ord('S'):
                _save(out_path, _rebuild_payload(payload, frames_by_idx))
                print(f"[*] Saved.")

            # ── commit and advance ────────────────────────────────────────
            elif key == ord(' '):
                frames_by_idx[frame_idx] = {
                    "frame_idx": frame_idx,
                    "objects": objects,
                }
                committed_count += 1
                if committed_count % AUTOSAVE_EVERY == 0:
                    _save(out_path, _rebuild_payload(payload, frames_by_idx))
                    print(f"[*] Auto-saved at {committed_count} commits.")
                i += 1
                selected = 0
                break

            # ── navigate selections ───────────────────────────────────────
            elif key == 81 or key == 2424832:  # LEFT
                if objects:
                    selected = (selected - 1) % len(objects)
            elif key == 83 or key == 2555904:  # RIGHT
                if objects:
                    selected = (selected + 1) % len(objects)

            # ── cycle team ────────────────────────────────────────────────
            elif key == ord('t') or key == ord('T'):
                if objects:
                    cur = objects[selected]["class"]
                    nxt = CLASSES[(CLASSES.index(cur) + 1) % len(CLASSES)]
                    objects[selected]["class"] = nxt

            # ── delete selected ───────────────────────────────────────────
            elif key in (255, ord('d'), ord('D'), 8):  # DEL or D or backspace
                if objects:
                    objects.pop(selected)
                    selected = min(selected, len(objects) - 1) if objects else 0

            # ── add mode toggle ───────────────────────────────────────────
            elif key == ord('a') or key == ord('A'):
                mode = "add" if mode == "select" else "select"
                ms.start = ms.end = None

            # ── finish drag in add mode ───────────────────────────────────
            if mode == "add" and ms.start and ms.end and not ms.drawing:
                x1, y1 = ms.start
                x2, y2 = ms.end
                if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                    # convert display coords to inference coords
                    dh2, dw2 = frame_infer.shape[:2]  # already INFER_W×INFER_H
                    sx = INFER_W / dw2
                    sy = INFER_H / dh2
                    bx = min(x1, x2) * sx
                    by = min(y1, y2) * sy
                    bw = abs(x2 - x1) * sx
                    bh = abs(y2 - y1) * sy
                    new_obj = {
                        "id": _next_id(objects),
                        "class": "team_a",
                        "bbox": [round(bx, 1), round(by, 1),
                                 round(bw, 1), round(bh, 1)],
                        "conf": 1.0,
                    }
                    objects.append(new_obj)
                    selected = len(objects) - 1
                ms.start = ms.end = None
                mode = "select"

    # All frames visited
    cap.release()
    cv2.destroyAllWindows()
    final = _rebuild_payload(payload, frames_by_idx)
    _save(out_path, final)
    print(f"[*] Annotation complete. {len(frames_by_idx)} frames saved → {out_path}")
    return final


def _rebuild_payload(base: dict, frames_by_idx: dict) -> dict:
    """Merge frames_by_idx back into base payload, sorted by frame_idx."""
    result = {k: v for k, v in base.items() if k != "frames"}
    result["frames"] = sorted(
        frames_by_idx.values(), key=lambda f: f["frame_idx"]
    )
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python tools/annotation_helper.py",
        description="Ground truth annotation tool for TactiVision Pro (Task D1).",
    )
    parser.add_argument("--video", required=True, type=Path,
                        help="Path to input video.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path. Default: tests/ground_truth/<stem>.json")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"Sample every N frames (default {DEFAULT_STRIDE}).")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index (default 0).")
    parser.add_argument("--end", type=int, default=None,
                        help="End frame index (default: last frame).")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Disable YOLO pre-fill; start each frame blank.")
    args = parser.parse_args(argv)

    video_path = args.video
    if not video_path.exists():
        print(f"[!] Video not found: {video_path}")
        sys.exit(1)

    out_path = args.out or (
        ROOT / "tests" / "ground_truth" / f"{video_path.stem}.json"
    )

    annotate(
        video_path=video_path,
        out_path=out_path,
        stride=args.stride,
        start_frame=args.start,
        end_frame=args.end,
        use_yolo=not args.no_yolo,
    )


if __name__ == "__main__":
    main()
