"""Pseudo-label match frames using RT-DETR for YOLOv8 fine-tuning.

Samples frames uniformly across the source videos, runs RT-DETR (COCO pretrained),
keeps only `person` (COCO id 0) and `sports ball` (COCO id 32), remaps to
{0: player, 1: ball}, and writes YOLO-format labels.

Usage:
    python scripts/finetune/pseudo_label.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import RTDETR

REPO = Path(__file__).resolve().parents[2]
VIDEOS = [
    REPO / "input_videos" / "test (2).mp4",
    REPO / "input_videos" / "liverpoolvstottenham.mp4",
    REPO / "input_videos" / "newcastlevsmanchesterunited.mp4",
]
OUT_ROOT = REPO / "data" / "finetune_v1"

COCO_PERSON = 0
COCO_BALL = 32
PLAYER_CONF = 0.50
BALL_CONF = 0.30
VAL_RATIO = 0.20


def sample_indices(total: int, n: int, seed: int = 0) -> list[int]:
    """Uniformly sample n distinct frame indices in [margin, total - margin)."""
    margin = int(total * 0.05)
    lo, hi = margin, max(margin + 1, total - margin)
    step = max(1, (hi - lo) // n)
    rng = random.Random(seed)
    base = list(range(lo, hi, step))[:n]
    jitter = [i + rng.randint(-step // 4, step // 4) for i in base]
    return sorted({max(0, min(total - 1, i)) for i in jitter})


def yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def process_video(video: Path, n_frames: int, model: RTDETR, rng: random.Random) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[skip] cannot open {video.name}")
        return 0, 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(total, n_frames)
    print(f"[{video.name}] total={total} sampling={len(idxs)}")

    stem = video.stem.replace(" ", "_").replace("(", "").replace(")", "")
    n_kept = 0
    n_player = 0
    n_ball = 0
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        res = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        lines = []
        for box in res.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if cls == COCO_PERSON and conf >= PLAYER_CONF:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                lines.append(yolo_line(0, x1, y1, x2, y2, w, h))
                n_player += 1
            elif cls == COCO_BALL and conf >= BALL_CONF:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                lines.append(yolo_line(1, x1, y1, x2, y2, w, h))
                n_ball += 1
        if not lines:
            continue
        split = "val" if rng.random() < VAL_RATIO else "train"
        img_path = OUT_ROOT / "images" / split / f"{stem}_f{fi:07d}.jpg"
        lbl_path = OUT_ROOT / "labels" / split / f"{stem}_f{fi:07d}.txt"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        n_kept += 1
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(idxs)} kept={n_kept}")
    cap.release()
    return n_kept, n_player, n_ball


def write_data_yaml() -> None:
    p = OUT_ROOT / "data.yaml"
    p.write_text(
        f"path: {OUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: player\n"
        "  1: ball\n",
        encoding="utf-8",
    )
    print(f"[wrote] {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=170, help="frames sampled per video")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    print("[load] RT-DETR-L (COCO)")
    model = RTDETR(str(REPO / "weights" / "rtdetr-l.pt"))
    rng = random.Random(args.seed)

    tot_kept = tot_p = tot_b = 0
    for v in VIDEOS:
        if not v.exists():
            print(f"[skip] missing {v}")
            continue
        k, p, b = process_video(v, args.per_video, model, rng)
        tot_kept += k
        tot_p += p
        tot_b += b
    write_data_yaml()
    print(f"\nDONE  frames_kept={tot_kept}  player_boxes={tot_p}  ball_boxes={tot_b}")


if __name__ == "__main__":
    main()
