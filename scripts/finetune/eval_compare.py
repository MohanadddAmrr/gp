"""Compare baseline yolov8n (COCO) against fine-tuned yolov8n on the val split.

Computes mAP@50 and mAP@50-95 per class against the pseudo-labelled val set.
The baseline runs in COCO class space and is remapped: person (0) -> 0,
sports ball (32) -> 1; all other classes are dropped.

Usage:
    python scripts/finetune/eval_compare.py --ft-weights runs/finetune/yolov8n_football_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "finetune_v1"
VAL_IMG_DIR = DATA_DIR / "images" / "val"
VAL_LBL_DIR = DATA_DIR / "labels" / "val"

COCO_PERSON = 0
COCO_BALL = 32
NUM_CLASSES = 2
CLASS_NAMES = ["player", "ball"]
IOU_THRESHOLDS = [0.5] + [0.5 + 0.05 * i for i in range(1, 10)]  # 0.50:0.95


@dataclass
class Det:
    cls: int
    score: float
    box: np.ndarray  # xyxy


def load_gt(img_path: Path) -> dict[int, list[np.ndarray]]:
    """Return {class: [xyxy boxes]}."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = VAL_LBL_DIR / (img_path.stem + ".txt")
    by_cls: dict[int, list[np.ndarray]] = {c: [] for c in range(NUM_CLASSES)}
    if not lbl.exists():
        return by_cls
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        by_cls[c].append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return by_cls


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def predict_baseline(model: YOLO, img_path: Path) -> list[Det]:
    res = model.predict(str(img_path), imgsz=640, conf=0.001, verbose=False)[0]
    dets: list[Det] = []
    for box in res.boxes:
        cls = int(box.cls.item())
        if cls == COCO_PERSON:
            mapped = 0
        elif cls == COCO_BALL:
            mapped = 1
        else:
            continue
        score = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
        dets.append(Det(mapped, score, xyxy))
    return dets


def predict_ft(model: YOLO, img_path: Path) -> list[Det]:
    res = model.predict(str(img_path), imgsz=640, conf=0.001, verbose=False)[0]
    dets: list[Det] = []
    for box in res.boxes:
        cls = int(box.cls.item())
        if cls not in (0, 1):
            continue
        score = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
        dets.append(Det(cls, score, xyxy))
    return dets


def compute_ap(matches: list[tuple[float, bool]], n_gt: int) -> float:
    """11-point or all-point AP. Uses all-point interpolation (COCO-style)."""
    if n_gt == 0:
        return 0.0
    matches.sort(key=lambda m: -m[0])
    tp = np.zeros(len(matches))
    fp = np.zeros(len(matches))
    for i, (_, hit) in enumerate(matches):
        if hit:
            tp[i] = 1
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # all-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate(name: str, predict_fn, imgs: list[Path]) -> dict:
    """Per-class AP at IoU 0.5 and mAP@0.5:0.95."""
    print(f"[{name}] evaluating on {len(imgs)} val images")
    # Collect per-class: list of (image_idx, det) and per-image gt boxes
    per_cls_dets: dict[int, list[tuple[int, Det]]] = {c: [] for c in range(NUM_CLASSES)}
    per_cls_gt: dict[int, dict[int, list[np.ndarray]]] = {c: {} for c in range(NUM_CLASSES)}
    n_gt_per_cls: dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}

    for img_idx, img_path in enumerate(imgs):
        gt = load_gt(img_path)
        for c in range(NUM_CLASSES):
            per_cls_gt[c][img_idx] = gt[c]
            n_gt_per_cls[c] += len(gt[c])
        for d in predict_fn(img_path):
            per_cls_dets[d.cls].append((img_idx, d))
        if (img_idx + 1) % 25 == 0:
            print(f"  ...{img_idx+1}/{len(imgs)}")

    results: dict = {"per_class": {}, "n_val_images": len(imgs)}
    aps_50_all = []
    aps_5095_all = []
    for c in range(NUM_CLASSES):
        ap_per_iou = []
        ap50_value = 0.0
        for iou_thr in IOU_THRESHOLDS:
            matches = match_at_iou(per_cls_dets[c], per_cls_gt[c], iou_thr)
            ap = compute_ap(matches, n_gt_per_cls[c])
            ap_per_iou.append(ap)
            if abs(iou_thr - 0.5) < 1e-6:
                ap50_value = ap
        ap_5095 = float(np.mean(ap_per_iou))
        results["per_class"][CLASS_NAMES[c]] = {
            "n_gt": n_gt_per_cls[c],
            "n_pred": len(per_cls_dets[c]),
            "ap50": ap50_value,
            "ap50_95": ap_5095,
        }
        aps_50_all.append(ap50_value)
        aps_5095_all.append(ap_5095)
    results["mAP50"] = float(np.mean(aps_50_all))
    results["mAP50_95"] = float(np.mean(aps_5095_all))
    return results


def match_at_iou(
    dets: list[tuple[int, Det]],
    gt_by_img: dict[int, list[np.ndarray]],
    iou_thr: float,
) -> list[tuple[float, bool]]:
    """Greedy IoU matching, sorted by confidence. Returns [(score, is_tp)]."""
    dets_sorted = sorted(dets, key=lambda x: -x[1].score)
    used: dict[int, set[int]] = {i: set() for i in gt_by_img}
    matches: list[tuple[float, bool]] = []
    for img_idx, d in dets_sorted:
        gt_boxes = gt_by_img.get(img_idx, [])
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if j in used.get(img_idx, set()):
                continue
            v = iou(d.box, g)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            used.setdefault(img_idx, set()).add(best_j)
            matches.append((d.score, True))
        else:
            matches.append((d.score, False))
    return matches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-weights", type=str, required=True)
    ap.add_argument(
        "--baseline-weights",
        type=str,
        default=str(REPO / "weights" / "yolov8n.pt"),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(REPO / "benchmarks" / "finetune_eval.json"),
    )
    args = ap.parse_args()

    imgs = sorted(VAL_IMG_DIR.glob("*.jpg"))
    assert imgs, f"no val images in {VAL_IMG_DIR}"

    baseline = YOLO(args.baseline_weights)
    ft = YOLO(args.ft_weights)

    res = {
        "baseline_yolov8n_coco": evaluate("baseline", lambda p: predict_baseline(baseline, p), imgs),
        "finetuned_yolov8n": evaluate("finetuned", lambda p: predict_ft(ft, p), imgs),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n=== RESULTS ===")
    for tag, r in res.items():
        print(f"{tag}:  mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}")
        for cls, m in r["per_class"].items():
            print(f"   {cls:8s}  AP50={m['ap50']:.4f}  AP50-95={m['ap50_95']:.4f}  gt={m['n_gt']}  pred={m['n_pred']}")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
