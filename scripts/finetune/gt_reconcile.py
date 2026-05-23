"""Reconcile baseline vs fine-tuned yolov8n against human ground truth.

Uses the project's established player-tracking metric (see
tests/test_accuracy_measurement.py::AccuracyMeasurer._measure_player_tracking):
  - detection point = bbox centroid
  - greedy nearest-GT match within a 50 px Euclidean threshold
  - precision / recall / F1 / MAE over all GT frames

The GT is centroid-only (no boxes), so this is a point-distance metric, not
mAP@50. It is directly comparable to the recorded baseline f1=0.797.

Usage:
    python scripts/finetune/gt_reconcile.py --ft-weights runs/finetune/yolov8n_football_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
GT_PATH = REPO / "tests" / "ground_truth" / "liverpoolvscity_ground_truth.json"
VIDEO_PATH = Path(r"C:\Users\mohan\Downloads\liverpoolvscity.mp4")
MATCH_THRESHOLD = 50.0  # px, same as the established evaluator
CONF = 0.5  # same as the recorded baseline


def load_gt() -> dict[int, list[tuple[float, float]]]:
    data = json.loads(GT_PATH.read_text())
    gt: dict[int, list[tuple[float, float]]] = {}
    for item in data:
        gt[item["frame_idx"]] = [
            (v["x"], v["y"]) for v in item["player_positions"].values()
        ]
    return gt


def detect_centroids(model: YOLO, frame: np.ndarray, player_cls: int) -> list[tuple[float, float]]:
    res = model.predict(frame, imgsz=640, conf=CONF, verbose=False)[0]
    pts: list[tuple[float, float]] = []
    for box in res.boxes:
        if int(box.cls.item()) != player_cls:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return pts


def score_frame(
    dets: list[tuple[float, float]],
    gts: list[tuple[float, float]],
) -> tuple[int, int, int, list[float]]:
    """Greedy nearest-GT matching within MATCH_THRESHOLD. Returns tp, fp, fn, errors."""
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    errors: list[float] = []
    for dx, dy in dets:
        best_dist = float("inf")
        best_gt = -1
        for gi, (gx, gy) in enumerate(gts):
            if gi in matched_gt:
                continue
            d = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
            if d < best_dist and d < MATCH_THRESHOLD:
                best_dist = d
                best_gt = gi
        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp += 1
            errors.append(best_dist)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, errors


def evaluate(name: str, model: YOLO, player_cls: int, gt: dict) -> dict:
    print(f"[{name}] player_cls={player_cls}  conf>{CONF}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    tp = fp = fn = 0
    all_err: list[float] = []
    for frame_idx in sorted(gt):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  [warn] cannot read frame {frame_idx}")
            continue
        dets = detect_centroids(model, frame, player_cls)
        f_tp, f_fp, f_fn, errs = score_frame(dets, gt[frame_idx])
        tp += f_tp
        fp += f_fp
        fn += f_fn
        all_err.extend(errs)
    cap.release()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mae = float(np.mean(all_err)) if all_err else 0.0
    rmse = float(np.sqrt(np.mean(np.square(all_err)))) if all_err else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mae_pixels": round(mae, 2),
        "rmse_pixels": round(rmse, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-weights", type=str, required=True)
    ap.add_argument("--baseline-weights", type=str, default=str(REPO / "weights" / "yolov8n.pt"))
    ap.add_argument("--out", type=str, default=str(REPO / "benchmarks" / "finetune_gt_reconcile.json"))
    args = ap.parse_args()

    assert GT_PATH.exists(), f"missing {GT_PATH}"
    assert VIDEO_PATH.exists(), f"missing {VIDEO_PATH}"
    gt = load_gt()
    total_gt_players = sum(len(v) for v in gt.values())
    print(f"GT: {len(gt)} frames, {total_gt_players} player annotations\n")

    baseline = YOLO(args.baseline_weights)
    ft = YOLO(args.ft_weights)

    res = {
        "video": str(VIDEO_PATH),
        "ground_truth_file": str(GT_PATH),
        "gt_frames": len(gt),
        "gt_player_annotations": total_gt_players,
        "metric": "point-distance F1 (centroid -> nearest GT within 50px), conf>0.5",
        "baseline_yolov8n_coco": evaluate("baseline", baseline, 0, gt),
        "finetuned_yolov8n": evaluate("finetuned", ft, 0, gt),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n=== HUMAN-GT RECONCILIATION ===")
    for tag in ("baseline_yolov8n_coco", "finetuned_yolov8n"):
        m = res[tag]
        print(
            f"{tag:24s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1_score']:.3f}  MAE={m['mae_pixels']:.1f}px  "
            f"(TP={m['true_positives']} FP={m['false_positives']} FN={m['false_negatives']})"
        )
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
