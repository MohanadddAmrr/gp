"""
Accuracy evaluator (P13) — scores a pipeline run against ground truth.

Produces the two metric families the project needs to *prove* improvements:

- Detection (for model fine-tuning, P18): precision / recall / F1 and mAP@50.
- Identity  (for Re-ID work, P10):        ID-switches and IDF1.

Both ground truth and predictions use one unified schema (see below). The
pseudo-label generator (`scripts/make_tracks.py`) emits this exact shape, so a
draft GT can be produced from the current pipeline and then hand-corrected.

Unified track-file schema
-------------------------
```jsonc
{
  "video": "arsenalvsfulham.mp4",
  "fps": 50.0,
  "frame_size": [640, 360],          // inference resolution; all bboxes here
  "coordinate_space": "inference_640x360",
  "source": "pipeline",              // "pipeline" | "pseudo" | "human"
  "frames": [
    {
      "frame_idx": 0,
      "objects": [
        // bbox = [x, y, w, h], top-left origin, pixels in frame_size space
        {"id": 12, "class": "team_a", "bbox": [x, y, w, h], "conf": 0.87}
      ]
    }
  ]
}
```
Class vocabulary: team_a | team_b | referee | goalkeeper | ball | ignore.
For *detection* metrics the four player classes collapse to "person"; "ball"
stays "ball"; "ignore" objects are dropped. For *identity* metrics only
"person" objects are considered (the ball has no meaningful identity).

Evaluation is anchored on GT frames: only `frame_idx` values present in the
ground-truth file are scored, so a sparse GT (e.g. 1 frame/sec) works fine.

CLI
---
    python -m services.accuracy_evaluator --gt GT.json --pred PRED.json [--out REPORT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

_PLAYER_CLASSES = {"team_a", "team_b", "referee", "goalkeeper"}
_IOU_THRESHOLD = 0.5


# --- schema loading -------------------------------------------------------


def load_tracks(path: Path) -> dict:
    """Load a unified track file and index its frames by `frame_idx`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict]] = {}
    for frame in payload.get("frames", []):
        by_frame[int(frame["frame_idx"])] = frame.get("objects", [])
    payload["_by_frame"] = by_frame
    return payload


def _detection_class(class_str: str) -> Optional[str]:
    """Collapse the 6-class vocabulary to the detector's view: person / ball.

    Returns None for `ignore` (and anything unrecognised) so those objects are
    dropped from detection scoring.
    """
    if class_str in _PLAYER_CLASSES:
        return "person"
    if class_str == "ball":
        return "ball"
    return None


# --- geometry -------------------------------------------------------------


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes (top-left origin)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = min(ax2, bx2) - max(ax, bx)
    inter_h = min(ay2, by2) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(
    gt_objs: list[dict], pred_objs: list[dict], iou_thr: float
) -> list[tuple[int, int, float]]:
    """Greedy IoU matching within one frame.

    Returns (gt_idx, pred_idx, iou) triples, highest-IoU pairs first, each GT
    and each prediction used at most once.
    """
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_objs):
        for pi, pred in enumerate(pred_objs):
            iou = _iou(gt["bbox"], pred["bbox"])
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, iou))
    return matches


# --- detection metrics ----------------------------------------------------


def _objects_of_class(objects: list[dict], det_class: str) -> list[dict]:
    return [o for o in objects if _detection_class(o.get("class", "")) == det_class]


def compute_detection_metrics(
    gt: dict, pred: dict, iou_thr: float = _IOU_THRESHOLD
) -> dict:
    """Precision / recall / F1 over all GT frames, person + ball combined."""
    gt_frames = gt["_by_frame"]
    pred_frames = pred["_by_frame"]

    tp = fp = fn = 0
    for frame_idx, gt_objs in gt_frames.items():
        pred_objs = pred_frames.get(frame_idx, [])
        for det_class in ("person", "ball"):
            g = _objects_of_class(gt_objs, det_class)
            p = _objects_of_class(pred_objs, det_class)
            matches = _greedy_match(g, p, iou_thr)
            tp += len(matches)
            fp += len(p) - len(matches)
            fn += len(g) - len(matches)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _average_precision(
    gt: dict, pred: dict, det_class: str, iou_thr: float
) -> tuple[float, int]:
    """All-point-interpolated AP for one detection class.

    Returns (ap, num_gt). AP is 0.0 when the class has no GT instances.
    """
    gt_frames = gt["_by_frame"]
    pred_frames = pred["_by_frame"]

    num_gt = sum(
        len(_objects_of_class(objs, det_class)) for objs in gt_frames.values()
    )
    if num_gt == 0:
        return 0.0, 0

    # Collect every prediction for this class across GT frames, with its conf.
    scored: list[tuple[float, int, dict]] = []  # (conf, frame_idx, pred_obj)
    for frame_idx in gt_frames:
        for obj in _objects_of_class(pred_frames.get(frame_idx, []), det_class):
            scored.append((float(obj.get("conf", 1.0)), frame_idx, obj))
    if not scored:
        return 0.0, num_gt
    scored.sort(key=lambda t: t[0], reverse=True)

    # Walk predictions high-conf first; a GT box may only be claimed once.
    claimed: dict[int, set[int]] = {}  # frame_idx -> set of gt indices used
    tp = [0] * len(scored)
    fp = [0] * len(scored)
    for rank, (_, frame_idx, pred_obj) in enumerate(scored):
        gt_objs = _objects_of_class(gt_frames.get(frame_idx, []), det_class)
        used = claimed.setdefault(frame_idx, set())
        best_iou, best_gi = 0.0, -1
        for gi, gt_obj in enumerate(gt_objs):
            if gi in used:
                continue
            iou = _iou(gt_obj["bbox"], pred_obj["bbox"])
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            used.add(best_gi)
            tp[rank] = 1
        else:
            fp[rank] = 1

    # Cumulative precision / recall, then all-point interpolation.
    cum_tp = cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for i in range(len(scored)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / num_gt)

    # Make precision monotonically decreasing from the right, then integrate.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(recalls)):
        if recalls[i] != prev_recall:
            ap += (recalls[i] - prev_recall) * precisions[i]
            prev_recall = recalls[i]
    return ap, num_gt


def compute_map(gt: dict, pred: dict, iou_thr: float = _IOU_THRESHOLD) -> dict:
    """mAP@iou_thr over the detection classes that have GT instances."""
    per_class: dict[str, float] = {}
    counts: dict[str, int] = {}
    for det_class in ("person", "ball"):
        ap, num_gt = _average_precision(gt, pred, det_class, iou_thr)
        if num_gt > 0:
            per_class[det_class] = round(ap, 4)
            counts[det_class] = num_gt
    m_ap = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "mAP": round(m_ap, 4),
        "iou_threshold": iou_thr,
        "per_class_ap": per_class,
        "per_class_gt_count": counts,
    }


# --- identity metrics -----------------------------------------------------


def _person_id_matches_per_frame(
    gt: dict, pred: dict, iou_thr: float
) -> dict[int, list[tuple[int, int]]]:
    """Per GT frame, the IoU-matched (gt_id, pred_id) pairs for person objects."""
    gt_frames = gt["_by_frame"]
    pred_frames = pred["_by_frame"]
    out: dict[int, list[tuple[int, int]]] = {}
    for frame_idx, gt_objs in gt_frames.items():
        g = _objects_of_class(gt_objs, "person")
        p = _objects_of_class(pred_frames.get(frame_idx, []), "person")
        pairs: list[tuple[int, int]] = []
        for gi, pi, _ in _greedy_match(g, p, iou_thr):
            pairs.append((int(g[gi]["id"]), int(p[pi]["id"])))
        out[frame_idx] = pairs
    return out


def compute_id_metrics(
    gt: dict, pred: dict, iou_thr: float = _IOU_THRESHOLD
) -> dict:
    """ID-switches and IDF1 for person tracks.

    - ID-switches: per the MOTChallenge definition — counted when a GT track is
      matched to a different predicted ID than the last frame it was matched.
    - IDF1: per-frame IoU>=thr association builds a (gt_id, pred_id)
      co-occurrence table; a global Hungarian assignment maximises IDTP; then
      IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN).
    """
    matches_per_frame = _person_id_matches_per_frame(gt, pred, iou_thr)
    frame_order = sorted(matches_per_frame)

    # --- ID-switches ------------------------------------------------------
    last_pred_for_gt: dict[int, int] = {}
    id_switches = 0
    matched_frame_count = 0
    for frame_idx in frame_order:
        pairs = matches_per_frame[frame_idx]
        if pairs:
            matched_frame_count += 1
        for gt_id, pred_id in pairs:
            prev = last_pred_for_gt.get(gt_id)
            if prev is not None and prev != pred_id:
                id_switches += 1
            last_pred_for_gt[gt_id] = pred_id

    # --- IDF1 -------------------------------------------------------------
    gt_frames = gt["_by_frame"]
    pred_frames = pred["_by_frame"]
    gt_track_frames: dict[int, int] = {}
    for frame_idx in frame_order:
        for obj in _objects_of_class(gt_frames[frame_idx], "person"):
            gt_track_frames[int(obj["id"])] = gt_track_frames.get(int(obj["id"]), 0) + 1
    pred_track_frames: dict[int, int] = {}
    for frame_idx in frame_order:
        for obj in _objects_of_class(pred_frames.get(frame_idx, []), "person"):
            pred_track_frames[int(obj["id"])] = pred_track_frames.get(int(obj["id"]), 0) + 1

    cooccur: dict[tuple[int, int], int] = {}
    for frame_idx in frame_order:
        for gt_id, pred_id in matches_per_frame[frame_idx]:
            cooccur[(gt_id, pred_id)] = cooccur.get((gt_id, pred_id), 0) + 1

    total_gt = sum(gt_track_frames.values())
    total_pred = sum(pred_track_frames.values())

    idtp = 0
    if cooccur:
        gt_ids = sorted(gt_track_frames)
        pred_ids = sorted(pred_track_frames)
        gt_index = {g: i for i, g in enumerate(gt_ids)}
        pred_index = {p: i for i, p in enumerate(pred_ids)}

        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            cost = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
            for (gt_id, pred_id), n in cooccur.items():
                cost[gt_index[gt_id], pred_index[pred_id]] = -n  # maximise IDTP
            rows, cols = linear_sum_assignment(cost)
            idtp = int(-cost[rows, cols].sum())
        except ImportError:  # pragma: no cover - scipy is in requirements.txt
            # Fallback: greedy assignment over co-occurrence counts.
            used_gt: set[int] = set()
            used_pred: set[int] = set()
            for (gt_id, pred_id), n in sorted(
                cooccur.items(), key=lambda kv: kv[1], reverse=True
            ):
                if gt_id in used_gt or pred_id in used_pred:
                    continue
                used_gt.add(gt_id)
                used_pred.add(pred_id)
                idtp += n

    idfn = total_gt - idtp
    idfp = total_pred - idtp
    idp = idtp / (idtp + idfp) if (idtp + idfp) else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) else 0.0
    idf1 = (
        2 * idtp / (2 * idtp + idfp + idfn)
        if (2 * idtp + idfp + idfn)
        else 0.0
    )

    return {
        "id_switches": id_switches,
        "id_switch_rate_per_100_frames": (
            round(100.0 * id_switches / matched_frame_count, 2)
            if matched_frame_count
            else 0.0
        ),
        "idf1": round(idf1, 4),
        "idp": round(idp, 4),
        "idr": round(idr, 4),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "gt_track_count": len(gt_track_frames),
        "pred_track_count": len(pred_track_frames),
    }


# --- top-level ------------------------------------------------------------


def evaluate(gt_path: Path, pred_path: Path, iou_thr: float = _IOU_THRESHOLD) -> dict:
    """Score a prediction file against a ground-truth file."""
    gt = load_tracks(gt_path)
    pred = load_tracks(pred_path)

    gt_frame_ids = set(gt["_by_frame"])
    pred_frame_ids = set(pred["_by_frame"])
    return {
        "ground_truth": str(gt_path),
        "predictions": str(pred_path),
        "video": gt.get("video"),
        "gt_source": gt.get("source"),
        "pred_source": pred.get("source"),
        "frames_scored": len(gt_frame_ids),
        "frames_in_pred_missing_from_gt": len(pred_frame_ids - gt_frame_ids),
        "gt_frames_missing_from_pred": len(gt_frame_ids - pred_frame_ids),
        "detection": compute_detection_metrics(gt, pred, iou_thr),
        "map": compute_map(gt, pred, iou_thr),
        "identity": compute_id_metrics(gt, pred, iou_thr),
    }


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.accuracy_evaluator",
        description="Score a pipeline run against ground truth (P13).",
    )
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth track file.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction track file.")
    parser.add_argument("--out", type=Path, help="Optional path to write the JSON report.")
    parser.add_argument(
        "--iou", type=float, default=_IOU_THRESHOLD, help="IoU match threshold."
    )
    args = parser.parse_args(argv)

    report = evaluate(args.gt, args.pred, iou_thr=args.iou)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    det = report["detection"]
    idm = report["identity"]
    print(f"video={report['video']}  frames_scored={report['frames_scored']}")
    print(
        f"detection  P={det['precision']:.3f} R={det['recall']:.3f} "
        f"F1={det['f1']:.3f}  mAP@{args.iou:g}={report['map']['mAP']:.3f}"
    )
    print(
        f"identity   IDF1={idm['idf1']:.3f}  ID-switches={idm['id_switches']} "
        f"({idm['id_switch_rate_per_100_frames']:.2f}/100 frames)"
    )
    if args.out:
        print(f"report written -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
