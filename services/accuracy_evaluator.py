"""
services/accuracy_evaluator.py — Tasks D2 (Days 9–11, Seif)

Produces the three metric families the project needs to *prove* improvements:

- Detection  (model comparison, §10.2): Precision / Recall / F1 and mAP@50.
- MOTA       (tracking quality,  §10.2): Multiple Object Tracking Accuracy —
             combines false positives, false negatives, and ID switches into
             one headline number.  Formula: MOTA = 1 − (FP+FN+IDSW)/GT
- IDF1       (identity quality,  §10.2): Identity F1 — penalises inconsistent
             IDs across time even when the bounding box is correct.
- ID-switches per minute — the flickering the doctor saw, quantified.

Both ground truth and predictions use one unified schema (see below). The
pseudo-label generator (`scripts/make_tracks.py`) emits this exact shape, so a
draft GT can be produced from the current pipeline and then hand-corrected.

Unified track-file schema
-------------------------
```jsonc
{
  "video": "arsenalvsfulham.mp4",
  "fps": 50.0,
  "frame_size": [640, 360],
  "coordinate_space": "inference_640x360",
  "source": "pipeline",
  "frames": [
    {
      "frame_idx": 0,
      "objects": [
        {"id": 12, "class": "team_a", "bbox": [x, y, w, h], "conf": 0.87}
      ]
    }
  ]
}
```
Class vocabulary: team_a | team_b | referee | goalkeeper | ball | ignore.
For *detection* metrics the four player classes collapse to "person"; "ball"
stays "ball"; "ignore" objects are dropped. For *identity* / MOTA metrics only
"person" objects are considered.

CLI
---
    python -m services.accuracy_evaluator --gt GT.json --pred PRED.json [--out REPORT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

_PLAYER_CLASSES  = {"team_a", "team_b", "referee", "goalkeeper"}
_IOU_THRESHOLD   = 0.5


# ── schema loading ────────────────────────────────────────────────────────────

def load_tracks(path: Path) -> dict:
    """Load a unified track file and index its frames by frame_idx."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict]] = {}
    for frame in payload.get("frames", []):
        by_frame[int(frame["frame_idx"])] = frame.get("objects", [])
    payload["_by_frame"] = by_frame
    return payload


def _detection_class(class_str: str) -> Optional[str]:
    """Collapse 6-class vocabulary → person / ball / None(ignore)."""
    if class_str in _PLAYER_CLASSES:
        return "person"
    if class_str == "ball":
        return "ball"
    return None


# ── geometry ──────────────────────────────────────────────────────────────────

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
    """Greedy IoU matching within one frame → (gt_idx, pred_idx, iou) triples."""
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


# ── detection metrics ─────────────────────────────────────────────────────────

def _objects_of_class(objects: list[dict], det_class: str) -> list[dict]:
    return [o for o in objects if _detection_class(o.get("class", "")) == det_class]


def compute_detection_metrics(
    gt: dict, pred: dict, iou_thr: float = _IOU_THRESHOLD
) -> dict:
    """Precision / Recall / F1 over all GT frames, person + ball combined."""
    gt_frames   = gt["_by_frame"]
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
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision":       round(precision, 4),
        "recall":          round(recall,    4),
        "f1":              round(f1,        4),
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _average_precision(
    gt: dict, pred: dict, det_class: str, iou_thr: float
) -> tuple[float, int]:
    """All-point-interpolated AP for one detection class."""
    gt_frames   = gt["_by_frame"]
    pred_frames = pred["_by_frame"]
    num_gt = sum(len(_objects_of_class(objs, det_class)) for objs in gt_frames.values())
    if num_gt == 0:
        return 0.0, 0
    scored: list[tuple[float, int, dict]] = []
    for frame_idx in gt_frames:
        for obj in _objects_of_class(pred_frames.get(frame_idx, []), det_class):
            scored.append((float(obj.get("conf", 1.0)), frame_idx, obj))
    if not scored:
        return 0.0, num_gt
    scored.sort(key=lambda t: t[0], reverse=True)
    claimed: dict[int, set[int]] = {}
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
    cum_tp = cum_fp = 0
    precisions: list[float] = []
    recalls:    list[float] = []
    for i in range(len(scored)):
        cum_tp += tp[i]; cum_fp += fp[i]
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / num_gt)
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
    """mAP@iou_thr over detection classes that have GT instances."""
    per_class: dict[str, float] = {}
    counts:    dict[str, int]   = {}
    for det_class in ("person", "ball"):
        ap, num_gt = _average_precision(gt, pred, det_class, iou_thr)
        if num_gt > 0:
            per_class[det_class] = round(ap, 4)
            counts[det_class]    = num_gt
    m_ap = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "mAP":               round(m_ap, 4),
        "iou_threshold":     iou_thr,
        "per_class_ap":      per_class,
        "per_class_gt_count": counts,
    }


# ── MOTA ──────────────────────────────────────────────────────────────────────
# Days 9–10: MOTA computation
#
# MOTA = 1 - (FP + FN + IDSW) / GT_total
#
# where GT_total = total number of ground-truth person objects across all frames.
# FP, FN are per-frame detection counts (person class only).
# IDSW = number of identity switches (same as compute_id_metrics).
#
# MOTA ranges from -∞ to 1.0.  Negative values occur when errors exceed GT count.
# A value of 0.70+ is our target (§10.4).

def compute_mota(
    gt: dict, pred: dict, iou_thr: float = _IOU_THRESHOLD
) -> dict:
    """
    MOTA — Multiple Object Tracking Accuracy (person class only).

    Returns
    -------
    dict with keys:
        mota            float   headline tracking score (−∞ to 1.0)
        motp            float   mean IoU of matched pairs (0.0 to 1.0)
        fp              int     false positives (spurious detections)
        fn              int     false negatives (missed detections)
        id_switches     int     identity flips
        gt_total        int     total GT person objects across all frames
        matched         int     total matched pairs (TP for MOTA)
        mota_components dict    {fp_rate, fn_rate, idsw_rate} — each /gt_total
    """
    gt_frames   = gt["_by_frame"]
    pred_frames = pred["_by_frame"]

    # ── per-frame detection counts (person only) ──────────────────────────
    fp = fn = 0
    matched_iou_sum = 0.0
    matched_total   = 0
    # Also track (gt_id → last_pred_id) for ID-switch counting
    last_pred_for_gt: dict[int, int] = {}
    id_switches = 0
    gt_total = 0

    for frame_idx in sorted(gt_frames):
        gt_objs   = _objects_of_class(gt_frames[frame_idx], "person")
        pred_objs = _objects_of_class(pred_frames.get(frame_idx, []), "person")
        gt_total += len(gt_objs)

        matches = _greedy_match(gt_objs, pred_objs, iou_thr)
        matched_gi = {gi for gi, _, _ in matches}
        matched_pi = {pi for _, pi, _ in matches}

        fp += len(pred_objs) - len(matches)
        fn += len(gt_objs)   - len(matches)

        for gi, pi, iou in matches:
            matched_iou_sum += iou
            matched_total   += 1
            gt_id   = int(gt_objs[gi]["id"])
            pred_id = int(pred_objs[pi]["id"])
            prev = last_pred_for_gt.get(gt_id)
            if prev is not None and prev != pred_id:
                id_switches += 1
            last_pred_for_gt[gt_id] = pred_id

    # ── MOTA formula ─────────────────────────────────────────────────────
    if gt_total == 0:
        mota = 0.0
    else:
        mota = 1.0 - (fp + fn + id_switches) / gt_total

    motp = matched_iou_sum / matched_total if matched_total > 0 else 0.0

    fp_rate   = round(fp          / gt_total, 4) if gt_total else 0.0
    fn_rate   = round(fn          / gt_total, 4) if gt_total else 0.0
    idsw_rate = round(id_switches / gt_total, 4) if gt_total else 0.0

    return {
        "mota":        round(mota, 4),
        "motp":        round(motp, 4),
        "fp":          fp,
        "fn":          fn,
        "id_switches": id_switches,
        "gt_total":    gt_total,
        "matched":     matched_total,
        "mota_components": {
            "fp_rate":   fp_rate,
            "fn_rate":   fn_rate,
            "idsw_rate": idsw_rate,
        },
    }


# ── identity metrics (IDF1) ───────────────────────────────────────────────────
# Day 10: IDF1 computation + ID-switch counter

def _person_id_matches_per_frame(
    gt: dict, pred: dict, iou_thr: float
) -> dict[int, list[tuple[int, int]]]:
    """Per GT frame → IoU-matched (gt_id, pred_id) pairs for person objects."""
    gt_frames   = gt["_by_frame"]
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
    """
    ID-switches and IDF1 for person tracks.

    ID-switches: MOTChallenge definition — a switch is counted when a GT track
    is matched to a different predicted ID than the last frame it was matched.

    IDF1: global Hungarian assignment maximises IDTP (identity true positives).
    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)
    """
    matches_per_frame = _person_id_matches_per_frame(gt, pred, iou_thr)
    frame_order = sorted(matches_per_frame)

    # ── ID-switches ───────────────────────────────────────────────────────
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

    # ── IDF1 ─────────────────────────────────────────────────────────────
    gt_frames   = gt["_by_frame"]
    pred_frames = pred["_by_frame"]

    gt_track_frames:   dict[int, int] = {}
    pred_track_frames: dict[int, int] = {}
    for frame_idx in frame_order:
        for obj in _objects_of_class(gt_frames[frame_idx], "person"):
            k = int(obj["id"])
            gt_track_frames[k] = gt_track_frames.get(k, 0) + 1
        for obj in _objects_of_class(pred_frames.get(frame_idx, []), "person"):
            k = int(obj["id"])
            pred_track_frames[k] = pred_track_frames.get(k, 0) + 1

    cooccur: dict[tuple[int, int], int] = {}
    for frame_idx in frame_order:
        for gt_id, pred_id in matches_per_frame[frame_idx]:
            cooccur[(gt_id, pred_id)] = cooccur.get((gt_id, pred_id), 0) + 1

    total_gt   = sum(gt_track_frames.values())
    total_pred = sum(pred_track_frames.values())
    idtp = 0

    if cooccur:
        gt_ids    = sorted(gt_track_frames)
        pred_ids  = sorted(pred_track_frames)
        gt_index  = {g: i for i, g in enumerate(gt_ids)}
        pred_index = {p: i for i, p in enumerate(pred_ids)}
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
            cost = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
            for (gt_id, pred_id), n in cooccur.items():
                cost[gt_index[gt_id], pred_index[pred_id]] = -n
            rows, cols = linear_sum_assignment(cost)
            idtp = int(-cost[rows, cols].sum())
        except ImportError:
            used_gt: set[int]   = set()
            used_pred: set[int] = set()
            for (gt_id, pred_id), n in sorted(
                cooccur.items(), key=lambda kv: kv[1], reverse=True
            ):
                if gt_id in used_gt or pred_id in used_pred:
                    continue
                used_gt.add(gt_id)
                used_pred.add(pred_id)
                idtp += n

    idfn = total_gt   - idtp
    idfp = total_pred - idtp
    idp  = idtp / (idtp + idfp) if (idtp + idfp) else 0.0
    idr  = idtp / (idtp + idfn) if (idtp + idfn) else 0.0
    idf1 = (
        2 * idtp / (2 * idtp + idfp + idfn)
        if (2 * idtp + idfp + idfn)
        else 0.0
    )

    return {
        "id_switches":                   id_switches,
        "id_switch_rate_per_100_frames": (
            round(100.0 * id_switches / matched_frame_count, 2)
            if matched_frame_count else 0.0
        ),
        "idf1":             round(idf1, 4),
        "idp":              round(idp,  4),
        "idr":              round(idr,  4),
        "idtp":             idtp,
        "idfp":             idfp,
        "idfn":             idfn,
        "gt_track_count":   len(gt_track_frames),
        "pred_track_count": len(pred_track_frames),
    }


# ── top-level evaluate ────────────────────────────────────────────────────────
# Day 11: Run accuracy on clip 1 + clip 2 — evaluate() bundles all metrics

def evaluate(gt_path: Path, pred_path: Path, iou_thr: float = _IOU_THRESHOLD) -> dict:
    """
    Score a prediction file against a ground-truth file.

    Returns a single dict containing detection, MOTA, identity (IDF1),
    and mAP metrics — everything the Accuracy Report tab needs.
    """
    gt   = load_tracks(gt_path)
    pred = load_tracks(pred_path)

    gt_frame_ids   = set(gt["_by_frame"])
    pred_frame_ids = set(pred["_by_frame"])

    return {
        "ground_truth":                    str(gt_path),
        "predictions":                     str(pred_path),
        "video":                           gt.get("video"),
        "fps":                             gt.get("fps", 25.0),
        "gt_source":                       gt.get("source"),
        "pred_source":                     pred.get("source"),
        "frames_scored":                   len(gt_frame_ids),
        "frames_in_pred_missing_from_gt":  len(pred_frame_ids - gt_frame_ids),
        "gt_frames_missing_from_pred":     len(gt_frame_ids  - pred_frame_ids),
        "detection": compute_detection_metrics(gt, pred, iou_thr),
        "mota":      compute_mota(gt, pred, iou_thr),
        "map":       compute_map(gt, pred, iou_thr),
        "identity":  compute_id_metrics(gt, pred, iou_thr),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.accuracy_evaluator",
        description=(
            "Score a pipeline run against ground truth.\n"
            "Produces: MOTA, IDF1, ID-switches, Detection F1, mAP@50."
        ),
    )
    parser.add_argument("--gt",  required=True, type=Path,
                        help="Ground-truth track file (unified schema).")
    parser.add_argument("--pred", required=True, type=Path,
                        help="Prediction track file (per_frame_tracks.json).")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Optional path to write the JSON report.")
    parser.add_argument("--iou", type=float, default=_IOU_THRESHOLD,
                        help=f"IoU match threshold (default {_IOU_THRESHOLD}).")
    args = parser.parse_args(argv)

    report = evaluate(args.gt, args.pred, iou_thr=args.iou)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    det  = report["detection"]
    mota = report["mota"]
    idm  = report["identity"]

    print(f"video={report['video']}  frames_scored={report['frames_scored']}")
    print(
        f"MOTA={mota['mota']:.3f}  MOTP={mota['motp']:.3f}  "
        f"FP={mota['fp']}  FN={mota['fn']}  IDSW={mota['id_switches']}"
    )
    print(
        f"IDF1={idm['idf1']:.3f}  "
        f"ID-switches={idm['id_switches']} "
        f"({idm['id_switch_rate_per_100_frames']:.2f}/100 frames)"
    )
    print(
        f"detection  P={det['precision']:.3f}  R={det['recall']:.3f}  "
        f"F1={det['f1']:.3f}  mAP@{args.iou:g}={report['map']['mAP']:.3f}"
    )
    if args.out:
        print(f"report written → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
