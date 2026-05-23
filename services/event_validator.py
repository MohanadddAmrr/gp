"""
services/event_validator.py — Task D2 extension (Seif)

Computes precision / recall / F1 per event type by comparing predicted events
against ground-truth events with a configurable time-tolerance window.

Used by:
  - services/accuracy_evaluator.py  (event accuracy section of the report)
  - demo/dashboard_pages/accuracy_report.py  (per-class P/R bar chart)

Event schema (both gt_events and pred_events are lists of dicts):
    {
        "type":       "pass" | "shot" | "sprint" | "dribble" | ...,
        "time_sec":   float,           # event timestamp in seconds
        "team":       "team_a" | "team_b" | None,   # optional
        "player_id":  int | None,                   # optional
        "confidence": float            # 0.0–1.0 (pred only; GT defaults to 1.0)
    }

Matching rule
-------------
A predicted event is a True Positive for event type T if there exists an
unmatched ground-truth event of the same type whose |time_sec difference| ≤
tolerance_sec.  Each GT event may be claimed by at most one prediction
(highest-confidence prediction wins when multiple preds compete for the same GT).

CLI
---
    python -m services.event_validator \\
        --gt tests/ground_truth/events_gt.json \\
        --pred demo_outputs/<stem>/events.json \\
        [--tolerance 2.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


# ── public API ────────────────────────────────────────────────────────────────


def evaluate_events(
    gt_events: list[dict],
    pred_events: list[dict],
    tolerance_sec: float = 2.0,
) -> dict:
    """
    Compute precision, recall, F1 per event type and overall.

    Parameters
    ----------
    gt_events : list[dict]
        Ground-truth events. Each must have at least ``type`` and ``time_sec``.
    pred_events : list[dict]
        Predicted events. Each must have at least ``type`` and ``time_sec``.
        Optional ``confidence`` field (default 1.0) is used to break ties.
    tolerance_sec : float
        Maximum absolute time difference for a prediction to match a GT event.

    Returns
    -------
    dict
        {
            "per_type": {
                "<event_type>": {
                    "precision": float,
                    "recall": float,
                    "f1": float,
                    "true_positives": int,
                    "false_positives": int,
                    "false_negatives": int,
                    "gt_count": int,
                    "pred_count": int,
                },
                ...
            },
            "overall": {
                "precision": float,
                "recall": float,
                "f1": float,
                "true_positives": int,
                "false_positives": int,
                "false_negatives": int,
            },
            "tolerance_sec": float,
            "gt_event_count": int,
            "pred_event_count": int,
        }
    """
    if tolerance_sec < 0:
        raise ValueError(f"tolerance_sec must be >= 0, got {tolerance_sec}")

    # Group by type
    gt_by_type: dict[str, list[dict]]   = _group_by_type(gt_events)
    pred_by_type: dict[str, list[dict]] = _group_by_type(pred_events)

    all_types = sorted(set(gt_by_type) | set(pred_by_type))

    per_type: dict[str, dict] = {}
    total_tp = total_fp = total_fn = 0

    for etype in all_types:
        gt_list   = gt_by_type.get(etype, [])
        pred_list = pred_by_type.get(etype, [])
        tp, fp, fn = _match_events(gt_list, pred_list, tolerance_sec)
        prec, rec, f1 = _prf(tp, fp, fn)
        per_type[etype] = {
            "precision":       round(prec, 4),
            "recall":          round(rec,  4),
            "f1":              round(f1,   4),
            "true_positives":  tp,
            "false_positives": fp,
            "false_negatives": fn,
            "gt_count":        len(gt_list),
            "pred_count":      len(pred_list),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    ov_prec, ov_rec, ov_f1 = _prf(total_tp, total_fp, total_fn)

    return {
        "per_type": per_type,
        "overall": {
            "precision":       round(ov_prec, 4),
            "recall":          round(ov_rec,  4),
            "f1":              round(ov_f1,   4),
            "true_positives":  total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        },
        "tolerance_sec":     tolerance_sec,
        "gt_event_count":    len(gt_events),
        "pred_event_count":  len(pred_events),
    }


# ── matching helpers ──────────────────────────────────────────────────────────


def _group_by_type(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by their ``type`` field."""
    groups: dict[str, list[dict]] = {}
    for ev in events:
        etype = str(ev.get("type", "unknown"))
        groups.setdefault(etype, []).append(ev)
    return groups


def _match_events(
    gt_list: list[dict],
    pred_list: list[dict],
    tolerance_sec: float,
) -> tuple[int, int, int]:
    """
    Greedy matching of predicted events to GT events by time proximity.

    Predictions sorted by confidence (desc) are matched first.
    Each GT event may be claimed at most once.

    Returns
    -------
    (true_positives, false_positives, false_negatives)
    """
    if not gt_list:
        return 0, len(pred_list), 0
    if not pred_list:
        return 0, 0, len(gt_list)

    # Sort predictions high-confidence first
    sorted_preds = sorted(
        pred_list,
        key=lambda e: float(e.get("confidence", 1.0)),
        reverse=True,
    )

    claimed_gt: set[int] = set()
    tp = 0

    for pred in sorted_preds:
        pred_t = float(pred.get("time_sec", 0.0))
        best_idx = -1
        best_dist = float("inf")

        for gi, gt in enumerate(gt_list):
            if gi in claimed_gt:
                continue
            dist = abs(float(gt.get("time_sec", 0.0)) - pred_t)
            if dist <= tolerance_sec and dist < best_dist:
                best_dist = dist
                best_idx = gi

        if best_idx >= 0:
            claimed_gt.add(best_idx)
            tp += 1

    fp = len(pred_list) - tp
    fn = len(gt_list) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m services.event_validator",
        description="Compare predicted events against ground-truth events (Task D2).",
    )
    parser.add_argument("--gt",   required=True, type=Path,
                        help="Ground-truth events JSON (list of event dicts).")
    parser.add_argument("--pred", required=True, type=Path,
                        help="Predicted events JSON (list of event dicts).")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Time window in seconds for matching (default 2.0).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    gt_events   = json.loads(args.gt.read_text(encoding="utf-8"))
    pred_events = json.loads(args.pred.read_text(encoding="utf-8"))

    if not isinstance(gt_events, list):
        print(f"[!] --gt must be a JSON array; got {type(gt_events).__name__}")
        return 1
    if not isinstance(pred_events, list):
        print(f"[!] --pred must be a JSON array; got {type(pred_events).__name__}")
        return 1

    report = evaluate_events(gt_events, pred_events, tolerance_sec=args.tolerance)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ov = report["overall"]
    print(f"Overall — P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
          f"F1={ov['f1']:.3f}  "
          f"TP={ov['true_positives']}  FP={ov['false_positives']}  "
          f"FN={ov['false_negatives']}")
    print(f"tolerance={args.tolerance}s  "
          f"gt_events={report['gt_event_count']}  "
          f"pred_events={report['pred_event_count']}")
    print()
    print(f"{'Type':<20} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 56)
    for etype, m in sorted(report["per_type"].items()):
        print(f"{etype:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['true_positives']:>5} "
              f"{m['false_positives']:>5} {m['false_negatives']:>5}")

    if args.out:
        print(f"\nReport written → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
