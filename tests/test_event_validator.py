"""
tests/test_event_validator.py — Task D2 (Seif)

Tests for services/event_validator.py
Covers: precision, recall, F1 per event type, tolerance window, tie-breaking,
        edge cases (empty lists, unknown types, zero tolerance).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.event_validator import evaluate_events


# ── helpers ──────────────────────────────────────────────────────────────────

def _ev(etype: str, time_sec: float, confidence: float = 1.0,
        team: str = None, player_id: int = None) -> dict:
    """Build a minimal event dict."""
    ev = {"type": etype, "time_sec": time_sec, "confidence": confidence}
    if team is not None:
        ev["team"] = team
    if player_id is not None:
        ev["player_id"] = player_id
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerfectMatch:

    def test_perfect_single_event(self):
        """Identical gt and pred lists → P=R=F1=1.0."""
        gt   = [_ev("pass", 10.0)]
        pred = [_ev("pass", 10.0)]
        r = evaluate_events(gt, pred)
        ov = r["overall"]
        assert ov["precision"] == 1.0
        assert ov["recall"]    == 1.0
        assert ov["f1"]        == 1.0
        assert ov["true_positives"]  == 1
        assert ov["false_positives"] == 0
        assert ov["false_negatives"] == 0

    def test_perfect_multiple_events_same_type(self):
        """Three passes, all matched perfectly → P=R=1.0."""
        gt   = [_ev("pass", t) for t in [5.0, 15.0, 30.0]]
        pred = [_ev("pass", t) for t in [5.0, 15.0, 30.0]]
        r = evaluate_events(gt, pred)
        assert r["overall"]["f1"] == 1.0
        assert r["per_type"]["pass"]["true_positives"] == 3

    def test_perfect_multiple_event_types(self):
        """Multiple event types all perfectly matched."""
        gt   = [_ev("pass", 5.0), _ev("shot", 20.0), _ev("sprint", 40.0)]
        pred = [_ev("pass", 5.0), _ev("shot", 20.0), _ev("sprint", 40.0)]
        r = evaluate_events(gt, pred)
        assert r["overall"]["f1"] == 1.0
        for etype in ("pass", "shot", "sprint"):
            assert r["per_type"][etype]["f1"] == 1.0


class TestNoMatch:

    def test_completely_wrong_times_no_match(self):
        """Prediction far outside tolerance → all FP + FN."""
        gt   = [_ev("pass", 5.0)]
        pred = [_ev("pass", 99.0)]
        r = evaluate_events(gt, pred, tolerance_sec=2.0)
        ov = r["overall"]
        assert ov["true_positives"]  == 0
        assert ov["false_positives"] == 1
        assert ov["false_negatives"] == 1
        assert ov["precision"] == 0.0
        assert ov["recall"]    == 0.0

    def test_wrong_event_type_not_matched(self):
        """Pred of type 'shot' cannot match GT of type 'pass'."""
        gt   = [_ev("pass", 10.0)]
        pred = [_ev("shot", 10.0)]
        r = evaluate_events(gt, pred)
        assert r["overall"]["true_positives"] == 0
        assert r["per_type"]["pass"]["false_negatives"] == 1
        assert r["per_type"]["shot"]["false_positives"] == 1


class TestPartialMatch:

    def test_half_matched_recall_half(self):
        """2 GT events, 1 matched → recall=0.5, precision=1.0."""
        gt   = [_ev("pass", 5.0), _ev("pass", 30.0)]
        pred = [_ev("pass", 5.0)]
        r = evaluate_events(gt, pred)
        pt = r["per_type"]["pass"]
        assert pt["recall"]    == 0.5
        assert pt["precision"] == 1.0
        assert pt["false_negatives"] == 1

    def test_extra_predictions_lower_precision(self):
        """1 GT event, 2 preds → precision=0.5, recall=1.0."""
        gt   = [_ev("shot", 20.0)]
        pred = [_ev("shot", 20.0), _ev("shot", 21.0)]
        r = evaluate_events(gt, pred, tolerance_sec=2.0)
        pt = r["per_type"]["shot"]
        assert pt["precision"] == 0.5
        assert pt["recall"]    == 1.0
        assert pt["false_positives"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tolerance window
# ═══════════════════════════════════════════════════════════════════════════════

class TestToleranceWindow:

    def test_within_tolerance_is_match(self):
        """Pred exactly at tolerance boundary → still a match."""
        gt   = [_ev("pass", 10.0)]
        pred = [_ev("pass", 12.0)]   # exactly 2.0 seconds off
        r = evaluate_events(gt, pred, tolerance_sec=2.0)
        assert r["overall"]["true_positives"] == 1

    def test_beyond_tolerance_is_miss(self):
        """Pred one millisecond beyond tolerance → no match."""
        gt   = [_ev("pass", 10.0)]
        pred = [_ev("pass", 12.001)]
        r = evaluate_events(gt, pred, tolerance_sec=2.0)
        assert r["overall"]["true_positives"] == 0

    def test_zero_tolerance_requires_exact_time(self):
        """tolerance_sec=0 → only events at the exact same timestamp match."""
        gt   = [_ev("pass", 10.0), _ev("pass", 20.0)]
        pred = [_ev("pass", 10.0), _ev("pass", 20.001)]  # second slightly off
        r = evaluate_events(gt, pred, tolerance_sec=0.0)
        pt = r["per_type"]["pass"]
        assert pt["true_positives"] == 1
        assert pt["false_positives"] == 1

    def test_negative_tolerance_raises(self):
        """Negative tolerance_sec is invalid and must raise ValueError."""
        with pytest.raises(ValueError):
            evaluate_events([], [], tolerance_sec=-1.0)

    def test_large_tolerance_matches_all(self):
        """With a very large tolerance, all events of the same type match."""
        gt   = [_ev("sprint", t) for t in [1.0, 60.0, 300.0]]
        pred = [_ev("sprint", t) for t in [1.5, 61.0, 299.0]]
        r = evaluate_events(gt, pred, tolerance_sec=999.0)
        assert r["per_type"]["sprint"]["true_positives"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_gt_all_pred_are_fp(self):
        """No GT events → all predictions are false positives."""
        pred = [_ev("pass", 5.0), _ev("shot", 10.0)]
        r = evaluate_events([], pred)
        assert r["overall"]["true_positives"]  == 0
        assert r["overall"]["false_positives"] == 2
        assert r["overall"]["false_negatives"] == 0
        assert r["overall"]["precision"] == 0.0
        # recall is undefined (0 GT) → 0.0
        assert r["overall"]["recall"] == 0.0

    def test_empty_pred_all_gt_are_fn(self):
        """No predictions → all GT events are false negatives."""
        gt = [_ev("pass", 5.0), _ev("shot", 10.0)]
        r = evaluate_events(gt, [])
        assert r["overall"]["true_positives"]  == 0
        assert r["overall"]["false_positives"] == 0
        assert r["overall"]["false_negatives"] == 2
        assert r["overall"]["recall"]    == 0.0
        assert r["overall"]["precision"] == 0.0

    def test_both_empty_returns_zeros(self):
        """Empty inputs → all counts 0, no per_type entries."""
        r = evaluate_events([], [])
        assert r["overall"]["true_positives"]  == 0
        assert r["overall"]["false_positives"] == 0
        assert r["overall"]["false_negatives"] == 0
        assert r["per_type"] == {}

    def test_each_gt_matched_at_most_once(self):
        """Two preds close to the same GT → only one TP, one FP."""
        gt   = [_ev("pass", 10.0)]
        pred = [_ev("pass", 10.0), _ev("pass", 10.5)]
        r = evaluate_events(gt, pred, tolerance_sec=2.0)
        pt = r["per_type"]["pass"]
        assert pt["true_positives"]  == 1
        assert pt["false_positives"] == 1
        assert pt["false_negatives"] == 0

    def test_result_keys_are_present(self):
        """Result dict must always contain per_type, overall, tolerance_sec, counts."""
        r = evaluate_events([_ev("pass", 5.0)], [_ev("pass", 5.0)])
        assert "per_type" in r
        assert "overall" in r
        assert "tolerance_sec" in r
        assert "gt_event_count" in r
        assert "pred_event_count" in r

    def test_overall_counts_sum_across_types(self):
        """Overall TP/FP/FN must equal the sum across all per_type entries."""
        gt   = [_ev("pass", 5.0), _ev("shot", 10.0), _ev("pass", 20.0)]
        pred = [_ev("pass", 5.0), _ev("shot", 12.0), _ev("sprint", 30.0)]
        r = evaluate_events(gt, pred, tolerance_sec=2.0)

        sum_tp = sum(v["true_positives"]  for v in r["per_type"].values())
        sum_fp = sum(v["false_positives"] for v in r["per_type"].values())
        sum_fn = sum(v["false_negatives"] for v in r["per_type"].values())

        assert r["overall"]["true_positives"]  == sum_tp
        assert r["overall"]["false_positives"] == sum_fp
        assert r["overall"]["false_negatives"] == sum_fn

    def test_event_counts_in_result(self):
        """gt_event_count and pred_event_count reflect input list lengths."""
        gt   = [_ev("pass", t) for t in [1.0, 2.0, 3.0]]
        pred = [_ev("pass", t) for t in [1.0, 2.0]]
        r = evaluate_events(gt, pred)
        assert r["gt_event_count"]   == 3
        assert r["pred_event_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence-based tie-breaking
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceTieBreaking:

    def test_high_confidence_pred_wins_tie(self):
        """When two preds could match the same GT, the higher-confidence one wins."""
        gt = [_ev("shot", 10.0)]
        # Both within tolerance; pred_a has higher confidence → it claims the GT
        pred_a = _ev("shot", 10.5, confidence=0.9)
        pred_b = _ev("shot",  9.8, confidence=0.3)
        r = evaluate_events(gt, [pred_a, pred_b], tolerance_sec=2.0)
        pt = r["per_type"]["shot"]
        assert pt["true_positives"]  == 1
        assert pt["false_positives"] == 1

    def test_missing_confidence_defaults_to_one(self):
        """Events without a confidence key are treated as confidence=1.0."""
        gt   = [_ev("pass", 5.0)]
        pred = [{"type": "pass", "time_sec": 5.0}]   # no confidence key
        r = evaluate_events(gt, pred, tolerance_sec=1.0)
        assert r["overall"]["true_positives"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# F1 computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestF1Computation:

    def test_f1_harmonic_mean(self):
        """F1 = 2*P*R/(P+R) must hold for a known case."""
        # GT: 3 passes, pred: 2 correct + 1 wrong-time → TP=2, FP=1, FN=1
        gt   = [_ev("pass", t) for t in [5.0, 15.0, 30.0]]
        pred = [_ev("pass", 5.0), _ev("pass", 15.0), _ev("pass", 99.0)]
        r = evaluate_events(gt, pred, tolerance_sec=1.0)
        pt = r["per_type"]["pass"]
        assert pt["true_positives"]  == 2
        assert pt["false_positives"] == 1
        assert pt["false_negatives"] == 1

        expected_p = 2 / 3
        expected_r = 2 / 3
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert abs(pt["f1"] - expected_f1) < 1e-4

    def test_f1_zero_when_no_matches(self):
        """F1 is 0.0 when nothing matches."""
        gt   = [_ev("pass", 1.0)]
        pred = [_ev("pass", 50.0)]
        r = evaluate_events(gt, pred, tolerance_sec=0.5)
        assert r["overall"]["f1"] == 0.0

    def test_f1_zero_precision_zero_recall_no_crash(self):
        """When both P and R are 0, F1 should be 0.0 without division by zero."""
        r = evaluate_events([_ev("pass", 1.0)], [_ev("shot", 1.0)])
        # pass has FN=1, shot has FP=1 → overall P=0, R=0
        assert r["overall"]["f1"] == 0.0
