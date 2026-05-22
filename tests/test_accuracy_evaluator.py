"""
tests/test_accuracy_evaluator.py — Task D2 Day 13 (Seif)

Strengthened test suite for services/accuracy_evaluator.py.
Covers: MOTA computation, MOTP, IDF1, ID-switches, mAP, detection P/R/F1,
        edge cases, and end-to-end evaluate().

All expected values are computed by hand so the tests serve as regression
guards — if the formula changes, these catch it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.accuracy_evaluator import (
    compute_detection_metrics,
    compute_id_metrics,
    compute_map,
    compute_mota,
    evaluate,
    load_tracks,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _obj(obj_id, cls, bbox, conf=1.0):
    return {"id": obj_id, "class": cls, "bbox": bbox, "conf": conf}


def _tracks(*frames, fps=25.0):
    """Build a loaded-style track dict from (frame_idx, [objects]) pairs."""
    by_frame = {idx: objs for idx, objs in frames}
    return {
        "_by_frame": by_frame,
        "frames": [{"frame_idx": i, "objects": o} for i, o in frames],
        "fps": fps,
        "video": "test.mp4",
        "source": "human",
    }


def _write_tracks(tmp_path, stem, tracks_dict):
    """Serialise a tracks dict (without _by_frame) to JSON."""
    payload = {k: v for k, v in tracks_dict.items() if k != "_by_frame"}
    p = tmp_path / f"{stem}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ════════════════════════════════════════════════════════════════════════════
# Detection metrics
# ════════════════════════════════════════════════════════════════════════════

class TestDetectionMetrics:

    def test_perfect_match_all_ones(self):
        objs = [_obj(1, "team_a", [0, 0, 10, 10]),
                _obj(2, "team_b", [50, 50, 10, 10]),
                _obj(9, "ball",   [20, 20,  4,  4])]
        gt = _tracks((0, objs), (1, objs))
        pred = _tracks((0, objs), (1, objs))
        det = compute_detection_metrics(gt, pred)
        assert det["precision"] == 1.0
        assert det["recall"]    == 1.0
        assert det["f1"]        == 1.0
        assert det["false_positives"] == 0
        assert det["false_negatives"] == 0

    def test_missed_detection_lowers_recall(self):
        full    = [_obj(1, "team_a", [0, 0, 10, 10]),
                   _obj(2, "team_b", [50, 50, 10, 10])]
        partial = [_obj(1, "team_a", [0, 0, 10, 10])]
        det = compute_detection_metrics(
            _tracks((0, full)), _tracks((0, partial))
        )
        assert det["precision"] == 1.0
        assert det["recall"]    == 0.5
        assert det["false_negatives"] == 1

    def test_false_positive_lowers_precision(self):
        gt   = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        pred = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10]),
                             _obj(2, "team_b", [50, 50, 10, 10])]))
        det = compute_detection_metrics(gt, pred)
        assert det["recall"]          == 1.0
        assert det["precision"]       == 0.5
        assert det["false_positives"] == 1

    def test_low_iou_is_not_a_match(self):
        gt   = _tracks((0, [_obj(1, "team_a", [0,  0, 10, 10])]))
        pred = _tracks((0, [_obj(1, "team_a", [8,  8, 10, 10])]))  # IoU ~0.02
        det = compute_detection_metrics(gt, pred)
        assert det["true_positives"]  == 0
        assert det["false_positives"] == 1
        assert det["false_negatives"] == 1

    def test_ignore_class_dropped(self):
        gt = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10]),
                           _obj(2, "ignore",  [50, 50, 10, 10])]))
        pred = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        det = compute_detection_metrics(gt, pred)
        assert det["false_negatives"] == 0
        assert det["recall"]          == 1.0

    def test_goalkeeper_counted_as_person(self):
        gt   = _tracks((0, [_obj(1, "goalkeeper", [0, 0, 10, 10])]))
        pred = _tracks((0, [_obj(1, "goalkeeper", [0, 0, 10, 10])]))
        det = compute_detection_metrics(gt, pred)
        assert det["true_positives"] == 1

    def test_empty_predictions(self):
        gt   = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        pred = _tracks((0, []))
        det = compute_detection_metrics(gt, pred)
        assert det["true_positives"]  == 0
        assert det["false_negatives"] == 1
        assert det["precision"]       == 0.0
        assert det["recall"]          == 0.0

    def test_empty_gt_all_fp(self):
        gt   = _tracks((0, []))
        pred = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        det = compute_detection_metrics(gt, pred)
        assert det["false_positives"] == 1
        assert det["true_positives"]  == 0


# ════════════════════════════════════════════════════════════════════════════
# MOTA  (Days 9–10)
# ════════════════════════════════════════════════════════════════════════════

class TestMOTA:

    def test_perfect_tracking_mota_one(self):
        box = [0, 0, 10, 10]
        objs = [_obj(1, "team_a", box), _obj(2, "team_b", [50, 50, 10, 10])]
        gt = _tracks((0, objs), (1, objs))
        m = compute_mota(gt, _tracks((0, objs), (1, objs)))
        assert m["mota"] == 1.0
        assert m["motp"] == 1.0
        assert m["fp"]   == 0
        assert m["fn"]   == 0
        assert m["id_switches"] == 0

    def test_one_fp_reduces_mota(self):
        """1 FP on frame 0, 2 GT objects → MOTA = 1 - 1/4 = 0.75."""
        objs = [_obj(1, "team_a", [0, 0, 10, 10]),
                _obj(2, "team_b", [50, 50, 10, 10])]
        gt   = _tracks((0, objs))
        pred_objs = objs + [_obj(99, "team_a", [300, 300, 10, 10])]
        pred = _tracks((0, pred_objs))
        m = compute_mota(gt, pred)
        assert m["fp"] == 1
        assert m["mota"] == pytest.approx(1 - 1/2, abs=1e-3)

    def test_one_fn_reduces_mota(self):
        """1 FN on frame 0, 2 GT objects → MOTA = 1 - 1/2 = 0.5."""
        objs = [_obj(1, "team_a", [0, 0, 10, 10]),
                _obj(2, "team_b", [50, 50, 10, 10])]
        gt   = _tracks((0, objs))
        pred = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        m = compute_mota(gt, pred)
        assert m["fn"]   == 1
        assert m["mota"] == pytest.approx(0.5, abs=1e-3)

    def test_id_switch_reduces_mota(self):
        """1 IDSW, 2 GT total → MOTA = 1 - 1/2 = 0.5."""
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]),
                        (1, [_obj(1, "team_a", box)]))
        pred = _tracks((0, [_obj(100, "team_a", box)]),
                        (1, [_obj(200, "team_a", box)]))
        m = compute_mota(gt, pred)
        assert m["id_switches"] == 1
        assert m["mota"] == pytest.approx(0.5, abs=1e-3)

    def test_mota_can_be_negative(self):
        """More errors than GT objects → MOTA < 0."""
        gt   = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        # 5 false positives, 0 GT matched
        pred_objs = [_obj(i, "team_a", [200+i*30, 200, 10, 10]) for i in range(5)]
        pred = _tracks((0, pred_objs))
        m = compute_mota(gt, pred)
        assert m["mota"] < 0.0

    def test_mota_formula_components(self):
        """MOTA = 1 - (FP + FN + IDSW) / GT_total."""
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box), _obj(2, "team_b", [50,50,10,10])]),
                        (1, [_obj(1, "team_a", box), _obj(2, "team_b", [50,50,10,10])]))
        # pred: frame 0 perfect, frame 1 has 1 FP
        pred = _tracks(
            (0, [_obj(1, "team_a", box), _obj(2, "team_b", [50,50,10,10])]),
            (1, [_obj(1, "team_a", box), _obj(2, "team_b", [50,50,10,10]),
                 _obj(99, "team_a", [300,300,10,10])]),
        )
        m = compute_mota(gt, pred)
        fp, fn, idsw, gt_total = m["fp"], m["fn"], m["id_switches"], m["gt_total"]
        expected_mota = 1 - (fp + fn + idsw) / gt_total
        assert m["mota"] == pytest.approx(expected_mota, abs=1e-4)

    def test_motp_is_mean_iou(self):
        """MOTP = average IoU of matched pairs; perfect overlap → 1.0."""
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]))
        pred = _tracks((0, [_obj(1, "team_a", box)]))
        m = compute_mota(gt, pred)
        assert m["motp"] == pytest.approx(1.0, abs=1e-4)

    def test_mota_empty_gt_zero(self):
        """No GT objects → MOTA = 0.0 (not an error)."""
        gt   = _tracks((0, []))
        pred = _tracks((0, []))
        m = compute_mota(gt, pred)
        assert m["mota"]     == 0.0
        assert m["gt_total"] == 0

    def test_mota_components_dict_present(self):
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]))
        pred = _tracks((0, [_obj(1, "team_a", box)]))
        m = compute_mota(gt, pred)
        assert "mota_components" in m
        assert "fp_rate"   in m["mota_components"]
        assert "fn_rate"   in m["mota_components"]
        assert "idsw_rate" in m["mota_components"]


# ════════════════════════════════════════════════════════════════════════════
# IDF1 and ID-switches  (Day 10)
# ════════════════════════════════════════════════════════════════════════════

class TestIDMetrics:

    def test_perfect_idf1_is_one(self):
        box  = [0, 0, 10, 10]
        objs = [_obj(1, "team_a", box), _obj(2, "team_b", [50, 50, 10, 10])]
        gt   = _tracks((0, objs), (1, objs))
        idm  = compute_id_metrics(gt, _tracks((0, objs), (1, objs)))
        assert idm["idf1"]        == 1.0
        assert idm["id_switches"] == 0

    def test_id_switch_counted(self):
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]),
                        (1, [_obj(1, "team_a", box)]))
        pred = _tracks((0, [_obj(100, "team_a", box)]),
                        (1, [_obj(200, "team_a", box)]))
        idm = compute_id_metrics(gt, pred)
        assert idm["id_switches"] == 1

    def test_idf1_on_half_swap(self):
        box = [0, 0, 10, 10]
        gt   = _tracks(*[(i, [_obj(1, "team_a", box)]) for i in range(4)])
        pred = _tracks(
            (0, [_obj(100, "team_a", box)]),
            (1, [_obj(100, "team_a", box)]),
            (2, [_obj(200, "team_a", box)]),
            (3, [_obj(200, "team_a", box)]),
        )
        idm = compute_id_metrics(gt, pred)
        assert idm["id_switches"] == 1
        assert idm["idtp"]        == 2
        assert idm["idf1"]        == pytest.approx(0.5, abs=1e-4)

    def test_no_predictions_idf1_zero(self):
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]))
        pred = _tracks((0, []))
        idm = compute_id_metrics(gt, pred)
        assert idm["idf1"]        == 0.0
        assert idm["id_switches"] == 0

    def test_id_switch_rate_per_100_frames(self):
        """3 frames matched, 1 switch → rate = 100/3 ≈ 33.3."""
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]),
                        (1, [_obj(1, "team_a", box)]),
                        (2, [_obj(1, "team_a", box)]))
        pred = _tracks((0, [_obj(10, "team_a", box)]),
                        (1, [_obj(20, "team_a", box)]),  # switch here
                        (2, [_obj(20, "team_a", box)]))
        idm = compute_id_metrics(gt, pred)
        assert idm["id_switches"] == 1
        assert idm["id_switch_rate_per_100_frames"] == pytest.approx(100/3, abs=0.5)

    def test_multiple_id_switches(self):
        box = [0, 0, 10, 10]
        gt   = _tracks(*[(i, [_obj(1, "team_a", box)]) for i in range(6)])
        # Switch every other frame → 3 switches
        pred = _tracks(
            (0, [_obj(10, "team_a", box)]),
            (1, [_obj(20, "team_a", box)]),
            (2, [_obj(10, "team_a", box)]),
            (3, [_obj(20, "team_a", box)]),
            (4, [_obj(10, "team_a", box)]),
            (5, [_obj(20, "team_a", box)]),
        )
        idm = compute_id_metrics(gt, pred)
        assert idm["id_switches"] == 5  # 10→20, 20→10, 10→20, 20→10, 10→20

    def test_ball_not_counted_in_identity_metrics(self):
        """Ball objects must not affect IDF1 or ID-switch counting."""
        box  = [0, 0, 4, 4]
        objs = [_obj(99, "ball", box)]
        gt   = _tracks((0, objs), (1, objs))
        pred = _tracks((0, objs), (1, objs))
        idm  = compute_id_metrics(gt, pred)
        # No person objects → gt_track_count = 0
        assert idm["gt_track_count"] == 0
        assert idm["idf1"]           == 0.0

    def test_result_keys_complete(self):
        box = [0, 0, 10, 10]
        gt   = _tracks((0, [_obj(1, "team_a", box)]))
        pred = _tracks((0, [_obj(1, "team_a", box)]))
        idm  = compute_id_metrics(gt, pred)
        for key in ("id_switches", "id_switch_rate_per_100_frames",
                    "idf1", "idp", "idr", "idtp", "idfp", "idfn",
                    "gt_track_count", "pred_track_count"):
            assert key in idm, f"Missing key: {key}"


# ════════════════════════════════════════════════════════════════════════════
# mAP
# ════════════════════════════════════════════════════════════════════════════

class TestMAP:

    def test_perfect_map_is_one(self):
        objs = [_obj(1, "team_a", [0, 0, 10, 10]),
                _obj(9, "ball",   [20, 20, 4, 4])]
        gt   = _tracks((0, objs), (1, objs))
        pred = _tracks((0, objs), (1, objs))
        assert compute_map(gt, pred)["mAP"] == 1.0

    def test_map_degrades_with_fp(self):
        gt   = _tracks((0, [_obj(1, "team_a", [0,  0, 10, 10]),
                             _obj(2, "team_a", [50, 50, 10, 10])]))
        pred = _tracks((0, [_obj(9,  "team_a", [200, 200, 10, 10], conf=0.95),
                             _obj(1,  "team_a", [0,   0,  10, 10], conf=0.80),
                             _obj(2,  "team_a", [50,  50, 10, 10], conf=0.70)]))
        m_ap = compute_map(gt, pred)["mAP"]
        assert 0.0 < m_ap < 1.0

    def test_map_per_class_present(self):
        objs = [_obj(1, "team_a", [0,  0, 10, 10]),
                _obj(9, "ball",   [20, 20, 4,  4])]
        gt   = _tracks((0, objs))
        pred = _tracks((0, objs))
        result = compute_map(gt, pred)
        assert "per_class_ap" in result
        assert "person" in result["per_class_ap"]
        assert "ball"   in result["per_class_ap"]

    def test_map_zero_when_all_wrong(self):
        gt   = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
        pred = _tracks((0, [_obj(1, "team_a", [500, 500, 10, 10])]))
        assert compute_map(gt, pred)["mAP"] == 0.0


# ════════════════════════════════════════════════════════════════════════════
# End-to-end evaluate()  (Day 11)
# ════════════════════════════════════════════════════════════════════════════

class TestEvaluateEndToEnd:

    def test_evaluate_perfect_returns_ones(self, tmp_path):
        objs = [_obj(1, "team_a", [0, 0, 10, 10]),
                _obj(9, "ball",   [20, 20, 4, 4])]
        payload = {
            "video": "x.mp4", "fps": 25.0,
            "frame_size": [640, 360], "source": "human",
            "frames": [{"frame_idx": 0, "objects": objs}],
        }
        gt_p   = tmp_path / "gt.json"
        pred_p = tmp_path / "pred.json"
        gt_p.write_text(json.dumps(payload), encoding="utf-8")
        pred_p.write_text(json.dumps({**payload, "source": "pipeline"}),
                          encoding="utf-8")
        r = evaluate(gt_p, pred_p)
        assert r["frames_scored"]          == 1
        assert r["detection"]["f1"]        == 1.0
        assert r["map"]["mAP"]             == 1.0
        assert r["identity"]["idf1"]       == 1.0
        assert r["mota"]["mota"]           == 1.0

    def test_evaluate_report_keys_complete(self, tmp_path):
        payload = {
            "video": "x.mp4", "fps": 25.0,
            "frame_size": [640, 360], "source": "human",
            "frames": [{"frame_idx": 0,
                        "objects": [_obj(1, "team_a", [0, 0, 10, 10])]}],
        }
        gt_p   = tmp_path / "gt.json"
        pred_p = tmp_path / "pred.json"
        gt_p.write_text(json.dumps(payload),   encoding="utf-8")
        pred_p.write_text(json.dumps(payload), encoding="utf-8")
        r = evaluate(gt_p, pred_p)
        for key in ("ground_truth", "predictions", "video", "fps",
                    "frames_scored", "detection", "mota", "map", "identity"):
            assert key in r, f"Missing key in evaluate() result: {key}"

    def test_evaluate_mota_key_present(self, tmp_path):
        """MOTA must be a top-level key in evaluate() output (Day 9)."""
        payload = {
            "video": "x.mp4", "fps": 25.0,
            "frame_size": [640, 360], "source": "human",
            "frames": [{"frame_idx": 0,
                        "objects": [_obj(1, "team_a", [0, 0, 10, 10])]}],
        }
        gt_p   = tmp_path / "gt.json"
        pred_p = tmp_path / "pred.json"
        gt_p.write_text(json.dumps(payload),   encoding="utf-8")
        pred_p.write_text(json.dumps(payload), encoding="utf-8")
        r = evaluate(gt_p, pred_p)
        assert "mota" in r
        assert "mota" in r["mota"]
        assert "motp" in r["mota"]

    def test_evaluate_with_id_switch(self, tmp_path):
        box = [0, 0, 10, 10]
        gt_payload = {
            "video": "x.mp4", "fps": 25.0,
            "frame_size": [640, 360], "source": "human",
            "frames": [
                {"frame_idx": 0, "objects": [_obj(1, "team_a", box)]},
                {"frame_idx": 1, "objects": [_obj(1, "team_a", box)]},
            ],
        }
        pred_payload = {
            "video": "x.mp4", "fps": 25.0,
            "frame_size": [640, 360], "source": "pipeline",
            "frames": [
                {"frame_idx": 0, "objects": [_obj(10, "team_a", box)]},
                {"frame_idx": 1, "objects": [_obj(20, "team_a", box)]},
            ],
        }
        gt_p   = tmp_path / "gt.json"
        pred_p = tmp_path / "pred.json"
        gt_p.write_text(json.dumps(gt_payload),   encoding="utf-8")
        pred_p.write_text(json.dumps(pred_payload), encoding="utf-8")
        r = evaluate(gt_p, pred_p)
        assert r["identity"]["id_switches"] == 1
        assert r["mota"]["id_switches"]     == 1
        assert r["mota"]["mota"]            == pytest.approx(0.5, abs=1e-3)

    def test_load_tracks_indexes_by_frame_idx(self, tmp_path):
        payload = {
            "video": "x.mp4", "fps": 25.0,
            "frame_size": [640, 360], "source": "human",
            "frames": [
                {"frame_idx": 100, "objects": [_obj(1, "team_a", [0, 0, 10, 10])]},
                {"frame_idx": 200, "objects": []},
            ],
        }
        p = tmp_path / "tracks.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_tracks(p)
        assert 100 in loaded["_by_frame"]
        assert 200 in loaded["_by_frame"]
        assert len(loaded["_by_frame"][100]) == 1
