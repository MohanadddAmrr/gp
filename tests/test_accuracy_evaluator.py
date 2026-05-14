"""Known-answer tests for services.accuracy_evaluator (P13).

The evaluator's metrics (P/R/F1, mAP@50, ID-switches, IDF1) are hand-rolled, so
each is exercised here on tiny synthetic inputs where the correct answer is
computed by hand.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.accuracy_evaluator import (
    compute_detection_metrics,
    compute_id_metrics,
    compute_map,
    evaluate,
)


def _obj(obj_id, cls, bbox, conf=1.0):
    return {"id": obj_id, "class": cls, "bbox": bbox, "conf": conf}


def _tracks(*frames):
    """Build a loaded-style track dict from (frame_idx, [objects]) pairs."""
    by_frame = {idx: objs for idx, objs in frames}
    return {
        "_by_frame": by_frame,
        "frames": [{"frame_idx": i, "objects": o} for i, o in frames],
    }


def test_perfect_match_scores_one():
    objs = [
        _obj(1, "team_a", [0, 0, 10, 10]),
        _obj(2, "team_b", [50, 50, 10, 10]),
        _obj(9, "ball", [20, 20, 4, 4]),
    ]
    gt = _tracks((0, objs), (1, objs))
    pred = _tracks((0, objs), (1, objs))

    det = compute_detection_metrics(gt, pred)
    assert det["precision"] == 1.0
    assert det["recall"] == 1.0
    assert det["f1"] == 1.0
    assert det["false_positives"] == 0
    assert det["false_negatives"] == 0

    assert compute_map(gt, pred)["mAP"] == 1.0

    idm = compute_id_metrics(gt, pred)
    assert idm["id_switches"] == 0
    assert idm["idf1"] == 1.0


def test_missed_detection_lowers_recall():
    full = [_obj(1, "team_a", [0, 0, 10, 10]), _obj(2, "team_b", [50, 50, 10, 10])]
    partial = [_obj(1, "team_a", [0, 0, 10, 10])]
    det = compute_detection_metrics(_tracks((0, full)), _tracks((0, partial)))

    assert det["precision"] == 1.0  # no false positives
    assert det["recall"] == 0.5  # 1 of 2 found
    assert det["false_negatives"] == 1


def test_false_positive_lowers_precision():
    gt = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
    pred = _tracks((0, [
        _obj(1, "team_a", [0, 0, 10, 10]),
        _obj(2, "team_b", [50, 50, 10, 10]),
    ]))
    det = compute_detection_metrics(gt, pred)

    assert det["recall"] == 1.0
    assert det["precision"] == 0.5
    assert det["false_positives"] == 1


def test_low_iou_is_not_a_match():
    gt = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
    pred = _tracks((0, [_obj(1, "team_a", [8, 8, 10, 10])]))  # IoU ~0.02
    det = compute_detection_metrics(gt, pred)

    assert det["true_positives"] == 0
    assert det["false_positives"] == 1
    assert det["false_negatives"] == 1


def test_ignore_class_dropped_from_detection():
    gt = _tracks((0, [
        _obj(1, "team_a", [0, 0, 10, 10]),
        _obj(2, "ignore", [50, 50, 10, 10]),
    ]))
    pred = _tracks((0, [_obj(1, "team_a", [0, 0, 10, 10])]))
    det = compute_detection_metrics(gt, pred)

    # the 'ignore' object must not count as a missed detection
    assert det["false_negatives"] == 0
    assert det["recall"] == 1.0


def test_id_switch_counted():
    box = [0, 0, 10, 10]
    gt = _tracks((0, [_obj(1, "team_a", box)]), (1, [_obj(1, "team_a", box)]))
    # same GT player, predicted ID flips 100 -> 200
    pred = _tracks((0, [_obj(100, "team_a", box)]), (1, [_obj(200, "team_a", box)]))

    assert compute_id_metrics(gt, pred)["id_switches"] == 1


def test_idf1_on_half_swap():
    box = [0, 0, 10, 10]
    gt = _tracks(*[(i, [_obj(1, "team_a", box)]) for i in range(4)])
    pred = _tracks(
        (0, [_obj(100, "team_a", box)]),
        (1, [_obj(100, "team_a", box)]),
        (2, [_obj(200, "team_a", box)]),
        (3, [_obj(200, "team_a", box)]),
    )
    idm = compute_id_metrics(gt, pred)

    assert idm["id_switches"] == 1
    # best assignment links GT 1 to one predicted ID -> IDTP = 2 of 4 frames
    assert idm["idtp"] == 2
    assert idm["idf1"] == 0.5


def test_map_degrades_with_high_conf_false_positive():
    gt = _tracks((0, [
        _obj(1, "team_a", [0, 0, 10, 10]),
        _obj(2, "team_a", [50, 50, 10, 10]),
    ]))
    pred = _tracks((0, [
        _obj(9, "team_a", [200, 200, 10, 10], conf=0.95),  # FP, highest conf
        _obj(1, "team_a", [0, 0, 10, 10], conf=0.80),
        _obj(2, "team_a", [50, 50, 10, 10], conf=0.70),
    ]))
    m_ap = compute_map(gt, pred)["mAP"]

    assert 0.0 < m_ap < 1.0


def test_evaluate_end_to_end(tmp_path):
    objs = [_obj(1, "team_a", [0, 0, 10, 10]), _obj(9, "ball", [20, 20, 4, 4])]
    payload = {
        "video": "x.mp4",
        "fps": 25.0,
        "frame_size": [640, 360],
        "source": "human",
        "frames": [{"frame_idx": 0, "objects": objs}],
    }
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    gt_path.write_text(json.dumps(payload), encoding="utf-8")
    pred_path.write_text(
        json.dumps({**payload, "source": "pipeline"}), encoding="utf-8"
    )

    report = evaluate(gt_path, pred_path)
    assert report["frames_scored"] == 1
    assert report["detection"]["f1"] == 1.0
    assert report["map"]["mAP"] == 1.0
    assert report["identity"]["idf1"] == 1.0
