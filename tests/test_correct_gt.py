"""Tests for the pure geometry / IO logic of scripts.correct_gt.

The OpenCV highgui shell (`_Editor`, `run_editor`) needs a display and is
verified interactively; everything below is the pure logic it is built on, so
each case here has a hand-computed expected answer.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.correct_gt import (
    apply_move,
    apply_resize,
    find_object_at,
    hit_test,
    load_gt,
    next_object_id,
    normalized_box,
    point_in_box,
    save_gt,
)


def test_point_in_box_includes_edges():
    box = [10, 10, 20, 20]
    assert point_in_box(15, 15, box)
    assert point_in_box(10, 10, box)  # corner, inclusive
    assert point_in_box(30, 30, box)  # opposite corner, inclusive
    assert not point_in_box(5, 5, box)
    assert not point_in_box(31, 30, box)


def test_hit_test_corners_then_interior_then_miss():
    box = [10, 10, 20, 20]  # corners at (10,10) (30,10) (10,30) (30,30)
    assert hit_test(10, 10, box) == "nw"
    assert hit_test(30, 10, box) == "ne"
    assert hit_test(10, 30, box) == "sw"
    assert hit_test(30, 30, box) == "se"
    assert hit_test(13, 13, box) == "nw"  # within the 6px handle radius
    assert hit_test(20, 20, box) == "move"  # interior, clear of every corner
    assert hit_test(100, 100, box) is None


def test_apply_move():
    assert apply_move([10, 10, 20, 20], 5, -3) == [15, 7, 20, 20]


def test_apply_resize_each_corner():
    box = [10, 10, 20, 20]
    assert apply_resize(box, "se", 5, 5) == [10, 10, 25, 25]
    assert apply_resize(box, "nw", 5, 5) == [15, 15, 15, 15]
    assert apply_resize(box, "ne", 5, -4) == [10, 6, 25, 24]
    assert apply_resize(box, "sw", -5, 5) == [5, 10, 25, 25]


def test_apply_resize_clamps_to_min_size():
    box = [10, 10, 20, 20]
    resized = apply_resize(box, "se", -100, -100, min_size=2.0)
    assert resized == [10, 10, 2.0, 2.0]


def test_apply_resize_rejects_non_corner():
    with pytest.raises(ValueError):
        apply_resize([10, 10, 20, 20], "move", 1, 1)


def test_normalized_box_from_any_corner_order():
    assert normalized_box(30, 40, 10, 20) == [10, 20, 20, 20]
    assert normalized_box(10, 10, 30, 40) == [10, 10, 20, 30]


def test_find_object_at_picks_topmost_on_overlap():
    # boxes large enough to have interior space clear of the 6px handle zones
    objects = [{"bbox": [0, 0, 40, 40]}, {"bbox": [20, 20, 40, 40]}]
    # (30,30) is inside both; the later-drawn box (index 1) is on top
    assert find_object_at(objects, 30, 30) == (1, "move")
    # (10,10) is only inside box 0
    assert find_object_at(objects, 10, 10) == (0, "move")
    # a clear miss
    assert find_object_at(objects, 200, 200) == (None, None)
    # a corner of the topmost box wins over the interior of the lower one
    assert find_object_at(objects, 20, 20) == (1, "nw")


def test_next_object_id():
    payload = {"frames": [
        {"objects": [{"id": 3}, {"id": 7}]},
        {"objects": [{"id": 5}]},
    ]}
    assert next_object_id(payload) == 8
    assert next_object_id({"frames": []}) == 1
    assert next_object_id({}) == 1


def test_save_gt_flips_source_to_human_and_round_trips(tmp_path):
    payload = {
        "video": "x.mp4",
        "frame_size": [640, 360],
        "source": "pseudo",
        "frames": [{"frame_idx": 0, "objects": [
            {"id": 1, "class": "team_a", "bbox": [1, 2, 3, 4], "conf": 0.9},
        ]}],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_gt(path)
    save_gt(loaded, path)
    result = load_gt(path)

    assert result["source"] == "human"
    assert result["frames"] == payload["frames"]
    assert result["video"] == "x.mp4"
