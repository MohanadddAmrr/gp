"""
tools/seif_dryrun_check.py — Day 19 (Seif)

Run this script before every dry-run and before the defense to confirm
ALL of Seif's acceptance criteria are met with zero manual inspection.

Usage:
    python tools/seif_dryrun_check.py

Exits 0 if everything is green. Exits 1 and prints blockers if not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.accuracy_evaluator import compute_mota, compute_id_metrics, evaluate
from services.ground_truth_manager import GroundTruthManager
from services.event_validator import evaluate_events

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

passed = 0
failed = 0
warnings = 0


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  {GREEN}✅ {msg}{RESET}")


def fail(msg: str) -> None:
    global failed
    failed += 1
    print(f"  {RED}❌ BLOCKER: {msg}{RESET}")


def warn(msg: str) -> None:
    global warnings
    warnings += 1
    print(f"  {YELLOW}⚠️  WARNING: {msg}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEIF DRY-RUN CHECK — TactiVision Pro")
print("=" * 60)

# ── Check 1: All code files exist ────────────────────────────────────────────
print("\n[ 1 ] Code files")
code_files = [
    "services/accuracy_evaluator.py",
    "services/ground_truth_manager.py",
    "services/event_validator.py",
    "tools/annotation_helper.py",
    "demo/dashboard_pages/__init__.py",
    "demo/dashboard_pages/accuracy_report.py",
]
for f in code_files:
    p = ROOT / f
    if p.exists():
        ok(f"{f} ({p.stat().st_size // 1024}KB)")
    else:
        fail(f"{f} MISSING")

# ── Check 2: All test files exist with enough tests ──────────────────────────
print("\n[ 2 ] Test files")
test_files = {
    "tests/test_accuracy_evaluator.py": 20,
    "tests/test_ground_truth.py":       20,
    "tests/test_event_validator.py":    20,
    "tests/test_tracker_accuracy.py":   10,
}
for f, min_tests in test_files.items():
    p = ROOT / f
    if not p.exists():
        fail(f"{f} MISSING")
        continue
    n = p.read_text().count("def test_")
    if n >= min_tests:
        ok(f"{f} ({n} tests)")
    else:
        fail(f"{f} only has {n} tests (need >= {min_tests})")

# ── Check 3: AC3 — MOTA/IDF1 computed correctly ──────────────────────────────
print("\n[ 3 ] AC3 — MOTA / IDF1 / ID-switches (Acceptance Criterion 3)")

def _obj(oid, cls, bbox):
    return {"id": oid, "class": cls, "bbox": bbox, "conf": 1.0}


box = [0, 0, 10, 10]
objs = [_obj(1, "team_a", box), _obj(2, "team_b", [50, 50, 10, 10])]
gt_dict   = {"_by_frame": {0: objs, 1: objs}, "fps": 25.0}
pred_dict = {"_by_frame": {0: objs, 1: objs}, "fps": 25.0}

try:
    m = compute_mota(gt_dict, pred_dict)
    if m["mota"] == 1.0:
        ok(f"compute_mota() perfect case → MOTA={m['mota']:.3f} ✓")
    else:
        fail(f"compute_mota() perfect case returned MOTA={m['mota']} (expected 1.0)")
except Exception as e:
    fail(f"compute_mota() crashed: {e}")

try:
    idm = compute_id_metrics(gt_dict, pred_dict)
    if idm["idf1"] == 1.0 and idm["id_switches"] == 0:
        ok(f"compute_id_metrics() perfect case → IDF1={idm['idf1']:.3f} ID-sw={idm['id_switches']} ✓")
    else:
        fail(f"IDF1={idm['idf1']}, ID-switches={idm['id_switches']} (expected 1.0, 0)")
except Exception as e:
    fail(f"compute_id_metrics() crashed: {e}")

# ID-switch detection
gt2   = {"_by_frame": {0: [_obj(1, "team_a", box)], 1: [_obj(1, "team_a", box)]}}
pred2 = {"_by_frame": {0: [_obj(10, "team_a", box)], 1: [_obj(20, "team_a", box)]}}
try:
    idm2 = compute_id_metrics(gt2, pred2)
    if idm2["id_switches"] == 1:
        ok(f"ID-switch detection → {idm2['id_switches']} switch detected ✓")
    else:
        fail(f"Expected 1 ID-switch, got {idm2['id_switches']}")
except Exception as e:
    fail(f"ID-switch test crashed: {e}")

# ── Check 4: AC4 — Ground truth files exist ───────────────────────────────────
print("\n[ 4 ] AC4 — Ground truth data (Acceptance Criterion 4)")
gt_dir = ROOT / "tests" / "ground_truth"
mgr = GroundTruthManager(gt_dir=gt_dir)
clips = mgr.list_clips()

if len(clips) >= 3:
    ok(f"{len(clips)} annotated clips found (need >= 3)")
else:
    fail(f"Only {len(clips)} clips found in {gt_dir} (need >= 3)")

for clip in clips:
    if clip.n_frames_annotated >= 50:
        ok(f"  {clip.stem}: {clip.n_frames_annotated} frames annotated")
    else:
        warn(f"  {clip.stem}: only {clip.n_frames_annotated} frames (recommend >= 50)")

# ── Check 5: Accuracy reports exist ──────────────────────────────────────────
print("\n[ 5 ] Accuracy reports")
rep_dir = ROOT / "tests" / "accuracy_reports"
reports = sorted(rep_dir.glob("accuracy_report_*.json"),
                 key=lambda p: p.stat().st_mtime, reverse=True) if rep_dir.exists() else []

if reports:
    ok(f"{len(reports)} report(s) found, most recent: {reports[0].name}")
    # Load latest and check aggregate keys
    try:
        latest = json.loads(reports[0].read_text())
        agg = latest.get("aggregate", {})
        needed = ["mota", "idf1", "detection_f1", "map", "id_switches_per_minute"]
        missing_keys = [k for k in needed if k not in agg]
        if not missing_keys:
            ok(f"Latest report has all required aggregate keys")
            mota = agg.get("mota", 0)
            idf1 = agg.get("idf1", 0)
            f1   = agg.get("detection_f1", 0)
            print(f"      MOTA={mota:.3f}  IDF1={idf1:.3f}  F1={f1:.3f}")
            if mota >= 0.55:
                ok(f"MOTA={mota:.3f} meets minimum target (>=0.55)")
            else:
                warn(f"MOTA={mota:.3f} below minimum target (0.55)")
            if idf1 >= 0.50:
                ok(f"IDF1={idf1:.3f} meets minimum target (>=0.50)")
            else:
                warn(f"IDF1={idf1:.3f} below minimum target (0.50)")
        else:
            fail(f"Latest report missing keys: {missing_keys}")
    except Exception as e:
        fail(f"Could not parse latest report: {e}")
else:
    fail("No accuracy reports found — run evaluation first")

# ── Check 6: Event GT files exist ────────────────────────────────────────────
print("\n[ 6 ] Event ground truth files")
event_files = list(gt_dir.glob("events_*.json"))
if len(event_files) >= 3:
    ok(f"{len(event_files)} event GT files found")
    for ef in event_files:
        events = json.loads(ef.read_text())
        ok(f"  {ef.name}: {len(events)} events")
else:
    warn(f"Only {len(event_files)} event GT files found (recommend >= 3)")

# ── Check 7: Dashboard tab imports cleanly ───────────────────────────────────
print("\n[ 7 ] Dashboard tab import")
try:
    from demo.dashboard_pages.accuracy_report import (
        run_evaluation_all_clips,
        _aggregate,
        render,
    )
    ok("accuracy_report.py imports cleanly")
    ok("render() function exists")
    ok("run_evaluation_all_clips() function exists")
    ok("_aggregate() function exists")
except Exception as e:
    fail(f"accuracy_report.py import error: {e}")

# ── Check 8: Docs exist ───────────────────────────────────────────────────────
print("\n[ 8 ] Documentation")
docs = {
    "docs/ACCURACY_VALIDATION_GUIDE.md":   200,
    "docs/SEIF_ACCURACY_WALKTHROUGH.md":   50,
    "docs/USER_EVALUATION_QUESTIONNAIRE.md": 50,
    "docs/SEIF_THESIS_SECTIONS.md":        200,
    "tests/ground_truth/README.md":        50,
    "tests/accuracy_reports/README.md":    20,
}
for f, min_lines in docs.items():
    p = ROOT / f
    if not p.exists():
        fail(f"{f} MISSING")
    else:
        lines = len(p.read_text().splitlines())
        if lines >= min_lines:
            ok(f"{f} ({lines} lines)")
        else:
            warn(f"{f} only {lines} lines (recommend >= {min_lines})")

# ── Check 9: event_validator works ───────────────────────────────────────────
print("\n[ 9 ] Event validator")
try:
    gt_events   = [{"type": "pass", "time_sec": 10.0, "confidence": 1.0}]
    pred_events = [{"type": "pass", "time_sec": 10.5, "confidence": 0.9}]
    r = evaluate_events(gt_events, pred_events, tolerance_sec=2.0)
    if r["overall"]["true_positives"] == 1:
        ok("evaluate_events() correctly matches within tolerance")
    else:
        fail(f"evaluate_events() returned TP={r['overall']['true_positives']} (expected 1)")
except Exception as e:
    fail(f"evaluate_events() crashed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULT: {passed} passed  |  {warnings} warnings  |  {failed} blockers")
print("=" * 60)

if failed == 0 and warnings == 0:
    print(f"{GREEN}✅ ALL CHECKS PASSED — ready for dry-run / defense{RESET}")
elif failed == 0:
    print(f"{YELLOW}⚠️  No blockers but {warnings} warning(s) — review before defense{RESET}")
else:
    print(f"{RED}❌ {failed} BLOCKER(S) must be fixed before the defense{RESET}")

sys.exit(0 if failed == 0 else 1)
