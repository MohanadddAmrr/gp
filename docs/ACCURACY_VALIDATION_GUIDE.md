# Accuracy Validation Guide — TactiVision Pro (Task D, Seif)

This guide explains **how we measure, reproduce, and interpret** tracking and
detection accuracy in TactiVision Pro. Every number shown in the Accuracy
Report dashboard tab (D4) is produced by the framework documented here.

---

## 1. Why We Measure

The previous review identified "tracking accuracy was very low" as a concern,
but without a number. This framework turns that qualitative statement into:

- **MOTA** — a single headline tracking score (higher = better)
- **IDF1** — how consistently the same player keeps the same ID
- **ID-switches / 100 frames** — the flickering the doctor saw, quantified
- **mAP@50** — detection quality per model (feeds the Model Comparison tab)
- **Event P/R/F1** — pass/shot/sprint detection accuracy vs human labels

---

## 2. File Locations

| Path | Purpose |
|------|---------|
| `tools/annotation_helper.py` | Keyboard-driven GT annotation tool |
| `tests/ground_truth/` | All human-annotated ground truth JSON files |
| `tests/ground_truth/README.md` | Format spec and annotation conventions |
| `services/accuracy_evaluator.py` | MOTA / IDF1 / mAP@50 computation |
| `services/ground_truth_manager.py` | GT file discovery, loading, export |
| `services/event_validator.py` | Event-level P/R/F1 computation |
| `tests/test_accuracy_evaluator.py` | Known-answer tests for the evaluator |
| `tests/test_ground_truth.py` | Format + manager tests |
| `tests/test_event_validator.py` | Event validator tests |
| `tests/accuracy_reports/` | Timestamped JSON reports (auto-generated) |

---

## 3. Ground Truth Format

All GT files follow the **unified track-file schema** shared with
`services/accuracy_evaluator.py` and `scripts/make_tracks.py`:

```jsonc
{
  "video": "arsenalvsfulham.mp4",
  "fps": 50.0,
  "frame_size": [640, 360],           // inference resolution (640×360)
  "coordinate_space": "inference_640x360",
  "source": "human",                  // "human" | "pseudo" | "pipeline"
  "frames": [
    {
      "frame_idx": 25,                // 0-based frame index in the video
      "objects": [
        {
          "id": 1,                    // stable integer ID across frames
          "class": "team_a",          // see vocabulary below
          "bbox": [x, y, w, h],       // top-left origin, pixels in frame_size space
          "conf": 1.0                 // always 1.0 for human annotations
        }
      ]
    }
  ]
}
```

### Class Vocabulary

| Class | Meaning |
|-------|---------|
| `team_a` | Outfield player, home team |
| `team_b` | Outfield player, away team |
| `referee` | Match official |
| `goalkeeper` | Goalkeeper (either team) |
| `ball` | Football |
| `ignore` | Object to exclude from scoring (crowd, advertising boards) |

**Detection scoring** collapses `team_a`, `team_b`, `referee`, `goalkeeper`
→ `person`. `ignore` objects are dropped entirely.

**Identity scoring** uses only `person`-class objects.

---

## 4. Step-by-Step: Creating Ground Truth

### 4.1 Choose Your Clips

Pick **3 clips** that cover diverse conditions:

| Clip | Recommended type |
|------|-----------------|
| Clip 1 | Tight broadcast camera, good lighting |
| Clip 2 | Wide tactical camera, more players visible |
| Clip 3 | Low-light, rainy, or high-motion sequence |

Each clip should be **~5 minutes**. At 1 frame/second (stride=25 at 25 fps)
that gives ~300 annotated frames per clip — enough for statistically
meaningful MOTA/IDF1 values.

### 4.2 Run the Annotation Tool

```bash
# Annotate the first 5 minutes of a clip (frames 0–7500 at 25 fps)
python tools/annotation_helper.py \
    --video input_videos/arsenalvsfulham.mp4 \
    --stride 25 \
    --end 7500

# Output saved automatically to:
# tests/ground_truth/arsenalvsfulham.json
```

**Keyboard controls inside the tool:**

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next detection in this frame |
| `T` | Cycle class: team_a → team_b → referee → goalkeeper → ball → ignore |
| `A` | Enter add-mode; drag mouse to draw a new bounding box |
| `DEL` or `D` | Delete the selected detection |
| `SPACE` | Commit frame and advance |
| `S` | Save without advancing |
| `Q` | Quit and save |

**YOLO pre-fill:** The tool automatically runs YOLOv8n on each frame and
pre-populates bounding boxes. Your job is to correct classes (all persons
start as `team_a`) and remove any false detections.

**Resume support:** If you quit mid-way, re-running the same command picks up
where you left off. Already-annotated frames are skipped automatically.

**Auto-save:** Every 10 committed frames the file is written to disk.

### 4.3 Annotation Conventions (for consistency)

1. **Every visible player must be annotated** — even those partially occluded
   at the edge of frame.
2. **IDs must be stable across frames** — if player #3 is team_a in frame 25,
   they must be team_a with id=3 in frame 50. Use the same ID the YOLO
   pre-fill assigned when possible.
3. **Goalkeepers** — label as `goalkeeper`, not `team_a` / `team_b`. The
   evaluator handles class collapsing.
4. **Ball** — label only if clearly visible. Do not guess.
5. **`ignore`** — use for advertising boards, spectators on the pitch,
   partially visible players where the bbox would be misleading.
6. **Two annotators** must agree on frame 0 of each clip to align ID
   conventions before continuing independently.

### 4.4 Minimum Coverage Target

| Phase | Target |
|-------|--------|
| End of Day 3 | ≥ 1 clip, ≥ 80% of sampled frames |
| End of Day 13 | ≥ 3 clips, each ≥ 80% |
| Defense | ≥ 3 clips, 100% |

Check coverage at any time:

```python
from services.ground_truth_manager import GroundTruthManager
mgr = GroundTruthManager()
print(mgr.coverage_percent("arsenalvsfulham", total_video_frames=9000))
```

---

## 5. Running the Evaluator

### 5.1 Prerequisites

The pipeline must have been run on the same video first, producing:

```
demo/demo_outputs/<video_stem>/per_frame_tracks.json
```

This file is written automatically by `demo/run_demo.py` (M3, Day 5).

### 5.2 Command-Line Evaluation

```bash
python -m services.accuracy_evaluator \
    --gt  tests/ground_truth/arsenalvsfulham_clip.json \
    --pred demo/demo_outputs/arsenalvsfulham/per_frame_tracks.json \
    --out  tests/accuracy_reports/arsenalvsfulham_$(date +%Y%m%d_%H%M%S).json
```

Example output:

```
video=arsenalvsfulham.mp4  frames_scored=90
detection  P=0.871  R=0.834  F1=0.852  mAP@0.5=0.789
identity   IDF1=0.673  ID-switches=12 (1.45/100 frames)
report written -> tests/accuracy_reports/arsenalvsfulham_20260210_142301.json
```

### 5.3 Evaluating Event Detection

```bash
python -m services.event_validator \
    --gt   tests/ground_truth/events_arsenalvsfulham.json \
    --pred demo/demo_outputs/arsenalvsfulham/events.json \
    --tolerance 2.0
```

### 5.4 Batch Evaluation (All Clips)

From the Accuracy Report dashboard tab, click **"Run evaluation now"** to
evaluate all clips with existing GT and save a timestamped report.

Or from the command line:

```python
from pathlib import Path
from services.ground_truth_manager import GroundTruthManager
from services.accuracy_evaluator import evaluate

mgr = GroundTruthManager()
for summary in mgr.list_clips():
    pred_path = Path("demo/demo_outputs") / summary.stem / "per_frame_tracks.json"
    if pred_path.exists():
        report = evaluate(summary.gt_path, pred_path)
        print(f"{summary.stem}: MOTA={report['identity']['idf1']:.3f}")
```

---

## 6. Understanding the Metrics

### 6.1 Detection Metrics

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| Precision | TP / (TP + FP) | Of all detections, how many were real objects? |
| Recall | TP / (TP + FN) | Of all real objects, how many did we detect? |
| F1 | 2·P·R / (P+R) | Harmonic mean — overall detection quality |
| mAP@50 | Area under P/R curve at IoU≥0.5 | Standard detection benchmark |

A detection is a **True Positive** (TP) when the predicted bounding box
overlaps the GT box with IoU ≥ 0.50.

### 6.2 Identity / Tracking Metrics

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| IDF1 | 2·IDTP / (2·IDTP + IDFP + IDFN) | How consistently is each player tracked with the same ID? |
| ID-switches | Count of GT→pred ID flips | The flickering the doctor observed |
| ID-sw / 100 fr | ID-sw ÷ matched frames × 100 | Normalised switch rate |

**IDF1** is the primary headline metric for the discussion because it directly
answers "does the same player keep the same ID across the whole clip?"

**ID-switches** are counted per the MOTChallenge definition: a switch is
recorded when a GT track is matched to a different predicted ID than the last
frame it was matched.

### 6.3 Event Metrics

| Metric | Meaning |
|--------|---------|
| Precision | Of all detected events of type T, how many matched a GT event? |
| Recall | Of all GT events of type T, how many were detected? |
| F1 | Harmonic mean of P and R |
| Tolerance | Two events match if \|time_A − time_B\| ≤ tolerance_sec (default 2.0 s) |

---

## 7. Targets for the Defense

| Metric | Minimum (pass) | Stretch (aim for) |
|--------|---------------|-------------------|
| MOTA | 0.55 | 0.70 |
| IDF1 | 0.50 | 0.65 |
| ID-switches / 100 frames | < 5.0 | < 2.0 |
| mAP@50 (best model) | 0.65 | 0.75 |
| Detection F1 | 0.75 | 0.85 |

These targets are **not aspirational** — they are the bar the doctor will ask
about. Every number must be reproducible by running:

```bash
python -m services.accuracy_evaluator --gt <GT_PATH> --pred <PRED_PATH>
```

---

## 8. Report Structure

Each timestamped JSON report (`tests/accuracy_reports/<ts>.json`) contains:

```jsonc
{
  "ground_truth": "tests/ground_truth/arsenalvsfulham_clip.json",
  "predictions": "demo/demo_outputs/arsenalvsfulham/per_frame_tracks.json",
  "video": "arsenalvsfulham.mp4",
  "frames_scored": 90,
  "detection": { "precision": ..., "recall": ..., "f1": ..., ... },
  "map": { "mAP": ..., "per_class_ap": { "person": ..., "ball": ... } },
  "identity": { "id_switches": ..., "idf1": ..., "idtp": ..., ... }
}
```

The dashboard tab (D4) loads the most recent report by default.

---

## 9. Troubleshooting

### "per_frame_tracks.json not found"
Run `demo/run_demo.py` on the video first. The file is written to
`demo/demo_outputs/<video_stem>/per_frame_tracks.json` automatically.

### "Ground truth file not found"
Run `tools/annotation_helper.py` on the video and commit the resulting JSON
to `tests/ground_truth/`.

### Very low IDF1 (< 0.30)
- Check that GT IDs are stable across frames (common annotation mistake).
- Verify that `reid_enabled: true` is set in `config.yaml`.
- Confirm the tracker is set to `botsort` (lower ID-switches than bytetrack
  on our footage — see Gate 1 review A/B numbers).

### High precision, low recall
The model misses real objects. Lower `confidence_threshold` in `config.yaml`
under `detection`.

### High recall, low precision
Too many false detections. Raise `confidence_threshold`.

---

## 10. Reproducibility Checklist (Before the Discussion)

Run this sequence the morning of the defense to confirm numbers are live:

```bash
# 1. Verify GT files exist
python -c "from services.ground_truth_manager import GroundTruthManager; [print(s.stem, s.n_frames_annotated) for s in GroundTruthManager().list_clips()]"

# 2. Run evaluator on all clips
for stem in arsenalvsfulham_clip liverpoolvscity_ground_truth; do
  python -m services.accuracy_evaluator \
    --gt  tests/ground_truth/${stem}.json \
    --pred demo/demo_outputs/${stem}/per_frame_tracks.json
done

# 3. Run all tests (must be green)
python -m pytest tests/test_accuracy_evaluator.py tests/test_ground_truth.py tests/test_event_validator.py -v
```

All three commands must succeed for the Accuracy Report tab to show live
numbers during the demo.
