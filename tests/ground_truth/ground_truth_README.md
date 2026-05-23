# Ground Truth — README

This folder contains human-annotated ground truth files used by
`services/accuracy_evaluator.py` to compute tracking and detection metrics
(MOTA, IDF1, ID-switches, mAP@50).

**Owner:** Seif (Task D1 / D3)
**Consumed by:** Mohanad's `services/multi_model_evaluator.py` (mAP@50 column),
accuracy report dashboard tab (D4), and the CLI evaluator.

---

## File Naming Convention

```
<video_stem>.json
```

Examples:
- `arsenalvsfulham_clip.json`  → annotated from `input_videos/arsenalvsfulham.mp4`
- `liverpoolvscity_clip.json`  → annotated from `input_videos/liverpoolvscity.mp4`

The stem must match what `services/ground_truth_manager.py` resolves when
called with `mgr.load("<video_stem>")`.

---

## JSON Schema

Every file in this folder must conform to this schema. Files that do not will
be silently skipped by `GroundTruthManager.list_clips()`.

```jsonc
{
  "video": "arsenalvsfulham.mp4",          // original video filename
  "fps": 50.0,                             // video frame rate
  "frame_size": [640, 360],                // [width, height] in inference space
  "coordinate_space": "inference_640x360", // always this string
  "source": "human",                       // "human" | "pseudo" | "pipeline"
  "frames": [
    {
      "frame_idx": 25,                     // 0-based index in the original video
      "objects": [
        {
          "id": 1,                         // stable integer ID (same player = same id across frames)
          "class": "team_a",              // see class vocabulary below
          "bbox": [x, y, w, h],           // top-left origin, pixels, in frame_size space
          "conf": 1.0                     // always 1.0 for human annotations
        }
      ]
    }
  ]
}
```

### Class Vocabulary

| Class | Use for |
|-------|---------|
| `team_a` | Outfield player, home / left team |
| `team_b` | Outfield player, away / right team |
| `referee` | Match official |
| `goalkeeper` | Goalkeeper (either team) |
| `ball` | Football |
| `ignore` | Exclude from scoring (crowd, partial view, unclear) |

> **Tip:** Use `goalkeeper` not `team_a` / `team_b` for keepers — the
> evaluator handles class collapsing automatically.

---

## Annotation Tool

Use `tools/annotation_helper.py` to create or extend any file in this folder:

```bash
# Annotate at 1 frame/second (stride=25 at 25 fps)
python tools/annotation_helper.py \
    --video input_videos/arsenalvsfulham.mp4 \
    --stride 25

# Annotate a specific range only
python tools/annotation_helper.py \
    --video input_videos/arsenalvsfulham.mp4 \
    --stride 25 --start 0 --end 7500

# Resume an interrupted session (default behaviour — idempotent)
python tools/annotation_helper.py \
    --video input_videos/arsenalvsfulham.mp4 \
    --stride 25
```

The tool writes to `tests/ground_truth/<video_stem>.json` automatically.
Already-annotated frames are skipped on resume.

---

## Annotation Conventions

Follow these rules so that every team member's annotations are consistent
and the evaluator produces meaningful numbers:

1. **All visible players must be labelled** — even those partially occluded
   at the frame edge.
2. **IDs must be stable** — assign the same integer ID to the same physical
   player across all annotated frames. When YOLO pre-fill assigns an ID on
   frame 25, use the same ID on frame 50 for the same player.
3. **Two annotators must agree on frame 0** of each clip before annotating
   independently, to align ID conventions.
4. **Goalkeepers** — always label as `goalkeeper`, never `team_a` / `team_b`.
5. **Ball** — label only when the ball is clearly visible and unambiguous.
6. **`ignore`** — use liberally for: advertising boards, spectators on the
   pitch, partially-visible players where the bbox would mislead the
   evaluator.
7. **Bounding boxes** — draw the box around the full player torso from head
   to feet. Do not box only the head or legs.

---

## Coverage Targets

| Milestone | Requirement |
|-----------|-------------|
| Gate 1 (Day 7) | ≥ 1 clip at ≥ 80% frame coverage |
| Gate 2 (Day 14) | ≥ 3 clips, each at ≥ 80% |
| Defense | ≥ 3 clips, 100% |

Check coverage:

```python
from services.ground_truth_manager import GroundTruthManager
mgr = GroundTruthManager()
for s in mgr.list_clips():
    print(s.stem, mgr.coverage_percent(s.stem, total_video_frames=9000))
```

---

## Current Files

| File | Clip | Frames | Source | Coverage |
|------|------|--------|--------|----------|
| `arsenalvsfulham_clip.json` | arsenalvsfulham.mp4 | 90 | pipeline (to be human-corrected) | ~5 min |
| `liverpoolvscity_ground_truth.json` | liverpoolvscity.mp4 | 20 | human (legacy format) | ~1 min |

> **Note:** `liverpoolvscity_ground_truth.json` uses the old format (list of
> player_positions dicts). It is kept for reference but is **not** compatible
> with `accuracy_evaluator.py`. Use the unified schema above for all new GT.

---

## Running the Evaluator Against a GT File

```bash
# Prerequisite: run_demo.py must have processed the same video first
# (produces demo/demo_outputs/<stem>/per_frame_tracks.json)

python -m services.accuracy_evaluator \
    --gt  tests/ground_truth/arsenalvsfulham_clip.json \
    --pred demo/demo_outputs/arsenalvsfulham/per_frame_tracks.json
```

See `docs/ACCURACY_VALIDATION_GUIDE.md` for full instructions.
