# Handshake: Mohanad ↔ Seif — Ground Truth & Per-Frame Tracks

**Coordination point** from §8.3 of the 21-day plan. Two contracts for Seif's
`services/accuracy_evaluator.py` (Day 8 of the 14-day plan) to consume what
my pipeline produces.

## Contract A — Per-Frame Tracks (I produce, Seif consumes)

**Path:** `demo/demo_outputs/<video_stem>/per_frame_tracks.json`

**When:** lands as part of M3 (Re-ID polish, Day 5 of the 14-day plan).

**Shape:**

```jsonc
{
  "video": "test (2).mp4",
  "fps": 25.0,
  "frame_size": [640, 360],
  "tracker": "botsort",                 // or "bytetrack"
  "frames": [
    {
      "frame_idx": 0,
      "tracks": [
        {
          "track_id": 12,
          "bbox": [x, y, w, h],          // pixels, top-left origin
          "class": "person",             // "person" | "ball"
          "conf": 0.87
        }
      ]
    }
  ]
}
```

Same coordinate system as the resized inference frame (640×360, since
`run_demo.py` resizes before tracking). Seif's evaluator must scale to GT
frame size if different.

## Contract B — Ground Truth (Seif produces, I consume in M1 mAP)

**Path:** `tests/ground_truth/<video_stem>.json`

**When:** Seif's D1 (annotation tool) lands first GT clip on Day 5.

**Shape** (matches §7.4 D1 spec from the 21-day plan):

```jsonc
{
  "video": "test (2).mp4",
  "fps": 25.0,
  "frame_size": [w, h],
  "frames": [
    {
      "frame_idx": 25,
      "objects": [
        {"id": 1, "class": "team_a", "bbox": [x, y, w, h]}
      ]
    }
  ]
}
```

Class vocabulary: `team_a | team_b | referee | goalkeeper | ball | ignore`.

## Open questions (resolve in 5-min sync at Day 6 standup)

1. **Coordinate system for GT.** Original video resolution or 640×360 inference
   frame? My vote: original video resolution; my evaluator scales bboxes.
2. **`ignore` regions.** Are these per-frame polygons (crowd, scoreboards) or
   just ignored bboxes around them? Affects how `accuracy_evaluator` filters
   FPs.
3. **Sampling cadence.** Seif's plan is 1 frame/sec (25 fps → frame_idx
   multiples of 25). Confirm — my `per_frame_tracks.json` will dump every
   processed frame; Seif's evaluator only consumes the ones with GT.
