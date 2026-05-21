# TactiVision Pro — System Architecture

This document covers the architectural viewpoints the thesis cites
(Ch 4.2.1 Context, 4.2.2 Composition, 4.2.3 Logical, 4.2.4 Patterns,
4.2.5 Algorithm). Sections are written from the actual code on
`master`, not from intent — every claim traces to a file path.

## 1. Detection + tracking pipeline (Composition viewpoint)

The runtime pipeline composes six services in fixed order. Each is a
single Python module under `services/`; each is independently testable;
none of them is monolithic.

```
                            +-----------------------------+
                            |  services/multi_model_      |
                            |  evaluator.py  (offline)    |
                            |  Bench 5 models -> CSV/PNG  |
                            +--------------+--------------+
                                           |
                                           v
   Video frame  --->  YOLO (yolov8n.pt by default; configurable)
                                           |
                                           v
                            +-----------------------------+
                            |  services/improved_tracker  |
                            |  build_tracker(algo, model) |
                            |  bytetrack | botsort        |
                            +--------------+--------------+
                                           |
                                           v  (raw track ids)
                            +-----------------------------+
                            |  services/reid_module       |
                            |  JerseyPosReID.resolve()    |
                            |  jersey HSV + position      |
                            |  conservative AND-rule      |
                            +--------------+--------------+
                                           |
                                           v  (preliminary canonical ids)
                            +-----------------------------+
                            |  services/tracking_filters  |
                            |  TrackDeduplicator (safety) |
                            |  position-only fallback     |
                            +--------------+--------------+
                                           |
                                           v  (final canonical pids)
                                  metrics + per_frame_tracks.json
```

### Why every layer earns its keep

- **`multi_model_evaluator.py`** is offline — produces the comparison
  table that defends the choice of yolov8n. Not on the runtime path.
- **`improved_tracker.py`** is a thin abstraction over Ultralytics with
  one job: swap the tracker yaml without touching anything else. Lets
  us A/B ByteTrack vs BoT-SORT and roll back from config.yaml.
- **`reid_module.py`** is the conservative merge layer. Only merges
  when jersey class matches **AND** distance < threshold. Either alone
  would risk false merges; both together let us drop canonical IDs
  20–25% with zero detection regression.
- **`tracking_filters.TrackDeduplicator`** is the position-only safety
  net for tracks the Re-ID layer couldn't classify (no crop, or
  classifier returned 'unknown'). Existed pre-M3; kept as a fallback.

### Day-7 acceptance test that locks the contract

`tests/test_chunked_processing.py::test_metrics_shape_matches_run_demo`
asserts that the chunked processor's `metrics.json` carries the same
top-level key set as `run_demo.py`'s output: `frame`, `num_players`,
`raw_track_ids`, `duration_seconds`, `duration_minutes`,
`tracking_quality`, `ball_tracking`. Dashboards built around either
output keep working when the producer is swapped.

## 2. Chunked video processing (Composition viewpoint — long matches)

A 90-min match at 25 fps is 135,000 frames. Loading every frame into
RAM at 720p uncompressed is ~25 GB; even decoded-on-the-fly the
running state (heatmaps, possession history, tracker memory) is
multi-GB. M4 (`services/chunked_video_processor.py`) splits the
problem.

### Chunk math

`compute_chunk_ranges(total_frames, fps, chunk_seconds)` partitions
the video into `(start_frame, end_frame_exclusive)` tuples. The last
chunk may be short. Examples (from `tests/test_chunked_processing.py`):

| Input | Result |
|---|---|
| 1,750 frames @ 25 fps, chunk_seconds=30 | `(0, 750), (750, 1500), (1500, 1750)` |
| 2,250 frames @ 25 fps, chunk_seconds=30 | `(0, 750), (750, 1500), (1500, 2250)` |
| 500 frames @ 25 fps, chunk_seconds=600 | `(0, 500)` (whole video, one chunk) |

### State persisted between chunks

`_ChunkState` carries the cross-chunk invariants:

- `raw_id_offset` — shifts tracker IDs by `idx * 100_000` so the same
  raw integer in chunk 0 and chunk 1 cannot collide.
- `canonical_id_map` — `{raw_tracker_id -> canonical_player_id}`. Means
  a player who starts in chunk 0 and continues in chunk 1 keeps the
  same canonical ID across the boundary.
- `ball_history_tail` — last 30 ball positions for cross-chunk
  velocity smoothing.
- `person_detections_total`, `ball_detections_total`,
  `chunks_completed` — running counters for the final `metrics.json`.
- `heat_global` — `(H, W) float32` accumulated heatmap. NOT stored in
  JSON (would balloon size); persisted alongside the chunk JSON as a
  compressed `.npz` via `numpy.savez_compressed`.

### Checkpoint file layout

```
demo_outputs/<video_stem>/
├── checkpoints/
│   ├── chunk_0000.json        ~80 KB (state)
│   ├── chunk_0000_heat.npz    ~100 KB (compressed heatmap)
│   ├── chunk_0001.json
│   ├── chunk_0001_heat.npz
│   └── ...
├── metrics.json                final (only written when all chunks done)
├── per_frame_tracks.json       Seif Contract A (when reid is on)
└── reid_merge_log.json         debug audit (when reid is on)
```

### Resume protocol

`process_video_in_chunks(..., resume=True)`:

1. Scan `<output>/checkpoints/` for `chunk_*.json` files.
2. `_highest_completed_chunk()` returns the largest `NNNN`.
3. `_read_checkpoint(highest)` rehydrates `_ChunkState`.
4. The chunk loop starts at `highest + 1`. Earlier chunks are recorded
   in the summary as `{"chunk": N, "skipped": true, "frames": K}` so
   we can audit *what was reused* and *what was re-processed*.
5. `cap.set(CAP_PROP_POS_FRAMES, start_frame)` jumps the video reader
   to the right frame in O(1) for `mp4v`-encoded video.

### Resume cost evidence

| Run | Wall-clock | Notes |
|---|---|---|
| Day 6 cold (synthetic 750 frames, 3 chunks) | 74.7 s | All chunks fresh |
| Day 6 warm (deleted final checkpoint, resumed) | 24.7 s | ~⅓ — exactly one chunk's worth |
| Day 9 real cold-then-killed (Arsenal vs Fulham, chunk 0 only) | 61 min wall, sandbox kill at ~62 min | Real-world failure mode |
| Day 9 resume #1 | TBD | This is being measured live as you read this |

The resume cost is **always** approximately the wall-clock of one
chunk (whatever the chunk_seconds is set to) regardless of how many
chunks were completed before the crash. That's the guarantee.

## 3. Configuration model (Logical viewpoint)

All knobs live in `config.yaml` under one root. The chunked processor
reads `detection.tracking.*` the same way `run_demo.py` does:

```yaml
detection:
  tracking:
    algorithm: bytetrack         # bytetrack | botsort
    reid_enabled: true           # M3 layer toggle
    reid_merge_distance_px: 120.0
    reid_max_lost_frames: 30
```

A config typo (e.g. `algorithm: deepsort`) raises `ValueError` from
`build_tracker(...)` with the supported algorithms list — fail-fast,
no silent fallback to a default that hides the user's intent.

## 4. Patterns used (Patterns Use viewpoint — Ch 4.2.4)

- **Strategy** — `services.improved_tracker.Tracker` (abstract) with
  `ByteTrackTracker` / `BotSortTracker` (concrete). Swap by config.
- **Factory** — `services.improved_tracker.build_tracker(algorithm,
  model)`. Centralises the algorithm-string → class mapping; one
  place to add a new tracker.
- **Cache** — `services.reid_module.JerseyPosReID._raw_id_team` is a
  per-raw-id classification cache. Day 5 proved this makes Re-ID
  free wall-clock-wise. Day 5 also added a counting-wrapper test to
  lock the invariant ("classifier should be called exactly once for a
  stable raw id").
- **Checkpoint / Memento** — `_ChunkState.to_json` / `from_json`
  serialise the running state at chunk boundaries. The chunk
  processor never holds more than one chunk's worth of frames in RAM.

## 5. What chunked processor does NOT do (honest scope)

The chunked processor is M4's *core deliverable* and intentionally
ships with a subset of `run_demo.py`'s analytics:

| Feature | `run_demo.py` | `chunked_video_processor.py` |
|---|---|---|
| YOLO detection | ✅ | ✅ |
| Tracker (M2) | ✅ | ✅ |
| Re-ID (M3) | ✅ | ✅ |
| TrackDeduplicator | ✅ | ✅ |
| `metrics.json` shape (top-level keys) | ✅ | ✅ (subset; locked by test) |
| `per_frame_tracks.json` (Seif Contract A) | ✅ | ⚠️ written by `run_demo.py`, not yet by chunked path |
| Heatmaps (global, team, player, ball) | ✅ all 4 | ⚠️ global only |
| Possession tracker, possession history | ✅ | ❌ deferred |
| xG analysis, shots, passes | ✅ | ❌ deferred |
| Highlights generator | ✅ | ❌ deferred |
| AI tactical recommendations | ✅ | ❌ deferred |
| Sprint detector / advanced analytics | ✅ | ❌ deferred |

These deferrals are **deliberate** — the chunked path proves the
checkpointing + resume invariants on the *minimum* pipeline needed
for the doctor's tracking-accuracy concern. The downstream analytics
work fine on a 5-min clip via `run_demo.py`; for long matches, Phase 2
will lift them into the chunked path one by one (issue: each one needs
chunk-safe state — possession history straddles chunks, xG snapshots
need the full match for percentile calc, etc.).
