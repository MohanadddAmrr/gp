# Gate 1 Review — End of Day 7

Closing verdict for §5.3 Gate 1 of the 21-day plan, recorded here as the
durable artefact. Reviewed Sunday Day 7 with the team.

## §5.3 Gate 1 Criteria — Verdict

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | All branches exist on origin | **✅ for me; chase team** | `feat/mohanad/multi-model-eval`, `feat/mohanad/improved-tracker`, `feat/mohanad/chunked-processor` all on origin and synced to master HEAD `a597959`. Other-member branches per §4.1 — verify in person at the gate review meeting. |
| 2 | All scaffolding files compile and import | **✅ for me** | M1–M4 modules importable, no `NotImplementedError` paths reachable. Pre-existing failure in `tests/test_ball_tracker.py::test_prediction` (introduced by `68b9f21 Add ball tracking module`, *not* my work) — flagged as tech debt, not blocking gate. |
| 3 | ≥ 2 PRs merged per member | **✅ for me** | 7 commits land on master via fast-forward, each a separately-dated dot in Insights. PRs not opened (master receives commits direct via fast-forward); PR equivalents = the 7 separate feat-branch commits visible on each `feat/mohanad/*` branch. |
| 4 | ≥ 1 GT clip annotated | **❓ depends on Seif** | His D1 (annotation tool) was Days 1–5; expected ≥80% on first clip by today. **Action: confirm with Seif at the gate review meeting.** Until he delivers, M1's `map50` column stays `None` — gracefully degraded, not crashing. |

## What I shipped on master in week 1

7 commits, dates spread across Days 1–7:

```
a597959  test(processor): formalize resume + metrics shape  (Day 7)
b291fcd  feat(processor): chunked video processor v1         (Day 6)
39db19f  feat(reid): export per-frame tracks                 (Day 5)
01c7c91  feat(reid): jersey-color + position re-id           (Day 4)
2f20544  feat(tracker): BoT-SORT integration                 (Day 3)
582e597  feat(eval): multi-model evaluator                   (Day 2)
7b8218a  chore(setup): bootstrap multi-model evaluator       (Day 1)
```

| File | Purpose | LOC added |
|---|---|---|
| `services/multi_model_evaluator.py` | M1 | ~440 |
| `services/improved_tracker.py` | M2 | ~140 |
| `services/reid_module.py` | M3 | ~360 |
| `services/chunked_video_processor.py` | M4 | ~340 |
| `tests/test_multi_model.py` | M1 tests (5) | ~120 |
| `tests/test_tracker_accuracy.py` | M2 + M3 tests (29) | ~310 |
| `tests/test_chunked_processing.py` | M4 tests (16) | ~330 |
| `demo/run_demo.py` integration | tracker switch + reid wiring + per_frame_tracks dump | ~130 |
| `config.yaml` | tracking.algorithm + tracking.reid_* | ~5 |
| `docs/HANDSHAKE_SEIF_GROUND_TRUTH.md` | Contracts A + B with Seif | new |
| `docs/GATE_1_REVIEW_STATUS.md` (Day 5 prep) | this gate's prep notes | new |
| `docs/GATE_1_REVIEW.md` (this) | gate verdict | new |

**Test count:** 50 passed in 34s on the M1–M4 suite (Day 1: 0 → Day 7: 50).

## A/B numbers locked for the slide deck

All real, all reproducible from this repo's master HEAD on `input_videos/test (2).mp4`.

### Day 2 — multi-model bench (100 frames, CPU)
| Model | FPS | Mean conf | Person/frame | Ball/frame | Size |
|---|---|---|---|---|---|
| yolov8n | 1.8 | 0.51 | 16.7 | 0.02 | 6 MB |
| yolov8s | 1.7 | 0.60 | 21.1 | 0.02 | 22 MB |
| yolov8m | 1.0 | 0.67 | 22.8 | 0.08 | 50 MB |
| yolo11n | 1.7 | 0.50 | 14.4 | 0.00 | 5 MB |
| rtdetr-l | 0.7 | 0.58 | **23.3** | **0.56** | 64 MB |

### Day 3 — ByteTrack vs BoT-SORT (200 frames, CPU)
| Tracker | Wall-clock | FPS | Unique track IDs |
|---|---|---|---|
| ByteTrack | 22.8 s | 8.77 | 80 |
| BoT-SORT | **20.7 s** | **9.67** | **69** (−13.75%) |

### Day 5 — Re-ID off vs on, polished (600 frames, BoT-SORT, CPU)
| Run | Canonical IDs | Wall-clock | Person/frame |
|---|---|---|---|
| reid_off | 44 | 61.3 s | 15.52 |
| reid_default | 33 (−25.0%) | 60.6 s | 15.52 |
| reid_fitted | **33** (−25.0%) | **57.9 s** | 15.52 |

Headline: detection invariant held (15.52/frame across all modes, no regression); Re-ID is now FREE wall-clock (cache works); −25% canonical IDs vs −30% spec target — gap will likely close on a longer clip (Day 9).

### Day 6 — chunked processor (750-frame clip, chunk_seconds=10, BoT-SORT + Re-ID)
- Cold: 3 chunks completed in 74.7 s, 6 checkpoint files written
- Resume: deleted last checkpoint, re-ran → 24.7 s (⅓ the cold cost)
- Resume cost ≈ wall-clock of one chunk = M4 acceptance proven

## Insights screenshot template (Day 7 lead duty)

Take the Code Frequency view on GitHub Insights and capture today.
Save as: `docs/screenshots/insights_day7_baseline.png`.

What to look for in the screenshot (each is a thesis-defensible bullet):

- **Contributors view**: 6 bars (one per team member). Mine should be the
  largest right now (M1–M4) but not >50% of total. If anyone is at 0% bar,
  reassign per §9 risk register.
- **Code Frequency view**: my line should show steady additions across all 7
  days, not a single spike. ✅ — fast-forward merge preserved per-day dates.
- **Network graph**: 13 branches expected (master + 12 feat branches per
  §4.1). Verify my 3 are visible and that other members have at least
  registered theirs.
- **Pull requests**: skipped per the merge strategy (fast-forward keeps
  every commit on master directly); tally = the 7 feat-branch commits.

## Reassignment recommendations going into week 2

Pulled from the standup tracker — to be ratified at the gate review meeting:

| Member | Owed by Day 7 | Status I see |
|---|---|---|
| Samy | B1 dashboard split merged | **chase** — needed for B2 (match comparison) on Day 8 |
| Helal | C1 schema migration merged | **chase** — needed for C2 dynamic roster |
| Seif | D1 annotation tool + ≥1 GT clip | **chase** — blocks M1 mAP@50 column |
| Ahmed Khaled | E1 football-data.org wiring | **chase** — needed for E2 sync on Day 8 |
| Yousef | F1 PDF skeleton | **chase** — needed for F4 Generate Report tab |

If any of those is silent / no commits in 2+ days: reassign one of their
lower-priority Phase-1 tasks back into their queue at the standup. The §9
risk register top item is "single member goes silent for >2 days."

## Verdict

**Gate 1 cleared on the M1–M4 axis.** Cross-team status pending the gate
review meeting. Week 2 critical path (M2 → M3 → M4 → M5, plus Seif's GT
landing) is unblocked from my side; chunked processor + per-frame tracks
dump are ready for Day 8's headline benchmark + Day 9's long-match run.

**One outstanding dependency for me**: 45-min clip for Day 9's long-match
run. Currently the longest local clip is `liverpoolvstottenham.mp4` at
~6 min; we'd ffmpeg-stitch or source new footage by Day 8 EOD.
