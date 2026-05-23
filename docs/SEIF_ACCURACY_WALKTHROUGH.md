# Seif — Accuracy Walkthrough (Discussion Day Script)
## Days 20–21 Prep | Task D, TactiVision Pro

This document is your personal script for the 5-minute accuracy walkthrough
during the defense (§11.1, minutes 6–11 of the demo).

---

## Your Three Sentences (§12.1)

> "I built ground truth annotation for 3 clips and an evaluator that computes
> MOTA, IDF1, and ID-switches per minute.
> Our MOTA is **0.896**, up from no baseline at the last review.
> The numbers are live in the Accuracy Report tab and reproducible with one
> command."

Practise saying these three sentences until they feel natural.
The doctor will then likely ask one of the questions below.

---

## Likely Doctor Questions + Your Answers

### Q1: "What does MOTA actually measure?"

**Your answer:**
MOTA — Multiple Object Tracking Accuracy — combines three error types into
one number:
- **False Positives** (detections with no matching real player)
- **False Negatives** (real players the detector missed)
- **ID-switches** (the flickering where the same player gets a different ID)

Formula: `MOTA = 1 − (FP + FN + ID-switches) / total GT objects`

A score of 1.0 is perfect. Our score of **0.896** means roughly 90% of all
player-frame observations were tracked correctly with a stable identity.

---

### Q2: "How did you create the ground truth?"

**Your answer:**
We used `tools/annotation_helper.py` — a keyboard-driven tool that:
1. Opens a video and samples 1 frame per second
2. Pre-fills bounding boxes using YOLOv8n
3. Lets the annotator correct classes (team_a/team_b/goalkeeper/ball) and
   fix or delete any wrong detections using keyboard shortcuts
4. Saves the result to `tests/ground_truth/<video>.json`

We annotated **3 clips** totalling **450 frames** of ground truth.

---

### Q3: "Why is IDF1 lower than MOTA?"

**Your answer:**
MOTA rewards detecting players in the right place each frame.
IDF1 additionally penalises *inconsistent identity* — even if a player is
always detected in the right position, if their ID keeps changing, IDF1
drops.

Our current IDF1 of 0.501 reflects that BoT-SORT with Re-ID reduces
ID-switches significantly compared to plain ByteTrack, but long occlusions
(players running behind each other) still cause some identity flips. This is
an honest result — we measured it, we know where it comes from, and the
Re-ID module in `services/reid_module.py` is the mitigation.

---

### Q4: "Can the doctor re-run the evaluation himself?"

**Your answer — and then show this command:**
```bash
python -m services.accuracy_evaluator \
    --gt  tests/ground_truth/arsenalvsfulham_clip.json \
    --pred demo/demo_outputs/arsenalvsfulham/per_frame_tracks.json
```
Output appears in ~3 seconds. The Accuracy Report tab also has a
**"Run evaluation now"** button that does the same thing with one click.

---

## Dashboard Walkthrough Steps (Minutes 6–11)

1. Open dashboard → click **"Accuracy Report"** tab
2. Point to the 5 KPI cards: **MOTA 0.896 ✅ · IDF1 0.501 🟡 · F1 0.949 ✅ · mAP 0.630 🟡 · ID-sw/min**
3. Scroll to **Per-Clip Results** table — explain the 3 clips
4. Scroll to **Per-Class Event Accuracy** chart — explain pass/shot/sprint P/R/F1
5. Open **"Reproduce these numbers"** expander — show the CLI command
6. Say: *"The doctor can re-run this command himself and get the same numbers"*

---

## Anti-Patterns to Avoid (§12.3)

- ❌ Do NOT say *"we tried to measure accuracy"* — say *"we measured X and got Y"*
- ❌ Do NOT apologise for IDF1 being below the stretch target — say
  *"IDF1 is 0.501, which meets the minimum bar of 0.50. The stretch target
  of 0.65 requires longer occlusion handling — that's Phase 2."*
- ❌ Do NOT let Mohanad answer the MOTA question — this is your module

---

## Day 21 Freeze Checklist

Before the dry-run #2, confirm:

- [ ] `tests/ground_truth/` has 3 JSON clips
- [ ] `tests/accuracy_reports/` has at least 1 report JSON
- [ ] Dashboard Accuracy Report tab opens without errors
- [ ] KPI cards show numbers (not zeros)
- [ ] Per-clip table shows 3 rows
- [ ] "Reproduce these numbers" expander shows correct CLI commands
- [ ] You can say your 3 sentences from memory
- [ ] You know the answer to all 4 doctor questions above
