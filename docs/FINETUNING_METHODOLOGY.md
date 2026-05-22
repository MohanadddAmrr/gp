# Detection Fine-Tuning Methodology (P18)

The doctor's review explicitly asked why the detection model had not been
fine-tuned for football. P18 answers that with a measured fine-tuning run:
the stock COCO `yolov8n` detector is fine-tuned on football frames and
scored, before and after, against both a held-out split and human ground
truth.

## Approach — self-distillation from RT-DETR

There is no hand-labelled football detection dataset in this project. Rather
than annotate one from scratch, P18 uses **self-distillation**: the strongest
detector already benchmarked here — RT-DETR-L — generates pseudo-labels, and
the fast `yolov8n` is fine-tuned to reproduce them.

This is a deliberate trade-off. RT-DETR-L is the most accurate detector in
`docs/MODEL_COMPARISON_REPORT.md` (highest mean confidence, far the best on
the ball) but the slowest — 63 MB, ~1.4 FPS on CPU. `yolov8n` is 6 MB and
~9 FPS. Distillation transfers RT-DETR's detection ability into the model
the pipeline can actually afford to run.

## Stage 1 — Pseudo-labelled dataset

`scripts/finetune/pseudo_label.py` samples frames uniformly across the three
match clips, runs RT-DETR-L (COCO weights), keeps `person` and `sports ball`
detections, and writes YOLO-format labels remapped to `{0: player, 1: ball}`.

| Parameter | Value |
|---|---|
| Source clips | `test (2).mp4`, `liverpoolvstottenham.mp4`, `newcastlevsmanchesterunited.mp4` |
| Teacher | RT-DETR-L, COCO weights (`weights/rtdetr-l.pt`) |
| Confidence gate | player ≥ 0.50, ball ≥ 0.30 |
| Frames kept | 509 (with ≥ 1 detection) |
| Train / val split | 409 / 100 (80 / 20) |
| Train boxes | 5,570 player + 461 ball |
| Val boxes | 1,250 player + 98 ball |
| Classes | `0: player`, `1: ball` |

## Stage 2 — Training

`scripts/finetune/train.py` fine-tunes `yolov8n` on the pseudo-labelled set.

| Parameter | Value |
|---|---|
| Base weights | `yolov8n.pt` (COCO) |
| Epochs | 30 (early-stopping patience 10; ran full 30) |
| Image size | 640 |
| Batch | 8 |
| Hardware | CPU only — 1.68 h wall clock |
| Output | `runs/finetune/yolov8n_football_v1/weights/best.pt` |

Training-time validation mAP@50 climbed 0.352 (epoch 1) → 0.653 (epoch 30),
still rising at the final epoch.

## Stage 3 — Evaluation

Two independent evaluations, because each answers a different question.

### Eval A — held-out pseudo-label split

`scripts/finetune/eval_compare.py` scores baseline `yolov8n` (COCO, classes
remapped) and the fine-tuned model on the 100-image val split, with an
all-point-interpolation mAP calculator.

| Metric | Baseline yolov8n | Fine-tuned | Δ |
|---|---|---|---|
| mAP@50 | 0.507 | **0.647** | +0.140 (+27.6%) |
| mAP@50-95 | 0.369 | 0.460 | +0.091 |
| Player AP@50 | 0.905 | 0.965 | +0.060 |
| Ball AP@50 | 0.109 | 0.329 | +0.220 (3×) |

**Caveat:** the val labels here are RT-DETR pseudo-labels, and the fine-tuned
model was trained to reproduce them. Eval A therefore measures *agreement
with the teacher*, and the fine-tuned model has a structural advantage. It
confirms the distillation worked; it is not an absolute accuracy claim.

### Eval B — human ground truth (the credible result)

`scripts/finetune/gt_reconcile.py` removes the circularity by scoring both
models against **human-annotated** ground truth
(`tests/ground_truth/liverpoolvscity_ground_truth.json`, 20 frames, 224
player annotations). It uses the project's established player-tracking
metric — detection centroid matched to the nearest GT point within 50 px —
identical to `tests/test_accuracy_measurement.py`.

| Metric | Baseline yolov8n | Fine-tuned | Δ |
|---|---|---|---|
| F1 | 0.797 | **0.888** | +0.091 (+11.4%) |
| Recall | 0.719 | 0.906 | +0.187 |
| Precision | 0.894 | 0.871 | −0.023 |
| Localization MAE | 11.6 px | 10.5 px | −1.1 px |
| Missed players (FN) | 63 | 21 | −42 |

The baseline reproduces the recorded 2026-02-03 accuracy report exactly
(P 0.894 / R 0.719 / F1 0.797), which confirms the comparison is sound. The
fine-tuning gain is almost entirely **recall**: the model finds 42 more of
the 224 real players — small, distant, partially occluded — that stock
`yolov8n` missed, at a small precision cost.

## Honest caveats

- **The "0.950 mAP@50 baseline" is unsubstantiated.** No such figure exists
  anywhere in the repo, git history, or benchmark files. The real on-record
  human-GT baseline is **F1 = 0.797**; that is the number the fine-tuned
  model beats. Eval A's 0.647 mAP@50 is on a different ruler (pseudo-labels)
  and must not be compared to a 0.950 figure.
- **Ball detection is still weak** (AP@50 0.33 even after fine-tuning) — it
  tripled, but the ball remains the weakest subsystem. See P15 (RT-DETR
  ball-only second pass).
- **Small evaluation footprint:** 509 training frames, a single 20-frame
  human-GT clip. Larger, more diverse data would tighten the numbers.
- Eval B is a point-distance F1, not a box mAP@50, because the human GT is
  centroid-only. A box-labelled clip would be needed for a textbook mAP@50.
- CPU-only training; a GPU run could fine-tune the larger `yolov8s`/`m`.

## Reproduction

```bash
python scripts/finetune/pseudo_label.py --per-video 170
python scripts/finetune/train.py --epochs 30 --batch 8
python scripts/finetune/eval_compare.py \
    --ft-weights runs/finetune/yolov8n_football_v1/weights/best.pt
python scripts/finetune/gt_reconcile.py \
    --ft-weights runs/finetune/yolov8n_football_v1/weights/best.pt
```

Results land in `benchmarks/finetune_eval.json` and
`benchmarks/finetune_gt_reconcile.json`.

## Summary for the defense

Fine-tuning distilled RT-DETR — the most accurate detector benchmarked here
(63 MB, ~1.4 FPS) — into `yolov8n` (6 MB, ~9 FPS). On human ground truth it
lifts detection **F1 from 0.797 to 0.888** (+9 points, driven by recall),
recovering most of the accuracy gap at roughly 6× the inference speed. This
is a direct, measured answer to "why didn't you fine-tune?".
