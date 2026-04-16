# Accuracy Testing Guide for TactiVision Pro

This guide explains how to measure the actual accuracy of the tracking and detection systems.

## Overview

The accuracy measurement framework compares system outputs against **ground truth** (manually annotated data) to calculate:
- **Precision**: How many detections were correct
- **Recall**: How many actual objects were detected
- **F1 Score**: Harmonic mean of precision and recall
- **MAE/RMSE**: Position error in pixels

## Quick Start

### Step 1: Create Ground Truth Annotations

First, you need to manually annotate some frames from your video:

```bash
python tests/test_accuracy_measurement.py --create-ground-truth --video input_videos/liverpoolvscity.mp4 --frames 100
```

This will:
- Open the video
- Show 100 evenly spaced frames
- Let you click to mark player positions
- Save annotations to `tests/ground_truth/`

**Controls:**
- **Left Click**: Add/select player
- **Right Click**: Remove player
- **'a'**: Switch to Team A
- **'b'**: Switch to Team B
- **'n'**: Next frame
- **'p'**: Previous frame
- **'s'**: Save current frame
- **'q'**: Quit

### Step 2: Measure Accuracy

Once you have ground truth, run the accuracy measurement:

```bash
python tests/test_accuracy_measurement.py --measure --video input_videos/liverpoolvscity.mp4
```

This will:
- Process each annotated frame
- Compare system detections to ground truth
- Calculate precision, recall, F1, MAE, RMSE
- Save results to `tests/accuracy_reports/`

### Step 3: View Results

To see the latest accuracy report:

```bash
python tests/test_accuracy_measurement.py --report
```

## Understanding the Metrics

### Detection Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of all detections, how many were correct? |
| **Recall** | TP / (TP + FN) | Of all actual objects, how many were detected? |
| **F1 Score** | 2 × (P × R) / (P + R) | Overall detection quality |

Where:
- **TP (True Positive)**: Correct detection
- **FP (False Positive)**: False alarm (detected something that wasn't there)
- **FN (False Negative)**: Missed detection

### Tracking Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MAE** | Mean position error (pixels) | < 20 pixels |
| **RMSE** | Root mean square error (pixels) | < 30 pixels |
| **ID Switches** | Number of times tracker swapped IDs | < 5 per 100 frames |

## What Accuracy to Expect

Based on the YOLO model and tracking algorithms used:

| Component | Expected Precision | Expected Recall | Notes |
|-----------|-------------------|-----------------|-------|
| **Player Detection** | 85-95% | 80-90% | Depends on video quality |
| **Player Tracking** | 80-90% | 75-85% | ID switches may occur |
| **Ball Tracking** | 60-75% | 50-70% | Ball is small and fast |
| **Pass Detection** | 70-85% | 65-80% | Depends on possession accuracy |
| **Shot Detection** | 75-90% | 70-85% | Usually clearer than passes |
| **Sprint Detection** | 80-90% | 75-85% | Depends on speed calculation |

## Improving Accuracy

If your accuracy is lower than expected:

### 1. Player Detection/Tracking
- Use higher resolution video
- Adjust YOLO confidence threshold
- Improve lighting conditions
- Use YOLOv8m or YOLOv8l instead of YOLOv8n

### 2. Ball Tracking
- Enable color-based detection
- Adjust ball detection parameters
- Use higher frame rate video

### 3. Pass/Shot Detection
- Improve possession tracking first
- Adjust event detection thresholds
- Review possession change logic

### 4. Sprint Detection
- Verify speed calibration (pixels to meters)
- Check frame rate is correct
- Adjust sprint threshold if needed

## Example Workflow

```bash
# 1. Annotate 100 frames from your test video
python tests/test_accuracy_measurement.py --create-ground-truth \
    --video input_videos/liverpoolvscity.mp4 \
    --frames 100

# 2. Measure accuracy
python tests/test_accuracy_measurement.py --measure \
    --video input_videos/liverpoolvscity.mp4

# 3. View results
python tests/test_accuracy_measurement.py --report

# 4. (Optional) Measure again after making improvements
python tests/test_accuracy_measurement.py --measure \
    --video input_videos/liverpoolvscity.mp4
```

## Tips for Good Ground Truth

1. **Annotate diverse frames**: Include different game situations (attack, defense, set pieces)
2. **Be consistent**: Click on the same body part for each player (e.g., center of torso)
3. **Include edge cases**: Frames with occlusions, distant players, etc.
4. **Annotate all visible players**: Even if they're partially occluded
5. **Use at least 50-100 frames**: More frames = more reliable accuracy metrics

## Troubleshooting

### "Ground truth file not found"
Run `--create-ground-truth` first before `--measure`

### "No accuracy reports found"
Run `--measure` first before `--report`

### Very low accuracy (< 50%)
- Check that video resolution matches ground truth
- Verify YOLO model is loaded correctly
- Check that ground truth annotations are correct

### High precision but low recall
- System is being too conservative
- Lower detection thresholds in config

### High recall but low precision
- System has many false detections
- Increase detection thresholds in config
