#!/usr/bin/env bash
# Overnight full-match production. Runs all 3 matches end-to-end with
# auto heatmap regen + tracked video render. Sequential (NOT parallel)
# because 2 simultaneous demos exhaust the 7.7 GB RAM and OOM.
#
# Each match is wrapped in its own try block so a single failure doesn't
# kill the rest of the pipeline. Per-match logs go to logs/overnight_<ts>/.

set -u  # error on undefined vars but DON'T `set -e` — we want soft failures

PY=.venv/Scripts/python.exe
LOGS=logs/overnight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"
SUMMARY="$LOGS/SUMMARY.txt"
echo "Overnight full-match runs started $(date)" | tee "$SUMMARY"
echo "Logs directory: $LOGS" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

mem_snapshot() {
    $PY -c "import psutil; v=psutil.virtual_memory(); print(f'Available RAM: {v.available/1e9:.1f}GB / {v.total/1e9:.1f}GB ({v.percent}% used)')"
}

run_match() {
    local name="$1"
    local video="$2"
    local stem="$3"
    local t0=$(date +%s)

    echo "" | tee -a "$SUMMARY"
    echo "========================================================" | tee -a "$SUMMARY"
    echo "[$(date '+%H:%M:%S')] $name" | tee -a "$SUMMARY"
    echo "  Video: $video" | tee -a "$SUMMARY"
    echo "  $(mem_snapshot)" | tee -a "$SUMMARY"
    echo "========================================================" | tee -a "$SUMMARY"

    # 1. Process (full video, no --max-seconds)
    if $PY demo/run_demo.py \
        --video "$video" \
        --chunk-size 1 \
        > "$LOGS/${stem}_process.log" 2>&1; then
        echo "[$(date '+%H:%M:%S')] ✓ $name PROCESSING done" | tee -a "$SUMMARY"
    else
        echo "[$(date '+%H:%M:%S')] ✗ $name PROCESSING FAILED — see $LOGS/${stem}_process.log" | tee -a "$SUMMARY"
        return 1
    fi

    # 2. Regen heatmaps
    if $PY scripts/regen_heatmaps.py --match-dir "demo/demo_outputs/${stem}" \
        > "$LOGS/${stem}_heatmaps.log" 2>&1; then
        echo "[$(date '+%H:%M:%S')] ✓ $name HEATMAPS done" | tee -a "$SUMMARY"
    else
        echo "[$(date '+%H:%M:%S')] ✗ $name HEATMAPS FAILED" | tee -a "$SUMMARY"
    fi

    # 3. Render tracked video
    if $PY scripts/visualize_tracking.py \
        --video "$video" \
        --tracks "demo/demo_outputs/${stem}/per_frame_tracks.json" \
        --metrics "demo/demo_outputs/${stem}/metrics.json" \
        --output "demo/demo_outputs/${stem}/tracked_output.mp4" \
        > "$LOGS/${stem}_render.log" 2>&1; then
        echo "[$(date '+%H:%M:%S')] ✓ $name VIDEO RENDER done" | tee -a "$SUMMARY"
    else
        echo "[$(date '+%H:%M:%S')] ✗ $name VIDEO RENDER FAILED" | tee -a "$SUMMARY"
    fi

    local elapsed=$(($(date +%s) - t0))
    echo "[$(date '+%H:%M:%S')] $name elapsed: $((elapsed / 60))m $((elapsed % 60))s" | tee -a "$SUMMARY"
}

# Order: shortest first so worst case (AF) is last and any earlier failure
# is reported quickly. NM is shortest (~14 min), then L/T (~16 min), AF
# is the giant (~52 min, expected ~3h+ processing).
run_match "Newcastle vs Man Utd"   "input_videos/newcastlevsmanchesterunited.mp4" "newcastlevsmanchesterunited"
run_match "Liverpool vs Tottenham" "input_videos/liverpoolvstottenham.mp4"        "liverpoolvstottenham"
run_match "Arsenal vs Fulham"      "input_videos/arsenalvsfulham.mp4"             "arsenalvsfulham"

echo "" | tee -a "$SUMMARY"
echo "========================================================" | tee -a "$SUMMARY"
echo "[$(date '+%H:%M:%S')] === OVERNIGHT RUN COMPLETE ===" | tee -a "$SUMMARY"
echo "$(mem_snapshot)" | tee -a "$SUMMARY"
echo "========================================================" | tee -a "$SUMMARY"
