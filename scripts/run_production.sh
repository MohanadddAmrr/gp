#!/usr/bin/env bash
# Sequential production pipeline for all 3 demo matches.
# Each match: process -> regen heatmaps -> render tracked video.
set -e

PY=.venv/Scripts/python.exe
LOGS=logs/production_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"
echo "Production logs -> $LOGS"

run_match() {
    local name="$1"
    local video="$2"
    local max="$3"
    local start="$4"
    local stem="$5"

    echo
    echo "========================================================"
    echo "[$(date +%H:%M:%S)] $name | start=${start}s | duration=${max}s"
    echo "========================================================"

    $PY demo/run_demo.py \
        --video "$video" \
        --max-seconds "$max" \
        --start-seconds "$start" \
        --chunk-size 1 \
        > "$LOGS/${stem}.log" 2>&1

    echo "[$(date +%H:%M:%S)] $name processing done. Regenerating heatmaps..."
    $PY scripts/regen_heatmaps.py \
        --match-dir "demo/demo_outputs/${stem}" \
        >> "$LOGS/${stem}.log" 2>&1

    echo "[$(date +%H:%M:%S)] $name regenerating tracking video..."
    $PY scripts/visualize_tracking.py \
        --video "$video" \
        --tracks "demo/demo_outputs/${stem}/per_frame_tracks.json" \
        --metrics "demo/demo_outputs/${stem}/metrics.json" \
        --output "demo/demo_outputs/${stem}/tracked_output.mp4" \
        >> "$LOGS/${stem}.log" 2>&1

    echo "[$(date +%H:%M:%S)] $name DONE."
}

run_match "Liverpool vs Tottenham" "input_videos/liverpoolvstottenham.mp4" 600 0 "liverpoolvstottenham"
run_match "Arsenal vs Fulham"     "input_videos/arsenalvsfulham.mp4"     600 1500 "arsenalvsfulham"
run_match "Newcastle vs Man Utd"  "input_videos/newcastlevsmanchesterunited.mp4" 600 0 "newcastlevsmanchesterunited"

echo
echo "[$(date +%H:%M:%S)] === ALL 3 MATCHES DONE ==="
