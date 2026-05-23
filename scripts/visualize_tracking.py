"""
Tracking Visualizer
===================
Renders a watchable video with player/ball boxes and identities, using the
per_frame_tracks.json produced by run_demo.py, enriched with the player
identities from metrics.json.

Originally authored by Ahmed Khaled (feat/e). Restored and adapted for the
integrated pipeline:
  - bbox is [x1, y1, w, h] (top-left + size), not centre-format
  - scaling uses frame_size recorded in the tracks file
  - run_demo processes every Nth frame (default stride 2); boxes are held
    across the skipped frames so the output plays at normal speed/length
  - labels are enriched from metrics.json -> player_identities

Usage:
    python scripts/visualize_tracking.py \
        --video   "input_videos/liverpoolvstottenham.mp4" \
        --tracks  "demo/demo_outputs/liverpoolvstottenham/per_frame_tracks.json" \
        --metrics "demo/demo_outputs/liverpoolvstottenham/metrics.json" \
        --output  "demo/demo_outputs/liverpoolvstottenham/tracked_output.mp4"
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Team colours (BGR for OpenCV)
TEAM_COLORS = {
    "A": (60, 60, 220),    # red  (home)
    "B": (220, 130, 30),   # blue (away)
    "?": (160, 160, 160),  # grey (unknown)
}
BALL_COLOR = (0, 255, 255)    # yellow
TEXT_COLOR = (255, 255, 255)  # white
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_tracks(path: Path):
    """Return (frames_by_idx, proc_w, proc_h)."""
    raw = json.load(open(path, encoding="utf-8"))
    frames: dict[int, list] = {}
    proc_w = proc_h = 0
    if isinstance(raw, dict):
        size = raw.get("frame_size") or [0, 0]
        proc_w, proc_h = int(size[0]), int(size[1])
        for entry in raw.get("frames", []):
            frames[int(entry.get("frame_idx", 0))] = entry.get("tracks", [])
    elif isinstance(raw, list):
        for entry in raw:
            idx = int(entry.get("frame_idx", entry.get("frame", 0)))
            frames[idx] = entry.get("tracks", entry.get("detections", []))
    return frames, proc_w, proc_h


def load_identities(path: Path | None) -> dict:
    """track_id (str) -> {team, number, name, display}."""
    if not path or not Path(path).exists():
        return {}
    data = json.load(open(path, encoding="utf-8"))
    pi = data.get("player_identities", {}) or {}
    return {str(k): v for k, v in pi.items()}


def draw_frame(frame, dets, fw, fh, proc_w, proc_h, identities):
    sx = fw / proc_w if proc_w else 1.0
    sy = fh / proc_h if proc_h else 1.0
    for det in dets:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox  # top-left + size, in processing coords
        x1, y1 = int(x * sx), int(y * sy)
        x2, y2 = int((x + w) * sx), int((y + h) * sy)

        tid = det.get("track_id", "?")
        if det.get("class") == "ball":
            color, label = BALL_COLOR, "Ball"
        else:
            ident = identities.get(str(tid))
            if ident:
                team = str(ident.get("team", "?"))
                label = ident.get("display") or f"P{tid}"
            else:
                team = str(det.get("team", "?"))
                label = f"P{tid}"
            color = TEAM_COLORS.get(team, TEAM_COLORS["?"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.45,
                    TEXT_COLOR, 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description="TactiVision Tracking Visualizer")
    ap.add_argument("--video", required=True, help="Original video path")
    ap.add_argument("--tracks", required=True, help="per_frame_tracks.json path")
    ap.add_argument("--metrics", default=None, help="metrics.json (for identities)")
    ap.add_argument("--output", required=True, help="Output .mp4 path")
    ap.add_argument("--stride", type=int, default=2,
                    help="Frames run_demo skips (default 2 = every 2nd frame)")
    args = ap.parse_args()

    video_path = Path(args.video)
    tracks_path = Path(args.tracks)
    output_path = Path(args.output)

    if not video_path.exists():
        sys.exit(f"ERROR: video not found: {video_path}")
    if not tracks_path.exists():
        sys.exit(f"ERROR: tracks file not found: {tracks_path}")

    frames_data, proc_w, proc_h = load_tracks(tracks_path)
    identities = load_identities(Path(args.metrics) if args.metrics else None)
    print(f"Loaded {len(frames_data)} tracked frames "
          f"(processing res {proc_w}x{proc_h}); "
          f"{len(identities)} identified players")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = max(1, args.stride)
    n_track = max(frames_data) + 1 if frames_data else 0
    total_out = n_track * stride
    print(f"Video {fw}x{fh}@{fps:.1f} | rendering {total_out} frames "
          f"(stride {stride})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    vidx = 0
    while vidx < total_out:
        ret, frame = cap.read()
        if not ret:
            break
        dets = frames_data.get(vidx // stride, [])
        frame = draw_frame(frame, dets, fw, fh, proc_w, proc_h, identities)

        players = sum(1 for d in dets if d.get("class") != "ball")
        has_ball = any(d.get("class") == "ball" for d in dets)
        cv2.putText(frame, f"Frame {vidx} | Players: {players} | "
                    f"Ball: {'yes' if has_ball else 'no'}",
                    (10, 25), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, "Team A (Home)", (10, fh - 40), FONT, 0.5,
                    TEAM_COLORS["A"], 2)
        cv2.putText(frame, "Team B (Away)", (10, fh - 18), FONT, 0.5,
                    TEAM_COLORS["B"], 2)

        writer.write(frame)
        vidx += 1
        if vidx % 250 == 0:
            print(f"  {vidx}/{total_out} ({100 * vidx / total_out:.0f}%)")

    cap.release()
    writer.release()
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nDone -> {output_path} ({size_mb:.1f} MB, {vidx} frames)")


if __name__ == "__main__":
    main()
