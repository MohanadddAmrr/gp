"""Live tracking demo player.

Plays a source video at real-time speed in an OpenCV window with the
saved tracking + identity overlays drawn live, frame by frame. Reads
per_frame_tracks.json + metrics.json from the matching demo output
directory — no YOLO inference at demo time, so it never lags or
stutters on the presentation machine.

Controls:
  SPACE      pause / resume
  Q / Esc    quit
  Left/Right seek -5s / +5s
  R          restart from beginning
  H          toggle HUD (player count, ball seen, scoreboard)

Usage (defaults to Liverpool vs Tottenham):
    python scripts/live_demo.py
    python scripts/live_demo.py --match arsenalvsfulham
    python scripts/live_demo.py --video input_videos/X.mp4 \
                                --match-dir demo/demo_outputs/X
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

# Team colours (BGR for OpenCV)
TEAM_COLORS = {
    "A": (60, 60, 220),    # red
    "B": (220, 130, 30),   # blue
    "REF": (40, 200, 255), # yellow (referee)
    "?":   (160, 160, 160),
}
BALL_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_tracks(path: Path):
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


def load_identities(path: Path):
    if not path.exists():
        return {}, {"A": "Team A", "B": "Team B"}
    data = json.load(open(path, encoding="utf-8"))
    pi = {str(k): v for k, v in (data.get("player_identities") or {}).items()}
    teams = data.get("team_names") or {"A": "Team A", "B": "Team B"}
    return pi, teams


def draw_overlay(frame, dets, fw, fh, proc_w, proc_h, identities,
                 frame_idx, total_frames, fps, teams, show_hud):
    sx = fw / max(proc_w, 1)
    sy = fh / max(proc_h, 1)
    n_players = 0
    has_ball = False

    for det in dets:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x1, y1 = int(x * sx), int(y * sy)
        x2, y2 = int((x + w) * sx), int((y + h) * sy)

        tid = det.get("track_id", "?")
        is_ball = det.get("class") == "ball"
        if is_ball:
            has_ball = True
            color, label = BALL_COLOR, "Ball"
        else:
            n_players += 1
            ident = identities.get(str(tid), {})
            team = str(ident.get("team", "?"))
            color = TEAM_COLORS.get(team, TEAM_COLORS["?"])
            label = ident.get("display") or f"P{tid}"

        thickness = 3 if is_ball else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)),
                      (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), FONT, 0.5,
                    TEXT_COLOR, 1, cv2.LINE_AA)

    if show_hud:
        # Scoreboard at top
        bar_h = 38
        cv2.rectangle(frame, (0, 0), (fw, bar_h), (10, 15, 25), -1)
        cv2.rectangle(frame, (0, bar_h), (fw, bar_h + 2), (218, 41, 28), -1)
        cv2.putText(frame, f"{teams.get('A','A')}", (16, 25),
                    FONT, 0.7, (90, 90, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, "vs", (180, 25), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{teams.get('B','B')}", (230, 25),
                    FONT, 0.7, (240, 160, 60), 2, cv2.LINE_AA)
        t_sec = frame_idx / max(fps, 1.0)
        cv2.putText(frame, f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}",
                    (fw - 110, 25), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom HUD
        info = f"Frame {frame_idx}/{total_frames}  |  Players: {n_players}  |  Ball: {'yes' if has_ball else 'no'}"
        (tw, th), _ = cv2.getTextSize(info, FONT, 0.55, 1)
        cv2.rectangle(frame, (10, fh - th - 18), (10 + tw + 12, fh - 5),
                      (10, 15, 25), -1)
        cv2.putText(frame, info, (16, fh - 12), FONT, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)

        # Legend
        legend_y = fh - th - 50
        cv2.rectangle(frame, (fw - 220, legend_y - 20), (fw - 10, legend_y + 50),
                      (10, 15, 25), -1)
        for i, (team, name) in enumerate([('A', teams.get('A', 'A')),
                                          ('B', teams.get('B', 'B'))]):
            y = legend_y + i * 22
            cv2.rectangle(frame, (fw - 210, y - 8), (fw - 192, y + 6),
                          TEAM_COLORS[team], -1)
            cv2.putText(frame, name[:22], (fw - 185, y + 4), FONT, 0.5,
                        (235, 235, 235), 1, cv2.LINE_AA)

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Source video path")
    ap.add_argument("--match-dir", default=None,
                    help="Demo output dir (per_frame_tracks.json + metrics.json)")
    ap.add_argument("--match", default="liverpoolvstottenham",
                    help="Match stem (used to derive --video and --match-dir if not given)")
    ap.add_argument("--stride", type=int, default=2,
                    help="Frame stride used at processing time (default 2)")
    ap.add_argument("--window-w", type=int, default=1280, help="Display window width")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="Optional starting frame to skip ahead in the source")
    args = ap.parse_args()

    video_path = Path(args.video) if args.video else (ROOT / "input_videos" / f"{args.match}.mp4")
    match_dir = Path(args.match_dir) if args.match_dir else (ROOT / "demo" / "demo_outputs" / args.match)

    if not video_path.exists():
        sys.exit(f"ERROR: video not found: {video_path}")
    if not match_dir.exists():
        sys.exit(f"ERROR: match dir not found: {match_dir}")

    tracks_path = match_dir / "per_frame_tracks.json"
    metrics_path = match_dir / "metrics.json"
    if not tracks_path.exists():
        sys.exit(f"ERROR: per_frame_tracks.json not found in {match_dir}")

    print(f"Loading tracking data from {match_dir.name}...")
    frames_data, proc_w, proc_h = load_tracks(tracks_path)
    identities, teams = load_identities(metrics_path)
    print(f"  {len(frames_data)} tracked frames (processing res {proc_w}x{proc_h})")
    print(f"  {len(identities)} identified player tracks")
    print(f"  Teams: {teams.get('A','A')} vs {teams.get('B','B')}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale display window
    target_w = args.window_w
    scale = target_w / fw_src
    fw, fh = target_w, int(fh_src * scale)
    print(f"Video {fw_src}x{fh_src}@{fps:.1f}, {total} frames "
          f"-> display {fw}x{fh}")

    win = f"TactiVision Live — {args.match}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, fw, fh)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frame_period = 1.0 / max(fps, 1.0)
    paused = False
    show_hud = True
    print("\nControls: SPACE pause/resume | Q/Esc quit | <-/-> seek 5s | R restart | H HUD")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back so the live demo never goes to a black screen
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            t_start = time.time()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        if fw != fw_src:
            disp = cv2.resize(frame, (fw, fh))
        else:
            disp = frame.copy()

        dets = frames_data.get(frame_idx // args.stride, [])
        disp = draw_overlay(disp, dets, fw, fh, proc_w, proc_h, identities,
                            frame_idx, total, fps, teams, show_hud)

        cv2.imshow(win, disp)

        # Sleep to real-time (skip if paused, since we wait on key)
        if not paused:
            elapsed = time.time() - t_start
            wait_ms = max(1, int((frame_period - elapsed) * 1000))
        else:
            wait_ms = 30

        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):  # q / Esc
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == 83 or key == ord('d'):  # right
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx + int(fps * 5), total - 1))
        elif key == 81 or key == ord('a'):  # left
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - int(fps * 5), 0))
        elif cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live demo closed.")


if __name__ == "__main__":
    main()
