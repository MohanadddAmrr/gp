"""
Lightweight match processing - NO YOLO/PyTorch required.
Uses OpenCV background subtraction for player detection.
Needs ~200 MB RAM vs ~800 MB for YOLO pipeline.

Usage:
    python scripts/run_match_lite.py input_videos/liverpoolvstottenham.mp4
    python scripts/run_match_lite.py input_videos/match.mp4 --team-a "Liverpool" --team-b "Tottenham Hotspur"
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_BASE = ROOT / "demo" / "demo_outputs"


def extract_teams_from_filename(filename, db_manager):
    """Extract team names from video filename using DB lookup."""
    from demo.run_demo import extract_teams_from_filename as _extract
    return _extract(filename, db_manager)


def make_colored_heatmap(heat, path, cmap=cv2.COLORMAP_JET):
    """Save a heatmap as a colored PNG."""
    if heat.max() > 0:
        norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(heat, dtype=np.uint8)
    colored = cv2.applyColorMap(norm, cmap)
    cv2.imwrite(str(path), colored)


def detect_players_bgsubtract(frame, bg_subtractor, min_area=500):
    """Detect player-like blobs using background subtraction."""
    mask = bg_subtractor.apply(frame)
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / max(w, 1)
        # Filter for roughly person-shaped blobs (taller than wide)
        if 1.0 < aspect < 5.0 and h > 20:
            cx = x + w // 2
            cy = y + h // 2
            detections.append((cx, cy, x, y, x + w, y + h))
    return detections


def assign_team_by_color(frame, x1, y1, x2, y2, team_a_color=None, team_b_color=None):
    """Simple team assignment by dominant jersey color in the bounding box."""
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return "A"
    # Use average hue to distinguish teams
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    # Simple split: lower hue = team A (reds), higher hue = team B (blues)
    return "A" if avg_hue < 90 else "B"


def process_video_lite(video_path, db_manager, team_a_name=None, team_b_name=None,
                       max_seconds=0, frame_skip=3):
    """Process video with lightweight OpenCV-only detection."""
    print(f"\nProcessing (lite mode): {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process at reduced resolution
    proc_w, proc_h = 640, 360

    if max_seconds > 0:
        total_frames = min(total_frames, int(max_seconds * fps))

    print(f"  Video: {orig_width}x{orig_height} @ {fps:.1f} FPS")
    print(f"  Processing at {proc_w}x{proc_h}, skip={frame_skip}, frames={total_frames}")

    # Resolve team names
    if not team_a_name or not team_b_name:
        fn_a, fn_b = extract_teams_from_filename(video_path.name, db_manager)
        if fn_a and fn_b:
            team_a_name = team_a_name or fn_a
            team_b_name = team_b_name or fn_b
            print(f"  Teams from filename: {team_a_name} vs {team_b_name}")

    if not team_a_name:
        team_a_name = "Team A"
    if not team_b_name:
        team_b_name = "Team B"

    # Load rosters from DB
    roster_a = db_manager.get_roster_for_team(team_a_name) if db_manager else []
    roster_b = db_manager.get_roster_for_team(team_b_name) if db_manager else []
    print(f"  Rosters: {team_a_name} ({len(roster_a)} players), {team_b_name} ({len(roster_b)} players)")

    # Load team colors from config
    import yaml
    team_colors = {}
    try:
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        team_colors = cfg.get("team_colors", {})
    except Exception:
        pass

    color_a = team_colors.get(team_a_name, "#ef4444")
    color_b = team_colors.get(team_b_name, "#60a5fa")

    # Output directory
    out_dir = OUTPUT_BASE / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data structures
    heat_global = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_team_A = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_team_B = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_ball = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_per_player = {}  # pid -> heatmap (capped at 30)

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    tracks = {}  # pid -> list of (cx, cy, t)
    team_assign = {}  # pid -> "A" or "B"
    next_pid = 1
    prev_detections = []  # for simple tracking

    possession_a_frames = 0
    possession_b_frames = 0
    total_tracked_frames = 0

    frame_idx = 0
    processed = 0
    start_time = time.time()

    print("  Processing frames...")

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Skip frames
        if frame_idx % frame_skip != 0:
            continue

        # Resize
        small = cv2.resize(frame, (proc_w, proc_h))
        t = frame_idx / fps

        # Detect players
        detections = detect_players_bgsubtract(small, bg_sub)
        processed += 1

        # Simple nearest-neighbor tracking
        used_prev = set()
        matched = []
        for (cx, cy, x1, y1, x2, y2) in detections:
            best_pid = None
            best_dist = 80  # max match distance in pixels
            for i, (pcx, pcy, ppid) in enumerate(prev_detections):
                if i in used_prev:
                    continue
                dist = np.hypot(cx - pcx, cy - pcy)
                if dist < best_dist:
                    best_dist = dist
                    best_pid = ppid
                    best_idx = i

            if best_pid is not None:
                used_prev.add(best_idx)
                pid = best_pid
            else:
                pid = next_pid
                next_pid += 1

            matched.append((cx, cy, pid))

            # Team assignment
            if pid not in team_assign:
                team_assign[pid] = assign_team_by_color(small, x1, y1, x2, y2)

            team = team_assign[pid]

            # Track history (keep last 5)
            if pid not in tracks:
                tracks[pid] = []
            tracks[pid].append((cx, cy, t))
            if len(tracks[pid]) > 5:
                tracks[pid] = tracks[pid][-5:]

            # Heatmaps
            if 0 <= cx < proc_w and 0 <= cy < proc_h:
                heat_global[cy, cx] += 1
                if team == "A":
                    heat_team_A[cy, cx] += 1
                else:
                    heat_team_B[cy, cx] += 1

                if pid not in heat_per_player and len(heat_per_player) < 30:
                    heat_per_player[pid] = np.zeros((proc_h, proc_w), dtype=np.float32)
                if pid in heat_per_player:
                    heat_per_player[pid][cy, cx] += 1

        prev_detections = matched

        # Simple possession: team with most players near center
        team_a_center = sum(1 for _, _, pid in matched if team_assign.get(pid) == "A")
        team_b_center = sum(1 for _, _, pid in matched if team_assign.get(pid) == "B")
        if team_a_center + team_b_center > 0:
            total_tracked_frames += 1
            if team_a_center >= team_b_center:
                possession_a_frames += 1
            else:
                possession_b_frames += 1

        # Progress logging
        if processed % 200 == 0:
            elapsed = time.time() - start_time
            pct = frame_idx / total_frames * 100
            print(f"    Frame {frame_idx}/{total_frames} ({pct:.0f}%) - "
                  f"{processed} processed - {elapsed:.0f}s elapsed")

        # Periodic GC
        if processed % 500 == 0:
            gc.collect()

    cap.release()
    elapsed = time.time() - start_time
    print(f"  Done: {processed} frames processed in {elapsed:.1f}s")

    # Calculate stats
    num_players = len(set(team_assign.keys()))
    team_a_ids = {pid for pid, t in team_assign.items() if t == "A"}
    team_b_ids = {pid for pid, t in team_assign.items() if t == "B"}

    poss_total = max(possession_a_frames + possession_b_frames, 1)
    poss_a = round(possession_a_frames / poss_total * 100, 1)
    poss_b = round(possession_b_frames / poss_total * 100, 1)

    duration_seconds = frame_idx / fps

    # Build player identities from DB rosters
    player_identities = {}
    pid_counter = 1
    for roster, team_label in [(roster_a, "A"), (roster_b, "B")]:
        for p in roster:
            pid_str = str(pid_counter)
            player_identities[pid_str] = {
                "team": team_label,
                "number": p.get("jersey_number"),
                "name": p.get("name", f"Player {pid_counter}"),
                "display": f"#{p.get('jersey_number', '?')} {p.get('name', 'Unknown')}",
            }
            pid_counter += 1

    # Build tracks data
    track_data = []
    for pid, positions in tracks.items():
        if len(positions) < 2:
            continue
        total_dist = sum(
            np.hypot(positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            for i in range(1, len(positions))
        )
        track_data.append({
            "id": pid,
            "team": team_assign.get(pid, "A"),
            "total_distance_px": round(total_dist, 1),
            "last_step_px": 0.0,
        })

    # Build metrics
    metrics = {
        "frame": frame_idx,
        "num_players": num_players,
        "raw_track_ids": next_pid - 1,
        "avg_step_px": 0.0,
        "ball_detected": False,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60, 1),
        "tracking_quality": {
            "canonical_players": num_players,
            "raw_yolo_tracks": next_pid - 1,
            "dedup_ratio": round(num_players / max(next_pid - 1, 1), 2),
            "speed_readings_capped": 0,
        },
        "player_identities": player_identities,
        "tracks": track_data,
        "team_names": {"A": team_a_name, "B": team_b_name},
        "team_colors": {"A": color_a, "B": color_b},
        "score": {"A": 0, "B": 0},
        "possession": {
            "team_possession_percentage": {"A": poss_a, "B": poss_b},
            "total_frames": total_tracked_frames,
        },
        "pass_detection": {
            "total_passes": 0,
            "pass_accuracy": 0.0,
            "passes_by_team": {"A": 0, "B": 0},
        },
        "shot_detection": {
            "total_shots": 0,
            "shots_by_team": {"A": 0, "B": 0},
        },
        "sprint_detection": {
            "total_sprints": 0,
        },
        "xg_analysis": {},
        "tactical_analysis": {},
        "ball_tracking": {
            "total_detections": 0,
            "detection_rate": 0.0,
        },
    }

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {metrics_path}")

    # Save heatmaps
    make_colored_heatmap(heat_global, out_dir / "heatmap_global.png")
    make_colored_heatmap(heat_team_A, out_dir / "heatmap_team_A.png")
    make_colored_heatmap(heat_team_B, out_dir / "heatmap_team_B.png")
    print(f"  Saved heatmaps to {out_dir}")

    # Save per-player heatmaps
    for pid, hmap in heat_per_player.items():
        make_colored_heatmap(hmap, out_dir / f"heatmap_player_{pid}.png")

    # Save info.json
    info = {"video_path": str(video_path.resolve()), "width": orig_width, "height": orig_height, "fps": fps}
    (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Save to database
    if db_manager:
        try:
            match_id = db_manager.create_match(
                video_path=str(video_path.resolve()),
                team_a=team_a_name,
                team_b=team_b_name,
                duration_seconds=duration_seconds,
                score_a=0,
                score_b=0,
                fps=fps,
                width=orig_width,
                height=orig_height,
            )
            print(f"  Match saved to DB: match_id={match_id}")
        except Exception as e:
            print(f"  DB save warning: {e}")

    # Also write root metrics.json for dashboard auto-load
    root_metrics = OUTPUT_BASE / "metrics.json"
    root_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Summary:")
    print(f"    {team_a_name} vs {team_b_name}")
    print(f"    Duration: {duration_seconds:.0f}s ({duration_seconds/60:.1f} min)")
    print(f"    Players detected: {num_players}")
    print(f"    Possession: {team_a_name} {poss_a}% - {poss_b}% {team_b_name}")
    print(f"    Output: {out_dir}")
    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(description="Lightweight match processing (no YOLO)")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--team-a", help="Team A name")
    parser.add_argument("--team-b", help="Team B name")
    parser.add_argument("--duration", type=float, default=0, help="Process first N seconds (0=full)")
    parser.add_argument("--skip", type=int, default=3, help="Process every Nth frame (default 3)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    from services.database_manager import DatabaseManager
    db = DatabaseManager()
    db.initialize_database()
    print(f"DB initialized: matches.db")

    process_video_lite(
        video_path,
        db_manager=db,
        team_a_name=args.team_a,
        team_b_name=args.team_b,
        max_seconds=args.duration,
        frame_skip=args.skip,
    )


if __name__ == "__main__":
    main()
