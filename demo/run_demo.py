import json
import sys
from pathlib import Path

# FIX: Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
from services.ball_tracker import BallTracker
from services.possession_tracker import PossessionTracker
from services.ball_detector import ColorBallDetector
from services.event_detector import EventDetector


# ============================================================
# PATHS & MODEL
# ============================================================
ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT.parent / "input_videos"
OUTPUT_BASE = ROOT / "demo_outputs"

MODEL_NAME = "yolov8n.pt"  # YOLO model file
PERSON_CLASS = 0           # COCO: person
BALL_CLASS = 32            # COCO: sports ball


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_colored_heatmap(hm: np.ndarray, out_path: Path):
    """Convert a 2D float heatmap to a colored PNG."""
    if hm.max() <= 0:
        h, w = hm.shape
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        blank[:, :] = (0, 80, 0)
        cv2.imwrite(str(out_path), blank)
        return

    norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), colored)


def process_video(video_path: Path, model: YOLO):
    print(f"\nProcessing: {video_path.name}")
    vid_id = video_path.stem.replace(" ", "_")
    out_dir = OUTPUT_BASE / vid_id
    ensure_dir(out_dir)

    # --------------------------------------------------------
    # BASIC VIDEO INFO
    # --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  FPS: {fps:.1f}, size: {width}x{height}")

    # --------------------------------------------------------
    # HEATMAPS + TRACK CONTAINERS
    # --------------------------------------------------------
    heat_global = np.zeros((height, width), dtype=np.float32)
    heat_team_A = np.zeros_like(heat_global)
    heat_team_B = np.zeros_like(heat_global)
    heat_per_player = {}  # pid -> heatmap

    track_history = {}    # pid -> list of (cx, cy, time_sec)
    frame_idx = 0
    ball_seen = False

    # --------------------------------------------------------
    # TRACKERS + DETECTORS INITIALIZATION (ENHANCED)
    # --------------------------------------------------------
    ball_tracker = BallTracker(max_history=30, velocity_window=3)
    possession_tracker = PossessionTracker(distance_threshold=50.0)
    event_detector = EventDetector(
        min_velocity_mps=1.0, 
        min_distance_m=5.0, 
        max_distance_m=45.0,
        shot_velocity_threshold_mps=5.0,
        shot_max_velocity_mps=200.0,
        shot_angle_threshold_deg=30.0
    )

    color_ball_detector = ColorBallDetector()  # Hybrid: fallback detector
    
    # Calculate meter_per_px early (needed for pass/shot detection)
    meter_per_px = 105.0 / width if width > 0 else 1.0  # Standard pitch = 105m wide
    
    ball_position_history = []
    heat_ball = np.zeros((height, width), dtype=np.float32)
    
    # Track detection method statistics
    yolo_detections = 0
    color_detections = 0
    predicted_detections = 0

    # --------------------------------------------------------
    # YOLO TRACKING: PLAYERS + BALL
    # --------------------------------------------------------
    results = model.track(
        source=str(video_path),
        stream=True,
        show=False,
        classes=[PERSON_CLASS, BALL_CLASS],
        conf=0.05,  # LOW CONFIDENCE FOR BALL DETECTION
        verbose=False,
        persist=True,
    )

    win_video = "TactiVision - Video"
    win_heat = "TactiVision - Heatmap"
    cv2.namedWindow(win_video, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_heat, cv2.WINDOW_NORMAL)

    stop_early = False

    for res in results:
        if stop_early:
            break

        frame_idx += 1
        frame = res.orig_img.copy()
        boxes = res.boxes

        if boxes is None or boxes.id is None:
            cv2.imshow(win_video, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                stop_early = True
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        t = frame_idx / fps

        # --------- UPDATE TRACKS & HEATMAPS (PLAYERS ONLY) ----------
        for tid, box, cls_id in zip(ids, xyxy, cls_arr):
            if cls_id != PERSON_CLASS:
                continue

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_history.setdefault(tid, []).append((cx, cy, t))

            ix, iy = int(cx), int(cy)
            if 0 <= ix < width and 0 <= iy < height:
                heat_global[iy, ix] += 1.0

            if tid not in heat_per_player:
                heat_per_player[tid] = np.zeros_like(heat_global)
            heat_per_player[tid][iy, ix] += 1.0

        # --------- ENHANCED BALL DETECTION: YOLO + Color + Prediction ----------
        ball_detected_this_frame = False
        ball_box = None
        detection_method = None

        # STEP 1: Try YOLO ball detection first (fast, accurate when visible)
        for tid, box, cls_id in zip(ids, xyxy, cls_arr):
            if cls_id == BALL_CLASS:
                ball_detected_this_frame = True
                ball_box = box
                detection_method = "YOLO"
                yolo_detections += 1
                break

        # STEP 2: If YOLO failed, try color-based detection (slower, works for small balls)
        if not ball_detected_this_frame:
            color_result = color_ball_detector.detect_best(frame)
            if color_result is not None:
                ball_box = color_result
                ball_detected_this_frame = True
                detection_method = "Color"
                color_detections += 1

        # STEP 3: Use trajectory prediction to fill gaps
        tracked, method = ball_tracker.update_with_prediction(
            ball_box, frame_idx, t, width, height, max_missing_frames=5
        )
        
        # Initialize possession_context for this frame
        possession_context = None
        player_positions = {}
        
        if tracked:
            if method == "predicted":
                predicted_detections += 1
                detection_method = "Predicted"
            
            ball_seen = True
            ball_position = ball_tracker.get_position()
            
            if ball_position:
                vel = ball_tracker.get_velocity()
                
                # Store position history
                ball_position_history.append({
                    'frame': frame_idx,
                    'x': float(ball_position[0]),
                    'y': float(ball_position[1]),
                    'timestamp': float(t),
                    'velocity': float(vel),
                    'detection_method': detection_method
                })
                
                # Update ball heatmap
                ix, iy = int(ball_position[0]), int(ball_position[1])
                if 0 <= ix < width and 0 <= iy < height:
                    heat_ball[iy, ix] += 1.0
                
                # --------- ENHANCED POSSESSION DETECTION WITH CONTEXT ----------
                # Build player_positions dict: {player_id: (x, y, team)}
                for pid, pbox, pcls in zip(ids, xyxy, cls_arr):
                    if pcls != PERSON_CLASS:
                        continue
                    px = (pbox[0] + pbox[2]) / 2.0
                    py = (pbox[1] + pbox[3]) / 2.0
                    # Determine team (left=A, right=B)
                    team = 'A' if px < width / 2.0 else 'B'
                    player_positions[int(pid)] = (float(px), float(py), team)
                
                # Detect possession with tactical context
                possession_context = possession_tracker.detect_possession_with_context(
                    ball_pos=ball_position,
                    player_positions=player_positions,
                    frame_idx=frame_idx,
                    timestamp=t,
                    frame_width=width,
                    frame_height=height
                )
                
                # --------- PASS DETECTION ----------
                ball_velocity_mps = ball_tracker.get_velocity() * meter_per_px
                pass_event = event_detector.detect_pass(
                    current_possessor=possession_tracker.get_current_possessor(),
                    current_team=possession_tracker.get_current_team(),
                    player_positions=player_positions,
                    ball_velocity_mps=ball_velocity_mps,
                    meter_per_px=meter_per_px,
                    frame_idx=frame_idx,
                    timestamp=t,
                    frame_width=width
                )
                
                if pass_event:
                    outcome = pass_event['outcome']
                    print(f"[PASS] {pass_event['passer_id']} -> {pass_event['receiver_id']} "
                          f"({outcome}, {pass_event['distance_m']}m, {pass_event['direction']})")
                
                # --------- SHOT DETECTION ----------
                ball_direction = ball_tracker.get_direction()
                shot_event = event_detector.detect_shot(
                    ball_position=ball_position,
                    ball_direction=(float(ball_direction[0]), float(ball_direction[1])),
                    ball_velocity_mps=ball_velocity_mps,
                    frame_idx=frame_idx,
                    timestamp=t,
                    frame_width=width,
                    frame_height=height
                )
                
                if shot_event:
                    print(f"[SHOT] Player {shot_event['shooter_id']} (Team {shot_event['shooter_team']}) "
                          f"- {shot_event['velocity_mps']:.1f} m/s, angle: {shot_event['angle_to_goal_deg']:.1f}°")

        # --------- DRAW PLAYERS + BALL ----------
        # Get current possessor for highlighting
        current_possessor = possession_tracker.get_current_possessor()
        
        # Draw players: green boxes with P{id}, highlight possessor in cyan
        for tid, box, cls_id in zip(ids, xyxy, cls_arr):
            x1, y1, x2, y2 = map(int, box)

            if cls_id == PERSON_CLASS:
                label = f"P{tid}"
                
                # Highlight current possessor
                if current_possessor == tid:
                    color = (255, 255, 0)  # Cyan for possessor
                    label += " [POSS]"
                else:
                    color = (0, 255, 0)  # Green for others
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        
        # Draw ball if tracked
        if tracked:
            ball_bbox = ball_tracker.get_bbox()
            if ball_bbox is not None:
                x1, y1, x2, y2 = map(int, ball_bbox)
                vel = ball_tracker.get_velocity()
                
                # Color code by detection method
                if detection_method == "YOLO":
                    ball_color = (0, 255, 255)  # Yellow
                elif detection_method == "Color":
                    ball_color = (255, 0, 255)  # Magenta
                else:  # Predicted
                    ball_color = (0, 165, 255)  # Orange
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)
                vel_text = f"Ball {vel:.0f}px/s [{detection_method}]"
                cv2.putText(
                    frame,
                    vel_text,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    ball_color,
                    2,
                )

        # --------- ENHANCED STATUS LINE WITH TACTICAL INFO ----------
        current_player_count = int((cls_arr == PERSON_CLASS).sum())
        current_team = possession_tracker.get_current_team()
        
        # Get possession context
        poss_text = "Loose ball"
        if current_team:
            poss_pct_A = possession_tracker.get_possession_percentage().get('A', 0)
            poss_pct_B = possession_tracker.get_possession_percentage().get('B', 0)
            
            # Get pressure and zone if available
            pressure_info = ""
            zone_info = ""
            if possession_context:
                pressure = possession_context.get('pressure', 0)
                zone = possession_context.get('zone', '')
                pressure_info = f" | Press: {pressure}"
                zone_info = f" | Zone: {zone}"
            
            poss_text = f"Team {current_team} ({poss_pct_A:.0f}%-{poss_pct_B:.0f}%){pressure_info}{zone_info}"
        
        status_text = f"Frame: {frame_idx}   Players: {current_player_count}   Possession: {poss_text}"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # show video
        cv2.imshow(win_video, frame)

        # show LIVE HEATMAP (global)
        if frame_idx % 5 == 0:
            if heat_global.max() > 0:
                norm = cv2.normalize(
                    heat_global, None, 0, 255, cv2.NORM_MINMAX
                ).astype("uint8")
                colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            else:
                colored = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.imshow(win_heat, colored)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            stop_early = True

    cv2.destroyAllWindows()

    print(f"  Frames processed: {frame_idx}")
    if frame_idx == 0:
        print("  [!] No frames processed, skipping.")
        return

    # --------------------------------------------------------
    # SIMPLE TEAM SPLIT (LEFT vs RIGHT)
    # --------------------------------------------------------
    team_A_ids = set()
    team_B_ids = set()
    mid_x = width / 2.0

    for pid, pts in track_history.items():
        xs = [p[0] for p in pts]
        mean_x = float(np.mean(xs))
        if mean_x < mid_x:
            team_A_ids.add(pid)
        else:
            team_B_ids.add(pid)

    for pid, hm in heat_per_player.items():
        if pid in team_A_ids:
            heat_team_A += hm
        else:
            heat_team_B += hm

    # Save heatmaps
    make_colored_heatmap(heat_global, out_dir / "heatmap_global.png")
    make_colored_heatmap(heat_team_A, out_dir / "heatmap_team_A.png")
    make_colored_heatmap(heat_team_B, out_dir / "heatmap_team_B.png")
    make_colored_heatmap(heat_ball, out_dir / "heatmap_ball.png")
    for pid, hm in heat_per_player.items():
        make_colored_heatmap(hm, out_dir / f"heatmap_player_{pid}.png")

    # ============================================================
    # METRICS (distance / speed / workload / involvement)
    # ============================================================
    tracks_list = []
    all_step_lengths = []

    attacking_third_A = width * (2.0 / 3.0)
    attacking_third_B = width * (1.0 / 3.0)
    central_y_min = height * 0.25
    central_y_max = height * 0.75

    for pid, pts in track_history.items():
        if len(pts) < 2:
            continue

        pts = sorted(pts, key=lambda p: p[2])
        team_label = "A" if pid in team_A_ids else "B"

        total_d_px = 0.0
        speeds_mps = []
        attacking_count = 0
        central_attacking_count = 0
        total_points = len(pts)

        for (x0, y0, t0), (x1, y1, t1) in zip(pts[:-1], pts[1:]):
            dx = x1 - x0
            dy = y1 - y0
            step = float(np.hypot(dx, dy))
            total_d_px += step
            all_step_lengths.append(step)

            dt = max(t1 - t0, 1e-3)
            speed_mps = step * meter_per_px / dt
            speeds_mps.append(speed_mps)

        for (x, y, t) in pts:
            if team_label == "A":
                in_attacking = x > attacking_third_A
            else:
                in_attacking = x < attacking_third_B

            in_central = central_y_min < y < central_y_max

            if in_attacking:
                attacking_count += 1
                if in_central:
                    central_attacking_count += 1

        total_d_m = total_d_px * meter_per_px
        total_time = pts[-1][2] - pts[0][2]
        avg_speed_mps = (total_d_m / total_time) if total_time > 0 else 0.0
        max_speed_mps = max(speeds_mps) if speeds_mps else 0.0

        minutes_played = total_time / 60.0 if total_time > 0 else 1e-3
        workload_score = total_d_m / minutes_played

        if total_points > 0:
            attacking_third_ratio = attacking_count / total_points
            central_attacking_ratio = central_attacking_count / total_points
        else:
            attacking_third_ratio = 0.0
            central_attacking_ratio = 0.0

        involvement_index = 0.7 * attacking_third_ratio + 0.3 * central_attacking_ratio

        tracks_list.append(
            {
                "player_id": int(pid),
                "team": team_label,
                "total_distance_px": float(total_d_px),
                "total_distance_m": float(total_d_m),
                "avg_speed_mps": float(avg_speed_mps),
                "max_speed_mps": float(max_speed_mps),
                "workload_score": float(workload_score),
                "attacking_third_ratio": float(attacking_third_ratio),
                "central_attacking_ratio": float(central_attacking_ratio),
                "involvement_index": float(involvement_index),
            }
        )

    avg_step_px = float(np.mean(all_step_lengths)) if all_step_lengths else 0.0

    # ============================================================
    # ENHANCED METRICS WITH ALL NEW FEATURES
    # ============================================================
    ball_velocities = [p['velocity'] for p in ball_position_history if 'velocity' in p]
    
    # Get possession statistics (now includes zones, pressure, duration)
    possession_stats = possession_tracker.get_statistics()
    possession_percentage = possession_tracker.get_possession_percentage()
    player_possession_stats = possession_tracker.get_player_possession_stats()
    possession_history = possession_tracker.get_possession_history()
    
    # Get pass detection statistics
    pass_stats = event_detector.get_pass_statistics()
    pass_events = event_detector.get_pass_events()
    passing_network = event_detector.get_passing_network()
    
    # Get shot detection statistics
    shot_stats = event_detector.get_shot_statistics()
    shot_events = event_detector.get_shot_events()
    
    metrics = {
        "frame": frame_idx,
        "num_players": len(track_history),
        "avg_step_px": avg_step_px,
        "ball_detected": bool(ball_seen),
        "ball_tracking": {
            "total_detections": len(ball_position_history),
            "detection_rate": len(ball_position_history) / frame_idx if frame_idx > 0 else 0,
            "avg_velocity_px_s": float(np.mean(ball_velocities)) if ball_velocities else 0.0,
            "max_velocity_px_s": float(np.max(ball_velocities)) if ball_velocities else 0.0,
            "yolo_detections": yolo_detections,
            "color_detections": color_detections,
            "predicted_detections": predicted_detections,
            "position_history": ball_position_history
        },
        "possession": {
            "team_possession_percentage": possession_percentage,
            "player_stats": {
                str(k): v for k, v in player_possession_stats.items()
            },
            "possession_history": possession_history,
            "total_possession_changes": len(possession_history),
            "possession_rate": possession_stats.get('possession_rate', 0.0),
            "zone_stats": possession_stats.get('zone_stats', {}),
            "pressure_stats": possession_stats.get('pressure_stats', {}),
            "duration_stats": possession_stats.get('duration_stats', {}),
            "zone_changes": possession_stats.get('zone_changes', 0)
        },
        "pass_detection": pass_stats,
        "pass_events": pass_events[-50:],
        "passing_network": passing_network,
        "shot_detection": shot_stats,
        "shot_events": shot_events,
        "tracks": tracks_list,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved metrics.json and heatmaps to: {out_dir}")
    print(f"  Ball detections: {len(ball_position_history)}/{frame_idx} frames ({len(ball_position_history)/frame_idx*100:.1f}%)")
    print(f"    - YOLO: {yolo_detections}, Color: {color_detections}, Predicted: {predicted_detections}")
    print(f"  Possession - Team A: {possession_percentage['A']:.1f}%, Team B: {possession_percentage['B']:.1f}%")
    print(f"  Possession changes: {len(possession_history)}")
    
    # Print tactical insights
    duration_stats = possession_stats.get('duration_stats', {})
    if duration_stats.get('total_possessions', 0) > 0:
        print(f"  Possession style - Short: {duration_stats.get('short_pct', 0):.0f}%, Medium: {duration_stats.get('medium_pct', 0):.0f}%, Long: {duration_stats.get('long_pct', 0):.0f}%")
    
    # Print pass detection summary
    print(f"  Passes detected: {pass_stats['total_passes']} (Completed: {pass_stats['completed_passes']}, Intercepted: {pass_stats['intercepted_passes']})")
    if pass_stats['total_passes'] > 0:
        print(f"  Pass accuracy: {pass_stats['pass_accuracy']:.1f}%")
        print(f"  Pass direction - Forward: {pass_stats['direction']['forward_pct']:.0f}%, Backward: {pass_stats['direction']['backward_pct']:.0f}%, Lateral: {pass_stats['direction']['lateral_pct']:.0f}%")
        print(f"  Avg pass distance: {pass_stats['distance']['avg_m']:.1f}m")
    
    # Print shot detection summary
    print(f"  Shots detected: {shot_stats['total_shots']} (Team A: {shot_stats['team_shots']['A']}, Team B: {shot_stats['team_shots']['B']})")
    if shot_stats['total_shots'] > 0:
        print(f"  Avg shot velocity: {shot_stats['velocity']['avg_mps']:.1f} m/s, Max: {shot_stats['velocity']['max_mps']:.1f} m/s")


def main():
    ensure_dir(OUTPUT_BASE)

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    if not VIDEO_DIR.exists():
        print(f"[!] VIDEO_DIR not found: {VIDEO_DIR}")
        return

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"[!] No .mp4 files found in {VIDEO_DIR}")
        return

    for vp in videos:
        process_video(vp, model)


if __name__ == "__main__":
    main()
