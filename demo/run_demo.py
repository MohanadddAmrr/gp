import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

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
    # YOLO TRACKING: PLAYERS + BALL
    # --------------------------------------------------------
    results = model.track(
        source=str(video_path),
        stream=True,
        show=False,                 # we draw ourselves
        classes=[PERSON_CLASS, BALL_CLASS],
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

        # --------- DRAW PLAYERS + BALL ----------
        # players: green boxes with P{id}
        for tid, box, cls_id in zip(ids, xyxy, cls_arr):
            x1, y1, x2, y2 = map(int, box)

            if cls_id == PERSON_CLASS:
                label = f"P{tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            elif cls_id == BALL_CLASS:
                # Draw ball in yellow
                ball_seen = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Ball",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # status line
        current_player_count = int((cls_arr == PERSON_CLASS).sum())
        status_text = f"Frame: {frame_idx}   Players: {current_player_count}"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
    for pid, hm in heat_per_player.items():
        make_colored_heatmap(hm, out_dir / f"heatmap_player_{pid}.png")

    # ============================================================
    # METRICS (distance / speed / workload / involvement)
    # ============================================================
    meter_per_px = 105.0 / width if width > 0 else 1.0

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

    metrics = {
        "frame": frame_idx,
        "num_players": len(track_history),
        "avg_step_px": avg_step_px,
        "ball_detected": bool(ball_seen),
        "tracks": tracks_list,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved metrics.json and heatmaps to: {out_dir}")


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
