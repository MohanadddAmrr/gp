"""Standalone heatmap regenerator.

Reads per_frame_tracks.json + metrics.json from a demo output directory
and rewrites heatmap_global.png + heatmap_team_A.png + heatmap_team_B.png
using the (now fixed) make_colored_heatmap renderer in demo.run_demo.

Avoids re-running the full YOLO/tracker pipeline just to update visuals.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.run_demo import make_colored_heatmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    d = Path(args.match_dir)
    tracks = json.load(open(d / "per_frame_tracks.json", encoding="utf-8"))
    metrics = json.load(open(d / "metrics.json", encoding="utf-8"))
    identities = metrics.get("player_identities", {})

    # tracks may be wrapped {frame_size, frames: [...]} or a bare list.
    if isinstance(tracks, dict):
        proc_w = int(tracks.get("frame_size", [args.width, args.height])[0])
        proc_h = int(tracks.get("frame_size", [args.width, args.height])[1])
        frames = tracks.get("frames", [])
    else:
        proc_w, proc_h = args.width, args.height
        frames = tracks

    w, h = args.width, args.height
    sx = w / max(proc_w, 1)
    sy = h / max(proc_h, 1)

    hg = np.zeros((h, w), dtype=np.float32)
    ha = np.zeros((h, w), dtype=np.float32)
    hb = np.zeros((h, w), dtype=np.float32)

    pid_team = {str(k): v.get("team") for k, v in identities.items()}

    for entry in frames:
        for det in entry.get("tracks", []):
            if det.get("class") != "person":
                continue
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, bw, bh = bbox
            cx = int((x + bw / 2.0) * sx)
            cy = int((y + bh / 2.0) * sy)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            hg[cy, cx] += 1
            team = pid_team.get(str(det.get("track_id")))
            if team == "A":
                ha[cy, cx] += 1
            elif team == "B":
                hb[cy, cx] += 1

    print(f"  global max={hg.max():.0f}  A max={ha.max():.0f}  B max={hb.max():.0f}")

    make_colored_heatmap(hg, d / "heatmap_global.png")
    make_colored_heatmap(ha, d / "heatmap_team_A.png")
    make_colored_heatmap(hb, d / "heatmap_team_B.png")
    print(f"  wrote heatmaps to {d}")


if __name__ == "__main__":
    main()
