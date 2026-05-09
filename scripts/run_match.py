"""
Run match processing from the command line.

Usage:
    python scripts/run_match.py input_videos/liverpoolvstottenham.mp4
    python scripts/run_match.py input_videos/match.mp4 --team-a "Liverpool" --team-b "Chelsea"
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Process a match video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--team-a", help="Team A name (optional, auto-detected from filename)")
    parser.add_argument("--team-b", help="Team B name (optional, auto-detected from filename)")
    parser.add_argument("--chunk", type=float, default=0.1, help="Chunk size in minutes (default 0.1)")
    parser.add_argument("--duration", type=float, default=0, help="Only process first N seconds (0 = full video)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    from ultralytics import YOLO
    from services.database_manager import DatabaseManager
    from demo.run_demo import process_video

    model = YOLO("yolov8n.pt")
    db = DatabaseManager()
    db.initialize_database()

    process_video(
        video_path,
        model,
        db_manager=db,
        chunk_size_minutes=args.chunk,
        memory_limit_percent=95,
        team_a_name=args.team_a,
        team_b_name=args.team_b,
        max_seconds=args.duration,
    )


if __name__ == "__main__":
    main()
