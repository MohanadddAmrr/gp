"""Fine-tune YOLOv8n on the pseudo-labelled football dataset (CPU).

Usage:
    python scripts/finetune/train.py --epochs 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
DATA_YAML = REPO / "data" / "finetune_v1" / "data.yaml"
WEIGHTS = REPO / "weights" / "yolov8n.pt"
PROJECT = REPO / "runs" / "finetune"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", type=str, default="yolov8n_football_v1")
    args = ap.parse_args()

    assert DATA_YAML.exists(), f"missing {DATA_YAML}"
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(PROJECT),
        name=args.name,
        exist_ok=True,
        patience=10,
        workers=2,
        verbose=True,
    )


if __name__ == "__main__":
    main()
