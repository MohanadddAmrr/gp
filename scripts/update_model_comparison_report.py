"""Refresh the headline table in docs/MODEL_COMPARISON_REPORT.md from a
benchmarks/multi_model/<UTC ts>/results.csv produced by services.multi_model_evaluator.

Usage:
    python scripts/update_model_comparison_report.py [BENCH_DIR]

If BENCH_DIR is omitted, the most recent benchmarks/multi_model/<ts>/ folder is used.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


REPORT = Path("docs/MODEL_COMPARISON_REPORT.md")
ROOT_BENCH = Path("benchmarks/multi_model")


def _latest_run() -> Path:
    runs = sorted([p for p in ROOT_BENCH.iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit(f"No runs under {ROOT_BENCH}")
    return runs[-1]


def _aggregate(csv_path: Path) -> list[dict]:
    """Average per-(model, video) rows into per-model means."""
    by_model: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_model.setdefault(row["model"], []).append(row)

    out: list[dict] = []
    for model, rows in by_model.items():
        n = len(rows)
        out.append({
            "model": model,
            "videos": n,
            "fps": sum(float(r["fps"]) for r in rows) / n,
            "mean_confidence": sum(float(r["mean_confidence"]) for r in rows) / n,
            "person_per_frame": sum(float(r["detections_per_frame_person"]) for r in rows) / n,
            "ball_per_frame": sum(float(r["detections_per_frame_ball"]) for r in rows) / n,
            "size_mb": sum(float(r["weights_size_mb"]) for r in rows) / n,
            "map50": ", ".join(r["map50"] or "—" for r in rows) or "—",
        })
    # Stable order matching SUPPORTED_MODELS
    order = {n: i for i, n in enumerate(["yolov8n", "yolov8s", "yolov8m", "yolo11n", "rtdetr-l"])}
    out.sort(key=lambda r: order.get(r["model"], 99))
    return out


def _render_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Avg FPS | Avg conf | Person/frame | Ball/frame | mAP@50 | Size |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        map50 = r["map50"] if r["map50"].strip() else "pending GT"
        lines.append(
            f"| {r['model']} | {r['fps']:.2f} | {r['mean_confidence']:.2f} | "
            f"{r['person_per_frame']:.1f} | {r['ball_per_frame']:.2f} | "
            f"{map50} | {r['size_mb']:.0f} MB |"
        )
    return "\n".join(lines)


def _update_report(table_md: str, run_dir: Path) -> None:
    text = REPORT.read_text(encoding="utf-8")
    old_table_marker = "| Model | Avg FPS | Avg conf | Person/frame | Ball/frame | mAP@50 | Size |"
    if old_table_marker not in text:
        raise SystemExit("Could not locate headline table marker in report.")
    # Replace the OLD headline placeholder block (header + sep + 5 rows = 7 lines).
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(old_table_marker))
    new_lines = lines[:start] + table_md.split("\n") + lines[start + 7 :]
    new_text = "\n".join(new_lines)

    # Update the placeholder note to point at the actual run dir.
    new_text = new_text.replace(
        "> Replace once the Day 8 bench finishes (running in background as of\n"
        "> 2026-05-03). Fields below are placeholders to make the table layout\n"
        "> reviewable while the run completes.",
        f"> Source: `{run_dir.as_posix()}/results.csv` (3 videos × 5 models × 300 frames).",
    )
    REPORT.write_text(new_text, encoding="utf-8")
    print(f"Refreshed {REPORT} from {run_dir}")


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]) if len(argv) > 1 else _latest_run()
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"No results.csv at {csv_path}")
    rows = _aggregate(csv_path)
    table_md = _render_table(rows)
    print(table_md)
    print()
    _update_report(table_md, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
