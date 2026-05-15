"""
Ground-truth correction viewer (P13) — turn a pseudo-labelled draft into a
hand-verified ground-truth file.

`scripts/make_tracks.py` produces a draft track file (boxes + IDs + classes
from the pipeline). This tool draws those boxes over the real video frames so
a human can fix them: move/resize boxes, delete false positives, add missed
players, and correct IDs and class labels. On save, `source` is flipped to
"human" — that is the signal `services.accuracy_evaluator` treats the file as
real ground truth.

It is intentionally minimal (OpenCV highgui, no extra dependencies) — not a
replacement for CVAT / Label Studio, just enough to correct ~90 frames.

Controls
--------
    n / p        next / previous frame (auto-saves)
    left-drag    on a box interior  -> move
                 on a corner handle -> resize
                 on empty space     -> draw a new box
    d / x        delete the selected box
    1..6         set selected box class: team_a team_b referee goalkeeper
                 ball ignore
    i            then type digits + Enter -> set the selected box's ID
                 (Esc cancels ID entry)
    s            save now
    q            save and quit

CLI
---
    python -m scripts.correct_gt --video VIDEO --gt tests/ground_truth/clip.json

The corrected file is written back to the --gt path (the original draft is
preserved in git history); pass --output to write elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASS_VOCAB = ("team_a", "team_b", "referee", "goalkeeper", "ball", "ignore")
CLASS_BY_KEY = {ord(str(i + 1)): name for i, name in enumerate(CLASS_VOCAB)}
MIN_SIZE = 2.0
HANDLE = 6.0  # corner grab radius, in GT-space pixels


# --- pure logic (no cv2 import — unit tested in tests/test_correct_gt.py) ---


def load_gt(path: Path) -> dict:
    """Load a track file (the unified schema from accuracy_evaluator)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_gt(payload: dict, path: Path) -> None:
    """Write the track file back, flipping `source` to 'human'."""
    payload = {**payload, "source": "human"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def point_in_box(px: float, py: float, box: list[float]) -> bool:
    """True when (px, py) lies inside the [x, y, w, h] box."""
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def hit_test(px: float, py: float, box: list[float], handle: float = HANDLE) -> Optional[str]:
    """Classify what part of `box` the point hits.

    Returns one of "nw"/"ne"/"sw"/"se" (corner handles), "move" (interior), or
    None (miss). Corner handles take priority over the interior.
    """
    x, y, w, h = box
    corners = {"nw": (x, y), "ne": (x + w, y), "sw": (x, y + h), "se": (x + w, y + h)}
    for name, (cx, cy) in corners.items():
        if abs(px - cx) <= handle and abs(py - cy) <= handle:
            return name
    if point_in_box(px, py, box):
        return "move"
    return None


def apply_move(box: list[float], dx: float, dy: float) -> list[float]:
    """Translate a box by (dx, dy)."""
    x, y, w, h = box
    return [x + dx, y + dy, w, h]


def apply_resize(
    box: list[float], handle: str, dx: float, dy: float, min_size: float = MIN_SIZE
) -> list[float]:
    """Resize a box by dragging one corner handle by (dx, dy).

    Width/height are clamped to `min_size`; this minimal tool does not support
    dragging a corner all the way through the opposite edge (no flip).
    """
    x, y, w, h = box
    if handle == "nw":
        x, y, w, h = x + dx, y + dy, w - dx, h - dy
    elif handle == "ne":
        y, w, h = y + dy, w + dx, h - dy
    elif handle == "sw":
        x, w, h = x + dx, w - dx, h + dy
    elif handle == "se":
        w, h = w + dx, h + dy
    else:
        raise ValueError(f"not a corner handle: {handle!r}")
    return [x, y, max(min_size, w), max(min_size, h)]


def normalized_box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Build an [x, y, w, h] box from two opposite corners in any order."""
    x, y = min(x0, x1), min(y0, y1)
    return [x, y, abs(x1 - x0), abs(y1 - y0)]


def find_object_at(
    objects: list[dict], px: float, py: float, handle: float = HANDLE
) -> tuple[Optional[int], Optional[str]]:
    """Topmost object under the cursor.

    Returns (index, hit_kind). Iterates last-to-first so the most recently
    drawn box (visually on top) wins an overlap. (None, None) on a miss.
    """
    for idx in range(len(objects) - 1, -1, -1):
        kind = hit_test(px, py, objects[idx]["bbox"], handle)
        if kind is not None:
            return idx, kind
    return None, None


def next_object_id(payload: dict) -> int:
    """An ID not used by any object anywhere in the file."""
    max_id = 0
    for frame in payload.get("frames", []):
        for obj in frame.get("objects", []):
            max_id = max(max_id, int(obj.get("id", 0)))
    return max_id + 1


# --- GUI shell (OpenCV highgui — not unit tested; verify interactively) ----


class _Editor:
    """Holds editor state and drives the OpenCV window."""

    def __init__(self, video_path: Path, gt_path: Path, output_path: Path):
        import cv2  # lazy: keeps the pure logic importable without a display

        self.cv2 = cv2
        self.gt_path = gt_path
        self.output_path = output_path
        self.payload = load_gt(gt_path)
        self.frames: list[dict] = self.payload.get("frames", [])
        if not self.frames:
            raise RuntimeError(f"{gt_path} has no frames")
        gt_w, gt_h = self.payload.get("frame_size", [640, 360])
        self.gt_w, self.gt_h = float(gt_w), float(gt_h)

        self.images = self._load_frame_images(video_path)
        sample = next(iter(self.images.values()))
        nat_h, nat_w = sample.shape[:2]
        self.scale_x = nat_w / self.gt_w
        self.scale_y = nat_h / self.gt_h

        self.cur = 0
        self.selected: Optional[int] = None
        self.drag_mode: Optional[str] = None  # "move"|"nw"|.. |"new"
        self.drag_start: tuple[float, float] = (0.0, 0.0)
        self.drag_orig_box: Optional[list[float]] = None
        self.id_buffer: Optional[str] = None  # not None => typing an ID
        self.window = "correct_gt"

    def _load_frame_images(self, video_path: Path) -> dict[int, "object"]:
        """Decode the video once, sequentially, caching the frames the GT
        references. Sequential decode avoids the per-seek H.264 degradation.

        We use `grab()` (no numpy copy) for the ~95% of frames we don't keep
        and only `retrieve()` for the wanted ones — fully reading every frame
        into Python keeps thousands of 1280x720x3 buffers alive long enough
        to OOM ffmpeg's internal frame pool on this video.
        """
        cv2 = self.cv2
        wanted = sorted(f["frame_idx"] for f in self.frames)
        wanted_set = set(wanted)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        images: dict[int, object] = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        last = wanted[-1]
        for idx in range(wanted[0], last + 1):
            if idx in wanted_set:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                images[idx] = frame
            else:
                if not cap.grab():
                    break
        cap.release()
        if not images:
            raise RuntimeError("decoded no frames the GT references")
        return images

    # -- coordinate mapping ------------------------------------------------

    def _to_gt(self, dx: int, dy: int) -> tuple[float, float]:
        return dx / self.scale_x, dy / self.scale_y

    # -- current-frame helpers --------------------------------------------

    @property
    def _objects(self) -> list[dict]:
        return self.frames[self.cur]["objects"]

    def _save(self) -> None:
        save_gt(self.payload, self.output_path)

    # -- mouse -------------------------------------------------------------

    def _on_mouse(self, event, mx, my, flags, _param) -> None:
        cv2 = self.cv2
        gx, gy = self._to_gt(mx, my)

        if event == cv2.EVENT_LBUTTONDOWN:
            idx, kind = find_object_at(self._objects, gx, gy)
            if idx is not None:
                self.selected = idx
                self.drag_mode = kind
                self.drag_orig_box = list(self._objects[idx]["bbox"])
            else:
                self.selected = None
                self.drag_mode = "new"
                self.drag_orig_box = None
            self.drag_start = (gx, gy)

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_mode:
            dx, dy = gx - self.drag_start[0], gy - self.drag_start[1]
            if self.drag_mode == "new":
                pass  # the rubber-band box is computed at render time
            elif self.drag_mode == "move":
                self._objects[self.selected]["bbox"] = apply_move(
                    self.drag_orig_box, dx, dy
                )
            else:  # a corner handle
                self._objects[self.selected]["bbox"] = apply_resize(
                    self.drag_orig_box, self.drag_mode, dx, dy
                )

        elif event == cv2.EVENT_LBUTTONUP and self.drag_mode:
            if self.drag_mode == "new":
                box = normalized_box(self.drag_start[0], self.drag_start[1], gx, gy)
                if box[2] >= MIN_SIZE and box[3] >= MIN_SIZE:
                    self._objects.append({
                        "id": next_object_id(self.payload),
                        "class": "ignore",
                        "bbox": [round(v, 1) for v in box],
                        "conf": 1.0,
                    })
                    self.selected = len(self._objects) - 1
            self.drag_mode = None
            self.drag_orig_box = None

    # -- rendering ---------------------------------------------------------

    def _render(self):
        cv2 = self.cv2
        frame = self.frames[self.cur]
        img = self.images[frame["frame_idx"]].copy()

        for idx, obj in enumerate(self._objects):
            x, y, w, h = obj["bbox"]
            p1 = (int(x * self.scale_x), int(y * self.scale_y))
            p2 = (int((x + w) * self.scale_x), int((y + h) * self.scale_y))
            selected = idx == self.selected
            color = (0, 255, 255) if selected else (0, 200, 0)
            cv2.rectangle(img, p1, p2, color, 2 if selected else 1)
            cv2.putText(
                img, f"{obj['id']}:{obj['class']}", (p1[0], max(p1[1] - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        hud = (
            f"frame {self.cur + 1}/{len(self.frames)} "
            f"(idx={frame['frame_idx']})  objects={len(self._objects)}"
        )
        if self.selected is not None and self.selected < len(self._objects):
            sel = self._objects[self.selected]
            hud += f"  | selected id={sel['id']} class={sel['class']}"
        if self.id_buffer is not None:
            hud += f"  | ID: {self.id_buffer}_"
        cv2.putText(img, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img, "n/p frame  drag move/resize/new  d del  1-6 class  i id  s save  q quit",
            (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return img

    # -- key handling ------------------------------------------------------

    def _handle_id_key(self, key: int) -> None:
        """Digit-entry sub-mode for setting the selected object's ID."""
        if key in (13, 10):  # Enter
            if self.id_buffer and self.selected is not None:
                self._objects[self.selected]["id"] = int(self.id_buffer)
            self.id_buffer = None
        elif key == 27:  # Esc
            self.id_buffer = None
        elif key == 8:  # Backspace
            self.id_buffer = self.id_buffer[:-1]
        elif 48 <= key <= 57:  # 0-9
            self.id_buffer += chr(key)

    def run(self) -> None:
        cv2 = self.cv2
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

        while True:
            cv2.imshow(self.window, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:  # no key
                continue

            if self.id_buffer is not None:
                self._handle_id_key(key)
                continue

            if key in (ord("q"), ord("p"), ord("n"), ord("s")):
                self._save()
            if key == ord("q"):
                break
            elif key == ord("n"):
                self.cur = min(self.cur + 1, len(self.frames) - 1)
                self.selected = None
            elif key == ord("p"):
                self.cur = max(self.cur - 1, 0)
                self.selected = None
            elif key in (ord("d"), ord("x")) and self.selected is not None:
                if self.selected < len(self._objects):
                    del self._objects[self.selected]
                self.selected = None
            elif key in CLASS_BY_KEY and self.selected is not None:
                if self.selected < len(self._objects):
                    self._objects[self.selected]["class"] = CLASS_BY_KEY[key]
            elif key == ord("i") and self.selected is not None:
                self.id_buffer = ""

        cv2.destroyAllWindows()


def run_editor(video_path: Path, gt_path: Path, output_path: Optional[Path] = None) -> None:
    """Open the correction window for `gt_path` over `video_path`."""
    editor = _Editor(video_path, gt_path, output_path or gt_path)
    editor.run()


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.correct_gt",
        description="Hand-correct a pseudo-labelled ground-truth track file.",
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video.")
    parser.add_argument("--gt", required=True, type=Path, help="Track file to correct.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the corrected file (default: overwrite --gt).",
    )
    args = parser.parse_args(argv)

    try:
        run_editor(args.video, args.gt, args.output)
    except Exception as exc:  # highgui has no display, video missing, etc.
        print(f"correct_gt: {exc}", file=sys.stderr)
        return 1
    print(f"saved -> {args.output or args.gt}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
