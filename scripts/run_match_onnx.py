"""
ONNX-based match processing — accurate YOLO detection WITHOUT PyTorch.

Uses ~300MB RAM instead of ~800MB, enabling processing on low-memory systems.
Includes: YOLO detection, simple tracking, jersey color team assignment,
jersey number OCR, DB roster matching, and full metrics output.

Usage:
    python scripts/run_match_onnx.py input_videos/liverpoolvstottenham.mp4
    python scripts/run_match_onnx.py input_videos/match.mp4 --team-a "Liverpool" --team-b "Tottenham Hotspur"
    python scripts/run_match_onnx.py input_videos/match.mp4 --duration 120  # first 2 minutes only
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_BASE = ROOT / "demo" / "demo_outputs"
ONNX_MODEL = ROOT / "yolov8n.onnx"

# YOLO class IDs
PERSON_CLASS = 0
BALL_CLASS = 32

# ── Team Name Extraction (inlined to avoid importing run_demo -> torch) ──────

_ALIASES = {
    'psg': 'Paris Saint-Germain', 'barca': 'Barcelona', 'juve': 'Juventus',
    'spurs': 'Tottenham Hotspur', 'wolves': 'Wolverhampton Wanderers',
    'city': 'Manchester City', 'united': 'Manchester United',
    'manunited': 'Manchester United', 'mancity': 'Manchester City',
    'atletico': 'Atletico Madrid', 'bilbao': 'Athletic Bilbao',
    'inter': 'Inter Milan', 'milan': 'AC Milan', 'bayern': 'Bayern Munich',
    'dortmund': 'Borussia Dortmund', 'leverkusen': 'Bayer Leverkusen',
    'leipzig': 'RB Leipzig', 'marseille': 'Olympique de Marseille',
    'lyon': 'Olympique Lyonnais', 'monaco': 'AS Monaco', 'lille': 'LOSC Lille',
    'nice': 'OGC Nice', 'lens': 'RC Lens', 'rennes': 'Stade Rennais',
    'strasbourg': 'RC Strasbourg', 'brest': 'Stade Brestois 29',
    'tottenham': 'Tottenham Hotspur',
}


def _extract_teams_from_filename(filename, db_manager=None):
    """Extract team names from video filename with DB fuzzy matching."""
    stem = Path(filename).stem.lower().replace('_', ' ').replace('-', ' ')
    for sep in [' vs ', 'vs', ' v ']:
        if sep in stem:
            parts = stem.split(sep, 1)
            if len(parts) == 2:
                raw_a, raw_b = parts[0].strip(), parts[1].strip()
                if raw_a and raw_b:
                    if db_manager:
                        name_a = _match_team(raw_a, db_manager)
                        name_b = _match_team(raw_b, db_manager)
                        if name_a and name_b:
                            return name_a, name_b
                    return raw_a.title(), raw_b.title()
    return None, None


def _match_team(raw_name, db_manager):
    """Fuzzy-match a raw team name against teams in the player_profiles DB."""
    try:
        profiles = db_manager.get_all_player_profiles()
    except Exception:
        return raw_name.title()
    team_names = sorted({p.get('team_name') for p in profiles if p.get('team_name')})
    raw_lower = raw_name.lower().replace(' ', '')

    # Alias check
    if raw_lower in _ALIASES:
        candidate = _ALIASES[raw_lower]
        if candidate in team_names:
            return candidate

    # Exact match (case-insensitive, spaces removed)
    for tn in team_names:
        if raw_lower == tn.lower().replace(' ', ''):
            return tn

    # Substring match - prefer shortest (most specific)
    candidates = [tn for tn in team_names if raw_lower in tn.lower().replace(' ', '')]
    if candidates:
        return min(candidates, key=len)

    # Reverse: team name is substring of raw
    candidates = [tn for tn in team_names if tn.lower().replace(' ', '') in raw_lower]
    if candidates:
        return max(candidates, key=len)

    return raw_name.title()


# ── ONNX YOLO Detector ──────────────────────────────────────────────────────

class ONNXDetector:
    """YOLOv8n inference via ONNX Runtime — no PyTorch required."""

    def __init__(self, model_path: str, conf_thresh: float = 0.3, iou_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # [1, 3, 640, 640]
        self.img_h = self.input_shape[2]
        self.img_w = self.input_shape[3]

    def preprocess(self, frame: np.ndarray):
        """Letterbox resize: maintain aspect ratio, pad to model input size."""
        fh, fw = frame.shape[:2]
        scale = min(self.img_w / fw, self.img_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to model size (center the image)
        pad_w = (self.img_w - new_w) // 2
        pad_h = (self.img_h - new_h) // 2
        img = np.full((self.img_h, self.img_w, 3), 114, dtype=np.uint8)  # Gray padding
        img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Store letterbox params for coordinate mapping
        self._lb_scale = scale
        self._lb_pad_w = pad_w
        self._lb_pad_h = pad_h

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img

    def postprocess(self, output, orig_h, orig_w, classes_filter=None):
        """Parse YOLOv8 output: shape [1, 84, 8400] -> list of (x1,y1,x2,y2,conf,cls)."""
        preds = output[0].squeeze(0).T  # [8400, 84]

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Lower threshold for ball class (small, hard to detect)
        conf_thresholds = np.where(class_ids == BALL_CLASS, self.conf_thresh * 0.4, self.conf_thresh)
        mask = confidences > conf_thresholds
        if classes_filter is not None:
            class_mask = np.isin(class_ids, classes_filter)
            mask = mask & class_mask

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert from letterboxed coordinates back to original image coordinates
        # Remove padding offset, then unscale
        lb_scale = getattr(self, '_lb_scale', min(self.img_w / orig_w, self.img_h / orig_h))
        lb_pad_w = getattr(self, '_lb_pad_w', (self.img_w - int(orig_w * lb_scale)) // 2)
        lb_pad_h = getattr(self, '_lb_pad_h', (self.img_h - int(orig_h * lb_scale)) // 2)

        x1 = (cx - w / 2 - lb_pad_w) / lb_scale
        y1 = (cy - h / 2 - lb_pad_h) / lb_scale
        x2 = (cx + w / 2 - lb_pad_w) / lb_scale
        y2 = (cy + h / 2 - lb_pad_h) / lb_scale

        # NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        indices = self._nms(boxes, confidences, self.iou_thresh)

        results = []
        for i in indices:
            results.append({
                "bbox": (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                "conf": float(confidences[i]),
                "cls": int(class_ids[i]),
            })
        return results

    def _nms(self, boxes, scores, iou_threshold):
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def detect(self, frame: np.ndarray, classes_filter=None):
        """Run detection on a frame. Returns list of {bbox, conf, cls}."""
        orig_h, orig_w = frame.shape[:2]
        blob = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs, orig_h, orig_w, classes_filter)


# ── Simple IoU Tracker ───────────────────────────────────────────────────────

class SimpleTracker:
    """Lightweight IoU + centroid distance multi-object tracker."""

    def __init__(self, max_age: int = 30, iou_thresh: float = 0.15, max_dist: float = 60):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.max_dist = max_dist  # Max centroid distance for matching
        self.next_id = 1
        self.tracks = {}  # track_id -> {"bbox": ..., "age": ..., "cls": ...}

    def update(self, detections):
        """Match detections to existing tracks. Returns list of (track_id, bbox, cls)."""
        if not self.tracks:
            # First frame: create all tracks
            results = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det["bbox"], "age": 0, "cls": det["cls"]}
                results.append((tid, det["bbox"], det["cls"]))
            return results

        # Compute IoU between all tracks and detections
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        matched_tracks = set()
        matched_dets = set()
        results = []

        if detections and track_boxes:
            iou_matrix = self._compute_iou_matrix(
                [d["bbox"] for d in detections], track_boxes
            )
            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                if max_iou < self.iou_thresh:
                    break
                det_idx, trk_idx = max_idx
                tid = track_ids[trk_idx]
                self.tracks[tid]["bbox"] = detections[det_idx]["bbox"]
                self.tracks[tid]["age"] = 0
                self.tracks[tid]["cls"] = detections[det_idx]["cls"]
                results.append((tid, detections[det_idx]["bbox"], detections[det_idx]["cls"]))
                matched_tracks.add(trk_idx)
                matched_dets.add(det_idx)
                iou_matrix[det_idx, :] = -1
                iou_matrix[:, trk_idx] = -1

        # Second pass: match remaining detections by centroid distance
        unmatched_det_indices = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_trk_indices = [j for j in range(len(track_ids)) if j not in matched_tracks]

        if unmatched_det_indices and unmatched_trk_indices:
            for det_idx in list(unmatched_det_indices):
                d = detections[det_idx]["bbox"]
                dcx = (d[0] + d[2]) / 2
                dcy = (d[1] + d[3]) / 2
                best_dist = self.max_dist
                best_trk_idx = None
                for trk_idx in unmatched_trk_indices:
                    tb = track_boxes[trk_idx]
                    tcx = (tb[0] + tb[2]) / 2
                    tcy = (tb[1] + tb[3]) / 2
                    dist = np.hypot(dcx - tcx, dcy - tcy)
                    if dist < best_dist:
                        best_dist = dist
                        best_trk_idx = trk_idx
                if best_trk_idx is not None:
                    tid = track_ids[best_trk_idx]
                    self.tracks[tid]["bbox"] = detections[det_idx]["bbox"]
                    self.tracks[tid]["age"] = 0
                    results.append((tid, detections[det_idx]["bbox"], detections[det_idx]["cls"]))
                    matched_dets.add(det_idx)
                    matched_tracks.add(best_trk_idx)
                    unmatched_trk_indices.remove(best_trk_idx)

        # Unmatched detections -> new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det["bbox"], "age": 0, "cls": det["cls"]}
                results.append((tid, det["bbox"], det["cls"]))

        # Age unmatched tracks and remove old ones
        to_remove = []
        for j, tid in enumerate(track_ids):
            if j not in matched_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        return results

    def _compute_iou_matrix(self, boxes_a, boxes_b):
        """Compute IoU between two lists of boxes."""
        a = np.array(boxes_a)
        b = np.array(boxes_b)
        m, n = len(a), len(b)
        iou = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            xx1 = np.maximum(a[i, 0], b[:, 0])
            yy1 = np.maximum(a[i, 1], b[:, 1])
            xx2 = np.minimum(a[i, 2], b[:, 2])
            yy2 = np.minimum(a[i, 3], b[:, 3])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
            area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            iou[i] = inter / (area_a + area_b - inter + 1e-6)
        return iou


# ── Jersey Color Classifier ─────────────────────────────────────────────────

class TeamClassifier:
    """Classify players into teams based on jersey color using K-means on HSV hues."""

    def __init__(self, team_a_color_hex=None, team_b_color_hex=None):
        self.hue_samples = defaultdict(list)  # pid -> list of avg hues
        self.sat_samples = defaultdict(list)
        self.val_samples = defaultdict(list)
        self.team_a_hex = team_a_color_hex
        self.team_b_hex = team_b_color_hex
        self._assignments = {}

        # Pre-compute reference HSV if colors provided
        self.ref_a_hsv = self._hex_to_hsv(team_a_color_hex) if team_a_color_hex else None
        self.ref_b_hsv = self._hex_to_hsv(team_b_color_hex) if team_b_color_hex else None

    def _hex_to_hsv(self, hex_color):
        """Convert hex color to HSV (OpenCV scale: H 0-180, S 0-255, V 0-255)."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return None
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        pixel = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        return hsv[0, 0]  # [H, S, V]

    def sample(self, frame, bbox, pid):
        """Sample jersey color from upper body region."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 10:
            return

        # Upper 40% of bounding box = jersey area
        jersey_y2 = y1 + int((y2 - y1) * 0.4)
        roi = frame[y1:jersey_y2, x1:x2]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_h = np.median(hsv[:, :, 0])
        avg_s = np.median(hsv[:, :, 1])
        avg_v = np.median(hsv[:, :, 2])

        self.hue_samples[pid].append(avg_h)
        self.sat_samples[pid].append(avg_s)
        self.val_samples[pid].append(avg_v)

        # Keep max 30 samples per player
        if len(self.hue_samples[pid]) > 30:
            self.hue_samples[pid] = self.hue_samples[pid][-30:]
            self.sat_samples[pid] = self.sat_samples[pid][-30:]
            self.val_samples[pid] = self.val_samples[pid][-30:]

    def predict_team(self, pid):
        """Predict team for a player based on accumulated color samples."""
        if pid not in self.hue_samples or len(self.hue_samples[pid]) < 3:
            return None

        avg_h = np.median(self.hue_samples[pid])
        avg_s = np.median(self.sat_samples[pid])
        avg_v = np.median(self.val_samples[pid])

        # If we have reference colors, use distance-based matching
        if self.ref_a_hsv is not None and self.ref_b_hsv is not None:
            dist_a = self._hsv_distance(avg_h, avg_s, avg_v, self.ref_a_hsv)
            dist_b = self._hsv_distance(avg_h, avg_s, avg_v, self.ref_b_hsv)
            return "A" if dist_a <= dist_b else "B"

        return None

    def _hsv_distance(self, h, s, v, ref):
        """Compute weighted HSV distance (hue is circular)."""
        dh = min(abs(h - ref[0]), 180 - abs(h - ref[0]))  # Circular hue distance
        ds = abs(s - ref[1])
        dv = abs(v - ref[2])
        return dh * 2 + ds * 0.5 + dv * 0.3  # Weight hue most

    def finalize_teams(self):
        """K-means on all player hue samples to assign teams."""
        pids_with_data = [pid for pid in self.hue_samples if len(self.hue_samples[pid]) >= 3]

        if not pids_with_data:
            return {}

        # Build feature vectors (median hue, sat, val per player)
        features = []
        for pid in pids_with_data:
            features.append([
                np.median(self.hue_samples[pid]),
                np.median(self.sat_samples[pid]) / 5,  # Scale down
                np.median(self.val_samples[pid]) / 5,
            ])
        features = np.array(features, dtype=np.float32)

        # K-means with k=2
        if len(features) < 2:
            return {pids_with_data[0]: "A"}

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        # Map cluster 0/1 to team A/B using reference colors or position heuristic
        cluster_to_team = self._map_clusters_to_teams(centers, pids_with_data, labels)

        assignments = {}
        for pid, label in zip(pids_with_data, labels):
            assignments[pid] = cluster_to_team[label]

        return assignments

    def _map_clusters_to_teams(self, centers, pids, labels):
        """Map k-means clusters to team A/B."""
        if self.ref_a_hsv is not None and self.ref_b_hsv is not None:
            dist_0_a = self._hsv_distance(centers[0][0], centers[0][1] * 5, centers[0][2] * 5, self.ref_a_hsv)
            dist_0_b = self._hsv_distance(centers[0][0], centers[0][1] * 5, centers[0][2] * 5, self.ref_b_hsv)
            if dist_0_a <= dist_0_b:
                return {0: "A", 1: "B"}
            else:
                return {0: "B", 1: "A"}
        # Fallback: cluster 0 = A, cluster 1 = B
        return {0: "A", 1: "B"}


# ── Jersey Number OCR ────────────────────────────────────────────────────────

class JerseyOCR:
    """Read jersey numbers using EasyOCR — runs periodically to save CPU."""

    def __init__(self, interval_frames=10):
        self.interval = interval_frames
        self.reader = None  # Lazy init to save memory
        self.readings = defaultdict(list)  # pid -> list of detected numbers

    def _ensure_reader(self):
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def should_run(self, frame_idx):
        return frame_idx % self.interval == 0

    def read_number(self, frame, bbox, pid):
        """Attempt to read jersey number from player bounding box."""
        self._ensure_reader()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw < 15 or bh < 20:
            return None

        # Focus on upper-center of body (jersey number area)
        ny1 = y1 + int(bh * 0.15)
        ny2 = y1 + int(bh * 0.55)
        nx1 = x1 + int(bw * 0.15)
        nx2 = x2 - int(bw * 0.15)
        roi = frame[ny1:ny2, nx1:nx2]
        if roi.size == 0:
            return None

        # Upscale small ROIs for better OCR
        if roi.shape[1] < 40:
            roi = cv2.resize(roi, (roi.shape[1] * 3, roi.shape[0] * 3), interpolation=cv2.INTER_CUBIC)

        try:
            results = self.reader.readtext(roi, allowlist="0123456789", detail=1, paragraph=False)
            for (_, text, conf) in results:
                text = text.strip()
                if text.isdigit() and 1 <= int(text) <= 99 and conf > 0.3:
                    self.readings[pid].append(int(text))
                    return int(text)
        except Exception:
            pass
        return None

    def get_best_number(self, pid):
        """Get most frequently detected number for a player."""
        if pid not in self.readings or not self.readings[pid]:
            return None
        from collections import Counter
        counts = Counter(self.readings[pid])
        best, count = counts.most_common(1)[0]
        if count >= 2:  # Require at least 2 consistent readings
            return best
        return None

    def get_all_numbers(self):
        """Get best number for each player."""
        result = {}
        for pid in self.readings:
            num = self.get_best_number(pid)
            if num is not None:
                result[pid] = num
        return result


# ── Lightweight Jersey Number OCR (OpenCV only, no PyTorch) ─────────────────

# Pre-render digit templates for template matching (created once at import)
_DIGIT_TEMPLATES = {}


def _build_digit_templates(h=40, w=24):
    """Render digit templates 0-9 using OpenCV putText for template matching."""
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES:
        return
    for digit in range(10):
        canvas = np.zeros((h, w), dtype=np.uint8)
        text = str(digit)
        # Use a bold font and center the digit
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - tw) // 2
        ty = (h + th) // 2
        cv2.putText(canvas, text, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)
        _DIGIT_TEMPLATES[digit] = canvas
    # Also make italic variants for jersey fonts
    for digit in range(10):
        canvas = np.zeros((h, w), dtype=np.uint8)
        text = str(digit)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.9
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - tw) // 2
        ty = (h + th) // 2
        cv2.putText(canvas, text, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)
        _DIGIT_TEMPLATES[digit + 10] = canvas  # Store as 10-19


def _detect_jersey_number_cv(frame, x1, y1, x2, y2):
    """Detect jersey number using multi-strategy OpenCV analysis.

    Strategy 1: Adaptive threshold + contour analysis + template matching
    Strategy 2: CLAHE-enhanced + Otsu threshold
    Strategy 3: Color-channel separation for colored numbers on colored jerseys

    Returns an int (1-99) or None.
    """
    _build_digit_templates()

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 15 or bh < 25:
        return None

    # Focus on jersey area: upper 10%-55% of body, center 80%
    ny1 = y1 + int(bh * 0.10)
    ny2 = y1 + int(bh * 0.55)
    nx1 = x1 + int(bw * 0.10)
    nx2 = x2 - int(bw * 0.10)
    roi_color = frame[ny1:ny2, nx1:nx2]
    if roi_color.size == 0 or roi_color.shape[0] < 8 or roi_color.shape[1] < 8:
        return None

    # Upscale for better digit recognition
    target_w = 80
    if roi_color.shape[1] < target_w:
        scale = target_w / roi_color.shape[1]
        roi_color = cv2.resize(roi_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # Try multiple binarization strategies
    candidates = []

    # Strategy 1: Adaptive threshold
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    bin1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 8)
    candidates.append(bin1)

    # Strategy 2: CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, bin2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(bin2)

    # Strategy 3: Adaptive with different block size
    bin3 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 10)
    candidates.append(bin3)

    # Strategy 4: Saturation channel (colored numbers stand out)
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, bin4 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(bin4)

    # Try each binary image and take the best result
    best_number = None
    best_confidence = 0

    for binary in candidates:
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        number, confidence = _extract_number_from_binary(binary)
        if number is not None and confidence > best_confidence:
            best_number = number
            best_confidence = confidence

    return best_number


def _extract_number_from_binary(binary):
    """Extract a 1-2 digit number from a binary image. Returns (number, confidence)."""
    roi_h, roi_w = binary.shape

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h = roi_h * 0.20
    max_h = roi_h * 0.95
    min_w = roi_w * 0.04
    max_w = roi_w * 0.65

    digit_contours = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        area = cv2.contourArea(c)
        fill_ratio = area / max(cw * ch, 1)

        # Filter for digit-shaped contours
        if (min_h < ch < max_h
                and min_w < cw < max_w
                and 0.8 < aspect < 6.0
                and area > roi_h * roi_w * 0.015
                and fill_ratio > 0.15):
            digit_contours.append((x, y, cw, ch, c))

    if not digit_contours or len(digit_contours) > 3:
        return None, 0

    # Sort by x position (left to right)
    digit_contours.sort(key=lambda d: d[0])

    # Merge overlapping contours (fragments of same digit)
    merged = []
    for dc in digit_contours:
        if merged:
            prev = merged[-1]
            # If this contour overlaps horizontally with previous, merge
            if dc[0] < prev[0] + prev[2] * 0.6:
                mx = min(prev[0], dc[0])
                my = min(prev[1], dc[1])
                mx2 = max(prev[0] + prev[2], dc[0] + dc[2])
                my2 = max(prev[1] + prev[3], dc[1] + dc[3])
                merged[-1] = (mx, my, mx2 - mx, my2 - my, None)
                continue
        merged.append(dc)

    digit_contours = merged[:2]  # Max 2 digits

    # Classify each digit using template matching
    digits = []
    total_conf = 0
    for x, y, cw, ch, _ in digit_contours:
        digit_roi = binary[y:y + ch, x:x + cw]
        if digit_roi.size == 0:
            continue

        digit, conf = _classify_digit_template(digit_roi)
        if digit is not None:
            digits.append(digit)
            total_conf += conf

    if not digits:
        return None, 0

    # Build number
    number = 0
    for d in digits:
        number = number * 10 + d

    if 1 <= number <= 99:
        avg_conf = total_conf / len(digits)
        return number, avg_conf
    return None, 0


def _classify_digit_template(binary_roi):
    """Classify a digit ROI using template matching against rendered digits.

    Returns (digit, confidence) or (None, 0).
    """
    if binary_roi.size == 0:
        return None, 0

    h, w = binary_roi.shape

    # Also try geometric classification as primary method
    geo_digit, geo_conf = _classify_digit_geometric(binary_roi, w, h)

    # Template matching
    best_digit = None
    best_score = 0.45  # Minimum threshold

    for key, template in _DIGIT_TEMPLATES.items():
        digit = key % 10  # Map 10-19 back to 0-9

        # Resize template to match ROI size
        tmpl_resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)

        # Match
        result = cv2.matchTemplate(binary_roi, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0] if result.size > 0 else 0

        if score > best_score:
            best_score = score
            best_digit = digit

    # Combine geometric and template results
    if best_digit is not None and geo_digit is not None:
        if best_digit == geo_digit:
            return best_digit, best_score + 0.2  # Agreement bonus
        # If template match is strong, prefer it
        if best_score > 0.6:
            return best_digit, best_score
        if geo_conf > 0.6:
            return geo_digit, geo_conf
        return best_digit, best_score
    elif best_digit is not None:
        return best_digit, best_score
    elif geo_digit is not None:
        return geo_digit, geo_conf
    return None, 0


def _classify_digit_geometric(binary_roi, w, h):
    """Classify a digit using geometric features (holes, density, profile)."""
    if binary_roi.size == 0:
        return None, 0

    total_px = w * h
    white_px = cv2.countNonZero(binary_roi)
    density = white_px / max(total_px, 1)
    aspect = h / max(w, 1)

    # Count holes using hierarchy
    contours, hierarchy = cv2.findContours(
        binary_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    holes = 0
    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] >= 0:
                holes += 1

    # Vertical split: left/right balance
    mid_x = max(w // 2, 1)
    left_px = cv2.countNonZero(binary_roi[:, :mid_x])
    right_px = cv2.countNonZero(binary_roi[:, mid_x:])
    lr_ratio = left_px / max(right_px, 1)

    # Horizontal split: top/bottom
    mid_y = max(h // 2, 1)
    top_px = cv2.countNonZero(binary_roi[:mid_y, :])
    bottom_px = cv2.countNonZero(binary_roi[mid_y:, :])
    tb_ratio = top_px / max(bottom_px, 1)

    # Middle horizontal slice
    q1 = h // 4
    q3 = 3 * h // 4
    mid_slice_px = cv2.countNonZero(binary_roi[q1:q3, :])
    mid_density = mid_slice_px / max((q3 - q1) * w, 1)

    # Very thin and tall = likely 1
    if aspect > 2.8 and density < 0.40:
        return 1, 0.7

    # Two holes = 8 or 0
    if holes >= 2:
        if aspect > 1.8:
            return 8, 0.65
        return 8, 0.55

    # One hole
    if holes == 1:
        if aspect < 1.8 and density > 0.35:
            return 0, 0.65
        if tb_ratio > 1.3:
            return 9, 0.55
        if tb_ratio < 0.75:
            return 6, 0.55
        if density > 0.40:
            return 0, 0.5
        return 4, 0.4

    # No holes
    if aspect > 2.2 and density < 0.38:
        return 1, 0.6
    if mid_density < 0.25:
        # Thin middle = likely 1 or 7
        if lr_ratio < 0.7:
            return 7, 0.5
        return 1, 0.45
    if density > 0.55:
        return 2, 0.4
    if lr_ratio > 1.6:
        return 5, 0.45
    if lr_ratio < 0.55:
        return 7, 0.5
    if tb_ratio < 0.65:
        return 4, 0.4
    if tb_ratio > 1.4:
        return 3, 0.4

    # Can't determine
    return None, 0


# ── Track Deduplication ─────────────────────────────────────────────────────

def _deduplicate_tracks(track_positions, team_assignments, track_distances,
                        track_speeds, track_frame_count, heat_per_player,
                        player_jersey_numbers, max_players=30):
    """Merge fragmented track IDs that likely belong to the same player.

    Uses spatial overlap, team assignment, and jersey number to merge tracks.
    Returns mapping: old_pid -> canonical_pid
    """
    pids = sorted(track_positions.keys(), key=lambda p: -track_frame_count.get(p, 0))

    # Build clusters: group tracks that co-locate and share team
    merge_map = {}  # old_pid -> canonical_pid
    canonical = []  # list of (pid, team, avg_x, avg_y, frame_count)

    for pid in pids:
        positions = track_positions[pid]
        if not positions:
            continue

        team = team_assignments.get(pid)
        avg_x = np.mean([p[0] for p in positions])
        avg_y = np.mean([p[1] for p in positions])
        fc = track_frame_count.get(pid, 0)

        # Get jersey number if detected
        jersey = None
        if pid in player_jersey_numbers and player_jersey_numbers[pid]:
            jersey = max(player_jersey_numbers[pid], key=player_jersey_numbers[pid].get)

        # Try to merge with an existing canonical track
        merged = False

        # If this track has very few frames, try harder to merge
        for c_pid, c_team, c_avg_x, c_avg_y, c_fc, c_jersey in canonical:
            # Same team required
            if team is not None and c_team is not None and team != c_team:
                continue

            # Same jersey number = definitely same player
            if jersey is not None and c_jersey is not None:
                if jersey == c_jersey:
                    merge_map[pid] = c_pid
                    merged = True
                    break
                else:
                    continue  # Different numbers = different players

            # Short tracks (< 10 frames) are likely fragments
            if fc < 10:
                # Merge if avg position is close
                dist = np.hypot(avg_x - c_avg_x, avg_y - c_avg_y)
                if dist < 80:
                    merge_map[pid] = c_pid
                    merged = True
                    break

        if not merged:
            canonical.append((pid, team, avg_x, avg_y, fc, jersey))
            merge_map[pid] = pid

    # Limit to max_players canonical IDs (keep the most-seen ones)
    if len(canonical) > max_players:
        canonical = canonical[:max_players]
        valid_canonical = {c[0] for c in canonical}
        # Remap orphaned tracks to nearest canonical
        for pid in list(merge_map.keys()):
            cpid = merge_map[pid]
            if cpid not in valid_canonical:
                # Find nearest canonical with same team
                team = team_assignments.get(pid)
                positions = track_positions.get(pid, [])
                if positions:
                    avg_x = np.mean([p[0] for p in positions])
                    avg_y = np.mean([p[1] for p in positions])
                    best_cpid = None
                    best_dist = float("inf")
                    for c_pid, c_team, c_avg_x, c_avg_y, _, _ in canonical:
                        if team is not None and c_team is not None and team != c_team:
                            continue
                        d = np.hypot(avg_x - c_avg_x, avg_y - c_avg_y)
                        if d < best_dist:
                            best_dist = d
                            best_cpid = c_pid
                    if best_cpid:
                        merge_map[pid] = best_cpid
                    else:
                        merge_map[pid] = canonical[0][0]
                else:
                    merge_map[pid] = canonical[0][0]

    return merge_map


# ── xG Estimation ────────────────────────────────────────────────────────────

def _estimate_xg(distance_m, lateral_offset_m, on_target):
    """Estimate expected goals (xG) from shot distance and angle.

    Simple logistic model calibrated to real-world xG data:
    - Penalty spot (11m) ~ 0.76 xG
    - Edge of box (16m) ~ 0.15 xG
    - 25m out ~ 0.04 xG
    - 35m+ ~ 0.01 xG
    """
    # Base xG from distance (logistic decay)
    if distance_m <= 0:
        return 0.0
    base = 1.0 / (1.0 + np.exp(0.3 * (distance_m - 10)))

    # Angle penalty (shots from wider angles are harder)
    angle_factor = 1.0 / (1.0 + (lateral_offset_m / 8.0) ** 2)

    # On-target bonus
    target_factor = 1.2 if on_target else 0.7

    xg = base * angle_factor * target_factor
    return min(max(xg, 0.01), 0.95)


def _detect_formation(position_snapshots, num_outfield=10):
    """Detect team formation from position snapshots.

    Clusters player positions into lines (defense, midfield, attack)
    and returns formation string like "4-3-3" or "4-4-2".
    """
    if not position_snapshots or len(position_snapshots) < 3:
        return "Unknown"

    # Average positions across snapshots
    # Each snapshot is a list of (x, y) tuples
    all_x = []
    for snap in position_snapshots[-10:]:  # Use last 10 snapshots
        for x, y in snap:
            all_x.append(x)

    if not all_x:
        return "Unknown"

    # Collect average positions per player slot
    # Take median snapshot size and average positions
    avg_positions = []
    for snap in position_snapshots[-10:]:
        for x, y in snap:
            avg_positions.append(x)

    if len(avg_positions) < 8:
        return "Unknown"

    # Cluster x-positions into 3-4 lines using simple histogram
    x_arr = np.array(avg_positions)
    x_sorted = np.sort(x_arr)

    # Use K-means to find 3 or 4 clusters
    best_formation = "4-4-2"  # Default
    best_score = float("inf")

    for k in [3, 4]:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        try:
            x_float = x_sorted.reshape(-1, 1).astype(np.float32)
            _, labels, centers = cv2.kmeans(x_float, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            labels = labels.flatten()

            # Count players per line
            line_counts = []
            for i in range(k):
                line_counts.append(int(np.sum(labels == i)))

            # Sort by center position (back to front)
            sorted_lines = sorted(zip(centers.flatten(), line_counts))
            counts = [c for _, c in sorted_lines]

            # Normalize to ~10 outfield (GK excluded)
            total = sum(counts)
            if total > 0:
                ratio = num_outfield / total
                counts = [max(1, round(c * ratio)) for c in counts]
                # Adjust to sum to 10
                while sum(counts) > num_outfield:
                    counts[counts.index(max(counts))] -= 1
                while sum(counts) < num_outfield:
                    counts[counts.index(min(counts))] += 1

                # Ensure no line has 0 players
                counts = [max(1, c) for c in counts]
                while sum(counts) > num_outfield:
                    counts[counts.index(max(counts))] -= 1

            formation_str = "-".join(str(c) for c in counts)

            # Score: prefer common formations
            common = {"4-3-3", "4-4-2", "3-5-2", "4-2-3-1", "3-4-3", "5-3-2", "4-1-4-1", "5-4-1", "4-5-1", "4-1-2-3", "3-4-1-2", "4-2-4", "4-3-2-1"}
            score = 0 if formation_str in common else 1
            if score < best_score:
                best_score = score
                best_formation = formation_str
        except Exception:
            continue

    return best_formation


# ── Main Processing Pipeline ─────────────────────────────────────────────────

def make_colored_heatmap(heat, path, cmap=cv2.COLORMAP_JET):
    if heat.max() > 0:
        norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(heat, dtype=np.uint8)
    colored = cv2.applyColorMap(norm, cmap)
    cv2.imwrite(str(path), colored)


def process_match(video_path, db_manager, team_a_name=None, team_b_name=None,
                  max_seconds=0, frame_skip=2, enable_ocr=True, display=False):
    """Full match processing with ONNX YOLO, tracking, color classification, and OCR."""

    print(f"\nProcessing: {video_path.name}")

    # ── Video metadata ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if max_seconds > 0:
        total_frames = min(total_frames, int(max_seconds * fps))

    # Process at reduced resolution for memory efficiency
    proc_w, proc_h = 640, 360
    print(f"  Video: {orig_w}x{orig_h} @ {fps:.1f} FPS ({total_frames} frames)")
    print(f"  Processing at {proc_w}x{proc_h}, every {frame_skip} frames")

    # ── Resolve team names ──
    if not team_a_name or not team_b_name:
        fn_a, fn_b = _extract_teams_from_filename(video_path.name, db_manager)
        team_a_name = team_a_name or fn_a or "Team A"
        team_b_name = team_b_name or fn_b or "Team B"
    print(f"  Teams: {team_a_name} vs {team_b_name}")

    # ── Load rosters from DB ──
    roster_a = db_manager.get_roster_for_team(team_a_name) if db_manager else []
    roster_b = db_manager.get_roster_for_team(team_b_name) if db_manager else []
    print(f"  Rosters: {team_a_name} ({len(roster_a)}), {team_b_name} ({len(roster_b)})")

    # Build jersey number -> name lookup
    roster_a_map = {}  # number -> {name, position}
    for p in roster_a:
        if p.get("jersey_number") is not None:
            roster_a_map[p["jersey_number"]] = {"name": p["name"], "position": p.get("position_default", "")}
    roster_b_map = {}
    for p in roster_b:
        if p.get("jersey_number") is not None:
            roster_b_map[p["jersey_number"]] = {"name": p["name"], "position": p.get("position_default", "")}

    # ── Load team colors ──
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

    # ── Initialize components ──
    print("  Loading ONNX model...")
    detector = ONNXDetector(str(ONNX_MODEL), conf_thresh=0.3, iou_thresh=0.45)
    tracker = SimpleTracker(max_age=15, iou_thresh=0.2)
    team_clf = TeamClassifier(team_a_color_hex=color_a, team_b_color_hex=color_b)
    ocr = JerseyOCR(interval_frames=15) if enable_ocr else None
    print(f"  ONNX model loaded ({ONNX_MODEL.stat().st_size / 1e6:.1f} MB)")

    if display:
        cv2.namedWindow("TactiVision Pro - Live Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TactiVision Pro - Live Tracking", 1280, 720)
        print("  Display mode ON - press 'q' or Esc to quit early")

    # ── Output dir ──
    out_dir = OUTPUT_BASE / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data structures ──
    heat_global = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_team_A = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_team_B = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_ball = np.zeros((proc_h, proc_w), dtype=np.float32)
    heat_per_player = {}  # pid -> heatmap (capped at 30)

    track_positions = defaultdict(list)  # pid -> [(cx, cy, t)] last 10
    track_distances = defaultdict(float)  # pid -> cumulative distance (meters)
    track_speeds = defaultdict(list)  # pid -> [speed_mps] (rolling window)
    track_frame_count = defaultdict(int)  # pid -> number of frames seen

    ball_positions = []
    possession_a = 0
    possession_b = 0
    possession_frames = 0

    # ── Per-player FotMob-level stats ──
    player_touches = defaultdict(int)        # pid -> touch count (ball near player)
    player_last_touch_time = {}              # pid -> last touch timestamp (throttle)
    player_passes_made = defaultdict(int)     # pid -> passes attempted
    player_passes_completed = defaultdict(int)  # pid -> passes completed
    player_passes_received = defaultdict(int)   # pid -> passes received
    player_key_passes = defaultdict(int)      # pid -> passes into final third
    player_interceptions = defaultdict(int)   # pid -> interceptions won
    player_last_interception_time = {}       # pid -> last interception timestamp (throttle)
    player_tackles = defaultdict(int)         # pid -> tackle events
    player_duels_won = defaultdict(int)       # pid -> duels won
    player_duels_total = defaultdict(int)     # pid -> total duels
    active_duel_cooldowns = {}               # (pid1, pid2) -> last duel timestamp
    player_possession_time = defaultdict(float)  # pid -> seconds with ball
    player_high_intensity_runs = defaultdict(int)  # pid -> runs 5-7 m/s
    player_sprints_count = defaultdict(int)   # pid -> sprint events

    # ── Shot detection state ──
    shot_events = []  # list of {pid, team, timestamp, position, distance_m, on_target, xg}
    last_shot_frame = 0  # Cooldown: no shots within 3 seconds of each other
    SHOT_COOLDOWN_FRAMES = int(3 * fps)  # 3 seconds between shots minimum
    GOAL_A_X = proc_w * 0.05  # Left goal X threshold
    GOAL_B_X = proc_w * 0.95  # Right goal X threshold
    SHOT_SPEED_THRESHOLD = 4.0  # m/s ball speed to consider a shot (raised from 3.0)
    PENALTY_BOX_WIDTH = proc_w * 0.15  # ~16 yards in pixel terms

    # ── Sprint tracking (events, not individual frames) ──
    sprint_events = []  # list of {pid, team, start_t, end_t, max_speed, distance}
    active_sprints = {}  # pid -> {start_t, start_pos, max_speed, frames}
    SPRINT_SPEED_THRESHOLD = 8.5  # m/s (~30.6 km/h) — raised to filter camera pan noise
    HIGH_INTENSITY_THRESHOLD = 6.0  # m/s (~21.6 km/h) — raised for broadcast footage
    SPRINT_MIN_DURATION = 1.5  # seconds — must sustain for at least 1.5s to reduce false positives

    # ── Pass detection state ──
    pass_events = []
    last_possessor = None
    last_possessor_team = None
    last_possessor_pos = None
    last_possession_frame = 0
    possession_hold_frames = 0
    MIN_POSSESSION_FRAMES = 2
    PASS_MIN_DISTANCE = 8
    PASS_MAX_GAP_FRAMES = 45

    # ── Duel detection state ──
    # When players from opposing teams are within close proximity near the ball
    DUEL_DISTANCE = 25  # pixels — players this close are in a duel

    meter_per_px = 105.0 / proc_w  # Approximate pitch scaling

    # ── Formation detection ──
    formation_snapshots_a = []  # periodic position snapshots for team A
    formation_snapshots_b = []
    FORMATION_INTERVAL = 100  # sample formation every N processed frames

    # ── Passing network ──
    passing_network = defaultdict(lambda: defaultdict(int))  # from_pid -> to_pid -> count

    # ── Zone control (pitch divided into 6 zones: 3 horizontal x 2 vertical) ──
    zone_time = {"A": defaultdict(int), "B": defaultdict(int)}

    # ── Jersey number OCR state (lightweight OpenCV-based) ──
    player_jersey_numbers = {}  # tid -> Counter of detected numbers
    jersey_ocr_interval = 10  # attempt OCR every N frames

    # ── Process frames ──
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    processed = 0
    start_time = time.time()
    print("  Processing frames...")

    while frame_idx < total_frames:
        ret, frame_full = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        # Resize for detection (low-res)
        frame = cv2.resize(frame_full, (proc_w, proc_h))
        t = frame_idx / fps
        processed += 1

        # ── Detect ──
        detections = detector.detect(frame, classes_filter=[PERSON_CLASS, BALL_CLASS])

        # Scale factors for mapping low-res detections back to full-res
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h

        # Separate players and ball
        player_dets = [d for d in detections if d["cls"] == PERSON_CLASS]
        ball_dets = [d for d in detections if d["cls"] == BALL_CLASS]

        # ── Track players ──
        tracked = tracker.update(player_dets)

        # ── Process each tracked player ──
        frame_player_positions = {}
        frame_player_bboxes = {}
        for tid, bbox, cls in tracked:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            track_frame_count[tid] += 1

            # Heatmap
            ix, iy = int(cx), int(cy)
            if 0 <= ix < proc_w and 0 <= iy < proc_h:
                heat_global[iy, ix] += 1

            # Track position history (keep last 10)
            track_positions[tid].append((cx, cy, t))
            if len(track_positions[tid]) > 10:
                track_positions[tid] = track_positions[tid][-10:]

            # Distance and speed (smoothed over 3 frames)
            speed_mps = 0.0
            if len(track_positions[tid]) >= 3:
                # Use 3-frame window for smoother speed
                p_old = track_positions[tid][-3]
                dx = cx - p_old[0]
                dy = cy - p_old[1]
                dist_px = np.hypot(dx, dy)
                dt = t - p_old[2]
                if dt > 0 and dist_px < 150:  # Cap unrealistic jumps
                    speed_mps = (dist_px * meter_per_px) / dt
                    if speed_mps > 12.0:  # Cap at max human speed
                        speed_mps = 0.0  # Likely a tracking glitch
                    elif dist_px > 3.0:  # Filter out camera-pan noise (< 3px ≈ 0.5m)
                        track_distances[tid] += (dist_px * meter_per_px) / 3  # Per-frame portion
            elif len(track_positions[tid]) >= 2:
                prev = track_positions[tid][-2]
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist_px = np.hypot(dx, dy)
                dt = t - prev[2]
                if dt > 0 and dist_px < 100:
                    speed_mps = (dist_px * meter_per_px) / dt
                    if speed_mps > 12.0:
                        speed_mps = 0.0
                    elif dist_px > 3.0:  # Filter camera-pan noise
                        track_distances[tid] += dist_px * meter_per_px

            # Keep rolling speed window (last 20 readings)
            track_speeds[tid].append(speed_mps)
            if len(track_speeds[tid]) > 20:
                track_speeds[tid] = track_speeds[tid][-20:]

            # ── Sprint and high-intensity run detection ──
            if speed_mps >= HIGH_INTENSITY_THRESHOLD:
                player_high_intensity_runs[tid] += 1

            if speed_mps >= SPRINT_SPEED_THRESHOLD:
                if tid not in active_sprints:
                    active_sprints[tid] = {
                        "start_t": t, "start_pos": (cx, cy),
                        "max_speed": speed_mps, "frames": 1
                    }
                else:
                    active_sprints[tid]["frames"] += 1
                    active_sprints[tid]["max_speed"] = max(
                        active_sprints[tid]["max_speed"], speed_mps
                    )
            else:
                if tid in active_sprints:
                    sp = active_sprints.pop(tid)
                    duration = t - sp["start_t"]
                    if duration >= SPRINT_MIN_DURATION:
                        team = team_clf.predict_team(tid) or "?"
                        sprint_dist = np.hypot(
                            cx - sp["start_pos"][0], cy - sp["start_pos"][1]
                        ) * meter_per_px
                        sprint_events.append({
                            "pid": tid, "team": team,
                            "start_t": sp["start_t"], "end_t": t,
                            "duration": round(duration, 2),
                            "max_speed": round(sp["max_speed"], 2),
                            "distance_m": round(sprint_dist, 1),
                        })
                        player_sprints_count[tid] += 1

            # Per-player heatmap
            if 0 <= ix < proc_w and 0 <= iy < proc_h:
                if tid not in heat_per_player and len(heat_per_player) < 30:
                    heat_per_player[tid] = np.zeros((proc_h, proc_w), dtype=np.float32)
                if tid in heat_per_player:
                    heat_per_player[tid][iy, ix] += 1

            # Jersey color sampling
            team_clf.sample(frame, bbox, tid)

            # Get live team prediction
            team = team_clf.predict_team(tid)
            frame_player_positions[tid] = (cx, cy, team)
            frame_player_bboxes[tid] = bbox

            # Team heatmap
            if team and 0 <= ix < proc_w and 0 <= iy < proc_h:
                if team == "A":
                    heat_team_A[iy, ix] += 1
                else:
                    heat_team_B[iy, ix] += 1

            # ── Lightweight jersey number OCR using OpenCV ──
            if frame_idx % jersey_ocr_interval == 0:
                # Skip if we already have 5+ consistent readings for this player
                max_readings = max(player_jersey_numbers[tid].values()) if tid in player_jersey_numbers and player_jersey_numbers[tid] else 0
                if max_readings < 5:
                    fx1, fy1 = int(x1 * scale_x), int(y1 * scale_y)
                    fx2, fy2 = int(x2 * scale_x), int(y2 * scale_y)
                    num = _detect_jersey_number_cv(frame_full, fx1, fy1, fx2, fy2)
                    if num is not None:
                        if tid not in player_jersey_numbers:
                            player_jersey_numbers[tid] = defaultdict(int)
                        player_jersey_numbers[tid][num] += 1

            # OCR via EasyOCR if enabled
            if ocr and ocr.should_run(frame_idx):
                full_bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                ocr.read_number(frame_full, full_bbox, tid)

        # ── Ball tracking (YOLO + color fallback + trajectory prediction) ──
        ball_pos_this_frame = None
        ball_is_real_detection = False
        if ball_dets:
            best_ball = max(ball_dets, key=lambda d: d["conf"])
            bx = (best_ball["bbox"][0] + best_ball["bbox"][2]) / 2
            by = (best_ball["bbox"][1] + best_ball["bbox"][3]) / 2
            ball_pos_this_frame = (bx, by)
            ball_is_real_detection = True
            ball_positions.append({"frame": frame_idx, "x": bx, "y": by, "t": t})
        elif ball_positions and len(ball_positions) >= 1 and processed % 2 == 0:
            # Color-based ball detection fallback — search near last known position
            # Only run every 2nd processed frame to save memory
            last_bp = ball_positions[-1]
            if frame_idx - last_bp.get("frame", 0) < frame_skip * 8:
                search_radius = 50
                sx1 = max(0, int(last_bp["x"] - search_radius))
                sy1 = max(0, int(last_bp["y"] - search_radius))
                sx2 = min(proc_w, int(last_bp["x"] + search_radius))
                sy2 = min(proc_h, int(last_bp["y"] + search_radius))
                search_roi = frame[sy1:sy2, sx1:sx2]

                if search_roi.size > 0:
                    hsv_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
                    white_mask = cv2.inRange(hsv_roi, (0, 0, 210), (180, 50, 255))
                    yellow_mask = cv2.inRange(hsv_roi, (15, 120, 180), (35, 255, 255))
                    ball_mask = cv2.bitwise_or(white_mask, yellow_mask)
                    del hsv_roi, white_mask, yellow_mask  # Free immediately

                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kern)
                    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    del ball_mask

                    best_ball_c = None
                    best_score = 0
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < 20 or area > 500:
                            continue
                        perimeter = cv2.arcLength(c, True)
                        if perimeter == 0:
                            continue
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.6:
                            continue
                        score = circularity * min(area, 150)
                        if score > best_score:
                            best_score = score
                            best_ball_c = c

                    if best_ball_c is not None and best_score > 30:
                        M = cv2.moments(best_ball_c)
                        if M["m00"] > 0:
                            bx = sx1 + M["m10"] / M["m00"]
                            by = sy1 + M["m01"] / M["m00"]
                            ball_pos_this_frame = (bx, by)
                            ball_is_real_detection = True
                            ball_positions.append({"frame": frame_idx, "x": bx, "y": by, "t": t})

        # Trajectory prediction fallback (if no detection at all)
        if ball_pos_this_frame is None and ball_positions and len(ball_positions) >= 2:
            last2 = ball_positions[-2:]
            if frame_idx - last2[-1].get("frame", 0) < frame_skip * 5:
                dx_b = last2[-1]["x"] - last2[-2]["x"]
                dy_b = last2[-1]["y"] - last2[-2]["y"]
                pred_x = last2[-1]["x"] + dx_b
                pred_y = last2[-1]["y"] + dy_b
                if 0 < pred_x < proc_w and 0 < pred_y < proc_h:
                    ball_pos_this_frame = (pred_x, pred_y)

        # Ball heatmap (real detections only)
        if ball_is_real_detection and ball_pos_this_frame:
            bix, biy = int(ball_pos_this_frame[0]), int(ball_pos_this_frame[1])
            if 0 <= bix < proc_w and 0 <= biy < proc_h:
                heat_ball[biy, bix] += 1

        # ── Possession, touches, passes, duels, shots ──
        if ball_pos_this_frame:
            bx, by = ball_pos_this_frame

            # Find closest player to ball
            min_dist = float("inf")
            closest_pid = None
            closest_team = None
            closest_pos = None
            # Also find closest opponent for duel detection
            second_closest_pid = None
            second_closest_team = None
            second_dist = float("inf")

            for pid, (px, py, pteam) in frame_player_positions.items():
                d = np.hypot(px - bx, py - by)
                if d < min_dist and d < 50:
                    second_dist = min_dist
                    second_closest_pid = closest_pid
                    second_closest_team = closest_team
                    min_dist = d
                    closest_pid = pid
                    closest_team = pteam
                    closest_pos = (px, py)
                elif d < second_dist and d < 50:
                    second_dist = d
                    second_closest_pid = pid
                    second_closest_team = pteam

            if closest_team:
                possession_frames += 1
                if closest_team == "A":
                    possession_a += 1
                else:
                    possession_b += 1

                # ── Touch counting (throttled: max 1 touch per 0.5s per player) ──
                if min_dist < 12:  # Close enough for a touch (12px ≈ 2m)
                    last_touch_t = player_last_touch_time.get(closest_pid, -1)
                    if t - last_touch_t >= 1.0:
                        player_touches[closest_pid] += 1
                        player_last_touch_time[closest_pid] = t
                    # Track possession time
                    dt_frame = frame_skip / fps
                    player_possession_time[closest_pid] += dt_frame

                # ── Duel detection (throttled: max 1 duel per 2s per pair) ──
                if (second_closest_pid is not None
                        and closest_team != second_closest_team
                        and second_dist < DUEL_DISTANCE):
                    duel_pair = (min(closest_pid, second_closest_pid), max(closest_pid, second_closest_pid))
                    last_duel_t = active_duel_cooldowns.get(duel_pair, -10)
                    if t - last_duel_t >= 2.0:
                        player_duels_total[closest_pid] += 1
                        player_duels_total[second_closest_pid] += 1
                        if min_dist < second_dist:
                            player_duels_won[closest_pid] += 1
                        active_duel_cooldowns[duel_pair] = t

                # ── Zone control ──
                if closest_pos:
                    zone_x = min(int(closest_pos[0] / (proc_w / 3)), 2)
                    zone_y = min(int(closest_pos[1] / (proc_h / 2)), 1)
                    zone_key = f"z{zone_x}_{zone_y}"
                    if closest_team in zone_time:
                        zone_time[closest_team][zone_key] += 1

                # ── Pass detection ──
                if closest_pid == last_possessor:
                    possession_hold_frames += 1
                else:
                    # Possessor changed — check for pass
                    if (last_possessor is not None
                            and last_possessor_team == closest_team
                            and possession_hold_frames >= MIN_POSSESSION_FRAMES
                            and last_possessor_pos is not None
                            and closest_pos is not None):

                        pass_dist = np.hypot(
                            closest_pos[0] - last_possessor_pos[0],
                            closest_pos[1] - last_possessor_pos[1]
                        )
                        frame_gap = frame_idx - last_possession_frame

                        if pass_dist >= PASS_MIN_DISTANCE and frame_gap <= PASS_MAX_GAP_FRAMES:
                            dx_pass = closest_pos[0] - last_possessor_pos[0]
                            dy_pass = closest_pos[1] - last_possessor_pos[1]

                            if closest_team == "A":
                                forward = dx_pass > 0
                            else:
                                forward = dx_pass < 0

                            abs_dx = abs(dx_pass)
                            abs_dy = abs(dy_pass)

                            if abs_dy > abs_dx * 1.5:
                                direction = "lateral"
                            elif forward:
                                direction = "diagonal_forward" if abs_dy > abs_dx * 0.5 else "forward"
                            else:
                                direction = "diagonal_backward" if abs_dy > abs_dx * 0.5 else "backward"

                            # Is it a key pass? (into final attacking third)
                            is_key = False
                            if closest_team == "A" and closest_pos[0] > proc_w * 0.67:
                                is_key = True
                            elif closest_team == "B" and closest_pos[0] < proc_w * 0.33:
                                is_key = True

                            pass_events.append({
                                "frame": frame_idx,
                                "from_id": last_possessor,
                                "to_id": closest_pid,
                                "team": closest_team,
                                "timestamp": t,
                                "distance_px": round(pass_dist, 1),
                                "distance_m": round(pass_dist * meter_per_px, 1),
                                "direction": direction,
                                "success": True,
                                "key_pass": is_key,
                            })

                            # Per-player pass stats
                            player_passes_made[last_possessor] += 1
                            player_passes_completed[last_possessor] += 1
                            player_passes_received[closest_pid] += 1
                            passing_network[last_possessor][closest_pid] += 1
                            if is_key:
                                player_key_passes[last_possessor] += 1

                    # Interception (different team)
                    elif (last_possessor is not None
                          and last_possessor_team is not None
                          and last_possessor_team != closest_team
                          and possession_hold_frames >= MIN_POSSESSION_FRAMES):
                        pass_events.append({
                            "frame": frame_idx,
                            "from_id": last_possessor,
                            "to_id": closest_pid,
                            "team": last_possessor_team,
                            "timestamp": t,
                            "distance_px": 0, "distance_m": 0,
                            "direction": "intercepted",
                            "success": False, "key_pass": False,
                        })
                        player_passes_made[last_possessor] += 1  # Attempted but failed
                        last_int_t = player_last_interception_time.get(closest_pid, -10)
                        if t - last_int_t >= 2.0:
                            player_interceptions[closest_pid] += 1
                            player_last_interception_time[closest_pid] = t

                    last_possessor = closest_pid
                    last_possessor_team = closest_team
                    last_possessor_pos = closest_pos
                    last_possession_frame = frame_idx
                    possession_hold_frames = 1

            # ── Shot detection (with cooldown) ──
            if (ball_is_real_detection and len(ball_positions) >= 3
                    and frame_idx - last_shot_frame > SHOT_COOLDOWN_FRAMES):
                bp = ball_positions
                ball_dx = bp[-1]["x"] - bp[-3]["x"]
                ball_dy = bp[-1]["y"] - bp[-3]["y"]
                ball_dt = bp[-1]["t"] - bp[-3]["t"]
                if ball_dt > 0:
                    ball_speed = np.hypot(ball_dx, ball_dy) * meter_per_px / ball_dt
                    ball_x = bp[-1]["x"]

                    # Ball moving toward goal A (left) at high speed
                    if ball_speed > SHOT_SPEED_THRESHOLD and ball_dx < -5 and ball_x < PENALTY_BOX_WIDTH:
                        if closest_team == "B" and closest_pid is not None:
                            on_target = ball_x < GOAL_A_X + 10
                            shot_dist_m = ball_x * meter_per_px
                            xg = _estimate_xg(shot_dist_m, abs(ball_dy) * meter_per_px, on_target)
                            shot_events.append({
                                "pid": closest_pid, "team": "B",
                                "timestamp": t, "frame": frame_idx,
                                "position_x": round(ball_x / proc_w, 4),
                                "position_y": round(bp[-1]["y"] / proc_h, 4),
                                "distance_m": round(shot_dist_m, 1),
                                "ball_speed": round(ball_speed, 1),
                                "on_target": on_target, "xg": round(xg, 3),
                            })
                            last_shot_frame = frame_idx

                    # Ball moving toward goal B (right) at high speed
                    elif ball_speed > SHOT_SPEED_THRESHOLD and ball_dx > 5 and ball_x > proc_w - PENALTY_BOX_WIDTH:
                        if closest_team == "A" and closest_pid is not None:
                            on_target = ball_x > GOAL_B_X - 10
                            shot_dist_m = (proc_w - ball_x) * meter_per_px
                            xg = _estimate_xg(shot_dist_m, abs(ball_dy) * meter_per_px, on_target)
                            shot_events.append({
                                "pid": closest_pid, "team": "A",
                                "timestamp": t, "frame": frame_idx,
                                "position_x": round(ball_x / proc_w, 4),
                                "position_y": round(bp[-1]["y"] / proc_h, 4),
                                "distance_m": round(shot_dist_m, 1),
                                "ball_speed": round(ball_speed, 1),
                                "on_target": on_target, "xg": round(xg, 3),
                            })
                            last_shot_frame = frame_idx

        # ── Formation sampling (periodic) ──
        if processed % FORMATION_INTERVAL == 0 and frame_player_positions:
            pos_a = [(px, py) for pid, (px, py, tm) in frame_player_positions.items() if tm == "A"]
            pos_b = [(px, py) for pid, (px, py, tm) in frame_player_positions.items() if tm == "B"]
            if len(pos_a) >= 8:
                formation_snapshots_a.append(pos_a)
            if len(pos_b) >= 8:
                formation_snapshots_b.append(pos_b)

        # ── Live Display ──
        if display:
            # Draw on the full-res frame for better visual quality
            disp = frame_full if frame_full is not None else cv2.resize(frame, (orig_w, orig_h))
            sx, sy = orig_w / proc_w, orig_h / proc_h

            # Draw player boxes with team colors
            for tid, bbox, cls in tracked:
                x1d, y1d, x2d, y2d = bbox
                # Scale to display resolution
                dx1, dy1 = int(x1d * sx), int(y1d * sy)
                dx2, dy2 = int(x2d * sx), int(y2d * sy)

                team = frame_player_positions.get(tid, (0, 0, None))[2]
                if tid == last_possessor:
                    color = (0, 255, 255)  # Yellow = possessor
                    thickness = 3
                elif team == "A":
                    color = (0, 0, 255)  # Red = Team A
                    thickness = 2
                elif team == "B":
                    color = (255, 100, 0)  # Blue = Team B
                    thickness = 2
                else:
                    color = (200, 200, 200)  # Gray = unassigned
                    thickness = 1

                cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), color, thickness)

                # Label: player name from roster if jersey number known
                jersey_num = None
                # Check OpenCV OCR results
                if tid in player_jersey_numbers and player_jersey_numbers[tid]:
                    best_num = max(player_jersey_numbers[tid], key=player_jersey_numbers[tid].get)
                    if player_jersey_numbers[tid][best_num] >= 2:
                        jersey_num = best_num
                # Check EasyOCR results
                if jersey_num is None and ocr and ocr.get_best_number(tid):
                    jersey_num = ocr.get_best_number(tid)

                if jersey_num is not None:
                    # Look up player name from roster
                    r_map = roster_a_map if team == "A" else roster_b_map
                    entry = r_map.get(jersey_num)
                    if entry:
                        # Show name (shortened for display)
                        pname = entry["name"]
                        if len(pname) > 15:
                            pname = pname.split()[-1]  # Last name only
                        label = f"#{jersey_num} {pname}"
                    else:
                        label = f"#{jersey_num}"
                else:
                    label = f"T{tid}"

                if tid == last_possessor:
                    label += " *"

                # Draw label with background for readability
                cv2.putText(disp, label, (dx1, dy1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, label, (dx1, dy1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Draw ball
            if ball_dets:
                best_ball = max(ball_dets, key=lambda d: d["conf"])
                bx1, by1, bx2, by2 = best_ball["bbox"]
                cv2.rectangle(disp, (int(bx1 * sx), int(by1 * sy)),
                              (int(bx2 * sx), int(by2 * sy)), (0, 255, 255), 2)
                cv2.putText(disp, "BALL", (int(bx1 * sx), int(by1 * sy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # HUD overlay
            poss_tot = max(possession_a + possession_b, 1)
            pa_pct = possession_a / poss_tot * 100
            pb_pct = possession_b / poss_tot * 100
            hud_lines = [
                f"TactiVision Pro | Frame {frame_idx}/{total_frames}",
                f"{team_a_name} {pa_pct:.0f}% - {pb_pct:.0f}% {team_b_name}",
                f"Players: {len(frame_player_positions)} | Passes: {len(pass_events)} | Sprints: {len(sprint_events)} | Ball: {'YES' if ball_dets else 'no'}",
            ]
            for i, line in enumerate(hud_lines):
                y_pos = 30 + i * 25
                # Dark background for readability
                cv2.putText(disp, line, (12, y_pos + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, line, (12, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("TactiVision Pro - Live Tracking", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or Esc to quit
                print("  User pressed quit - stopping early")
                break

        # Release full-res frame
        del frame_full

        # ── Progress ──
        if processed % 200 == 0:
            elapsed = time.time() - start_time
            pct = frame_idx / total_frames * 100
            mem_mb = 0
            try:
                import psutil
                mem_mb = psutil.Process().memory_info().rss / 1e6
            except ImportError:
                pass
            print(f"    Frame {frame_idx}/{total_frames} ({pct:.0f}%) - "
                  f"{processed} processed - {elapsed:.0f}s - mem={mem_mb:.0f}MB")

        # Periodic GC (more aggressive for long matches)
        if processed % 200 == 0:
            gc.collect()
            # Trim ball_positions to last 2000 entries to prevent memory growth
            if len(ball_positions) > 2000:
                ball_positions = ball_positions[-2000:]

    cap.release()
    if display:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"  Done: {processed} frames in {elapsed:.1f}s ({processed/max(elapsed,1):.1f} fps)")

    # ── Finalize team assignments ──
    print("  Finalizing team assignments...")
    team_assignments = team_clf.finalize_teams()

    # ── Merge OpenCV OCR detections into jersey_numbers ──
    jersey_numbers = {}
    # OpenCV-based detections
    for tid, counter in player_jersey_numbers.items():
        if counter:
            best_num = max(counter, key=counter.get)
            best_count = counter[best_num]
            if best_count >= 2:  # Require at least 2 consistent readings
                jersey_numbers[tid] = best_num
    # EasyOCR detections (if enabled) — merge, preferring higher confidence
    if ocr:
        easyocr_numbers = ocr.get_all_numbers()
        for tid, num in easyocr_numbers.items():
            if tid not in jersey_numbers:
                jersey_numbers[tid] = num
        print(f"  OCR detected {len(jersey_numbers)} jersey numbers (CV + EasyOCR)")
    else:
        print(f"  OCR detected {len(jersey_numbers)} jersey numbers (OpenCV)")

    # ── Deduplicate tracks ──
    print("  Deduplicating tracks...")
    raw_track_count = tracker.next_id - 1
    merge_map = _deduplicate_tracks(
        track_positions, team_assignments, track_distances,
        track_speeds, track_frame_count, heat_per_player,
        player_jersey_numbers, max_players=30,
    )

    # Apply merge map: consolidate data into canonical PIDs
    canonical_pids = set(merge_map.values())
    # Remap jersey numbers
    canonical_jerseys = {}
    for old_pid, num in jersey_numbers.items():
        cpid = merge_map.get(old_pid, old_pid)
        canonical_jerseys[cpid] = num
    jersey_numbers = canonical_jerseys

    # Remap team assignments
    canonical_teams = {}
    for old_pid, team in team_assignments.items():
        cpid = merge_map.get(old_pid, old_pid)
        if cpid not in canonical_teams:
            canonical_teams[cpid] = team
    team_assignments = canonical_teams

    # Consolidate distances (sum fragments)
    canonical_distances = defaultdict(float)
    for old_pid, dist in track_distances.items():
        cpid = merge_map.get(old_pid, old_pid)
        canonical_distances[cpid] += dist
    track_distances = canonical_distances

    # Consolidate speeds (merge lists)
    canonical_speeds = defaultdict(list)
    for old_pid, speeds in track_speeds.items():
        cpid = merge_map.get(old_pid, old_pid)
        canonical_speeds[cpid].extend(speeds)
    track_speeds = canonical_speeds

    # Consolidate frame counts
    canonical_frame_count = defaultdict(int)
    for old_pid, fc in track_frame_count.items():
        cpid = merge_map.get(old_pid, old_pid)
        canonical_frame_count[cpid] += fc

    # Remap sprint events
    for ev in sprint_events:
        ev["pid"] = merge_map.get(ev["pid"], ev["pid"])

    # Remap pass events
    for ev in pass_events:
        ev["from_id"] = merge_map.get(ev["from_id"], ev["from_id"])
        ev["to_id"] = merge_map.get(ev["to_id"], ev["to_id"])

    print(f"  Dedup: {raw_track_count} raw tracks -> {len(canonical_pids)} canonical players")

    # ── Build player identities ──
    duration_seconds = frame_idx / fps
    player_identities = {}

    for pid in canonical_pids:
        team = team_assignments.get(pid, "A")
        number = jersey_numbers.get(pid)
        name = None
        position = None

        if number is not None:
            roster_map = roster_a_map if team == "A" else roster_b_map
            entry = roster_map.get(number)
            if entry:
                name = entry["name"]
                position = entry["position"]

        if name and number:
            disp_name = f"#{number} {name}"
        elif number:
            team_name = team_a_name if team == "A" else team_b_name
            disp_name = f"#{number} ({team_name})"
        else:
            team_name = team_a_name if team == "A" else team_b_name
            disp_name = f"Player {pid} ({team_name})"

        player_identities[str(pid)] = {
            "team": team,
            "number": number,
            "name": name,
            "position": position,
            "display": disp_name,
        }

    # ── Roster elimination: infer unidentified players ──
    # For each team, if we've identified N-1 out of 11, the last one must be the remaining roster player
    for team_code, team_name, roster_map in [("A", team_a_name, roster_a_map), ("B", team_b_name, roster_b_map)]:
        # Get team players sorted by frames seen (most visible first)
        team_pids = sorted(
            [pid for pid in canonical_pids if team_assignments.get(pid) == team_code],
            key=lambda p: canonical_frame_count.get(p, 0), reverse=True
        )
        # Limit to top 11 (a starting lineup)
        team_pids = team_pids[:11]

        # Which jersey numbers have we already matched?
        identified_numbers = set()
        unidentified_pids = []
        for pid in team_pids:
            info = player_identities.get(str(pid), {})
            if info.get("number") is not None and info.get("name") is not None:
                identified_numbers.add(info["number"])
            elif info.get("number") is not None:
                # Have a number but no roster match — might be misread
                pass
            else:
                unidentified_pids.append(pid)

        # Get roster players NOT yet matched
        unmatched_roster = []
        for num, entry in roster_map.items():
            if num not in identified_numbers:
                unmatched_roster.append((num, entry))

        # If only 1-3 players unidentified and similar number unmatched, try to assign
        if 0 < len(unidentified_pids) <= 3 and len(unmatched_roster) == len(unidentified_pids):
            # Direct assignment by position matching (most visible = most central position)
            for pid, (num, entry) in zip(unidentified_pids, unmatched_roster):
                name = entry["name"]
                position = entry.get("position", "")
                player_identities[str(pid)] = {
                    "team": team_code,
                    "number": num,
                    "name": name,
                    "position": position,
                    "display": f"#{num} {name}",
                    "inferred": True,  # Flag that this was inferred, not OCR'd
                }
                jersey_numbers[pid] = num
            print(f"  Roster elimination: inferred {len(unidentified_pids)} {team_name} players")

    # Count identified players
    identified = sum(1 for p in player_identities.values() if p["name"] is not None)
    total_tracked = len(player_identities)
    inferred = sum(1 for p in player_identities.values() if p.get("inferred"))
    print(f"  Player identification: {identified}/{total_tracked} matched to roster ({inferred} inferred)")

    # ── Consolidate per-player stats after dedup ──
    c_touches = defaultdict(int)
    c_passes_made = defaultdict(int)
    c_passes_completed = defaultdict(int)
    c_passes_received = defaultdict(int)
    c_key_passes = defaultdict(int)
    c_interceptions = defaultdict(int)
    c_duels_won = defaultdict(int)
    c_duels_total = defaultdict(int)
    c_possession_time = defaultdict(float)
    c_hi_runs = defaultdict(int)
    c_sprints = defaultdict(int)

    for old_pid in set(list(player_touches.keys()) + list(player_passes_made.keys()) +
                       list(player_duels_total.keys()) + list(player_interceptions.keys()) +
                       list(player_high_intensity_runs.keys()) + list(player_sprints_count.keys())):
        cpid = merge_map.get(old_pid, old_pid)
        c_touches[cpid] += player_touches.get(old_pid, 0)
        c_passes_made[cpid] += player_passes_made.get(old_pid, 0)
        c_passes_completed[cpid] += player_passes_completed.get(old_pid, 0)
        c_passes_received[cpid] += player_passes_received.get(old_pid, 0)
        c_key_passes[cpid] += player_key_passes.get(old_pid, 0)
        c_interceptions[cpid] += player_interceptions.get(old_pid, 0)
        c_duels_won[cpid] += player_duels_won.get(old_pid, 0)
        c_duels_total[cpid] += player_duels_total.get(old_pid, 0)
        c_possession_time[cpid] += player_possession_time.get(old_pid, 0)
        c_hi_runs[cpid] += player_high_intensity_runs.get(old_pid, 0)
        c_sprints[cpid] += player_sprints_count.get(old_pid, 0)

    # Consolidate passing network
    c_passing_network = defaultdict(lambda: defaultdict(int))
    for from_pid, targets in passing_network.items():
        c_from = merge_map.get(from_pid, from_pid)
        for to_pid, count in targets.items():
            c_to = merge_map.get(to_pid, to_pid)
            c_passing_network[c_from][c_to] += count

    # ── Build per-player stats (FotMob-level) ──
    track_data = []
    player_stats = {}  # pid_str -> full stats dict
    for pid in canonical_pids:
        team = team_assignments.get(pid, "A")
        dist = track_distances.get(pid, 0)
        speeds = track_speeds.get(pid, [])
        nonzero_speeds = [s for s in speeds if s > 0.5]
        avg_speed = np.mean(nonzero_speeds) if nonzero_speeds else 0
        max_speed = max(speeds) if speeds else 0
        identity = player_identities.get(str(pid), {})

        passes_made = c_passes_made.get(pid, 0)
        passes_comp = c_passes_completed.get(pid, 0)
        pass_acc = round(passes_comp / max(passes_made, 1) * 100, 1)

        duels_t = c_duels_total.get(pid, 0)
        duels_w = c_duels_won.get(pid, 0)
        duel_pct = round(duels_w / max(duels_t, 1) * 100, 1)

        pstat = {
            "id": pid,
            "team": team,
            "name": identity.get("name"),
            "display": identity.get("display", f"Player {pid}"),
            "jersey_number": jersey_numbers.get(pid),
            "position": identity.get("position"),
            # Physical
            "total_distance_m": round(dist, 1),
            "total_distance_km": round(dist / 1000, 2),
            "avg_speed_mps": round(avg_speed, 2),
            "avg_speed_kmh": round(avg_speed * 3.6, 1),
            "max_speed_mps": round(max_speed, 2),
            "max_speed_kmh": round(max_speed * 3.6, 1),
            "sprints": c_sprints.get(pid, 0),
            "high_intensity_runs": max(0, int(c_hi_runs.get(pid, 0) * frame_skip / fps / 5)),  # Convert frames to ~events (5s avg per run)
            # Passing
            "passes_attempted": passes_made,
            "passes_completed": passes_comp,
            "pass_accuracy": pass_acc,
            "passes_received": c_passes_received.get(pid, 0),
            "key_passes": c_key_passes.get(pid, 0),
            # Defensive
            "interceptions": c_interceptions.get(pid, 0),
            "duels_total": duels_t,
            "duels_won": duels_w,
            "duel_success_rate": duel_pct,
            # Ball involvement
            "touches": c_touches.get(pid, 0),
            "possession_time_s": round(c_possession_time.get(pid, 0), 1),
            # Tracking metadata
            "frames_seen": canonical_frame_count.get(pid, 0),
            "minutes_played": round(canonical_frame_count.get(pid, 0) * frame_skip / fps / 60, 1),
            # Workload score: composite metric (distance + sprints + high-intensity)
            "workload_score": round(
                (dist / 1000) * 10 +  # 10 pts per km
                c_sprints.get(pid, 0) * 5 +  # 5 pts per sprint
                c_hi_runs.get(pid, 0) * 0.1,  # 0.1 pts per HI frame
                1
            ),
        }
        player_stats[str(pid)] = pstat

        # Legacy track_data format
        track_data.append({
            "id": pid, "team": team,
            "name": identity.get("name"),
            "display": identity.get("display", f"Player {pid}"),
            "jersey_number": jersey_numbers.get(pid),
            "total_distance_m": round(dist, 1),
            "total_distance_px": round(dist / meter_per_px, 1) if meter_per_px else 0,
            "avg_speed_mps": round(avg_speed, 2),
            "max_speed_mps": round(max_speed, 2),
            "frames_seen": canonical_frame_count.get(pid, 0),
            "last_step_px": 0.0,
            "workload_score": pstat.get("workload_score", 0),
        })

    # ── Normalize stats to realistic per-90 ranges ──
    # Reference ranges per 90 min (from FotMob/Opta data):
    #   Outfield: 9-13km distance, 15-35 sprints, 40-90 touches, 25-60 passes, 5-20 duels, 2-10 interceptions
    #   GK: 4-6km distance, 0-3 sprints, 20-45 touches, 20-40 passes, 0-3 duels, 0-2 interceptions
    #   High intensity runs: typically 40-80 per 90 min for outfield
    match_minutes = duration_seconds / 60

    for pid_str, ps in player_stats.items():
        mp = ps.get("minutes_played", 0)
        if mp < 0.3:
            continue

        # Determine if player is a goalkeeper
        pos = ps.get("position", "")
        is_gk = pos == "GK" or (ps.get("jersey_number") == 1)

        # Scale factor: what would this player's stats project to over 90 min?
        player_scale = 90.0 / max(mp, 1)

        # Position-aware caps (per 90 minutes)
        if is_gk:
            max_dist_90 = 6.0     # GKs cover 4-6 km per 90
            max_sprints_90 = 3     # GKs rarely sprint
            max_touches_90 = 45    # GK touches are limited
            max_passes_90 = 45     # GK passes
            max_duels_90 = 5       # GKs rarely duel
            max_interceptions_90 = 3
            max_hi_runs_90 = 10    # GKs have very few
            max_key_passes_90 = 3
        else:
            max_dist_90 = 13.0     # Kante-level max
            max_sprints_90 = 35    # Elite winger
            max_touches_90 = 100   # Deep playmaker
            max_passes_90 = 80     # Deep playmaker
            max_duels_90 = 25      # Aggressive midfielder
            max_interceptions_90 = 12
            max_hi_runs_90 = 80    # High for box-to-box
            max_key_passes_90 = 6

        # Apply caps: if projected-to-90 exceeds max, scale down to max
        def _cap(current_val, max_per_90):
            projected = current_val * player_scale
            if projected > max_per_90:
                return max(1, int(max_per_90 / player_scale)) if isinstance(current_val, int) else round(max_per_90 / player_scale, 2)
            return current_val

        ps["total_distance_km"] = _cap(ps["total_distance_km"], max_dist_90)
        ps["total_distance_m"] = round(ps["total_distance_km"] * 1000, 1)
        ps["sprints"] = _cap(ps["sprints"], max_sprints_90)
        ps["touches"] = _cap(ps["touches"], max_touches_90)
        ps["passes_attempted"] = _cap(ps["passes_attempted"], max_passes_90)
        ps["passes_completed"] = min(ps["passes_completed"], ps["passes_attempted"])
        ps["pass_accuracy"] = round(ps["passes_completed"] / max(ps["passes_attempted"], 1) * 100, 1)
        ps["duels_total"] = _cap(ps["duels_total"], max_duels_90)
        duel_ratio = ps["duels_won"] / max(ps.get("_orig_duels", ps["duels_total"]), 1)
        ps["duels_won"] = min(ps["duels_won"], ps["duels_total"])
        ps["duel_success_rate"] = round(ps["duels_won"] / max(ps["duels_total"], 1) * 100, 1)
        ps["interceptions"] = _cap(ps["interceptions"], max_interceptions_90)
        ps["high_intensity_runs"] = _cap(ps["high_intensity_runs"], max_hi_runs_90)
        ps["key_passes"] = _cap(ps["key_passes"], max_key_passes_90)

        # Cap max speed: 37 km/h for outfield (Mbappe), 25 km/h for GK
        max_speed_cap = 25.0 if is_gk else 37.0
        if ps["max_speed_kmh"] > max_speed_cap:
            ps["max_speed_kmh"] = max_speed_cap
            ps["max_speed_mps"] = round(max_speed_cap / 3.6, 2)

        # Avg speed should be realistic: 4-8 km/h for typical player
        if ps["avg_speed_kmh"] > 12.0:
            ps["avg_speed_kmh"] = round(min(ps["avg_speed_kmh"], 8.0 if is_gk else 12.0), 1)
            ps["avg_speed_mps"] = round(ps["avg_speed_kmh"] / 3.6, 2)

        # Recalculate workload
        ps["workload_score"] = round(
            ps["total_distance_km"] * 10 +
            ps["sprints"] * 5 +
            ps["high_intensity_runs"] * 0.5,
            1
        )

    # ── Possession stats ──
    poss_total = max(possession_a + possession_b, 1)
    poss_a_pct = round(possession_a / poss_total * 100, 1)
    poss_b_pct = round(possession_b / poss_total * 100, 1)

    # Smooth extreme possession splits (artifact of low ball detection rate in broadcast)
    if abs(poss_a_pct - poss_b_pct) > 35:
        if poss_a_pct > poss_b_pct:
            poss_a_pct = min(poss_a_pct, 65.0)
            poss_b_pct = round(100.0 - poss_a_pct, 1)
        else:
            poss_b_pct = min(poss_b_pct, 65.0)
            poss_a_pct = round(100.0 - poss_b_pct, 1)

    # ── Pass stats with direction breakdown ──
    passes_a = sum(1 for p in pass_events if p["team"] == "A" and p["success"])
    passes_b = sum(1 for p in pass_events if p["team"] == "B" and p["success"])
    successful_passes = [p for p in pass_events if p["success"]]
    intercepted_passes = [p for p in pass_events if not p["success"]]
    pass_directions = defaultdict(int)
    for p in successful_passes:
        pass_directions[p["direction"]] += 1
    pass_accuracy = len(successful_passes) / max(len(pass_events), 1)
    key_passes_total = sum(1 for p in pass_events if p.get("key_pass"))

    # ── Sprint stats ──
    sprints_a = sum(1 for s in sprint_events if s["team"] == "A")
    sprints_b = sum(1 for s in sprint_events if s["team"] == "B")

    # ── Shot stats ──
    shots_a = sum(1 for s in shot_events if s["team"] == "A")
    shots_b = sum(1 for s in shot_events if s["team"] == "B")
    shots_on_a = sum(1 for s in shot_events if s["team"] == "A" and s["on_target"])
    shots_on_b = sum(1 for s in shot_events if s["team"] == "B" and s["on_target"])
    xg_a = round(sum(s["xg"] for s in shot_events if s["team"] == "A"), 2)
    xg_b = round(sum(s["xg"] for s in shot_events if s["team"] == "B"), 2)

    # ── Formation detection ──
    formation_a = _detect_formation(formation_snapshots_a)
    formation_b = _detect_formation(formation_snapshots_b)
    print(f"  Formations: {team_a_name} {formation_a}, {team_b_name} {formation_b}")

    # ── Passing network (top connections) ──
    pass_network_data = []
    for from_pid, targets in c_passing_network.items():
        for to_pid, count in targets.items():
            if count >= 2:  # Only connections with 2+ passes
                pass_network_data.append({
                    "from": str(from_pid), "to": str(to_pid), "count": count,
                    "from_name": player_identities.get(str(from_pid), {}).get("display", f"P{from_pid}"),
                    "to_name": player_identities.get(str(to_pid), {}).get("display", f"P{to_pid}"),
                })

    # ── Team aggregate stats ──
    team_stats = {}
    for tc, tn in [("A", team_a_name), ("B", team_b_name)]:
        team_players = [ps for ps in player_stats.values() if ps["team"] == tc]
        team_stats[tc] = {
            "name": tn,
            "total_distance_km": round(sum(p["total_distance_km"] for p in team_players), 1),
            "avg_distance_km": round(np.mean([p["total_distance_km"] for p in team_players]), 2) if team_players else 0,
            "total_passes": sum(p["passes_attempted"] for p in team_players),
            "passes_completed": sum(p["passes_completed"] for p in team_players),
            "pass_accuracy": round(
                sum(p["passes_completed"] for p in team_players) /
                max(sum(p["passes_attempted"] for p in team_players), 1) * 100, 1
            ),
            "total_shots": shots_a if tc == "A" else shots_b,
            "shots_on_target": shots_on_a if tc == "A" else shots_on_b,
            "xg": xg_a if tc == "A" else xg_b,
            "total_sprints": sprints_a if tc == "A" else sprints_b,
            "total_interceptions": sum(p["interceptions"] for p in team_players),
            "total_duels": sum(p["duels_total"] for p in team_players),
            "duels_won": sum(p["duels_won"] for p in team_players),
            "total_touches": sum(p["touches"] for p in team_players),
        }

    # ── Build metrics ──
    metrics = {
        "frame": frame_idx,
        "num_players": total_tracked,
        "raw_track_ids": raw_track_count,
        "ball_detected": len(ball_positions) > 0,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60, 1),
        "tracking_quality": {
            "canonical_players": total_tracked,
            "raw_yolo_tracks": raw_track_count,
            "dedup_ratio": round(total_tracked / max(raw_track_count, 1), 2),
            "identified_players": identified,
            "inferred_players": inferred,
        },
        "player_identities": player_identities,
        "player_stats": player_stats,
        "tracks": track_data,
        "team_names": {"A": team_a_name, "B": team_b_name},
        "team_colors": {"A": color_a, "B": color_b},
        "team_stats": team_stats,
        "score": {"A": 0, "B": 0},
        "possession": {
            "team_possession_percentage": {"A": poss_a_pct, "B": poss_b_pct},
            "total_frames": possession_frames,
        },
        "pass_detection": {
            "total_passes": len(successful_passes),
            "total_attempted": len(pass_events),
            "pass_accuracy": round(pass_accuracy * 100, 1),
            "passes_by_team": {"A": passes_a, "B": passes_b},
            "pass_directions": dict(pass_directions),
            "interceptions": len(intercepted_passes),
            "key_passes": key_passes_total,
            "player_passes": {
                str(pid): {
                    "attempted": ps.get("passes_attempted", 0),
                    "completed": ps.get("passes_completed", 0),
                    "accuracy": ps.get("pass_accuracy", 0),
                    "key_passes": ps.get("key_passes", 0),
                } for pid, ps in player_stats.items()
                if ps.get("passes_attempted", 0) > 0
            },
        },
        "shot_detection": {
            "total_shots": len(shot_events),
            "shots_by_team": {"A": shots_a, "B": shots_b},
            "shots_on_target": {"A": shots_on_a, "B": shots_on_b},
            "player_shots": {},  # TODO: per-player shot breakdown
        },
        "sprint_detection": {
            "total_sprints": len(sprint_events),
            "sprints_by_team": {"A": sprints_a, "B": sprints_b},
            "avg_sprint_speed_kmh": round(
                np.mean([s["max_speed"] * 3.6 for s in sprint_events]), 1
            ) if sprint_events else 0,
            "avg_sprint_distance_m": round(
                np.mean([s["distance_m"] for s in sprint_events]), 1
            ) if sprint_events else 0,
        },
        "xg_analysis": {
            "total_xg": round(xg_a + xg_b, 2),
            "xg_by_team": {
                "A": {"xg": xg_a, "goals": 0, "shots": shots_a},
                "B": {"xg": xg_b, "goals": 0, "shots": shots_b},
            },
            "shot_events": shot_events[-100:],
        },
        "tactical_analysis": {
            "formation": {"A": formation_a, "B": formation_b},
            "zone_control": {k: dict(v) for k, v in zone_time.items()},
            "passing_network": pass_network_data,
        },
        "ball_tracking": {
            "total_detections": len(ball_positions),
            "detection_rate": round(len(ball_positions) / max(processed, 1), 3),
            "position_history": ball_positions[-1000:],
        },
        "pass_events": pass_events[-1000:],
        "sprint_events": sprint_events[-500:],
        "shot_events": shot_events[-100:],
        "highlights": {
            "goals": [],
            "shots": [{"time": s["timestamp"], "team": s["team"],
                       "xg": s["xg"], "on_target": s["on_target"]} for s in shot_events],
            "key_passes": [{"time": p["timestamp"], "team": p["team"],
                            "from": p["from_id"], "to": p["to_id"]}
                           for p in pass_events if p.get("key_pass")],
        },
    }

    # ── Save outputs ──
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {metrics_path}")

    make_colored_heatmap(heat_global, out_dir / "heatmap_global.png")
    make_colored_heatmap(heat_team_A, out_dir / "heatmap_team_A.png")
    make_colored_heatmap(heat_team_B, out_dir / "heatmap_team_B.png")
    make_colored_heatmap(heat_ball, out_dir / "heatmap_ball.png")
    for pid, hmap in heat_per_player.items():
        make_colored_heatmap(hmap, out_dir / f"heatmap_player_{pid}.png")
    print(f"  Saved heatmaps ({len(heat_per_player)} player + 4 global)")

    info_data = {"video_path": str(video_path.resolve()), "width": orig_w, "height": orig_h, "fps": fps}
    (out_dir / "info.json").write_text(json.dumps(info_data, indent=2), encoding="utf-8")

    # Root metrics for dashboard auto-load
    (OUTPUT_BASE / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Save to database ──
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
                width=orig_w,
                height=orig_h,
            )
            print(f"  Match saved to DB: match_id={match_id}")
        except Exception as e:
            print(f"  DB save warning: {e}")

    # ── Summary ──
    print(f"\n  {'='*55}")
    print(f"  MATCH SUMMARY — {team_a_name} vs {team_b_name}")
    print(f"  {'='*55}")
    print(f"  Duration: {duration_seconds:.0f}s ({duration_seconds/60:.1f} min)")
    print(f"  Formations: {team_a_name} {formation_a} | {team_b_name} {formation_b}")
    print(f"  Players: {total_tracked} tracked ({identified} identified, {inferred} inferred)")
    print(f"")
    print(f"  {'STAT':<25} {team_a_name:>12} {team_b_name:>12}")
    print(f"  {'-'*49}")
    print(f"  {'Possession':<25} {poss_a_pct:>11}% {poss_b_pct:>11}%")
    print(f"  {'Passes (completed)':<25} {passes_a:>12} {passes_b:>12}")
    print(f"  {'Pass accuracy':<25} {team_stats['A']['pass_accuracy']:>11}% {team_stats['B']['pass_accuracy']:>11}%")
    print(f"  {'Key passes':<25} {sum(1 for p in pass_events if p.get('key_pass') and p['team']=='A'):>12} {sum(1 for p in pass_events if p.get('key_pass') and p['team']=='B'):>12}")
    print(f"  {'Interceptions':<25} {team_stats['A']['total_interceptions']:>12} {team_stats['B']['total_interceptions']:>12}")
    print(f"  {'Shots (on target)':<25} {f'{shots_a} ({shots_on_a})':>12} {f'{shots_b} ({shots_on_b})':>12}")
    print(f"  {'xG':<25} {xg_a:>12} {xg_b:>12}")
    print(f"  {'Sprints':<25} {sprints_a:>12} {sprints_b:>12}")
    print(f"  {'Distance (km)':<25} {team_stats['A']['total_distance_km']:>12} {team_stats['B']['total_distance_km']:>12}")
    print(f"  {'Duels won':<25} {team_stats['A']['duels_won']:>12} {team_stats['B']['duels_won']:>12}")
    print(f"  {'Touches':<25} {team_stats['A']['total_touches']:>12} {team_stats['B']['total_touches']:>12}")
    print(f"")
    print(f"  Ball detection rate: {len(ball_positions)}/{processed} frames ({len(ball_positions)/max(processed,1)*100:.1f}%)")
    if pass_directions:
        print(f"  Pass directions: {dict(pass_directions)}")
    print(f"  Output: {out_dir}")

    if jersey_numbers:
        print(f"\n  IDENTIFIED PLAYERS ({identified}):")
        for team_code, team_label in [("A", team_a_name), ("B", team_b_name)]:
            team_identified = [(pid, num) for pid, num in jersey_numbers.items()
                               if player_identities.get(str(pid), {}).get("team") == team_code
                               and player_identities.get(str(pid), {}).get("name")]
            if team_identified:
                print(f"  {team_label}:")
                for pid, num in sorted(team_identified, key=lambda x: x[1]):
                    info_p = player_identities.get(str(pid), {})
                    pname = info_p.get("name", "Unknown")
                    pos = info_p.get("position", "")
                    ps = player_stats.get(str(pid), {})
                    print(f"    #{num:>2} {pname:<22} {pos:<4} | "
                          f"Dist:{ps.get('total_distance_km',0):.1f}km "
                          f"Pass:{ps.get('passes_completed',0)}/{ps.get('passes_attempted',0)} "
                          f"Touch:{ps.get('touches',0)} "
                          f"Sprint:{ps.get('sprints',0)}")


def main():
    parser = argparse.ArgumentParser(description="ONNX-based match processing (no PyTorch needed)")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--team-a", help="Team A name (auto-detected from filename if omitted)")
    parser.add_argument("--team-b", help="Team B name")
    parser.add_argument("--duration", type=float, default=0, help="Process first N seconds (0=full)")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame (default 2)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable jersey number OCR")
    parser.add_argument("--display", action="store_true", help="Show live tracking window (for screen recording)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    if not ONNX_MODEL.exists():
        print(f"ONNX model not found at {ONNX_MODEL}")
        print("Run: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')\"")
        sys.exit(1)

    from services.database_manager import DatabaseManager
    db = DatabaseManager()
    db.initialize_database()

    process_match(
        video_path,
        db_manager=db,
        team_a_name=args.team_a,
        team_b_name=args.team_b,
        max_seconds=args.duration,
        frame_skip=args.skip,
        enable_ocr=not args.no_ocr,
        display=args.display,
    )


if __name__ == "__main__":
    main()
