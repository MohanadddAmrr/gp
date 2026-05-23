"""
Player Identity Module - Jersey Number OCR + Roster Config + Color Clustering

Identifies real players by combining three approaches:
1. ROSTER CONFIG: JSON file mapping jersey numbers to player names per team
2. JERSEY COLOR: K-means clustering on upper-body pixels to identify teams
3. JERSEY OCR: Reads jersey numbers from clearer frames using OCR

PIPELINE:
- Every frame: classify team by jersey color (fast)
- Periodically: attempt OCR on player bounding boxes (slower)
- When OCR reads a number: link canonical track ID to roster entry
- Fallback: if OCR fails, player shown as "Team A #unknown"

REQUIREMENTS:
- pip install easyocr  (for jersey number OCR)
- Roster JSON file (optional but recommended)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from pathlib import Path
import json


class JerseyColorClassifier:
    """
    Classifies players into teams based on jersey color.
    Now uses roster color hints when available for more accurate classification.
    """

    # HSV hue ranges for common jersey colors (0-180 scale in OpenCV)
    COLOR_RANGES = {
        "red": [(0, 15), (165, 180)],      # Red wraps around
        "blue": [(100, 130)],
        "white": None,                      # Use saturation check
        "black": None,                      # Use value check
        "yellow": [(20, 35)],
        "green": [(35, 85)],
        "orange": [(10, 25)],
        "purple": [(130, 160)],
        "pink": [(150, 170)],
        "cyan": [(85, 100)],
    }
    
        # Referee jersey colors (yellow, black, bright colors)
    REFEREE_COLORS = {
        "yellow": [(20, 35)],
        "bright_yellow": [(25, 40)],
        "black": None,  # Low value check
        "neon": [(35, 50)],  # Bright green/yellow
    }
    
    # Thresholds for referee detection
    REFEREE_SATURATION_LOW = 40  # Black jerseys
    REFEREE_SATURATION_HIGH = 200  # Bright yellow
    REFEREE_VALUE_LOW = 60  # Black jerseys
    REFEREE_YELLOW_HUE_MIN = 20
    REFEREE_YELLOW_HUE_MAX = 45


    def __init__(self, n_teams: int = 2, team_colors: dict = None):
        """
        Args:
            n_teams: Number of teams (default 2)
            team_colors: Optional dict like {"A": "red", "B": "blue"} from roster
        """
        self.n_teams = n_teams
        self.team_colors = team_colors or {}
        self._color_samples: Dict[int, List[int]] = defaultdict(list)
        self._saturation_samples: Dict[int, List[int]] = defaultdict(list)
        self._value_samples: Dict[int, List[int]] = defaultdict(list)
        self._team_assignments: Dict[int, str] = {}
        self._team_centroids: Optional[List[float]] = None

    def _get_hue_ranges_for_color(self, color_name: str):
        """Get HSV hue ranges for a color name."""
        if color_name is None:
            return None
        return self.COLOR_RANGES.get(color_name.lower())

    @staticmethod
    def _circular_mean_hue(hues) -> int:
        """Circular mean of OpenCV hues (0-180). A linear mean/median on the
        0/180 wrap is meaningless: Liverpool red samples at H=175 and H=5 are
        both 'red', but np.median([170, 175, 178, 0, 3, 5]) returns ~87,
        which is cyan-green. The circular calculation correctly returns ~0.
        """
        if hues is None or len(hues) == 0:
            return 0
        # Map cv2 hue 0..180 to a full 2pi cycle so 0 and 180 coincide
        angles = np.asarray(hues, dtype=np.float64) * (np.pi / 90.0)
        s = float(np.sin(angles).mean())
        c = float(np.cos(angles).mean())
        a = np.arctan2(s, c)
        if a < 0:
            a += 2.0 * np.pi
        return int(a * 90.0 / np.pi) % 180

    @staticmethod
    def _hex_to_hsv(hex_str: str):
        """Convert '#rrggbb' to OpenCV HSV tuple (H 0-180, S 0-255, V 0-255)."""
        try:
            h = hex_str.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            bgr = np.uint8([[[b, g, r]]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            return int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])
        except Exception:
            return None

    def _hue_matches_color(self, hue: int, color: str) -> bool:
        """Check if a hue value matches the expected color.

        Accepts either a color name (e.g. 'red') or a hex code ('#c8102e').
        Hex inputs are converted to HSV; matches require the hue to be within
        ~18 degrees of the target. Low-saturation hex (white/grey/black) is
        not matched on hue — saturation/value classification handles those.
        """
        if not color:
            return False
        # Hex-code path: this is what the roster/DB actually passes in
        if color.startswith('#'):
            target = self._hex_to_hsv(color)
            if target is None:
                return False
            h_target, s_target, _ = target
            if s_target < 40:
                return False  # white/grey/black — no meaningful hue
            # Cyclic distance on the 0-180 OpenCV hue circle
            diff = min((hue - h_target) % 180, (h_target - hue) % 180)
            return diff <= 18
        # Name-keyed path (legacy / direct color-name input)
        ranges = self._get_hue_ranges_for_color(color)
        if ranges is None:
            return False
        for low, high in ranges:
            if low <= hue <= high:
                return True
        return False

    def sample_jersey_color(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float], player_id: int
    ):
        """Sample the dominant jersey color from a player's bounding box.

        Tightened sampling band (20%-50% vertical, 25%-75% horizontal) to
        avoid head, shorts and grass leakage that previously pulled the
        sampled median hue toward grass-green (~60deg) and made every red
        Liverpool player appear closer to Tottenham yellow than to red.

        Also explicitly masks pitch-green pixels (hue 35-95 at mid-high
        saturation/value) before computing the dominant jersey hue.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        box_h = y2 - y1
        box_w = x2 - x1
        torso_y1 = y1 + int(box_h * 0.20)
        torso_y2 = y1 + int(box_h * 0.50)
        torso_x1 = x1 + int(box_w * 0.25)
        torso_x2 = x2 - int(box_w * 0.25)

        if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
            return

        torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso.size == 0:
            return

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hues = hsv[:, :, 0].flatten()
        sats = hsv[:, :, 1].flatten()
        vals = hsv[:, :, 2].flatten()

        # Pitch-green mask: hue 35-95 with mid/high sat is grass. We exclude
        # these pixels so the dominant hue reflects the JERSEY, not grass.
        # (Tottenham canary yellow is hue ~28 — safely outside this range.)
        grass_mask = (hues >= 35) & (hues <= 95) & (sats >= 60) & (sats <= 200)

        # Skin mask: orange-tan range with mid saturation — typical for face/arms.
        skin_mask = (hues >= 5) & (hues <= 25) & (sats >= 30) & (sats <= 130) & (vals >= 120)

        # Valid jersey pixel: not too desaturated (white/grey is OK as long as
        # vals are bounded), not grass, not skin.
        mask = (vals > 50) & (vals < 235) & (~grass_mask) & (~skin_mask)
        # Keep low-sat pixels (white kits) only if value is mid-bright.
        mask &= ((sats > 35) | ((sats <= 35) & (vals > 130)))

        valid_hues = hues[mask]
        valid_sats = sats[mask]
        valid_vals = vals[mask]

        if len(valid_hues) < 10:
            return

        # Dominant hue via 36-bin histogram (circular: bin 35 and bin 0 are
        # adjacent for the red wrap).
        hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
        dominant_hue = int(bin_edges[np.argmax(hist)])
        avg_sat = int(np.mean(valid_sats))
        avg_val = int(np.mean(valid_vals))

        self._color_samples[player_id].append(dominant_hue)
        self._saturation_samples[player_id].append(avg_sat)
        self._value_samples[player_id].append(avg_val)

    def predict_team_live(self, player_id: int) -> Optional[str]:
        """Get a live team prediction using roster color hints if available."""
        if player_id in self._team_assignments:
            return self._team_assignments.get(player_id)

        # If we have roster colors, use them for direct matching
        if self.team_colors and player_id in self._color_samples:
            samples = self._color_samples[player_id]
            if len(samples) >= 3:
                median_hue = self._circular_mean_hue(samples)
                
                # Check against each team's expected color
                for team, color in self.team_colors.items():
                    if self._hue_matches_color(median_hue, color):
                        self._team_assignments[player_id] = team
                        return team

        # Fallback to clustering
        eligible = {pid: hues for pid, hues in self._color_samples.items() if len(hues) >= 5}
        if len(eligible) < 6:
            return None

        pids = list(eligible.keys())
        medians = [self._circular_mean_hue(eligible[p]) for p in pids]
        hue_rad = np.array(medians, dtype=np.float32) * np.pi / 90.0
        hue_2d = np.column_stack([np.cos(hue_rad), np.sin(hue_rad)]).astype(np.float32)

        n_clusters = min(3, len(pids))
        if n_clusters < 2:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        try:
            _, labels, centers = cv2.kmeans(hue_2d, n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            return None

        labels = labels.flatten()
        cluster_counts = Counter(labels)
        top_clusters = [c[0] for c in cluster_counts.most_common(3)]
        
        if len(top_clusters) < 2:
            return None

        # If we have roster colors, match clusters to teams
        if self.team_colors:
            cluster_to_team = {}
            for cluster_id in top_clusters:
                cluster_pids = [pids[i] for i in range(len(pids)) if labels[i] == cluster_id]
                cluster_hues = [medians[i] for i in range(len(pids)) if labels[i] == cluster_id]
                if cluster_hues:
                    avg_hue = int(np.mean(cluster_hues))
                    for team, color in self.team_colors.items():
                        if self._hue_matches_color(avg_hue, color) and team not in cluster_to_team.values():
                            cluster_to_team[cluster_id] = team
                            break
            
            # Assign teams based on cluster matching
            for i, pid in enumerate(pids):
                cluster = labels[i]
                if cluster in cluster_to_team:
                    self._team_assignments[pid] = cluster_to_team[cluster]
                elif cluster == top_clusters[-1] and len(top_clusters) == 3:
                    self._team_assignments[pid] = 'REF'
        else:
            # No roster colors - use simple cluster assignment
            for i, pid in enumerate(pids):
                if labels[i] == top_clusters[0]:
                    self._team_assignments[pid] = 'A'
                elif labels[i] == top_clusters[1]:
                    self._team_assignments[pid] = 'B'
                else:
                    self._team_assignments[pid] = 'REF'

        return self._team_assignments.get(player_id)

    def finalize_teams(self) -> Dict[int, str]:
        """Final team assignment using all collected samples with improved referee detection."""
        if not self._color_samples:
            return {}

        player_hues = {}
        for pid, hues in self._color_samples.items():
            if len(hues) >= 3:
                player_hues[pid] = self._circular_mean_hue(hues)

        if len(player_hues) < 4:
            return {}

        pids = list(player_hues.keys())
        
        # First pass: Identify referees by color characteristics
        for pid in pids:
            if pid in self._team_assignments:
                continue
                
            hue = player_hues[pid]
            avg_sat = np.mean(self._saturation_samples.get(pid, [100]))
            avg_val = np.mean(self._value_samples.get(pid, [128]))
            
            # Check for referee characteristics. CRITICAL: if the player's
            # hue matches one of the configured TEAM colors (e.g. an away team
            # wearing yellow), they're a team player, not a referee. Without
            # this guard, yellow-kit away teams collapsed entirely into 'REF'
            # because yellow is the most common referee colour.
            matches_team_color = bool(self.team_colors) and any(
                self._hue_matches_color(hue, c) for c in self.team_colors.values()
            )

            is_referee = False
            if not matches_team_color:
                # Yellow referee jersey (most common)
                if (self.REFEREE_YELLOW_HUE_MIN <= hue <= self.REFEREE_YELLOW_HUE_MAX and
                    avg_sat > 100 and avg_val > 150):
                    is_referee = True
                # Black referee jersey (low saturation AND low value)
                elif avg_sat < self.REFEREE_SATURATION_LOW and avg_val < self.REFEREE_VALUE_LOW:
                    is_referee = True
                # Bright fluorescent (high saturation yellow-green)
                elif (30 <= hue <= 50 and avg_sat > 180 and avg_val > 200):
                    is_referee = True

            if is_referee:
                self._team_assignments[pid] = 'REF'
        
        # If roster colors available, use direct color matching for remaining players
        if self.team_colors:
            for pid in pids:
                if pid in self._team_assignments:
                    continue
                hue = player_hues[pid]
                for team, color in self.team_colors.items():
                    if self._hue_matches_color(hue, color):
                        self._team_assignments[pid] = team
                        break
            return self._team_assignments

        # Fallback to clustering for remaining unassigned players
        unassigned_pids = [p for p in pids if p not in self._team_assignments]
        if len(unassigned_pids) < 4:
            return self._team_assignments
            
        hue_values = np.array([player_hues[p] for p in unassigned_pids], dtype=np.float32).reshape(-1, 1)
        hue_rad = hue_values * np.pi / 90.0
        hue_2d = np.column_stack([np.cos(hue_rad), np.sin(hue_rad)]).astype(np.float32)

        n_clusters = 2  # Only team A and B for remaining players
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(hue_2d, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        sorted_clusters = sorted(range(n_clusters), key=lambda x: cluster_sizes[x], reverse=True)

        for i, pid in enumerate(unassigned_pids):
            if labels[i] == sorted_clusters[0]:
                self._team_assignments[pid] = 'A'
            else:
                self._team_assignments[pid] = 'B'

        return self._team_assignments


    def get_team(self, player_id: int) -> Optional[str]:
        """Get team assignment for a player."""
        return self._team_assignments.get(player_id)

    def finalize_teams_ocr_anchored(
        self,
        ocr_readings: Dict[int, List[int]],
        roster_players_A: Dict[str, str],
        roster_players_B: Dict[str, str],
    ) -> Dict[int, str]:
        """Per-player team finalization driven by HSV-distance + OCR override.

        Pure colour k-means clustering fails on broadcast footage: Liverpool
        red and Tottenham yellow samples overlap heavily after lighting,
        motion blur, jersey trim and crowd bleed, and k-means collapses them
        into one giant cluster + small outliers (verified empirically on 60s
        of Lpool vs Spurs: cluster sizes were ~49/14/2 instead of 11/11/~10).

        OCR-anchored cluster voting also fails because the two PL rosters
        share most numbers (1-10, 17, 19-21, 38 etc.) — across 60s of
        footage only ONE OCR'd number was unique to one roster. Not enough
        signal to label clusters.

        This routine classifies each player INDIVIDUALLY:
          - Sampled HSV vs each team's canonical hex code -> distance score
            (hue weighted 3:1 over S/V; if a team's target is desaturated,
            use S/V distance only).
          - OVERRIDE: if the player's OCR'd numbers contain one that exists
            in exactly one of the two rosters, that team wins regardless of
            colour distance.
          - REF gate: only if the player's nearest team is still far in HSV
            distance AND their saturation is low (black/grey/fluo).
        """
        # Build per-player features.
        pids: List[int] = []
        feats: Dict[int, Tuple[float, float, float]] = {}  # pid -> (hue, sat, val)
        for pid, hues in self._color_samples.items():
            if len(hues) < 3:
                continue
            sats = self._saturation_samples.get(pid, [])
            vals = self._value_samples.get(pid, [])
            if not sats or not vals:
                continue
            h = self._circular_mean_hue(hues)
            s = float(np.median(sats))
            v = float(np.median(vals))
            feats[pid] = (h, s, v)
            pids.append(pid)

        if not pids:
            return {}

        hex_a = self.team_colors.get('A')
        hex_b = self.team_colors.get('B')
        target_a = self._hex_to_hsv(hex_a) if hex_a and hex_a.startswith('#') else None
        target_b = self._hex_to_hsv(hex_b) if hex_b and hex_b.startswith('#') else None

        def _dist_to(target, h, s, v) -> float:
            if target is None:
                return 999.0
            h_t, s_t, v_t = target
            # Sat/Val distance always counts (handles white/black kits).
            sv_d = float(np.hypot((s - s_t) / 255.0, (v - v_t) / 255.0))
            # Desaturated target (white/grey/black) -> ignore hue entirely.
            if s_t < 40:
                return sv_d + 2.0 * (1.0 if s > 80 else 0.0)
            # Desaturated sample but coloured target -> hue is unreliable;
            # heavy penalty (the player's kit isn't this colour at all).
            if s < 40:
                return sv_d + 2.0
            cos_t = np.cos(h_t * np.pi / 90.0)
            sin_t = np.sin(h_t * np.pi / 90.0)
            cos_s = np.cos(h * np.pi / 90.0)
            sin_s = np.sin(h * np.pi / 90.0)
            hue_d = float(np.hypot(cos_s - cos_t, sin_s - sin_t))
            return hue_d * 3.0 + sv_d

        # Per-player OCR signal: which roster do their unique-number reads
        # point to? (None if no unique-roster numbers were read.)
        def _ocr_team(pid: int) -> Optional[str]:
            va = vb = 0
            for n in ocr_readings.get(pid, []):
                key = str(int(n))
                in_a = key in roster_players_A
                in_b = key in roster_players_B
                if in_a and not in_b:
                    va += 1
                elif in_b and not in_a:
                    vb += 1
            if va == 0 and vb == 0:
                return None
            return 'A' if va > vb else ('B' if vb > va else None)

        self._team_assignments = {}

        # Single-pass per-player classification against the canonical team
        # hex codes. We do NOT bootstrap "observed" centroids from OCR-
        # anchored players because OCR misreads on the wrong team (e.g. a
        # Liverpool player whose #11 gets read as #6) poison the centroid
        # for the opposing team and flip the rest of the classification.
        # OCR override is applied only when it AGREES with the colour
        # signal, or when the colour signal is weak (low saturation).
        for pid in pids:
            h, s, v = feats[pid]
            da = _dist_to(target_a, h, s, v)
            db = _dist_to(target_b, h, s, v)
            color_team = 'A' if da <= db else 'B'
            color_margin = abs(da - db)

            ocr_team = _ocr_team(pid)

            # REF gate: very dark or very desaturated AND neither team
            # colour is a close match.
            if min(da, db) > 1.5 and (v < 70 or s < 35) and ocr_team is None:
                self._team_assignments[pid] = 'REF'
                continue

            # OCR override only when the colour signal is weak (small margin
            # between teams) OR when OCR agrees with colour. A strong-margin
            # disagreement is treated as an OCR misread.
            if ocr_team is not None and (ocr_team == color_team or color_margin < 0.4):
                self._team_assignments[pid] = ocr_team
            else:
                self._team_assignments[pid] = color_team

        return self._team_assignments


class JerseyNumberOCR:
    """
    Reads jersey numbers from player bounding boxes using EasyOCR.
    Only attempts OCR periodically (every N frames) to save processing time.
    """

    def __init__(self, ocr_interval_frames: int = 5):
        self.ocr_interval = ocr_interval_frames
        self._reader = None
        self._number_readings: Dict[int, List[int]] = defaultdict(list)
        self._final_numbers: Dict[int, int] = {}

    def _ensure_reader(self):
        """Lazy-initialize EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            except ImportError:
                print("  [!] easyocr not installed. Jersey OCR disabled.")
                print("      Install with: pip install easyocr")
                self._reader = False
            except Exception as e:
                print(f"  [!] EasyOCR init failed: {e}")
                self._reader = False

    def should_attempt(self, frame_idx: int) -> bool:
        """Check if we should attempt OCR this frame."""
        return frame_idx % self.ocr_interval == 0

    def read_number(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float], player_id: int
    ) -> Optional[int]:
        """Attempt to read jersey number from player bbox."""
        self._ensure_reader()
        if self._reader is False:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_h = y2 - y1
        box_w = x2 - x1
        if box_h < 30 or box_w < 15:
            return None

        num_y1 = y1 + int(box_h * 0.1)
        num_y2 = y1 + int(box_h * 0.5)
        margin = int(box_w * 0.15)
        num_x1 = x1 + margin
        num_x2 = x2 - margin

        if num_x2 <= num_x1 or num_y2 <= num_y1:
            return None

        crop = frame[num_y1:num_y2, num_x1:num_x2]
        if crop.size == 0:
            return None

        scale = max(1, 60 // max(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        try:
            results = self._reader.readtext(
                thresh,
                allowlist='0123456789',
                detail=1,
                paragraph=False,
            )
        except Exception:
            return None

        for (bbox_ocr, text, conf) in results:
            if conf < 0.3:
                continue
            digits = ''.join(c for c in text if c.isdigit())
            if digits:
                num = int(digits)
                if 1 <= num <= 99:
                    self._number_readings[player_id].append(num)
                    return num

        return None

    def get_live_number(self, player_id: int) -> Optional[int]:
        """Get the current best jersey number for a player during processing.
        Uses majority voting from readings collected so far.
        """
        readings = self._number_readings.get(player_id, [])
        if not readings:
            return None
        counter = Counter(readings)
        best_num, best_count = counter.most_common(1)[0]
        # Require at least 2 readings or a single confident reading
        if best_count >= 2 or (best_count >= 1 and len(readings) <= 2):
            return best_num
        return None

    def finalize(self) -> Dict[int, int]:
        """Use majority voting to assign final jersey numbers."""
        for pid, readings in self._number_readings.items():
            if readings:
                counter = Counter(readings)
                best_num, best_count = counter.most_common(1)[0]
                # Lower threshold - require only 1 reading if it's the only one
                if best_count >= 1:
                    self._final_numbers[pid] = best_num
        return dict(self._final_numbers)

    def get_number(self, player_id: int) -> Optional[int]:
        """Get jersey number for a player, or None if not detected."""
        return self._final_numbers.get(player_id)


class RosterConfig:
    """Loads team/player roster from a JSON config file."""

    def __init__(self):
        self.match_name: str = ""
        self.teams: Dict[str, Dict] = {}

    def load(self, config_path: Path) -> bool:
        """Load roster from JSON file. Returns True if successful."""
        if not config_path.exists():
            return False
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.match_name = data.get("match", "")
            self.teams = data.get("teams", {})
            return True
        except Exception as e:
            print(f"  [!] Failed to load roster: {e}")
            return False

    def load_from_db(self, db_manager, team_a_name: str, team_b_name: str,
                     team_a_color: str = None, team_b_color: str = None) -> bool:
        """
        Load rosters from the database instead of a JSON file.

        Args:
            db_manager: DatabaseManager instance
            team_a_name: Name of team A (e.g. "Liverpool")
            team_b_name: Name of team B (e.g. "Manchester City")
            team_a_color: Optional jersey color for team A
            team_b_color: Optional jersey color for team B

        Returns:
            True if at least one team had players in the DB
        """
        self.match_name = f"{team_a_name} vs {team_b_name}"
        self.teams = {}

        for side, name, color in [("A", team_a_name, team_a_color),
                                   ("B", team_b_name, team_b_color)]:
            roster = db_manager.get_roster_for_team(name)
            players = {}
            for p in roster:
                if p.get("jersey_number") is not None:
                    players[str(p["jersey_number"])] = p["name"]

            team_entry = {"name": name, "players": players}
            if color:
                team_entry["jersey_color"] = color
            self.teams[side] = team_entry

        return self.is_loaded()

    def get_player_name(self, team: str, jersey_number: int) -> Optional[str]:
        """Look up player name by team and jersey number."""
        team_data = self.teams.get(team, {})
        players = team_data.get("players", {})
        return players.get(str(jersey_number))

    def get_team_name(self, team: str) -> str:
        """Get display name for a team."""
        return self.teams.get(team, {}).get("name", f"Team {team}")

    def is_loaded(self) -> bool:
        return bool(self.teams)


class PlayerIdentityManager:
    def __init__(self, roster_path: Optional[Path] = None, enable_ocr: bool = True,
                 db_manager=None, team_a_name: str = None, team_b_name: str = None,
                 team_a_color: str = None, team_b_color: str = None):
        self.roster = RosterConfig()

        # Priority: DB roster > JSON file
        if db_manager and team_a_name and team_b_name:
            loaded = self.roster.load_from_db(
                db_manager, team_a_name, team_b_name,
                team_a_color=team_a_color, team_b_color=team_b_color,
            )
            if loaded:
                a_count = len(self.roster.teams.get("A", {}).get("players", {}))
                b_count = len(self.roster.teams.get("B", {}).get("players", {}))
                print(f"  Roster loaded from DB: {self.roster.match_name}")
                print(f"    {team_a_name}: {a_count} players, {team_b_name}: {b_count} players")
        elif roster_path:
            loaded = self.roster.load(roster_path)
            if loaded:
                print(f"  Roster loaded: {self.roster.match_name}")

        # Extract team colors from roster
        team_colors = {}
        if self.roster.is_loaded():
            for team_id in ["A", "B"]:
                team_data = self.roster.teams.get(team_id, {})
                color = team_data.get("jersey_color")
                if color:
                    team_colors[team_id] = color
                    print(f"    Team {team_id} ({team_data.get('name', team_id)}): {color}")

        self.color_classifier = JerseyColorClassifier(team_colors=team_colors)
        self.ocr = JerseyNumberOCR(ocr_interval_frames=5) if enable_ocr else None

        self._identities: Dict[int, Dict[str, Any]] = {}


    def process_frame(
        self,
        frame: np.ndarray,
        player_bboxes: Dict[int, Tuple[float, float, float, float]],
        frame_idx: int,
    ):
        """
        Process a single frame - sample colors and optionally run OCR.

        Args:
            frame: The video frame (BGR)
            player_bboxes: {canonical_player_id: (x1,y1,x2,y2)}
            frame_idx: Current frame index
        """
        for pid, bbox in player_bboxes.items():
            self.color_classifier.sample_jersey_color(frame, bbox, pid)

            if self.ocr and self.ocr.should_attempt(frame_idx):
                self.ocr.read_number(frame, bbox, pid)

    def get_live_team(self, player_id: int) -> Optional[str]:
        """
        Get team prediction for a player during processing (before finalize).
        Uses color clustering with accumulated samples.
        Returns 'A', 'B', or None if not enough data yet.
        """
        return self.color_classifier.predict_team_live(player_id)

    def finalize(self) -> Dict[int, Dict[str, Any]]:
        """
        After video processing, compute final player identities.

        Returns:
            Dict mapping canonical_id -> {team, number, name, display}
        """
        # OCR-anchored team finalization: use jersey numbers (which only exist
        # in one roster) to label colour clusters, instead of trying to match
        # broadcast-shifted HSV samples to canonical team hex codes.
        ocr_readings: Dict[int, List[int]] = {}
        if self.ocr:
            ocr_readings = dict(self.ocr._number_readings)

        roster_a = self.roster.teams.get("A", {}).get("players", {}) if self.roster.is_loaded() else {}
        roster_b = self.roster.teams.get("B", {}).get("players", {}) if self.roster.is_loaded() else {}

        if self.roster.is_loaded() and ocr_readings:
            team_assignments = self.color_classifier.finalize_teams_ocr_anchored(
                ocr_readings, roster_a, roster_b
            )
            a_n = sum(1 for t in team_assignments.values() if t == 'A')
            b_n = sum(1 for t in team_assignments.values() if t == 'B')
            r_n = sum(1 for t in team_assignments.values() if t == 'REF')
            print(f"  Team finalize (OCR-anchored): A={a_n}, B={b_n}, REF={r_n}")
            # Diagnostic dump: per-player sampled HSV vs team assignment.
            try:
                from pathlib import Path as _P
                import json as _json
                _dump = []
                for _pid, _hues in self.color_classifier._color_samples.items():
                    if len(_hues) < 3:
                        continue
                    _h = self.color_classifier._circular_mean_hue(_hues)
                    _s = float(np.median(self.color_classifier._saturation_samples.get(_pid, [0])))
                    _v = float(np.median(self.color_classifier._value_samples.get(_pid, [0])))
                    _dump.append({
                        "pid": int(_pid),
                        "n_samples": len(_hues),
                        "h": int(_h), "s": int(_s), "v": int(_v),
                        "team": team_assignments.get(_pid),
                        "ocr": [int(x) for x in ocr_readings.get(_pid, [])],
                    })
                _P("logs").mkdir(exist_ok=True)
                with open("logs/hsv_dump.json", "w", encoding="utf-8") as _f:
                    _json.dump(_dump, _f, indent=2)
                print(f"  [diag] wrote logs/hsv_dump.json ({len(_dump)} players)")
            except Exception as _ex:
                print(f"  [diag] HSV dump failed: {_ex}")
        else:
            team_assignments = self.color_classifier.finalize_teams()

        jersey_numbers = {}
        if self.ocr:
            jersey_numbers = self.ocr.finalize()

        # Include ALL tracked players, not just those with jersey numbers
        all_pids = set(team_assignments.keys())

        # Also add players that only have jersey number detections
        all_pids.update(set(jersey_numbers.keys()))

        # Add any players that have been seen but not yet classified
        all_pids.update(set(self.color_classifier._color_samples.keys()))

        for pid in all_pids:
            team = team_assignments.get(pid, None)
            number = jersey_numbers.get(pid, None)
            name = None

            # Gate roster name lookup: only resolve a name if the OCR'd number
            # actually exists in this player's assigned team's roster. Without
            # this guard, a Liverpool player whose number was misread as 6 (a
            # number that exists in Tottenham's roster) gets labelled
            # "Radu Dragusin" — the bug visible in the verification screenshot.
            if team in ('A', 'B') and number and self.roster.is_loaded():
                roster_for_team = roster_a if team == 'A' else roster_b
                if str(int(number)) in roster_for_team:
                    name = roster_for_team[str(int(number))]
                else:
                    # OCR number doesn't match this team's roster — likely a
                    # bad read, drop it to avoid wrong-team name attribution.
                    number = None

            if name and number:
                display = f"{name} (#{number})"
            elif number and team in ('A', 'B'):
                team_name = self.roster.get_team_name(team) if self.roster.is_loaded() else f"Team {team}"
                display = f"{team_name} #{number}"
            elif team in ('A', 'B'):
                team_name = self.roster.get_team_name(team) if self.roster.is_loaded() else f"Team {team}"
                display = f"{team_name} (P{pid})"
            elif team == 'REF':
                display = f"Referee (P{pid})"
            else:
                display = f"P{pid}"

            self._identities[pid] = {
                "team": team if team else "unknown",
                "number": number,
                "name": name,
                "display": display,
            }

        return self._identities

    def _build_live_identity(self, player_id: int) -> Dict[str, Any]:
        """Build identity info using live data during frame processing.
        This allows displaying player names before finalize() is called.
        """
        # Get live team assignment
        team = self.color_classifier.predict_team_live(player_id)
        
        # Get live jersey number from OCR readings so far
        number = None
        if self.ocr:
            number = self.ocr.get_live_number(player_id)
        
        # Look up player name if we have team and number
        name = None
        if team and number and self.roster.is_loaded():
            name = self.roster.get_player_name(team, number)
        
        # Build display name
        if name and number:
            display = f"{name} (#{number})"
        elif number and team:
            team_name = self.roster.get_team_name(team) if self.roster.is_loaded() else f"Team {team}"
            display = f"{team_name} #{number}"
        elif team:
            team_name = self.roster.get_team_name(team) if self.roster.is_loaded() else f"Team {team}"
            display = f"{team_name} (P{player_id})"
        else:
            display = f"P{player_id}"
        
        return {
            "team": team if team else "unknown",
            "number": number,
            "name": name,
            "display": display,
        }

    def get_identity(self, player_id: int) -> Dict[str, Any]:
        """Get identity info for a player.
        Uses finalized identities if available, otherwise builds from live data.
        """
        if player_id in self._identities:
            return self._identities[player_id]
        
        # Return live identity during processing
        return self._build_live_identity(player_id)

    def get_display_name(self, player_id: int) -> str:
        """Get short display name for overlay.
        Uses finalized identities if available, otherwise builds from live data.
        """
        if player_id in self._identities:
            return self._identities[player_id].get("display", f"P{player_id}")
        
        # Build live display name during processing
        live_identity = self._build_live_identity(player_id)
        return live_identity.get("display", f"P{player_id}")

    def get_all_identities(self) -> Dict[int, Dict[str, Any]]:
        """Return all player identities."""
        return dict(self._identities)

    def get_team_for_player(self, player_id: int) -> Optional[str]:
        """Get team label for a player."""
        identity = self._identities.get(player_id, {})
        team = identity.get("team")
        return team if team != "unknown" else None
