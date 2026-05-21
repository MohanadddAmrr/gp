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
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    from services.roster_io_sentinel import track_json_file_read
except ImportError:
    def track_json_file_read() -> None:  # pragma: no cover
        pass


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

    def _hue_matches_color(self, hue: int, color_name: str) -> bool:
        """Check if a hue value matches the expected color."""
        ranges = self._get_hue_ranges_for_color(color_name)
        if ranges is None:
            return False
        for low, high in ranges:
            if low <= hue <= high:
                return True
        return False

    def sample_jersey_color(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float], player_id: int
    ):
        """Sample the dominant jersey color from a player's bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        torso_y2 = y1 + int((y2 - y1) * 0.4)
        box_w = x2 - x1
        margin = int(box_w * 0.2)
        torso_x1 = x1 + margin
        torso_x2 = x2 - margin

        if torso_x2 <= torso_x1 or torso_y2 <= y1:
            return

        torso = frame[y1:torso_y2, torso_x1:torso_x2]
        if torso.size == 0:
            return

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hues = hsv[:, :, 0].flatten()
        sats = hsv[:, :, 1].flatten()
        vals = hsv[:, :, 2].flatten()

        # Filter out low saturation (white/grey/black)
        mask = (sats > 40) & (vals > 50) & (vals < 230)
        valid_hues = hues[mask]
        valid_sats = sats[mask]
        valid_vals = vals[mask]

        if len(valid_hues) < 10:
            return

        # Dominant hue
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
                median_hue = int(np.median(samples))
                
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
        medians = [int(np.median(eligible[p])) for p in pids]
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
                player_hues[pid] = int(np.median(hues))

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
            
            # Check for referee characteristics
            is_referee = False
            
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
            track_json_file_read()
            return True
        except Exception as e:
            print(f"  [!] Failed to load roster: {e}")
            return False

    def load_from_db(self, db_manager, team_a_name: str, team_b_name: str,
                     team_a_color: str = None, team_b_color: str = None) -> bool:
        """
        Load rosters from the database using DatabaseManager CRUD methods.
        """
        self.match_name = f"{team_a_name} vs {team_b_name}"
        self.teams = {}

        # Get all teams to find matching IDs by name
        all_teams = db_manager.list_teams()
        team_map = {t['name'].lower(): t for t in all_teams}

        for side, name, color_fallback in [("A", team_a_name, team_a_color),
                                           ("B", team_b_name, team_b_color)]:
            players = {}
            color = color_fallback
            
            t_info = team_map.get(name.lower())
            if t_info:
                tid = t_info['team_id']
                if t_info.get('primary_color') and t_info['primary_color'] != '#FFFFFF':
                    color = t_info['primary_color']
                
                lookup = db_manager.get_roster_lookup(tid)
                players = {str(k): v for k, v in lookup.items()}

            team_entry = {"name": name, "players": players}
            if color:
                team_entry["jersey_color"] = color
            self.teams[side] = team_entry

        total_players = sum(len(t.get("players", {})) for t in self.teams.values())
        return total_players > 0

    def load_from_bundle(self, data: Dict[str, Any]) -> bool:
        """Load from a JSON-shaped dict (e.g. DynamicRosterManager.roster_identity_bundle_from_stem)."""
        if not data or not isinstance(data, dict):
            return False
        self.match_name = str(data.get("match") or "")
        self.teams = dict(data.get("teams") or {})
        return bool(self.teams)

    def load_from_match_roster(
        self,
        roster_data: Dict[str, Any],
        team_a_color: Optional[str] = None,
        team_b_color: Optional[str] = None,
    ) -> bool:
        """
        Populate roster from DatabaseManager.get_match_roster() shape:
        home_team / away_team with name, players (jersey str -> name), optional colors.
        """
        ht = roster_data.get("home_team") or {}
        at = roster_data.get("away_team") or {}
        if not ht.get("name") or not at.get("name"):
            return False
        hp = ht.get("players") or {}
        ap = at.get("players") or {}
        if not hp and not ap:
            return False

        self.match_name = f"{ht['name']} vs {at['name']}"
        self.teams = {
            "A": {"name": ht["name"], "players": {str(k): v for k, v in hp.items()}},
            "B": {"name": at["name"], "players": {str(k): v for k, v in ap.items()}},
        }

        def jersey_hint(team_row: Dict, override: Optional[str]) -> Optional[str]:
            if override:
                return override
            jc = team_row.get("jersey_color")
            if jc:
                return jc
            pc = team_row.get("primary_color")
            if isinstance(pc, str) and pc and not pc.startswith("#"):
                return pc
            return None

        h_color = jersey_hint(ht, team_a_color)
        a_color = jersey_hint(at, team_b_color)
        if h_color:
            self.teams["A"]["jersey_color"] = h_color
        if a_color:
            self.teams["B"]["jersey_color"] = a_color

        return True

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
                 team_a_color: str = None, team_b_color: str = None,
                 roster_manager=None, match_id: Optional[str] = None,
                 roster_bundle: Optional[Dict[str, Any]] = None):
        self.roster = RosterConfig()

        mid = str(match_id) if match_id is not None else None

        # 1) Sprint blueprint match_rosters + roster_players (via RosterManager.get_player_lookup)
        if roster_manager is not None and mid:
            lookup = roster_manager.get_player_lookup(mid)
            if lookup is not None:
                full = roster_manager.db.get_match_roster(mid)
                if full and self.roster.load_from_match_roster(
                    full,
                    team_a_color=team_a_color,
                    team_b_color=team_b_color,
                ):
                    a_count = len(self.roster.teams.get("A", {}).get("players", {}))
                    b_count = len(self.roster.teams.get("B", {}).get("players", {}))
                    ta = self.roster.teams.get("A", {}).get("name", "Team A")
                    tb = self.roster.teams.get("B", {}).get("name", "Team B")
                    print(f"  Roster loaded from database (match roster): {self.roster.match_name}")
                    print(f"    {ta}: {a_count} players, {tb}: {b_count} players")

        # 2) C2 dynamic lineup tables (lineup_teams / lineup_players / match_lineups)
        if not self.roster.is_loaded() and roster_bundle:
            if self.roster.load_from_bundle(roster_bundle):
                a_count = len(self.roster.teams.get("A", {}).get("players", {}))
                b_count = len(self.roster.teams.get("B", {}).get("players", {}))
                ta = self.roster.teams.get("A", {}).get("name", "Team A")
                tb = self.roster.teams.get("B", {}).get("name", "Team B")
                print(f"  Roster loaded from dynamic lineup (C2): {self.roster.match_name}")
                print(f"    {ta}: {a_count} players, {tb}: {b_count} players")

        # 3) Legacy: player_profiles keyed by team name
        if not self.roster.is_loaded() and db_manager and team_a_name and team_b_name:
            loaded = self.roster.load_from_db(
                db_manager, team_a_name, team_b_name,
                team_a_color=team_a_color, team_b_color=team_b_color,
            )
            if loaded:
                a_count = len(self.roster.teams.get("A", {}).get("players", {}))
                b_count = len(self.roster.teams.get("B", {}).get("players", {}))
                print(f"  Roster loaded from DB: {self.roster.match_name}")
                print(f"    {team_a_name}: {a_count} players, {team_b_name}: {b_count} players")

        # 4) JSON file on disk (deprecated when DB path exists elsewhere)
        if not self.roster.is_loaded() and roster_path:
            path = Path(roster_path)
            if path.exists():
                loaded = self.roster.load(path)
                if loaded:
                    logger.warning(
                        "Using deprecated JSON roster. Please migrate to database."
                    )
                    print(f"  Roster loaded: {self.roster.match_name}")
            else:
                logger.warning(
                    "Roster JSON path missing (%s). Continuing without named players.",
                    path,
                )

        if not self.roster.is_loaded():
            if roster_manager is not None and mid:
                logger.warning(
                    "No database match roster for match_id=%r and no usable JSON roster; continuing without named players.",
                    mid,
                )
            else:
                logger.info(
                    "No roster loaded; player names require OCR + color data or configure a DB/JSON roster."
                )

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

            if team and number and self.roster.is_loaded():
                name = self.roster.get_player_name(team, number)

            if name and number:
                display = f"{name} (#{number})"
            elif number and team:
                team_name = self.roster.get_team_name(team) if self.roster.is_loaded() else f"Team {team}"
                display = f"{team_name} #{number}"
            elif team:
                team_name = self.roster.get_team_name(team) if self.roster.is_loaded() else f"Team {team}"
                display = f"{team_name} (P{pid})"
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
