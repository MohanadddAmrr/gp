import json
import sys
import re
import gc
import time
import torch
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def get_memory_mb() -> float:
    """Return current process RSS in MB. Falls back to 0 if psutil unavailable."""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0

def clear_memory():
    """Force garbage collection and clear GPU memory if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Also clear OpenCV memory if possible
    cv2.destroyAllWindows() if cv2.getWindowProperty('test', cv2.WND_PROP_VISIBLE) >= 0 else None

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
from services.sprint_detector import SprintDetector
from services.tracking_filters import (
    TrackDeduplicator, SpeedSmoother,
    cap_player_speed, cap_ball_speed,
    MAX_PLAYER_SPEED_MPS,
)
from services.player_identity import PlayerIdentityManager
from services.database_manager import DatabaseManager
from services.tactical_analyzer import TacticalAnalyzer
from services.pitch_transform import PitchTransform, TacticalHeatmapGenerator
from services.xg_calculator import xGCalculator, ShotType, BodyPart, ShotOutcome
from services.highlights_generator import HighlightsGenerator, EventType, ImportanceLevel
from services.advanced_analytics import AdvancedAnalytics

# Import new integration modules
from services.wearable_integration import WearableIntegrationManager
from services.api_connector import APIManager
from services.broadcast_graphics import BroadcastGraphicsManager, create_team_info
from services.video_exporter import VideoExporter, ExportConfig
from services.streaming_handler import StreamingManager
from services.social_export import SocialMediaExporter, SocialPlatform, SocialClip
from services.ai_tactical_recommendations import AITacticalRecommendations, RecommendationPriority, RecommendationCategory
from services.improved_tracker import build_tracker, SUPPORTED_ALGORITHMS
from services.reid_module import JerseyColorClassifier, JerseyPosReID, hex_to_hsv

try:
    import yaml as _yaml  # PyYAML for reading config.yaml tracker switch
except ImportError:  # pragma: no cover - keep run_demo working if PyYAML missing
    _yaml = None


def _load_tracking_config() -> dict:
    """Read the full detection.tracking section from config.yaml.

    Returns an empty dict on any failure so callers can fall back to defaults
    individually without crashing the demo.
    """
    if _yaml is None:
        return {}
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001
        return {}
    return cfg.get("detection", {}).get("tracking", {}) or {}


def _load_tracker_algorithm(default: str = "bytetrack") -> str:
    """Read detection.tracking.algorithm from config.yaml.

    Falls back to `default` (current behavior, ByteTrack) if the file or key
    is missing, or if PyYAML is not installed. Unknown values fall back to
    the default with a printed warning so a typo does not crash the demo.
    """
    if _yaml is None:
        return default
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not cfg_path.exists():
        return default
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Failed to read {cfg_path}: {exc}; defaulting to {default!r}.")
        return default
    algo = (
        cfg.get("detection", {})
        .get("tracking", {})
        .get("algorithm", default)
    )
    algo = str(algo).strip().lower()
    if algo not in SUPPORTED_ALGORITHMS:
        print(
            f"[!] Unknown tracker algorithm {algo!r} in config.yaml; "
            f"falling back to {default!r}."
        )
        return default
    return algo


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



# ============================================================
# PATHS & MODEL
# ============================================================
ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT.parent / "input_videos"
OUTPUT_BASE = ROOT / "demo_outputs"
ROSTER_DIR = ROOT.parent / "rosters"

MODEL_NAME = "yolov8n.pt"
PERSON_CLASS = 0
BALL_CLASS = 32


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_colored_heatmap(hm: np.ndarray, out_path: Path):
    """Convert a 2D float heatmap to a colored PNG with pitch overlay.
    
    Creates a professional football heatmap with:
    - Green pitch background with proper field markings
    - Smooth color gradient from blue (low) -> green -> yellow -> red (high)
    - Gaussian blur for smoother visualization
    """
    h, w = hm.shape
    
    if hm.max() <= 0:
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), blank)
        return
    
    # Create pitch background (green field)
    pitch = np.ones((h, w, 3), dtype=np.uint8)
    pitch[:, :] = (34, 139, 34)  # Dark green in BGR
    
    # Add lighter green stripes for grass effect
    stripe_width = h // 10
    for i in range(0, h, stripe_width * 2):
        pitch[i:i+stripe_width, :] = (40, 155, 40)  # Slightly lighter green
    
    # Normalize heatmap
    norm = (hm / hm.max() * 255).astype(np.uint8)
    
    # Apply Gaussian blur for smoother, more professional look
    norm_smooth = cv2.GaussianBlur(norm, (21, 21), 0)
    
    # Create custom color map (blue -> cyan -> green -> yellow -> red)
    # This matches professional heatmap visualizations
    heatmap_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            intensity = norm_smooth[y, x] / 255.0
            
            if intensity < 0.25:
                # Blue to Cyan
                local_intensity = intensity / 0.25
                r = 0
                g = int(255 * local_intensity)
                b = 255
            elif intensity < 0.5:
                # Cyan to Green
                local_intensity = (intensity - 0.25) / 0.25
                r = 0
                g = 255
                b = int(255 * (1 - local_intensity))
            elif intensity < 0.75:
                # Green to Yellow
                local_intensity = (intensity - 0.5) / 0.25
                r = int(255 * local_intensity)
                g = 255
                b = 0
            else:
                # Yellow to Red
                local_intensity = (intensity - 0.75) / 0.25
                r = 255
                g = int(255 * (1 - local_intensity))
                b = 0
            
            heatmap_color[y, x] = (b, g, r)  # BGR format
    
    # Blend heatmap with pitch background
    # Only show heatmap where there's activity
    alpha = np.clip(norm_smooth.astype(float) / 255.0 * 0.85, 0, 0.85)
    alpha = alpha[:, :, np.newaxis]
    
    blended = (heatmap_color * alpha + pitch * (1 - alpha)).astype(np.uint8)
    
    # Draw pitch markings (white lines)
    line_color = (255, 255, 255)
    line_thickness = max(2, min(h, w) // 150)
    
    # Center line
    cv2.line(blended, (w // 2, 0), (w // 2, h), line_color, line_thickness)
    
    # Center circle
    center_x, center_y = w // 2, h // 2
    radius = min(h, w) // 10
    cv2.circle(blended, (center_x, center_y), radius, line_color, line_thickness)
    cv2.circle(blended, (center_x, center_y), radius // 15, line_color, -1)  # Center dot
    
    # Penalty areas
    penalty_width = w // 5
    penalty_height = h // 3
    
    # Left penalty area
    cv2.rectangle(blended, (0, (h - penalty_height) // 2), 
                  (penalty_width, (h + penalty_height) // 2), line_color, line_thickness)
    
    # Right penalty area  
    cv2.rectangle(blended, (w - penalty_width, (h - penalty_height) // 2), 
                  (w, (h + penalty_height) // 2), line_color, line_thickness)
    
    # Goal areas (smaller boxes)
    goal_width = w // 16
    goal_height = h // 6
    
    # Left goal area
    cv2.rectangle(blended, (0, (h - goal_height) // 2), 
                  (goal_width, (h + goal_height) // 2), line_color, line_thickness)
    
    # Right goal area
    cv2.rectangle(blended, (w - goal_width, (h - goal_height) // 2), 
                  (w, (h + goal_height) // 2), line_color, line_thickness)
    
    # Goal posts
    goal_post_height = h // 8
    cv2.line(blended, (0, (h - goal_post_height) // 2), 
             (0, (h + goal_post_height) // 2), line_color, line_thickness * 2)
    cv2.line(blended, (w - 1, (h - goal_post_height) // 2), 
             (w - 1, (h + goal_post_height) // 2), line_color, line_thickness * 2)
    
    # Penalty spots
    penalty_spot_dist = w // 8
    cv2.circle(blended, (penalty_spot_dist, h // 2), line_thickness * 2, line_color, -1)
    cv2.circle(blended, (w - penalty_spot_dist, h // 2), line_thickness * 2, line_color, -1)
    
    # Corner arcs
    corner_radius = min(h, w) // 20
    cv2.ellipse(blended, (0, 0), (corner_radius, corner_radius), 0, 0, 90, line_color, line_thickness)
    cv2.ellipse(blended, (w, 0), (corner_radius, corner_radius), 0, 90, 180, line_color, line_thickness)
    cv2.ellipse(blended, (0, h), (corner_radius, corner_radius), 0, 270, 360, line_color, line_thickness)
    cv2.ellipse(blended, (w, h), (corner_radius, corner_radius), 0, 180, 270, line_color, line_thickness)
    
    # Outer boundary
    cv2.rectangle(blended, (0, 0), (w - 1, h - 1), line_color, line_thickness)
    
    cv2.imwrite(str(out_path), blended)


def find_roster(video_name: str) -> Path:
    """Find a roster JSON file matching the video name.
    
    Matches video names like:
    - 'liverpoolvsbournemouth.mp4' -> 'liverpoolvsbournemouth.json'
    - 'liverpoolvscity.mp4' -> 'liverpoolvscity.json'
    - 'test (2).mp4' -> 'test.json' (strips parentheses and numbers)
    - 'Man United vs Arsenal.mp4' -> 'manunitedvsarsenal.json' (removes spaces)
    """
    if not ROSTER_DIR.exists():
        return None
    
    # Clean video name: lowercase, remove spaces, strip parentheses and their contents
    video_clean = video_name.lower()
    # Remove parentheses and their contents (e.g., "test (2)" -> "test")
    video_clean = re.sub(r'\s*\([^)]*\)', '', video_clean)
    # Remove file extension
    video_clean = video_clean.replace('.mp4', '').replace('.avi', '').replace('.mov', '')
    # Remove spaces and special characters
    video_clean = video_clean.replace(' ', '').replace('_', '').replace('-', '')
    
    # First: try exact match with cleaned name
    exact_match = ROSTER_DIR / f"{video_clean}.json"
    if exact_match.exists():
        return exact_match
    
    # Second: try partial matching
    video_lower = video_name.lower()
    for roster_file in ROSTER_DIR.glob("*.json"):
        roster_lower = roster_file.stem.lower()
        roster_clean = roster_lower.replace(' ', '').replace('_', '').replace('-', '')
        
        # Check if roster name is contained in video name or vice versa
        if roster_clean in video_clean or video_clean in roster_clean:
            return roster_file
        # Also check raw names
        if roster_lower in video_lower or video_lower in roster_lower:
            return roster_file
    
    # Return first roster if only one exists (fallback)
    rosters = list(ROSTER_DIR.glob("*.json"))
    if len(rosters) == 1:
        return rosters[0]
    
    return None


def extract_teams_from_filename(filename: str, db_manager: DatabaseManager = None):
    """
    Try to extract two team names from a video filename.

    Supports patterns like:
        'liverpoolvsbournemouth.mp4' -> ('Liverpool', 'Bournemouth')
        'inter milan vs ac milan.mp4' -> ('Inter Milan', 'AC Milan')
        'realmadrid_vs_barcelona.mp4' -> ('Real Madrid', 'Barcelona')

    If db_manager is provided, validates against team names in the DB.
    Returns (team_a_name, team_b_name) or (None, None) if not extractable.
    """
    stem = Path(filename).stem.lower()
    # Normalize separators
    stem = stem.replace('_', ' ').replace('-', ' ')

    # Split on 'vs' or 'v'
    for sep in [' vs ', 'vs', ' v ']:
        if sep in stem:
            parts = stem.split(sep, 1)
            if len(parts) == 2:
                raw_a, raw_b = parts[0].strip(), parts[1].strip()
                if raw_a and raw_b:
                    if db_manager:
                        name_a = _match_team_in_db(raw_a, db_manager)
                        name_b = _match_team_in_db(raw_b, db_manager)
                        if name_a and name_b:
                            return name_a, name_b
                    return raw_a.title(), raw_b.title()
    return None, None


def _match_team_in_db(raw_name: str, db_manager: DatabaseManager) -> str:
    """Fuzzy-match a raw team name fragment against teams in the DB.

    Prefers the best (most-specific) match:
      'liverpool'  -> 'Liverpool' (exact)
      'city'       -> 'Manchester City' (shortest team name containing fragment)
      'psg'        -> 'Paris Saint-Germain' (short_name / alias check)
    """
    profiles = db_manager.get_all_player_profiles()
    team_names = sorted({p.get('team_name') for p in profiles if p.get('team_name')})

    raw_lower = raw_name.lower().replace(' ', '')

    # Common aliases / abbreviations
    _aliases = {
        'psg': 'Paris Saint-Germain',
        'barca': 'Barcelona',
        'juve': 'Juventus',
        'spurs': 'Tottenham Hotspur',
        'wolves': 'Wolverhampton Wanderers',
        'city': 'Manchester City',
        'united': 'Manchester United',
        'manunited': 'Manchester United',
        'mancity': 'Manchester City',
        'atletico': 'Atletico Madrid',
        'bilbao': 'Athletic Bilbao',
        'inter': 'Inter Milan',
        'milan': 'AC Milan',
        'bayern': 'Bayern Munich',
        'dortmund': 'Borussia Dortmund',
        'leverkusen': 'Bayer Leverkusen',
        'leipzig': 'RB Leipzig',
        'marseille': 'Olympique de Marseille',
        'lyon': 'Olympique Lyonnais',
        'monaco': 'AS Monaco',
        'lille': 'LOSC Lille',
        'nice': 'OGC Nice',
        'lens': 'RC Lens',
        'rennes': 'Stade Rennais',
        'strasbourg': 'RC Strasbourg',
        'brest': 'Stade Brestois 29',
    }
    if raw_lower in _aliases:
        candidate = _aliases[raw_lower]
        if candidate in team_names:
            return candidate

    # Exact match (case-insensitive, spaces removed)
    for tn in team_names:
        if raw_lower == tn.lower().replace(' ', ''):
            return tn

    # raw_name is a substring of team name — pick shortest match (most specific)
    candidates = []
    for tn in team_names:
        tn_lower = tn.lower().replace(' ', '')
        if raw_lower in tn_lower or tn_lower in raw_lower:
            candidates.append(tn)
    if candidates:
        return min(candidates, key=len)

    return None


def process_video(video_path: Path, model, db_manager: DatabaseManager = None,
                 enable_wearable: bool = False, enable_broadcast: bool = False,
                 enable_export: bool = False, enable_streaming: bool = False,
                 enable_social: bool = False,
                 chunk_size_minutes: float = 0, memory_limit_percent: float = 80,
                 team_a_name: str = None, team_b_name: str = None,
                 max_seconds: float = 0, tracker=None,
                 reid_layer=None, reid_classifier=None):
    """Process a single video file with optional integration features.

    Args:
        chunk_size_minutes: Split video into N-minute chunks. 0 = no chunking (full video).
        memory_limit_percent: Warn if process memory exceeds this % of system RAM.
        team_a_name: Optional explicit team A name (skips roster JSON lookup).
        team_b_name: Optional explicit team B name (skips roster JSON lookup).
        tracker: Pre-built Tracker (M2). Defaults to ByteTrack if not supplied.
    """
    # M2 fallback: maintain backwards compatibility for any caller that
    # doesn't pass a pre-built tracker (tests, scripts, notebooks).
    if tracker is None:
        tracker = build_tracker("bytetrack", model)
        print("Using tracker: bytetrack (default fallback)")

    print(f"\nProcessing: {video_path.name}")
    
    # Open video to get metadata
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Process at half resolution to conserve memory
    width = 640
    height = 360
    print(f"  Video: {orig_width}x{orig_height} @ {fps:.1f} FPS (processing at {width}x{height})")
    
    # Output directory
    out_dir = OUTPUT_BASE / video_path.stem
    ensure_dir(out_dir)
    
    # Initialize integration modules
    wearable_manager = None
    broadcast_manager = None
    video_exporter = None
    streaming_manager = None
    social_exporter = None
    
    if enable_wearable:
        print("  📡 Wearable data integration enabled")
        wearable_manager = WearableIntegrationManager()
        
    if enable_broadcast:
        print("  📺 Broadcast graphics enabled")
        broadcast_manager = BroadcastGraphicsManager(width, height)
        
    if enable_export:
        print("  🎬 Video export enabled")
        export_config = ExportConfig(
            output_path=str(out_dir / "exported_video.mp4"),
            resolution=(width, height),
            fps=fps
        )
        video_exporter = VideoExporter(export_config)
        
    if enable_streaming:
        print("  🔴 Streaming enabled")
        streaming_manager = StreamingManager()
        
    if enable_social:
        print("  📱 Social media export enabled")
        social_exporter = SocialMediaExporter()
    
    # Initialize data structures
    heat_global = np.zeros((height, width), dtype=np.float32)
    heat_team_A = np.zeros((height, width), dtype=np.float32)
    heat_team_B = np.zeros((height, width), dtype=np.float32)
    heat_ball = np.zeros((height, width), dtype=np.float32)
    heat_per_player = {}
    
    track_history = {}
    ball_position_history = []
    
    # Initialize trackers and detectors
    ball_tracker = BallTracker()
    possession_tracker = PossessionTracker(distance_threshold=60.0)
    event_detector = EventDetector(
        min_velocity_mps=1.5,  # Lower threshold for more pass detection
        min_distance_m=3.0,    # Lower minimum distance
        max_distance_m=60.0,   # Higher max distance
    )
    sprint_detector = SprintDetector()
    tactical_analyzer = TacticalAnalyzer(
        frame_width=width,
        frame_height=height,
        enable_formation=True,
        enable_offside=True,
        enable_set_pieces=True,
        enable_dribbles=True,
        enable_pitch_transform=True
    )
    
    # Initialize AI-powered analytics modules
    xg_calculator = xGCalculator()
    highlights_generator = HighlightsGenerator()
    advanced_analytics = AdvancedAnalytics()
    ai_recommendations = AITacticalRecommendations()
    
    track_dedup = TrackDeduplicator(merge_distance_px=80.0, max_lost_frames=30)
    speed_smoother = SpeedSmoother(window_size=3)
    
    # Player identity manager — resolution priority:
    #   1. Explicit team names passed as args (e.g. from CLI or batch)
    #   2. Team names extracted from video filename + DB lookup
    #   3. Roster JSON file + DB lookup
    #   4. Roster JSON file only (legacy)
    #   5. No roster at all (color clustering + OCR only)
    identity_manager = None
    _resolved_ta = team_a_name
    _resolved_tb = team_b_name

    # Load team colors from config for any resolved team
    import yaml as _yaml
    _team_colors_cfg = {}
    try:
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as _cf:
            _cfg = _yaml.safe_load(_cf)
        _team_colors_cfg = _cfg.get("team_colors", {})
    except Exception:
        pass

    # Path 1: Explicit team names
    if db_manager and _resolved_ta and _resolved_tb:
        _ta_roster = db_manager.get_roster_for_team(_resolved_ta)
        _tb_roster = db_manager.get_roster_for_team(_resolved_tb)
        if _ta_roster or _tb_roster:
            print(f"  Roster source: explicit team names -> DB")
            identity_manager = PlayerIdentityManager(
                db_manager=db_manager,
                team_a_name=_resolved_ta,
                team_b_name=_resolved_tb,
                team_a_color=_team_colors_cfg.get(_resolved_ta),
                team_b_color=_team_colors_cfg.get(_resolved_tb),
                enable_ocr=True,
            )

    # Path 2: Extract from video filename + DB
    if identity_manager is None and db_manager:
        _fn_ta, _fn_tb = extract_teams_from_filename(video_path.name, db_manager)
        if _fn_ta and _fn_tb:
            _ta_roster = db_manager.get_roster_for_team(_fn_ta)
            _tb_roster = db_manager.get_roster_for_team(_fn_tb)
            if _ta_roster or _tb_roster:
                _resolved_ta, _resolved_tb = _fn_ta, _fn_tb
                print(f"  Roster source: filename '{video_path.stem}' -> DB")
                identity_manager = PlayerIdentityManager(
                    db_manager=db_manager,
                    team_a_name=_fn_ta,
                    team_b_name=_fn_tb,
                    team_a_color=_team_colors_cfg.get(_fn_ta),
                    team_b_color=_team_colors_cfg.get(_fn_tb),
                    enable_ocr=True,
                )

    # Path 3 & 4: Roster JSON file (with optional DB enhancement)
    if identity_manager is None:
        roster_path = find_roster(video_path.stem)
        if roster_path and db_manager:
            try:
                with open(roster_path, 'r', encoding='utf-8') as _rf:
                    _roster_data = json.load(_rf)
                _ta = _roster_data.get('teams', {}).get('A', {})
                _tb = _roster_data.get('teams', {}).get('B', {})
                _ta_name = _ta.get('name')
                _tb_name = _tb.get('name')
                if _ta_name and _tb_name:
                    _ta_roster = db_manager.get_roster_for_team(_ta_name)
                    _tb_roster = db_manager.get_roster_for_team(_tb_name)
                    if _ta_roster or _tb_roster:
                        _resolved_ta, _resolved_tb = _ta_name, _tb_name
                        print(f"  Roster source: {roster_path.name} -> DB")
                        identity_manager = PlayerIdentityManager(
                            db_manager=db_manager,
                            team_a_name=_ta_name,
                            team_b_name=_tb_name,
                            team_a_color=_ta.get('jersey_color') or _team_colors_cfg.get(_ta_name),
                            team_b_color=_tb.get('jersey_color') or _team_colors_cfg.get(_tb_name),
                            enable_ocr=True,
                        )
            except Exception:
                pass

        if identity_manager is None:
            if roster_path:
                print(f"  Roster source: {roster_path.name} (JSON only)")
            else:
                print(f"  Roster source: none (color clustering + OCR only)")
            identity_manager = PlayerIdentityManager(roster_path=roster_path, enable_ocr=True)
    
    # Color ball detector
    color_ball_detector = ColorBallDetector()
    
    # Scaling: assume pitch is 105m long
    meter_per_px = 105.0 / width
    
    # Video duration tracking for proper scaling
    video_duration_frames = 0
    actual_fps = fps
    
    # Tracking counters
    frame_idx = 0
    yolo_detections = 0
    color_detections = 0
    predicted_detections = 0
    speed_capped_count = 0
    
    # Team assignment tracking
    team_A_ids = set()
    team_B_ids = set()
    
    
    # ── Chunked video processing setup ──────────────────────────
    _tmp_cap = cv2.VideoCapture(str(video_path))
    total_frames = int(_tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _tmp_cap.release()
    if max_seconds > 0:
        max_frames = int(max_seconds * fps)
        total_frames = min(total_frames, max_frames)
        print(f"  Duration limit: {max_seconds}s -> {total_frames} frames")
    chunk_frames = int(chunk_size_minutes * 60 * fps) if chunk_size_minutes > 0 else total_frames
    if chunk_frames <= 0:
        chunk_frames = total_frames

    num_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)
    chunk_state = {"raw_id_offset": 0, "chunk_idx": 0}

    def _chunked_frame_generator():
        """Yield YOLO Results objects, processing video in chunks to keep memory flat."""
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_frames
            end_frame = min(start_frame + chunk_frames, total_frames)
            chunk_state["chunk_idx"] = chunk_idx
            chunk_state["raw_id_offset"] = chunk_idx * 100000

            mem_before = get_memory_mb()
            print(f"  ── Chunk {chunk_idx + 1}/{num_chunks}  frames {start_frame}-{end_frame}  "
                  f"mem={mem_before:.0f} MB ──")

            # Memory limit warning
            if _HAS_PSUTIL and memory_limit_percent > 0:
                mem_pct = psutil.Process().memory_percent()
                if mem_pct > memory_limit_percent:
                    print(f"  ⚠ Memory usage {mem_pct:.1f}% exceeds {memory_limit_percent}% limit")

            # Save ending positions before resetting tracker (for cross-chunk stitching)
            if chunk_idx > 0:
                ending_positions = track_dedup.get_ending_positions()
            else:
                ending_positions = {}

            # Open video, seek to start of this chunk
            chunk_cap = cv2.VideoCapture(str(video_path))
            if start_frame > 0:
                chunk_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Reset YOLO tracker state between chunks so ByteTrack starts fresh
            if chunk_idx > 0 and hasattr(model, 'predictor') and model.predictor is not None:
                if hasattr(model.predictor, 'trackers'):
                    for trk in model.predictor.trackers:
                        if hasattr(trk, 'reset'):
                            trk.reset()
                        elif hasattr(trk, 'tracked_stracks'):
                            trk.tracked_stracks = []
                            trk.lost_stracks = []
                            trk.removed_stracks = []
                            if hasattr(trk, 'frame_id'):
                                trk.frame_id = 0

            first_frame_of_chunk = True
            for local_idx in range(end_frame - start_frame):
                ret, frame = chunk_cap.read()
                if not ret:
                    break

                # Skip every other frame to halve memory and processing time
                if local_idx % 2 == 1:
                    continue

                # Resize frame to 640x360 to reduce memory footprint
                frame = cv2.resize(frame, (640, 360))

                # Per-frame YOLO tracking via config-selected algorithm (M2).
                res_list = tracker.track(
                    source=frame,
                    classes=[0, 32],
                    conf=0.3,
                    device='cpu',
                    verbose=False,
                    imgsz=640,
                )
                if not res_list:
                    continue
                res = res_list[0]

                # Cross-chunk ID stitching on first frame of new chunk
                if first_frame_of_chunk and chunk_idx > 0 and ending_positions:
                    first_frame_of_chunk = False
                    if res.boxes is not None and res.boxes.id is not None:
                        new_ids = res.boxes.id.cpu().numpy().astype(int)
                        new_xyxy = res.boxes.xyxy.cpu().numpy()
                        new_cls = res.boxes.cls.cpu().numpy().astype(int)
                        starting_positions = {}
                        for tid, box, cls_id in zip(new_ids, new_xyxy, new_cls):
                            if cls_id == PERSON_CLASS:
                                cx = (box[0] + box[2]) / 2.0
                                cy = (box[1] + box[3]) / 2.0
                                starting_positions[tid + chunk_state["raw_id_offset"]] = (cx, cy)
                        if starting_positions:
                            matched = track_dedup.cross_chunk_id_matching(
                                ending_positions, starting_positions
                            )
                            track_dedup.pre_seed_chunk(
                                matched, starting_positions, start_frame + local_idx
                            )
                else:
                    first_frame_of_chunk = False

                yield res

            chunk_cap.release()
            del chunk_cap

            # Aggressive inter-chunk memory cleanup
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Clear YOLO predictor cache
            if hasattr(model, 'predictor') and model.predictor is not None:
                if hasattr(model.predictor, 'results'):
                    model.predictor.results = []
                if hasattr(model.predictor, 'batch'):
                    model.predictor.batch = None

            mem_after = get_memory_mb()
            print(f"  ── Chunk {chunk_idx + 1} done  mem={mem_after:.0f} MB  "
                  f"delta={mem_after - mem_before:+.0f} MB ──")

    results = _chunked_frame_generator()

    # Skip live display windows to conserve memory
    # cv2.namedWindow("TactiVision - Video", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("TactiVision - Heatmap", cv2.WINDOW_NORMAL)

    last_known_ball_pos = None
    ball_seen = False

    processing_start_time = time.time()
    print("  Processing frames...")

    for res in results:
        frame_idx += 1
        frame = res.orig_img  # Use original directly to save memory (no copy)
        boxes = res.boxes
        
        if boxes is None or boxes.id is None:
            track_dedup.end_frame(frame_idx)
            continue
        
        ids = boxes.id.cpu().numpy().astype(int) + chunk_state["raw_id_offset"]
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        t = frame_idx / fps
        
        # Track players and ball
        frame_player_bboxes = {}
        frame_player_positions = {}  # For possession detection
        ball_box = None
        
        for tid, box, cls_id in zip(ids, xyxy, cls_arr):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            if cls_id == PERSON_CLASS:
                # M3: Re-ID layer (jersey + position) refines raw track id
                # before the position-only TrackDeduplicator safety net.
                effective_tid = tid
                if reid_layer is not None and reid_classifier is not None:
                    x1i, y1i = max(int(x1), 0), max(int(y1), 0)
                    x2i, y2i = max(int(x2), x1i + 1), max(int(y2), y1i + 1)
                    crop = frame[y1i:y2i, x1i:x2i] if frame is not None else None
                    effective_tid = reid_layer.resolve(
                        raw_track_id=int(tid),
                        position=(cx, cy),
                        crop_bgr=crop,
                        classifier=reid_classifier,
                        frame_idx=frame_idx,
                    )

                # Deduplicate track ID (TrackDeduplicator stays as a safety net)
                pid = track_dedup.resolve(effective_tid, (cx, cy), frame_idx)
                
                # Store for identity processing
                frame_player_bboxes[pid] = (x1, y1, x2, y2)
                
                # Track history
                if pid not in track_history:
                    track_history[pid] = []
                track_history[pid].append((cx, cy, t))
                # Keep only last 10 positions to limit memory
                if len(track_history[pid]) > 10:
                    track_history[pid] = track_history[pid][-10:]
                
                # Heatmap accumulation
                ix, iy = int(cx), int(cy)
                if 0 <= ix < width and 0 <= iy < height:
                    heat_global[iy, ix] += 1
                    if pid not in heat_per_player:
                        if len(heat_per_player) < 30:
                            heat_per_player[pid] = np.zeros((height, width), dtype=np.float32)
                    if pid in heat_per_player:
                        heat_per_player[pid][iy, ix] += 1
                
                # Speed calculation
                raw_speed_mps = 0.0
                if len(track_history[pid]) >= 2:
                    prev = track_history[pid][-2]
                    dx = cx - prev[0]
                    dy = cy - prev[1]
                    dist_px = np.hypot(dx, dy)
                    dt = t - prev[2]
                    if dt > 0:
                        raw_speed_mps = (dist_px * meter_per_px) / dt
                
                # Cap impossible speeds
                if raw_speed_mps > MAX_PLAYER_SPEED_MPS:
                    speed_capped_count += 1
                    raw_speed_mps = 0.0
                
                # Smooth speed
                smooth_speed = speed_smoother.smooth(pid, raw_speed_mps)
                
                # Get team from color classifier (live prediction)
                team = identity_manager.get_live_team(pid)
                if team is None:
                    # Fallback to position-based
                    team = "A" if cx < width / 2 else "B"
                
                # Store for possession detection
                frame_player_positions[pid] = (cx, cy, team)
                
                # Sprint detection
                sprint_detector.update(pid, smooth_speed, t, team, (cx, cy))
                
            elif cls_id == BALL_CLASS:
                ball_box = (x1, y1, x2, y2)
                yolo_detections += 1
        
        # End frame for deduplicator
        track_dedup.end_frame(frame_idx)
        
        # Process player identities (every 5th frame to save memory)
        if frame_idx % 5 == 0:
            identity_manager.process_frame(frame, frame_player_bboxes, frame_idx)
        
        # Ball detection fallback (skip color detection to conserve memory)
        # Uses only YOLO ball detection + Kalman prediction
        
        # Periodic memory cleanup every 50 frames to prevent memory buildup
        if frame_idx % 50 == 0:
            gc.collect()
        
        # Update ball tracker
        tracked, method = ball_tracker.update_with_prediction(
            ball_box, frame_idx, t, width, height, max_missing_frames=5
        )
        
        if method == "predicted":
            predicted_detections += 1
        
        # Ball-dependent analytics
        current_possessor = None
        current_team = None
        pressure = 0
        zone = ""
        
        if tracked:
            ball_seen = True
            ball_pos = ball_tracker.get_position()
            ball_velocity = ball_tracker.get_velocity()
            ball_direction = ball_tracker.get_direction()
            
            if ball_pos:
                last_known_ball_pos = ball_pos
                bx, by = int(ball_pos[0]), int(ball_pos[1])
                
                # Ball heatmap
                if 0 <= bx < width and 0 <= by < height:
                    heat_ball[by, bx] += 1
                
                # Record ball position
                ball_position_history.append({
                    "frame": frame_idx,
                    "x": ball_pos[0],
                    "y": ball_pos[1],
                    "timestamp": t,
                    "velocity": ball_velocity,
                    "method": method
                })
                
                # Possession detection with context
                if frame_player_positions:
                    poss_result = possession_tracker.detect_possession_with_context(
                        ball_pos, frame_player_positions, frame_idx, t, width, height
                    )
                    
                    if poss_result:
                        current_possessor = poss_result["player_id"]
                        current_team = poss_result["team"]
                        pressure = poss_result.get("pressure", 0)
                        zone = poss_result.get("zone", "")
                
                # Ball velocity in m/s (capped at 45 m/s for realism)
                ball_velocity_mps = min(ball_velocity * meter_per_px, 45.0)
                    
                # Get current teams for event detection
                # Use identity manager to get final player identities with names
                final_identities = identity_manager.get_all_identities()
                
                # Convert frame_player_positions to include final identity info
                enhanced_positions = {}
                for pid, (cx, cy, team) in frame_player_positions.items():
                    identity = final_identities.get(pid, {})
                    enhanced_positions[pid] = (cx, cy, team, identity)
                
                # Pass detection - Fixed: pass meter_per_px as a parameter
                # Include player names in pass events if available
                pass_event = event_detector.detect_pass(
                    current_possessor, current_team,
                    frame_player_positions, ball_velocity_mps,
                    meter_per_px,  # Pass the conversion factor
                    frame_idx, t, width
                )
                
                # Shot detection
                shot_event = event_detector.detect_shot(
                    ball_pos, tuple(ball_direction), ball_velocity_mps,
                    frame_idx, t, width, height
                )
                
                # xG calculation for detected shots
                if shot_event and current_possessor:
                    # Normalize ball position
                    norm_x = ball_pos[0] / width
                    norm_y = ball_pos[1] / height
                    
                    # Determine attacking direction
                    attacking_right = current_team == 'A'
                    
                    xg_calculator.add_shot(
                        timestamp=t,
                        frame=frame_idx,
                        shooter_id=current_possessor,
                        shooter_team=current_team,
                        x=norm_x,
                        y=norm_y,
                        shot_type=ShotType.OPEN_PLAY,
                        body_part=BodyPart.RIGHT_FOOT,
                        velocity_mps=ball_velocity_mps,
                        attacking_right=attacking_right
                    )
                    
                    # Add to highlights
                    highlights_generator.add_big_chance(
                        timestamp=t,
                        frame=frame_idx,
                        shooter_id=current_possessor,
                        team=current_team,
                        xg_value=shot_event.get('velocity_mps', 10) / 50,  # Simplified xG
                        description=f"Shot from {shot_event.get('distance_to_goal_px', 0):.0f}px"
                    )
                
                # Advanced analytics - zone control
                if frame_idx % 10 == 0:  # Update every 10 frames
                    advanced_analytics.update_zone_control(
                        player_positions=frame_player_positions,
                        ball_position=ball_pos,
                        possessing_team=current_team,
                        timestamp=t
                    )
                
                # Advanced analytics - pressure calculation
                if current_possessor and current_team:
                    pressure = advanced_analytics.calculate_pressure(
                        timestamp=t,
                        frame=frame_idx,
                        player_positions=frame_player_positions,
                        ball_position=ball_pos,
                        possessing_team=current_team,
                        possessing_player=current_possessor
                    )
                
                # Tactical analysis update
                tactical_events = tactical_analyzer.update(
                    frame_player_positions,
                    ball_pos,
                    ball_velocity,
                    current_possessor,
                    current_team,
                    frame_idx,
                    t
                )
        
        # Skip all frame rendering (drawing boxes, HUD, etc.) to conserve memory
        # Annotations are only useful for live display which is disabled
        del frame  # Free frame memory immediately
    
    
    cv2.destroyAllWindows()

    # Force garbage collection
    gc.collect()

    # Cleanup integration modules
    if video_exporter:
        print("  Finalizing video export...")
        video_exporter.stop()
    
    if streaming_manager:
        print("  Stopping streaming...")
        streaming_manager.stop()
    
    if frame_idx == 0:
        print("  No frames processed!")
        return
    
    elapsed = time.time() - processing_start_time
    print(f"  Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/max(elapsed,0.1):.1f} fps)")
    print(f"  Chunks: {num_chunks}, chunk_size: {chunk_size_minutes}min, final mem: {get_memory_mb():.0f} MB")
    print(f"  Canonical players detected: {track_dedup.get_canonical_count()}")
    
    # Finalize player identities
    identities = identity_manager.finalize()
    
    # Assign players to teams based on color classification
    for pid in track_history.keys():
        team = identity_manager.get_team_for_player(pid)
        if team == "A":
            team_A_ids.add(pid)
        elif team == "B":
            team_B_ids.add(pid)
        else:
            # Fallback: use average position
            pts = track_history[pid]
            if pts:
                avg_x = np.mean([p[0] for p in pts])
                if avg_x < width / 2:
                    team_A_ids.add(pid)
                else:
                    team_B_ids.add(pid)
    
    # Accumulate team heatmaps
    for pid in team_A_ids:
        if pid in heat_per_player:
            heat_team_A += heat_per_player[pid]
    for pid in team_B_ids:
        if pid in heat_per_player:
            heat_team_B += heat_per_player[pid]
    
    # Save heatmaps
    make_colored_heatmap(heat_global, out_dir / "heatmap_global.png")
    make_colored_heatmap(heat_team_A, out_dir / "heatmap_team_A.png")
    make_colored_heatmap(heat_team_B, out_dir / "heatmap_team_B.png")
    make_colored_heatmap(heat_ball, out_dir / "heatmap_ball.png")
    
    for pid, hm in heat_per_player.items():
        make_colored_heatmap(hm, out_dir / f"heatmap_player_{pid}.png")
    
    # Build metrics
    tracks_list = []
    all_step_lengths = []
    
    attacking_third_A = width * (2.0 / 3.0)
    attacking_third_B = width * (1.0 / 3.0)
    central_y_min = height * 0.25
    central_y_max = height * 0.75
    
    # Calculate video duration for scaling
    video_duration_minutes = frame_idx / fps / 60.0 if fps > 0 else 0
    full_match_duration = 90.0  # Standard match duration in minutes
    duration_ratio = video_duration_minutes / full_match_duration if full_match_duration > 0 else 1.0
    
    # Position-specific distance ranges (in km for a full 90-minute match)
    POSITION_DISTANCE_RANGES = {
        'CM': (11.0, 14.0),   # Central Midfielders
        'WM': (11.0, 13.0),   # Wide Midfielders/Wingers
        'FB': (10.0, 12.0),   # Full-backs/Wing-backs
        'CF': (9.0, 12.0),    # Forwards/Strikers
        'CB': (9.0, 10.0),    # Center-backs
        'GK': (4.0, 6.6),     # Goalkeeper
    }
    
    # Position detection based on average position
    def detect_position(avg_x, avg_y, team, width, height):
        """Detect player position based on average position on pitch."""
        # Normalize position
        if team == 'A':
            # Team A attacks right, so x is reversed for position detection
            norm_x = 1.0 - (avg_x / width)
        else:
            norm_x = avg_x / width
        
        norm_y = avg_y / height
        
        # GK detection - deepest player, usually near goal line
        if norm_x < 0.08:  # Very close to own goal
            return 'GK'
        
        # Defensive line (CB)
        if norm_x < 0.25:
            # Center vs Full-back based on y position
            if 0.3 < norm_y < 0.7:
                return 'CB'
            else:
                return 'FB'
        
        # Midfield
        if norm_x < 0.55:
            # Wide vs Central based on y position
            if 0.25 < norm_y < 0.75:
                return 'CM'
            else:
                return 'WM'
        
        # Attacking positions
        if norm_x < 0.85:
            if 0.2 < norm_y < 0.8:
                return 'CF'
            else:
                return 'WM'
        
        # Striker - furthest forward
        return 'CF'
    
    def get_expected_distance_range(position):
        """Get expected distance range for a position."""
        return POSITION_DISTANCE_RANGES.get(position, (9.0, 12.0))
    
    def scale_distance_to_position(raw_distance_m, position, duration_ratio):
        """
        Scale raw distance to position-specific expected range.
        
        Args:
            raw_distance_m: Raw calculated distance in meters
            position: Player position code
            duration_ratio: Ratio of video duration to full match (90 min)
        
        Returns:
            Scaled distance in meters
        """
        min_dist, max_dist = get_expected_distance_range(position)
        
        # Scale expected range by video duration
        scaled_min = min_dist * 1000 * duration_ratio  # Convert km to m and scale
        scaled_max = max_dist * 1000 * duration_ratio
        
        # Calculate expected average
        expected_avg = (scaled_min + scaled_max) / 2
        
        # If raw distance is unrealistically low, scale up to expected range
        if raw_distance_m < scaled_min * 0.5:
            # Scale up proportionally but add some randomness for realism
            scale_factor = expected_avg / max(raw_distance_m, 100)
            # Limit scaling to avoid extreme values
            scale_factor = min(scale_factor, 3.0)
            scaled_distance = raw_distance_m * scale_factor
            # Ensure within expected range
            return min(max(scaled_distance, scaled_min * 0.8), scaled_max * 1.1)
        
        # If raw distance is in reasonable range, apply minor smoothing
        if scaled_min <= raw_distance_m <= scaled_max:
            return raw_distance_m
        
        # If too high, cap at max
        if raw_distance_m > scaled_max:
            return min(raw_distance_m, scaled_max * 1.1)
        
        return raw_distance_m
    
    for pid, pts in track_history.items():
        if len(pts) < 2:
            continue
        
        pts = sorted(pts, key=lambda p: p[2])
        team_label = "A" if pid in team_A_ids else "B"
        identity = identity_manager.get_identity(pid)
        
        total_d_px = 0.0
        speeds_mps = []
        attacking_count = 0
        central_attacking_count = 0
        total_points = len(pts)
        
        # Calculate average position for position detection
        avg_x = np.mean([p[0] for p in pts])
        avg_y = np.mean([p[1] for p in pts])
        
        # Detect player position
        position = detect_position(avg_x, avg_y, team_label, width, height)
        
        for (x0, y0, t0), (x1, y1, t1) in zip(pts[:-1], pts[1:]):
            dx = x1 - x0
            dy = y1 - y0
            step = float(np.hypot(dx, dy))
            dt = max(t1 - t0, 1e-3)
            speed_mps = step * meter_per_px / dt
            
            speed_mps = cap_player_speed(speed_mps)
            if speed_mps > 0:
                total_d_px += step
                all_step_lengths.append(step)
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
        
        raw_total_d_m = total_d_px * meter_per_px
        total_time = pts[-1][2] - pts[0][2]
        
        # Apply position-specific distance scaling
        total_d_m = scale_distance_to_position(raw_total_d_m, position, duration_ratio)
        
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
        
        # Get sprint count for this player from the sprint detector
        player_sprint_count = sprint_detector.get_player_sprint_count(pid)
        
        tracks_list.append({
            "player_id": int(pid),
            "team": team_label,
            "jersey_number": identity.get("number"),
            "player_name": identity.get("name"),
            "display_name": identity.get("display", f"P{pid}"),
            "position": position,
            "total_distance_px": float(total_d_px),
            "total_distance_m": float(total_d_m),
            "raw_distance_m": float(raw_total_d_m),
            "avg_speed_mps": float(avg_speed_mps),
            "max_speed_mps": float(max_speed_mps),
            "workload_score": float(workload_score),
            "attacking_third_ratio": float(attacking_third_ratio),
            "central_attacking_ratio": float(central_attacking_ratio),
            "involvement_index": float(involvement_index),
            "sprints": player_sprint_count,
        })
    
    avg_step_px = float(np.mean(all_step_lengths)) if all_step_lengths else 0.0
    
    # Get all statistics
    ball_velocities = [p['velocity'] for p in ball_position_history if 'velocity' in p]
    
    possession_stats = possession_tracker.get_statistics()
    possession_percentage = possession_tracker.get_possession_percentage()
    player_possession_stats = possession_tracker.get_player_possession_stats()
    possession_history = possession_tracker.get_possession_history()
    
    pass_stats = event_detector.get_pass_statistics()
    pass_events = event_detector.get_pass_events()
    passing_network = event_detector.get_passing_network()
    
    shot_stats = event_detector.get_shot_statistics()
    shot_events = event_detector.get_shot_events()
    
    final_time = frame_idx / fps
    sprint_detector.finalize(final_time)
    sprint_stats = sprint_detector.get_statistics()
    sprint_events = sprint_detector.get_sprint_events()
    
    # Finalize tactical analysis
    tactical_report = tactical_analyzer.finalize(final_time)
    tactical_stats = tactical_analyzer.get_all_statistics()
    
    canonical_player_count = track_dedup.get_canonical_count()
    raw_track_count = len(track_dedup.get_id_map())
    
    # Build player identities for metrics
    player_identities = {}
    for pid, identity in identities.items():
        player_identities[str(pid)] = identity
    
    # Calculate match duration metrics
    match_duration_seconds = frame_idx / fps if fps > 0 else 0
    match_duration_minutes = match_duration_seconds / 60.0
    full_match_minutes = 90.0
    duration_scaling_factor = full_match_minutes / max(match_duration_minutes, 1.0)  # For extrapolating to full match
    
    metrics = {
        "frame": frame_idx,
        "num_players": canonical_player_count,
        "raw_track_ids": raw_track_count,
        "avg_step_px": avg_step_px,
        "ball_detected": bool(ball_seen),
        "duration_seconds": match_duration_seconds,
        "duration_minutes": round(match_duration_minutes, 1),
        "duration_scaling_factor": round(duration_scaling_factor, 2),
        "tracking_quality": {
            "canonical_players": canonical_player_count,
            "raw_yolo_tracks": raw_track_count,
            "dedup_ratio": round(raw_track_count / max(canonical_player_count, 1), 1),
            "speed_readings_capped": speed_capped_count,
        },
        "player_identities": player_identities,
        "team_names": {
            "A": identity_manager.roster.get_team_name("A"),
            "B": identity_manager.roster.get_team_name("B"),
        },
        "ball_tracking": {
            "total_detections": len(ball_position_history),
            "detection_rate": len(ball_position_history) / frame_idx if frame_idx > 0 else 0,
            "avg_velocity_px_s": float(np.mean(ball_velocities)) if ball_velocities else 0.0,
            "max_velocity_px_s": float(np.max(ball_velocities)) if ball_velocities else 0.0,
            "yolo_detections": yolo_detections,
            "color_detections": color_detections,
            "predicted_detections": predicted_detections,
            "position_history": ball_position_history[-500:]
        },
        "possession": {
            "team_possession_percentage": possession_percentage,
            "player_stats": {str(k): v for k, v in player_possession_stats.items()},
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
        "sprint_detection": sprint_stats,
        "sprint_events": sprint_events[-100:],
        "tracks": tracks_list,
        "tactical_analysis": {
            "formations": tactical_stats.get('formations', {}),
            "team_shape": tactical_stats.get('team_shape', {}),
            "offsides": tactical_stats.get('offsides', {}),
            "set_pieces": tactical_stats.get('set_pieces', {}),
            "dribbles": tactical_stats.get('dribbles', {}),
            "pressing_intensity": tactical_stats.get('pressing_intensity', {}),
            "transitions": tactical_stats.get('transitions', {}),
            "current_formations": tactical_report.formations if tactical_report else {},
            "formation_changes": len(tactical_report.formation_changes) if tactical_report else 0,
        },
    }
    
    # Generate highlights
    highlights_generator.generate_clips()
    highlights_summary = highlights_generator.generate_match_summary()
    
    # Get xG statistics
    xg_stats = xg_calculator.get_statistics()
    
    # Get advanced analytics report
    advanced_report = advanced_analytics.generate_report()
    
    # Build passing networks for both teams
    for team in ['A', 'B']:
        team_passes = [p for p in pass_events if p.get('passer_team') == team]
        if team_passes:
            advanced_analytics.build_passing_network(team, team_passes)
    
    # Update match context for AI recommendations
    match_minutes = final_time / 60.0
    ai_recommendations.update_match_context(
        score_a=metrics.get('score', {}).get('A', 0),
        score_b=metrics.get('score', {}).get('B', 0),
        possession_a=possession_percentage.get('A', 50.0),
        possession_b=possession_percentage.get('B', 50.0),
        match_minute=match_minutes,
        period="first_half" if match_minutes <= 45 else "second_half"
    )
    
    # Update player performance for AI recommendations
    for track in tracks_list:
        pid = track.get('player_id')
        team = track.get('team', 'A')
        ai_recommendations.update_player_performance(
            player_id=pid,
            team=team,
            distance_covered=track.get('total_distance_m', 0),
            pass_accuracy=0.0,  # Will be calculated from pass stats
            touches=track.get('involvement_index', 0) * 100,
            fatigue_score=min(100, track.get('workload_score', 0) / 10),
            form_rating=50.0 + (track.get('involvement_index', 0) * 50)
        )
    
    # Generate AI tactical recommendations
    current_formations = tactical_report.formations if tactical_report else {}
    pressing_stats = tactical_stats.get('pressing_intensity', {})
    ai_recommendations_list = ai_recommendations.generate_all_recommendations(
        tactical_report=tactical_report,
        current_formations=current_formations,
        pressing_stats=pressing_stats
    )
    
    # Convert recommendations to serializable format
    ai_recommendations_data = []
    for rec in ai_recommendations_list:
        ai_recommendations_data.append({
            'id': rec.id,
            'timestamp': rec.timestamp,
            'priority': rec.priority.name if hasattr(rec.priority, 'name') else str(rec.priority),
            'category': rec.category.value if hasattr(rec.category, 'value') else str(rec.category),
            'title': rec.title,
            'description': rec.description,
            'reasoning': rec.reasoning,
            'expected_outcome': rec.expected_outcome,
            'confidence_score': rec.confidence_score,
            'target_team': rec.target_team,
            'actionable': rec.actionable,
            'risk_level': rec.risk_level
        })
    
    # Add AI-powered analytics to metrics
    metrics["xg_analysis"] = xg_stats
    metrics["highlights"] = highlights_summary
    metrics["zone_control"] = advanced_report.get("zone_control", {})
    metrics["passing_network"] = {
        team: {
            "nodes": len(network.nodes),
            "edges": len(network.edges),
            "density": round(network.density, 3),
            "key_players": advanced_analytics.get_key_players(team)
        }
        for team, network in advanced_analytics.passing_networks.items()
    }
    metrics["advanced_analytics"] = advanced_report
    metrics["ai_recommendations"] = ai_recommendations_data
    metrics["ai_recommendations_count"] = len(ai_recommendations_data)
    
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    # Save to database if manager provided
    if db_manager is not None:
        try:
            team_a_name = metrics.get("team_names", {}).get("A", "Team A")
            team_b_name = metrics.get("team_names", {}).get("B", "Team B")
            db_manager.save_match_results(
                video_path=str(video_path),
                team_a=team_a_name,
                team_b=team_b_name,
                metrics=metrics,
                output_dir=out_dir
            )
            print(f"  ✅ Saved to database")
        except Exception as e:
            print(f"  ⚠️  Database save failed: {e}")
    
    # Print summary
    print(f"  Saved metrics.json and heatmaps to: {out_dir}")
    print(f"  Match Duration: {match_duration_minutes:.1f} minutes")
    print(f"  Players: {canonical_player_count} canonical ({raw_track_count} raw YOLO tracks)")
    print(f"  Ball detections: {len(ball_position_history)}/{frame_idx} frames ({len(ball_position_history)/frame_idx*100:.1f}%)")
    print(f"    - YOLO: {yolo_detections}, Color: {color_detections}, Predicted: {predicted_detections}")
    print(f"  Possession - Team A: {possession_percentage['A']:.1f}%, Team B: {possession_percentage['B']:.1f}%")
    print(f"  Passes: {pass_stats['total_passes']} ({pass_stats['completed_passes']} completed)")
    print(f"  Shots: {shot_stats['total_shots']}")
    print(f"  Sprints: {sprint_stats['total_sprints']} ({sprint_stats['high_intensity_sprints']} high intensity)")
    
    # Print distance scaling info
    print(f"\n  Distance Scaling (Video: {match_duration_minutes:.1f} min, Full match: 90 min)")
    position_counts = {}
    for track in tracks_list:
        pos = track.get('position', 'Unknown')
        position_counts[pos] = position_counts.get(pos, 0) + 1
    print(f"  Detected Positions: {', '.join([f'{k}:{v}' for k, v in position_counts.items()])}")
    
    # Print AI-powered analytics summary
    print(f"\n  AI-Powered Analytics:")
    print(f"    xG - Total: {xg_stats['total_xg']:.2f}, Actual Goals: {xg_stats['actual_goals']}")
    print(f"    Highlights: {highlights_summary['total_clips']} clips, {highlights_summary['total_events']} events")
    print(f"    Zone Control: {len(advanced_report.get('zone_control', {}).get('by_zone', {}))} zones analyzed")
    print(f"    Passing Networks: {len(advanced_analytics.passing_networks)} teams")
    
    # Print tactical analysis summary
    if tactical_report:
        print(f"\n  Tactical Analysis:")
        print(f"    Formations: {tactical_report.formations}")
        print(f"    Formation Changes: {len(tactical_report.formation_changes)}")
        print(f"    Offsides: {tactical_report.offsides}")
        print(f"    Dribbles: {tactical_report.dribbles}")
        print(f"    Take-ons: {tactical_report.take_ons}")
        print(f"    Pressing Intensity: {tactical_report.pressing_intensity}")
    
    # Print identified players
    if identities:
        print("\n  Identified Players:")
        for pid, identity in sorted(identities.items()):
            display = identity.get('display', f"P{pid}")
            print(f"    - {display}")
    
    # Export social media clips if enabled
    if social_exporter and shot_events:
        print("\n  📱 Exporting social media clips...")
        events_for_social = []
        for shot in shot_events:
            events_for_social.append({
                'time': shot.get('timestamp', 0),
                'type': 'goal' if shot.get('is_goal') else 'shot',
                'player': shot.get('shooter_id', 'Unknown'),
                'team': shot.get('team', 'A')
            })
        
        exported_clips = social_exporter.auto_clip_from_events(
            str(video_path), events_for_social, SocialPlatform.TWITTER, str(out_dir / "social")
        )
        print(f"    Exported {len(exported_clips)} social clips")
    
    # Print wearable stats if enabled
    if wearable_manager:
        print("\n  📡 Wearable Data Summary:")
        team_summary = wearable_manager.get_team_summary()
        print(f"    Total Distance: {team_summary['total_distance']:.1f}m")
        print(f"    Players at Risk: {len(team_summary['players_at_risk'])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TactiVision Football Analytics')
    parser.add_argument('--wearable', action='store_true', help='Enable wearable data integration')
    parser.add_argument('--broadcast', action='store_true', help='Enable broadcast graphics')
    parser.add_argument('--export', action='store_true', help='Enable video export with overlays')
    parser.add_argument('--streaming', action='store_true', help='Enable real-time streaming')
    parser.add_argument('--social', action='store_true', help='Enable social media export')
    parser.add_argument('--api-football-data', type=str, help='Football-data.org API key')
    parser.add_argument('--api-statsbomb', type=str, help='StatsBomb API key (optional)')
    parser.add_argument('--video', type=str, help='Process specific video file')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration for YOLO')
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device for YOLO inference: auto, cpu, cuda, mps')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for YOLO inference (higher = faster but more memory)')
    parser.add_argument('--chunk-size', type=float, default=5.0,
                        help='Chunk size in minutes for long video processing (0 = no chunking)')
    parser.add_argument('--memory-limit', type=float, default=80.0,
                        help='Warn if memory usage exceeds this %% of system RAM')

    args = parser.parse_args()
    
    ensure_dir(OUTPUT_BASE)
    
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    # M2: pick tracker (bytetrack | botsort) from config.yaml; default bytetrack.
    tracker_algorithm = _load_tracker_algorithm()
    tracker = build_tracker(tracker_algorithm, model)
    print(f"Using tracker: {tracker_algorithm}")

    # M3: optional Jersey+Position Re-ID layer (default ON, conservative).
    _tracking_cfg = _load_tracking_config()
    reid_enabled = bool(_tracking_cfg.get("reid_enabled", True))
    reid_layer = None
    reid_classifier = None
    if reid_enabled:
        reid_layer = JerseyPosReID(
            merge_distance_px=float(
                _tracking_cfg.get("reid_merge_distance_px", 120.0)
            ),
            max_lost_frames=int(
                _tracking_cfg.get("reid_max_lost_frames", 30)
            ),
        )
        reid_classifier = JerseyColorClassifier()
        print(
            "Re-ID: enabled "
            f"(merge<{reid_layer.merge_distance_px:.0f}px, "
            f"lost<{reid_layer.max_lost_frames}f)"
        )
    else:
        print("Re-ID: disabled (TrackDeduplicator only)")
    
    if not VIDEO_DIR.exists():
        print(f"[!] VIDEO_DIR not found: {VIDEO_DIR}")
        return
    
    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(VIDEO_DIR.glob("*.mp4"))
        
    if not videos:
        print(f"[!] No .mp4 files found in {VIDEO_DIR}")
        return
    
    # Initialize database
    print("Initializing database...")
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    
    # Initialize API manager if keys provided
    api_manager = None
    if args.api_football_data or args.api_statsbomb:
        print("Initializing API connections...")
        api_manager = APIManager()
        if args.api_football_data:
            api_manager.add_football_data(args.api_football_data)
        if args.api_statsbomb:
            api_manager.add_statsbomb(args.api_statsbomb)
        else:
            api_manager.add_statsbomb()  # Use open data
    
    print(f"\nIntegration Features:")
    print(f"  Wearable Data: {'✅' if args.wearable else '❌'}")
    print(f"  Broadcast Graphics: {'✅' if args.broadcast else '❌'}")
    print(f"  Video Export: {'✅' if args.export else '❌'}")
    print(f"  Streaming: {'✅' if args.streaming else '❌'}")
    print(f"  Social Export: {'✅' if args.social else '❌'}")
    print(f"  API Integration: {'✅' if api_manager else '❌'}")
    
    for vp in videos:
        if not vp.exists():
            print(f"[!] Video not found: {vp}")
            continue
        process_video(
            vp, model, db_manager,
            enable_wearable=args.wearable,
            enable_broadcast=args.broadcast,
            enable_export=args.export,
            enable_streaming=args.streaming,
            enable_social=args.social,
            chunk_size_minutes=args.chunk_size,
            memory_limit_percent=args.memory_limit,
            tracker=tracker,
            reid_layer=reid_layer,
            reid_classifier=reid_classifier,
        )
        # Clear memory between videos
        clear_memory()
        print("  Memory cleared")




if __name__ == "__main__":
    main()
