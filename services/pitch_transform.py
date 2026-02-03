"""
Pitch Transform / Bird's Eye View Module

Transforms video coordinates to tactical pitch view using homography.
Generates tactical visualizations and heatmaps on pitch template.

HOMOGRAPHY TRANSFORMATION:
- Maps video coordinates (pixels) to pitch coordinates (meters)
- Requires 4+ corresponding points between video and pitch
- Can use pitch lines (if visible) or manual calibration

PITCH DIMENSIONS (standard):
- Length: 105 meters
- Width: 68 meters
- Penalty area: 16.5m x 40.3m
- Goal area: 5.5m x 18.32m
- Center circle radius: 9.15m
- Penalty spot: 11m from goal line

FEATURES:
- Coordinate transformation (video to pitch)
- Tactical visualizations
- Heatmap generation on pitch template
- Support for automatic and manual calibration
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from dataclasses import dataclass
from collections import deque


@dataclass
class PitchDimensions:
    """Standard football pitch dimensions in meters."""
    length: float = 105.0  # Length (x-axis)
    width: float = 68.0    # Width (y-axis)
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.3
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    goal_width: float = 7.32


class PitchTransform:
    """
    Transforms video coordinates to tactical pitch coordinates.
    
    Uses homography to map between video pixel coordinates and
    real-world pitch coordinates (in meters).
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        output_width: int = 1050,  # Output image width (10px per meter)
        output_height: int = 680   # Output image height (10px per meter)
    ):
        """
        Initialize pitch transform.
        
        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            output_width: Output image width in pixels
            output_height: Output image height in pixels
        """
        self.pitch_dims = PitchDimensions(length=pitch_length, width=pitch_width)
        self.output_width = output_width
        self.output_height = output_height

        # Meters per pixel in output
        self.meters_per_px_x = pitch_length / output_width
        self.meters_per_px_y = pitch_width / output_height

        # Homography matrix (video to pitch)
        self._homography_matrix: Optional[np.ndarray] = None
        self._inverse_homography: Optional[np.ndarray] = None

        # Calibration points
        self._video_points: List[Tuple[float, float]] = []
        self._pitch_points: List[Tuple[float, float]] = []

        # Default pitch keypoints (in meters)
        self._default_pitch_points = [
            (0, 0),                    # Top-left corner
            (pitch_length / 2, 0),     # Top middle
            (pitch_length, 0),         # Top-right corner
            (0, pitch_width / 2),      # Left middle
            (pitch_length / 2, pitch_width / 2),  # Center
            (pitch_length, pitch_width / 2),       # Right middle
            (0, pitch_width),          # Bottom-left corner
            (pitch_length / 2, pitch_width),       # Bottom middle
            (pitch_length, pitch_width),           # Bottom-right corner
            (pitch_length * 0.11, pitch_width / 2),  # Left penalty spot
            (pitch_length * 0.89, pitch_width / 2),  # Right penalty spot
        ]

    def calibrate_from_points(
        self,
        video_points: List[Tuple[float, float]],
        pitch_points: List[Tuple[float, float]] = None
    ) -> bool:
        """
        Calibrate using corresponding video and pitch points.
        
        Args:
            video_points: List of (x, y) points in video coordinates
            pitch_points: List of (x, y) points in pitch coordinates (meters)
                         If None, uses default pitch points
            
        Returns:
            True if calibration successful
        """
        if len(video_points) < 4:
            return False

        self._video_points = video_points

        if pitch_points is None:
            # Use first N default points
            pitch_points = self._default_pitch_points[:len(video_points)]

        self._pitch_points = pitch_points

        # Convert to numpy arrays
        src_pts = np.array(video_points, dtype=np.float32)
        dst_pts = np.array(pitch_points, dtype=np.float32)

        # Calculate homography
        try:
            self._homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            self._inverse_homography = np.linalg.inv(self._homography_matrix)
            return True
        except Exception:
            return False

    def calibrate_from_pitch_lines(
        self,
        frame: np.ndarray,
        line_color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None
    ) -> bool:
        """
        Attempt automatic calibration from visible pitch lines.
        
        Args:
            frame: Video frame
            line_color_range: ((H_min, S_min, V_min), (H_max, S_max, V_max)) for line color
            
        Returns:
            True if calibration successful
        """
        # This is a simplified version - full implementation would use
        # line detection algorithms to find pitch lines

        # Default white line color range in HSV
        if line_color_range is None:
            line_color_range = ((0, 0, 200), (180, 30, 255))

        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create mask for white lines
            lower, upper = line_color_range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Look for large rectangular contours (pitch boundaries)
            # This is simplified - real implementation would be more sophisticated
            h, w = frame.shape[:2]

            # Use frame corners as approximation
            video_points = [
                (w * 0.1, h * 0.1),   # Top-left
                (w * 0.5, h * 0.05),  # Top center
                (w * 0.9, h * 0.1),   # Top-right
                (w * 0.05, h * 0.5),  # Left center
                (w * 0.5, h * 0.5),   # Center
                (w * 0.95, h * 0.5),  # Right center
                (w * 0.1, h * 0.9),   # Bottom-left
                (w * 0.5, h * 0.95),  # Bottom center
                (w * 0.9, h * 0.9),   # Bottom-right
            ]

            return self.calibrate_from_points(video_points)

        except Exception:
            return False

    def video_to_pitch(
        self,
        video_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transform video coordinates to pitch coordinates.
        
        Args:
            video_point: (x, y) in video pixels
            
        Returns:
            (x, y) in pitch meters or None if not calibrated
        """
        if self._homography_matrix is None:
            return None

        # Apply homography
        pt = np.array([[video_point[0], video_point[1], 1]], dtype=np.float32).T
        transformed = self._homography_matrix @ pt

        # Normalize
        x = transformed[0, 0] / transformed[2, 0]
        y = transformed[1, 0] / transformed[2, 0]

        return (x, y)

    def pitch_to_video(
        self,
        pitch_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transform pitch coordinates to video coordinates.
        
        Args:
            pitch_point: (x, y) in pitch meters
            
        Returns:
            (x, y) in video pixels or None if not calibrated
        """
        if self._inverse_homography is None:
            return None

        # Apply inverse homography
        pt = np.array([[pitch_point[0], pitch_point[1], 1]], dtype=np.float32).T
        transformed = self._inverse_homography @ pt

        # Normalize
        x = transformed[0, 0] / transformed[2, 0]
        y = transformed[1, 0] / transformed[2, 0]

        return (x, y)

    def transform_players(
        self,
        player_positions: Dict[int, Tuple[float, float, str]]
    ) -> Dict[int, Tuple[float, float, str]]:
        """
        Transform multiple player positions to pitch coordinates.
        
        Args:
            player_positions: {player_id: (x, y, team)}
            
        Returns:
            {player_id: (pitch_x, pitch_y, team)}
        """
        transformed = {}

        for player_id, (x, y, team) in player_positions.items():
            pitch_pos = self.video_to_pitch((x, y))
            if pitch_pos:
                transformed[player_id] = (pitch_pos[0], pitch_pos[1], team)

        return transformed

    def is_calibrated(self) -> bool:
        """Check if transform is calibrated."""
        return self._homography_matrix is not None

    def create_pitch_template(self, scale: int = 10) -> np.ndarray:
        """
        Create a blank pitch template image.
        
        Args:
            scale: Pixels per meter
            
        Returns:
            Blank pitch image (green field)
        """
        width = int(self.pitch_dims.length * scale)
        height = int(self.pitch_dims.width * scale)

        # Green field
        pitch = np.ones((height, width, 3), dtype=np.uint8) * np.array([34, 139, 34], dtype=np.uint8)

        # White lines
        line_color = (255, 255, 255)
        line_thickness = max(2, scale // 5)

        # Outer boundary
        cv2.rectangle(pitch, (0, 0), (width - 1, height - 1), line_color, line_thickness)

        # Center line
        center_x = width // 2
        cv2.line(pitch, (center_x, 0), (center_x, height - 1), line_color, line_thickness)

        # Center circle
        center_y = height // 2
        radius = int(self.pitch_dims.center_circle_radius * scale)
        cv2.circle(pitch, (center_x, center_y), radius, line_color, line_thickness)
        cv2.circle(pitch, (center_x, center_y), line_thickness, line_color, -1)

        # Left penalty area
        pa_w = int(self.pitch_dims.penalty_area_length * scale)
        pa_h = int(self.pitch_dims.penalty_area_width * scale)
        pa_y = (height - pa_h) // 2
        cv2.rectangle(pitch, (0, pa_y), (pa_w, pa_y + pa_h), line_color, line_thickness)

        # Right penalty area
        cv2.rectangle(pitch, (width - pa_w, pa_y), (width - 1, pa_y + pa_h), line_color, line_thickness)

        # Left goal area
        ga_w = int(self.pitch_dims.goal_area_length * scale)
        ga_h = int(self.pitch_dims.goal_area_width * scale)
        ga_y = (height - ga_h) // 2
        cv2.rectangle(pitch, (0, ga_y), (ga_w, ga_y + ga_h), line_color, line_thickness)

        # Right goal area
        cv2.rectangle(pitch, (width - ga_w, ga_y), (width - 1, ga_y + ga_h), line_color, line_thickness)

        # Penalty spots
        ps_dist = int(self.pitch_dims.penalty_spot_distance * scale)
        cv2.circle(pitch, (ps_dist, center_y), line_thickness * 2, line_color, -1)
        cv2.circle(pitch, (width - ps_dist, center_y), line_thickness * 2, line_color, -1)

        # Corner arcs (simplified as small circles)
        corner_radius = int(1 * scale)
        cv2.ellipse(pitch, (0, 0), (corner_radius, corner_radius), 0, 0, 90, line_color, line_thickness)
        cv2.ellipse(pitch, (width - 1, 0), (corner_radius, corner_radius), 0, 90, 180, line_color, line_thickness)
        cv2.ellipse(pitch, (0, height - 1), (corner_radius, corner_radius), 0, 270, 360, line_color, line_thickness)
        cv2.ellipse(pitch, (width - 1, height - 1), (corner_radius, corner_radius), 0, 180, 270, line_color, line_thickness)

        return pitch

    def create_heatmap_on_pitch(
        self,
        positions: List[Tuple[float, float]],
        sigma: float = 2.0,
        scale: int = 10
    ) -> np.ndarray:
        """
        Create a heatmap on the pitch template.
        
        Args:
            positions: List of (x, y) positions in pitch meters
            sigma: Gaussian blur sigma (in meters)
            scale: Pixels per meter
            
        Returns:
            Heatmap image
        """
        width = int(self.pitch_dims.length * scale)
        height = int(self.pitch_dims.width * scale)

        # Create blank heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Add points
        for x, y in positions:
            if 0 <= x <= self.pitch_dims.length and 0 <= y <= self.pitch_dims.width:
                px = int(x * scale)
                py = int(y * scale)
                if 0 <= px < width and 0 <= py < height:
                    heatmap[py, px] += 1

        # Apply Gaussian blur
        if sigma > 0:
            kernel_size = int(sigma * scale * 3) * 2 + 1
            heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma * scale)

        return heatmap

    def visualize_positions_on_pitch(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_position: Optional[Tuple[float, float]] = None,
        scale: int = 10,
        team_colors: Dict[str, Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Visualize player positions on pitch template.
        
        Args:
            player_positions: {player_id: (pitch_x, pitch_y, team)}
            ball_position: (pitch_x, pitch_y) or None
            scale: Pixels per meter
            team_colors: {team: (B, G, R)} color mapping
            
        Returns:
            Visualization image
        """
        # Create pitch template
        pitch = self.create_pitch_template(scale)

        # Default team colors
        if team_colors is None:
            team_colors = {
                'A': (0, 0, 255),    # Red
                'B': (255, 0, 0),    # Blue
            }

        # Draw players
        for player_id, (x, y, team) in player_positions.items():
            px = int(x * scale)
            py = int(y * scale)

            color = team_colors.get(team, (128, 128, 128))
            cv2.circle(pitch, (px, py), max(5, scale // 2), color, -1)
            cv2.circle(pitch, (px, py), max(5, scale // 2), (255, 255, 255), 1)

        # Draw ball
        if ball_position:
            bx = int(ball_position[0] * scale)
            by = int(ball_position[1] * scale)
            cv2.circle(pitch, (bx, by), max(4, scale // 3), (0, 255, 255), -1)
            cv2.circle(pitch, (bx, by), max(4, scale // 3), (0, 0, 0), 1)

        return pitch

    def visualize_formation_on_pitch(
        self,
        formation_lines: List[Any],
        team: str,
        scale: int = 10
    ) -> np.ndarray:
        """
        Visualize team formation on pitch template.
        
        Args:
            formation_lines: List of FormationLine objects
            team: Team label
            scale: Pixels per meter
            
        Returns:
            Visualization image
        """
        pitch = self.create_pitch_template(scale)

        # Color based on team
        color = (0, 0, 255) if team == 'A' else (255, 0, 0)

        # Draw formation lines
        for line in formation_lines:
            line_x = int(line.avg_x * scale)
            cv2.line(pitch, (line_x, 0), (line_x, pitch.shape[0]), color, 2)

            # Draw players in this line
            for player_id in line.players:
                # Note: Would need actual positions here
                pass

        return pitch

    def get_tactical_view_dimensions(self) -> Tuple[int, int]:
        """Get dimensions of tactical view in meters."""
        return (self.pitch_dims.length, self.pitch_dims.width)

    def reset(self) -> None:
        """Reset calibration."""
        self._homography_matrix = None
        self._inverse_homography = None
        self._video_points.clear()
        self._pitch_points.clear()


class TacticalHeatmapGenerator:
    """
    Generates tactical heatmaps on the bird's eye view.
    
    Accumulates position data over time and generates
    heatmaps showing player activity areas.
    """

    def __init__(
        self,
        pitch_transform: PitchTransform,
        temporal_window: int = 300,  # 10 seconds at 30fps
        scale: int = 10
    ):
        """
        Initialize heatmap generator.
        
        Args:
            pitch_transform: PitchTransform instance
            temporal_window: Number of frames to accumulate
            scale: Pixels per meter for output
        """
        self.transform = pitch_transform
        self.temporal_window = temporal_window
        self.scale = scale

        # Position history per player
        self._position_history: Dict[int, deque] = {}

        # Accumulated heatmaps
        self._team_heatmaps: Dict[str, np.ndarray] = {}
        self._player_heatmaps: Dict[int, np.ndarray] = {}

        # Initialize heatmaps
        self._init_heatmaps()

    def _init_heatmaps(self) -> None:
        """Initialize empty heatmaps."""
        dims = self.transform.get_tactical_view_dimensions()
        width = int(dims[0] * self.scale)
        height = int(dims[1] * self.scale)

        for team in ['A', 'B', 'REF']:
            self._team_heatmaps[team] = np.zeros((height, width), dtype=np.float32)

    def add_position(
        self,
        player_id: int,
        video_position: Tuple[float, float],
        team: str
    ) -> None:
        """
        Add a player position to the heatmap.
        
        Args:
            player_id: Player ID
            video_position: (x, y) in video coordinates
            team: Team label
        """
        # Transform to pitch coordinates
        pitch_pos = self.transform.video_to_pitch(video_position)
        if pitch_pos is None:
            return

        # Add to history
        if player_id not in self._position_history:
            self._position_history[player_id] = deque(maxlen=self.temporal_window)

        self._position_history[player_id].append(pitch_pos)

        # Update heatmaps
        x, y = pitch_pos
        px = int(x * self.scale)
        py = int(y * self.scale)

        if 0 <= px < self._team_heatmaps[team].shape[1] and 0 <= py < self._team_heatmaps[team].shape[0]:
            self._team_heatmaps[team][py, px] += 1

            if player_id not in self._player_heatmaps:
                height, width = self._team_heatmaps[team].shape
                self._player_heatmaps[player_id] = np.zeros((height, width), dtype=np.float32)

            self._player_heatmaps[player_id][py, px] += 1

    def get_team_heatmap(self, team: str, smoothed: bool = True) -> np.ndarray:
        """
        Get heatmap for a team.
        
        Args:
            team: Team label
            smoothed: Apply Gaussian smoothing
            
        Returns:
            Heatmap array
        """
        heatmap = self._team_heatmaps.get(team, np.array([]))

        if smoothed and heatmap.size > 0:
            kernel_size = 15
            heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 3)

        return heatmap

    def get_player_heatmap(self, player_id: int, smoothed: bool = True) -> np.ndarray:
        """
        Get heatmap for a player.
        
        Args:
            player_id: Player ID
            smoothed: Apply Gaussian smoothing
            
        Returns:
            Heatmap array
        """
        heatmap = self._player_heatmaps.get(player_id, np.array([]))

        if smoothed and heatmap.size > 0:
            kernel_size = 15
            heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 3)

        return heatmap

    def reset(self) -> None:
        """Reset all heatmaps."""
        self._position_history.clear()
        self._init_heatmaps()
        self._player_heatmaps.clear()
