"""
Set Piece Recognition Module

Detects and classifies set pieces in football matches:
- Corners (ball near corner + players in box)
- Free kicks (ball stationary + wall formation)
- Throw-ins (ball out of bounds + player position)
- Penalties (ball on penalty spot)

SET PIECE DETECTION LOGIC:

CORNER:
- Ball is near corner arc (within 30px of corner)
- Multiple players from both teams in penalty area
- Ball is stationary or moving slowly

FREE KICK:
- Ball is stationary for >1 second
- Wall of defending players formed (2+ players within 2m)
- Ball is in dangerous position (attacking third)

THROW-IN:
- Ball goes out of bounds (near touchline)
- Player positioned to take throw
- Ball re-enters from touchline

PENALTY:
- Ball is on penalty spot (within tolerance)
- Players positioned in/around box
- Minimal player movement (set piece setup)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math


class SetPieceType(Enum):
    """Types of set pieces."""
    CORNER = "corner"
    FREE_KICK = "free_kick"
    THROW_IN = "throw_in"
    PENALTY = "penalty"
    GOAL_KICK = "goal_kick"
    UNKNOWN = "unknown"


class SetPieceOutcome(Enum):
    """Possible outcomes of set pieces."""
    SHOT = "shot"
    PASS = "pass"
    CLEARED = "cleared"
    GOAL = "goal"
    POSSESSION_WON = "possession_won"
    POSSESSION_LOST = "possession_lost"
    IN_PLAY = "in_play"
    UNKNOWN = "unknown"


@dataclass
class SetPieceEvent:
    """Records a detected set piece."""
    set_piece_type: SetPieceType
    timestamp: float
    frame_idx: int
    team: str  # Team taking the set piece
    ball_position: Tuple[float, float]
    outcome: SetPieceOutcome = SetPieceOutcome.UNKNOWN
    outcome_timestamp: Optional[float] = None
    outcome_frame_idx: Optional[int] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SetPieceDetector:
    """
    Detects and classifies set pieces in football matches.
    
    Features:
    - Corner detection (ball near corner arc)
    - Free kick detection (stationary ball + wall)
    - Throw-in detection (ball out of bounds)
    - Penalty detection (ball on spot)
    - Outcome classification
    """

    # Pitch dimensions (in relative coordinates 0-1)
    PENALTY_AREA_WIDTH = 0.165  # 16.5m / 105m
    PENALTY_AREA_DEPTH = 0.403  # 40.3m / 68m
    PENALTY_SPOT_X = 0.11  # 11m from goal line
    CORNER_ARC_RADIUS_PX = 30.0
    TOUCHLINE_MARGIN_PX = 20.0

    def __init__(
        self,
        stationary_threshold_mps: float = 1.0,  # Ball velocity threshold for stationary
        stationary_frames: int = 10,  # Frames ball must be stationary
        corner_detection_radius_px: float = 50.0,
        penalty_spot_tolerance_px: float = 20.0,
        wall_detection_distance_px: float = 60.0,
        min_wall_players: int = 2,
    ):
        """
        Initialize set piece detector.
        
        Args:
            stationary_threshold_mps: Max velocity for ball to be considered stationary
            stationary_frames: Number of frames ball must be stationary
            corner_detection_radius_px: Radius around corner to detect corners
            penalty_spot_tolerance_px: Tolerance for penalty spot detection
            wall_detection_distance_px: Distance between players to form wall
            min_wall_players: Minimum players to form a wall
        """
        self.stationary_threshold = stationary_threshold_mps
        self.stationary_frames = stationary_frames
        self.corner_radius = corner_detection_radius_px
        self.penalty_tolerance = penalty_spot_tolerance_px
        self.wall_distance = wall_detection_distance_px
        self.min_wall_players = min_wall_players

        # State tracking
        self._ball_stationary_frames: int = 0
        self._last_ball_position: Optional[Tuple[float, float]] = None
        self._last_ball_velocity: float = 0.0
        self._ball_out_of_bounds: bool = False
        self._out_of_bounds_position: Optional[Tuple[float, float]] = None

        # Set piece detection state
        self._active_set_piece: Optional[SetPieceEvent] = None
        self._set_piece_history: List[SetPieceEvent] = []
        self._player_positions: Dict[int, Tuple[float, float, str]] = {}

        # Frame dimensions
        self._frame_width: int = 1920
        self._frame_height: int = 1080

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Set frame dimensions for coordinate calculations."""
        self._frame_width = width
        self._frame_height = height

    def update(
        self,
        ball_position: Optional[Tuple[float, float]],
        ball_velocity: float,
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float,
        current_possessor: Optional[int] = None,
        current_team: Optional[str] = None
    ) -> Optional[SetPieceEvent]:
        """
        Update set piece detection with current frame data.
        
        Args:
            ball_position: (x, y) ball position or None
            ball_velocity: Ball velocity in pixels/second
            player_positions: Dict of {player_id: (x, y, team)}
            frame_idx: Current frame index
            timestamp: Current timestamp
            current_possessor: Current ball possessor ID
            current_team: Current team in possession
            
        Returns:
            SetPieceEvent if set piece detected, None otherwise
        """
        self._player_positions = player_positions
        self._last_ball_velocity = ball_velocity

        # Update ball stationary tracking
        if ball_position:
            self._update_ball_state(ball_position, ball_velocity, frame_idx)

        # Check for active set piece outcome
        if self._active_set_piece:
            outcome = self._check_set_piece_outcome(
                ball_position, ball_velocity, frame_idx, timestamp
            )
            if outcome:
                self._active_set_piece.outcome = outcome
                self._active_set_piece.outcome_timestamp = timestamp
                self._active_set_piece.outcome_frame_idx = frame_idx
                self._set_piece_history.append(self._active_set_piece)
                completed_event = self._active_set_piece
                self._active_set_piece = None
                return completed_event

        # Detect new set pieces
        if ball_position:
            set_piece = self._detect_set_piece(
                ball_position, ball_velocity, player_positions,
                frame_idx, timestamp, current_team
            )

            if set_piece:
                self._active_set_piece = set_piece
                return set_piece

        return None

    def _update_ball_state(
        self,
        ball_position: Tuple[float, float],
        ball_velocity: float,
        frame_idx: int
    ) -> None:
        """Update ball state tracking."""
        self._last_ball_position = ball_position

        # Check if ball is stationary
        if ball_velocity < self.stationary_threshold:
            self._ball_stationary_frames += 1
        else:
            self._ball_stationary_frames = 0

        # Check if ball is out of bounds
        margin = self.TOUCHLINE_MARGIN_PX
        was_out = self._ball_out_of_bounds

        self._ball_out_of_bounds = (
            ball_position[0] < -margin or
            ball_position[0] > self._frame_width + margin or
            ball_position[1] < -margin or
            ball_position[1] > self._frame_height + margin
        )

        # Record where ball went out
        if self._ball_out_of_bounds and not was_out:
            self._out_of_bounds_position = ball_position

    def _detect_set_piece(
        self,
        ball_position: Tuple[float, float],
        ball_velocity: float,
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float,
        current_team: Optional[str]
    ) -> Optional[SetPieceEvent]:
        """
        Detect if current situation is a set piece.
        
        Returns:
            SetPieceEvent if detected, None otherwise
        """
        # Check for penalty
        penalty = self._detect_penalty(ball_position, player_positions, frame_idx, timestamp)
        if penalty:
            return penalty

        # Check for corner
        corner = self._detect_corner(ball_position, player_positions, frame_idx, timestamp, current_team)
        if corner:
            return corner

        # Check for free kick
        free_kick = self._detect_free_kick(ball_position, player_positions, frame_idx, timestamp, current_team)
        if free_kick:
            return free_kick

        # Check for throw-in
        throw_in = self._detect_throw_in(ball_position, player_positions, frame_idx, timestamp, current_team)
        if throw_in:
            return throw_in

        return None

    def _detect_penalty(
        self,
        ball_position: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float
    ) -> Optional[SetPieceEvent]:
        """Detect penalty kick situation."""
        # Penalty spot positions (relative to frame width)
        # Assuming standard pitch layout with goals on left and right
        left_penalty_spot_x = self._frame_width * self.PENALTY_SPOT_X
        right_penalty_spot_x = self._frame_width * (1 - self.PENALTY_SPOT_X)
        center_y = self._frame_height / 2

        # Check if ball is near either penalty spot
        near_left_spot = (
            abs(ball_position[0] - left_penalty_spot_x) < self.penalty_tolerance and
            abs(ball_position[1] - center_y) < self.penalty_tolerance
        )
        near_right_spot = (
            abs(ball_position[0] - right_penalty_spot_x) < self.penalty_tolerance and
            abs(ball_position[1] - center_y) < self.penalty_tolerance
        )

        if not (near_left_spot or near_right_spot):
            return None

        # Check for players in penalty area
        players_in_box = self._count_players_in_penalty_area(player_positions, near_left_spot)

        if players_in_box < 3:  # Need some players in box
            return None

        # Determine which team is taking the penalty
        team = 'A' if near_right_spot else 'B'  # Team A attacks right

        return SetPieceEvent(
            set_piece_type=SetPieceType.PENALTY,
            timestamp=timestamp,
            frame_idx=frame_idx,
            team=team,
            ball_position=ball_position,
            confidence=0.8,
            metadata={
                'penalty_spot': 'left' if near_left_spot else 'right',
                'players_in_box': players_in_box
            }
        )

    def _detect_corner(
        self,
        ball_position: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float,
        current_team: Optional[str]
    ) -> Optional[SetPieceEvent]:
        """Detect corner kick situation."""
        # Corner positions
        corners = [
            (0, 0, 'top_left'),
            (0, self._frame_height, 'bottom_left'),
            (self._frame_width, 0, 'top_right'),
            (self._frame_width, self._frame_height, 'bottom_right')
        ]

        for corner_x, corner_y, corner_name in corners:
            distance = math.sqrt(
                (ball_position[0] - corner_x) ** 2 +
                (ball_position[1] - corner_y) ** 2
            )

            if distance < self.corner_radius:
                # Check for players in penalty area
                is_left_corner = 'left' in corner_name
                players_in_box = self._count_players_in_penalty_area(
                    player_positions, is_left_corner
                )

                if players_in_box < 2:
                    return None

                # Determine team based on corner
                # Team A attacks right (takes corners from left)
                # Team B attacks left (takes corners from right)
                team = 'A' if is_left_corner else 'B'

                return SetPieceEvent(
                    set_piece_type=SetPieceType.CORNER,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    team=team,
                    ball_position=ball_position,
                    confidence=0.85,
                    metadata={
                        'corner': corner_name,
                        'players_in_box': players_in_box,
                        'distance_to_corner': distance
                    }
                )

        return None

    def _detect_free_kick(
        self,
        ball_position: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float,
        current_team: Optional[str]
    ) -> Optional[SetPieceEvent]:
        """Detect free kick situation."""
        # Ball must be stationary
        if self._ball_stationary_frames < self.stationary_frames:
            return None

        # Check for wall formation
        wall_detected, wall_team = self._detect_wall(player_positions, ball_position)

        if not wall_detected:
            return None

        # Determine attacking team (opposite of wall team)
        attacking_team = 'B' if wall_team == 'A' else 'A'

        # Check if in dangerous position (attacking third)
        if attacking_team == 'A':
            in_dangerous_area = ball_position[0] > self._frame_width * 0.6
        else:
            in_dangerous_area = ball_position[0] < self._frame_width * 0.4

        confidence = 0.7 if in_dangerous_area else 0.5

        return SetPieceEvent(
            set_piece_type=SetPieceType.FREE_KICK,
            timestamp=timestamp,
            frame_idx=frame_idx,
            team=attacking_team,
            ball_position=ball_position,
            confidence=confidence,
            metadata={
                'wall_detected': True,
                'wall_team': wall_team,
                'in_dangerous_area': in_dangerous_area,
                'stationary_frames': self._ball_stationary_frames
            }
        )

    def _detect_throw_in(
        self,
        ball_position: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float,
        current_team: Optional[str]
    ) -> Optional[SetPieceEvent]:
        """Detect throw-in situation."""
        # Ball was out of bounds and is now back in
        if not self._ball_out_of_bounds and self._out_of_bounds_position is not None:
            # Check if ball is near touchline
            near_touchline = (
                ball_position[1] < self.TOUCHLINE_MARGIN_PX * 2 or
                ball_position[1] > self._frame_height - self.TOUCHLINE_MARGIN_PX * 2
            )

            if near_touchline:
                # Determine which team takes the throw
                # This is simplified - in reality would need to track who touched last
                team = current_team if current_team else 'A'

                event = SetPieceEvent(
                    set_piece_type=SetPieceType.THROW_IN,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    team=team,
                    ball_position=ball_position,
                    confidence=0.6,
                    metadata={
                        'out_of_bounds_position': self._out_of_bounds_position
                    }
                )

                # Reset out of bounds tracking
                self._out_of_bounds_position = None
                return event

        return None

    def _detect_wall(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_position: Tuple[float, float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if a defensive wall has formed.
        
        Returns:
            Tuple of (wall_detected, wall_team)
        """
        # Group players by team near the ball
        teams_near_ball = {'A': [], 'B': []}

        for player_id, (x, y, team) in player_positions.items():
            distance = math.sqrt((x - ball_position[0]) ** 2 + (y - ball_position[1]) ** 2)
            if distance < self.wall_distance * 2:  # Within wall detection range
                if team in teams_near_ball:
                    teams_near_ball[team].append((player_id, x, y))

        # Check for wall formation (2+ players close together from same team)
        for team, players in teams_near_ball.items():
            if len(players) >= self.min_wall_players:
                # Check if they're in a line (wall formation)
                if len(players) >= 2:
                    # Simple check: are they within wall_distance of each other
                    for i, (pid1, x1, y1) in enumerate(players):
                        close_players = 1
                        for j, (pid2, x2, y2) in enumerate(players):
                            if i != j:
                                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                                if dist < self.wall_distance:
                                    close_players += 1
                        if close_players >= self.min_wall_players:
                            return True, team

        return False, None

    def _count_players_in_penalty_area(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        left_side: bool
    ) -> int:
        """Count players in penalty area."""
        count = 0

        # Define penalty area bounds
        if left_side:
            area_x_min = 0
            area_x_max = self._frame_width * self.PENALTY_AREA_WIDTH
        else:
            area_x_min = self._frame_width * (1 - self.PENALTY_AREA_WIDTH)
            area_x_max = self._frame_width

        area_y_min = self._frame_height * (1 - self.PENALTY_AREA_DEPTH) / 2
        area_y_max = self._frame_height * (1 + self.PENALTY_AREA_DEPTH) / 2

        for player_id, (x, y, team) in player_positions.items():
            if area_x_min <= x <= area_x_max and area_y_min <= y <= area_y_max:
                count += 1

        return count

    def _check_set_piece_outcome(
        self,
        ball_position: Optional[Tuple[float, float]],
        ball_velocity: float,
        frame_idx: int,
        timestamp: float
    ) -> Optional[SetPieceOutcome]:
        """
        Check if the active set piece has concluded and determine outcome.
        
        Returns:
            SetPieceOutcome if concluded, None otherwise
        """
        if not self._active_set_piece:
            return None

        # Check if ball is moving significantly (set piece taken)
        if ball_velocity > self.stationary_threshold * 3:
            # Set piece has been taken
            # Classify outcome based on what happens next
            # This is simplified - would need more context for accurate classification

            # If ball moves toward goal, likely a shot
            if self._active_set_piece.set_piece_type in [SetPieceType.FREE_KICK, SetPieceType.PENALTY]:
                return SetPieceOutcome.SHOT
            elif self._active_set_piece.set_piece_type == SetPieceType.CORNER:
                return SetPieceOutcome.PASS  # Corners usually crossed
            else:
                return SetPieceOutcome.IN_PLAY

        # Check if too much time has passed (set piece expired)
        time_elapsed = timestamp - self._active_set_piece.timestamp
        if time_elapsed > 10.0:  # 10 seconds timeout
            return SetPieceOutcome.UNKNOWN

        return None

    def get_active_set_piece(self) -> Optional[SetPieceEvent]:
        """Get currently active set piece."""
        return self._active_set_piece

    def get_set_piece_history(
        self,
        set_piece_type: SetPieceType = None,
        team: str = None
    ) -> List[SetPieceEvent]:
        """
        Get set piece history with optional filtering.
        
        Args:
            set_piece_type: Filter by type
            team: Filter by team
            
        Returns:
            List of SetPieceEvent objects
        """
        events = self._set_piece_history

        if set_piece_type:
            events = [e for e in events if e.set_piece_type == set_piece_type]

        if team:
            events = [e for e in events if e.team == team]

        return events

    def get_statistics(self) -> Dict[str, Any]:
        """Get set piece detection statistics."""
        stats = {
            'total_set_pieces': len(self._set_piece_history),
            'by_type': {},
            'by_team': {'A': 0, 'B': 0},
            'outcomes': {}
        }

        for event in self._set_piece_history:
            # Count by type
            type_name = event.set_piece_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1

            # Count by team
            if event.team in stats['by_team']:
                stats['by_team'][event.team] += 1

            # Count outcomes
            outcome_name = event.outcome.value
            stats['outcomes'][outcome_name] = stats['outcomes'].get(outcome_name, 0) + 1

        return stats

    def reset(self) -> None:
        """Reset all tracking data."""
        self._ball_stationary_frames = 0
        self._last_ball_position = None
        self._last_ball_velocity = 0.0
        self._ball_out_of_bounds = False
        self._out_of_bounds_position = None
        self._active_set_piece = None
        self._set_piece_history.clear()
        self._player_positions.clear()
