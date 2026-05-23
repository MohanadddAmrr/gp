"""
Offside Detection System Module

Calculates offside lines and detects potential offside situations.

OFFSIDE RULE:
A player is in an offside position if:
- They are nearer to the opponent's goal line than both the ball and the second-last opponent
- At the moment the ball is played to them

OFFSIDE DETECTION LOGIC:
1. Track second-last defender position (offside line)
2. Track goalkeeper position
3. Identify attacking players beyond the offside line
4. Detect when ball is played to an offside player
5. Visualize offside lines on pitch

IMPLEMENTATION NOTES:
- Uses x-coordinate for horizontal positioning
- Team A attacks right (increasing x), Team B attacks left (decreasing x)
- Offside line is at the second-last defender's x position
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class OffsideLine:
    """Represents the offside line at a specific moment."""
    timestamp: float
    frame_idx: int
    team: str  # Team that the offside line is FOR (attacking team)
    line_x: float  # X-coordinate of offside line
    second_last_defender_id: Optional[int]
    goalkeeper_id: Optional[int]
    defender_positions: List[Tuple[int, float]] = field(default_factory=list)  # (player_id, x)


@dataclass
class PotentialOffside:
    """Records a potential offside situation."""
    timestamp: float
    frame_idx: int
    attacking_team: str
    attacking_player_id: int
    attacker_x: float
    offside_line_x: float
    offside_distance: float
    ball_played: bool
    confirmed_offside: bool = False


class OffsideDetector:
    """
    Detects offside situations by tracking defender positions and attacking players.
    
    Key features:
    - Calculates offside line based on second-last defender
    - Tracks last defender and goalkeeper positions
    - Identifies attacking players in offside positions
    - Detects when ball is played to offside players
    - Provides visualization data for offside lines
    """

    def __init__(
        self,
        offside_margin_px: float = 10.0,  # Margin for offside detection
        history_window: int = 30,  # Frames to keep history
        min_defenders: int = 2,  # Minimum defenders needed for valid offside line
    ):
        """
        Initialize offside detector.
        
        Args:
            offside_margin_px: Margin in pixels for offside detection (tolerance)
            history_window: Number of frames to keep in history
            min_defenders: Minimum defenders needed to establish offside line
        """
        self.offside_margin = offside_margin_px
        self.history_window = history_window
        self.min_defenders = min_defenders

        # Current state
        self._current_offside_lines: Dict[str, OffsideLine] = {}  # Per defending team
        self._player_positions: Dict[int, Tuple[float, float, str]] = {}  # (x, y, team)

        # History
        self._offside_line_history: List[OffsideLine] = []
        self._potential_offsides: List[PotentialOffside] = []
        self._confirmed_offsides: List[Dict[str, Any]] = []

        # Ball tracking for offside confirmation
        self._last_ball_position: Optional[Tuple[float, float]] = None
        self._last_ball_velocity: float = 0.0
        self._ball_played_frame: Optional[int] = None

        # Offside-offence de-duplication. A single offside offence spans many
        # frames; without a cooldown the same offence is re-confirmed every
        # frame, which inflated the count into the hundreds.
        self._last_confirmed_offside_frame: int = -10000
        self.offside_cooldown_frames: int = 90  # ~3.5s at the processed frame rate

    def update_player_positions(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float
    ) -> None:
        """
        Update player positions and recalculate offside lines.
        
        Args:
            player_positions: Dict of {player_id: (x, y, team)}
            frame_idx: Current frame index
            timestamp: Current timestamp
        """
        self._player_positions = player_positions

        # Calculate offside lines for both teams
        for team in ['A', 'B']:
            offside_line = self._calculate_offside_line(team, frame_idx, timestamp)
            if offside_line:
                self._current_offside_lines[team] = offside_line
                self._offside_line_history.append(offside_line)

        # Check for potential offsides
        self._check_potential_offsides(frame_idx, timestamp)

    def _calculate_offside_line(
        self,
        defending_team: str,
        frame_idx: int,
        timestamp: float
    ) -> Optional[OffsideLine]:
        """
        Calculate offside line for a defending team.
        
        The offside line is at the position of the second-last defender
        (or the ball position if it's behind the second-last defender).
        
        Args:
            defending_team: Team defending ('A' or 'B')
            frame_idx: Current frame index
            timestamp: Current timestamp
            
        Returns:
            OffsideLine object or None if insufficient data
        """
        # Get defenders for this team
        defenders = []
        goalkeeper = None

        for player_id, (x, y, team) in self._player_positions.items():
            if team != defending_team:
                continue

            # Assume deepest player (furthest back) is goalkeeper
            defenders.append((player_id, x))

        if len(defenders) < self.min_defenders:
            return None

        # Sort defenders by x position
        # Team A defends left (lower x), Team B defends right (higher x)
        if defending_team == 'A':
            # Team A: defenders sorted by x ascending (deepest first)
            defenders.sort(key=lambda d: d[1])
        else:
            # Team B: defenders sorted by x descending (deepest first)
            defenders.sort(key=lambda d: d[1], reverse=True)

        # Second-last defender
        second_last = defenders[1] if len(defenders) >= 2 else defenders[0]
        goalkeeper = defenders[0]  # Deepest is usually GK

        # Offside line is at second-last defender
        offside_x = second_last[1]

        # Adjust for ball position (offside line is ball position if ball is behind defenders)
        if self._last_ball_position:
            ball_x = self._last_ball_position[0]
            if defending_team == 'A':
                # If ball is behind (left of) second-last defender, use ball position
                if ball_x < offside_x:
                    offside_x = ball_x
            else:
                # If ball is behind (right of) second-last defender, use ball position
                if ball_x > offside_x:
                    offside_x = ball_x

        return OffsideLine(
            timestamp=timestamp,
            frame_idx=frame_idx,
            team=defending_team,
            line_x=offside_x,
            second_last_defender_id=second_last[0],
            goalkeeper_id=goalkeeper[0],
            defender_positions=defenders
        )

    def _check_potential_offsides(self, frame_idx: int, timestamp: float) -> None:
        """
        Check for attacking players in offside positions.
        
        Args:
            frame_idx: Current frame index
            timestamp: Current timestamp
        """
        for team in ['A', 'B']:
            defending_team = 'B' if team == 'A' else 'A'

            if defending_team not in self._current_offside_lines:
                continue

            offside_line = self._current_offside_lines[defending_team]
            offside_x = offside_line.line_x

            # Check all attacking players
            for player_id, (x, y, player_team) in self._player_positions.items():
                if player_team != team:
                    continue

                # Check if player is beyond offside line
                is_offside = False
                if team == 'A':
                    # Team A attacks right (higher x)
                    # Offside if x > offside_line_x + margin
                    is_offside = x > (offside_x + self.offside_margin)
                else:
                    # Team B attacks left (lower x)
                    # Offside if x < offside_line_x - margin
                    is_offside = x < (offside_x - self.offside_margin)

                if is_offside:
                    offside_distance = abs(x - offside_x)

                    potential = PotentialOffside(
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        attacking_team=team,
                        attacking_player_id=player_id,
                        attacker_x=x,
                        offside_line_x=offside_x,
                        offside_distance=offside_distance,
                        ball_played=False
                    )

                    self._potential_offsides.append(potential)

        # Keep the list bounded. update_ball_state() only looks back ~30
        # frames, so older records are dead weight — and a full-match run
        # would otherwise accumulate hundreds of thousands of entries.
        if len(self._potential_offsides) > 600:
            del self._potential_offsides[:-600]

    def update_ball_state(
        self,
        ball_position: Tuple[float, float],
        ball_velocity: float,
        frame_idx: int,
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """
        Update ball state and check for offside confirmations.
        
        Args:
            ball_position: (x, y) ball position
            ball_velocity: Ball velocity in pixels/second
            frame_idx: Current frame index
            timestamp: Current timestamp
            
        Returns:
            Offside event dict if offside confirmed, None otherwise
        """
        self._last_ball_position = ball_position
        self._last_ball_velocity = ball_velocity

        # Detect if ball was played (significant velocity change or pass)
        ball_played = ball_velocity > 200  # Threshold for ball being played

        if not ball_played:
            return None

        # De-dup: one offside offence spans many frames. Only count one
        # offence per cooldown window.
        if frame_idx - self._last_confirmed_offside_frame < self.offside_cooldown_frames:
            return None

        # Check if ball was played to an offside player
        for potential in reversed(self._potential_offsides):
            # reversed() => newest first; once we hit one older than the
            # window every remaining record is older too, so stop (break,
            # not continue — continue made this O(n) per frame -> O(n^2)).
            if frame_idx - potential.frame_idx > 30:
                break

            # Check if ball is near the offside player
            for player_id, (x, y, team) in self._player_positions.items():
                if player_id != potential.attacking_player_id:
                    continue

                # Calculate distance from ball to player
                distance = math.sqrt(
                    (ball_position[0] - x) ** 2 +
                    (ball_position[1] - y) ** 2
                )

                # If ball is played to this player, confirm offside
                if distance < 50:  # Within 50 pixels
                    potential.ball_played = True
                    potential.confirmed_offside = True
                    self._last_confirmed_offside_frame = frame_idx

                    offside_event = {
                        'type': 'offside',
                        'timestamp': timestamp,
                        'frame_idx': frame_idx,
                        'attacking_team': potential.attacking_player_id,
                        'attacking_player_id': potential.attacking_player_id,
                        'offside_line_x': potential.offside_line_x,
                        'attacker_x': potential.attacker_x,
                        'offside_distance': potential.offside_distance,
                        'second_last_defender': self._current_offside_lines.get(
                            'B' if potential.attacking_team == 'A' else 'A'
                        ).second_last_defender_id if potential.attacking_team == 'A' else None
                    }

                    self._confirmed_offsides.append(offside_event)
                    return offside_event

        return None

    def get_offside_line(self, defending_team: str) -> Optional[float]:
        """
        Get current offside line x-coordinate for a defending team.
        
        Args:
            defending_team: Team defending ('A' or 'B')
            
        Returns:
            X-coordinate of offside line or None
        """
        if defending_team in self._current_offside_lines:
            return self._current_offside_lines[defending_team].line_x
        return None

    def get_offside_line_data(self, defending_team: str) -> Optional[Dict[str, Any]]:
        """
        Get complete offside line data for visualization.
        
        Args:
            defending_team: Team defending ('A' or 'B')
            
        Returns:
            Dictionary with offside line data for visualization
        """
        if defending_team not in self._current_offside_lines:
            return None

        line = self._current_offside_lines[defending_team]

        return {
            'line_x': line.line_x,
            'team': line.team,
            'second_last_defender_id': line.second_last_defender_id,
            'goalkeeper_id': line.goalkeeper_id,
            'defender_count': len(line.defender_positions),
            'defender_positions': line.defender_positions
        }

    def get_potential_offsides(
        self,
        team: str = None,
        confirmed_only: bool = False
    ) -> List[PotentialOffside]:
        """
        Get potential offside situations.
        
        Args:
            team: Filter by attacking team (optional)
            confirmed_only: Only return confirmed offsides
            
        Returns:
            List of PotentialOffside objects
        """
        offsides = self._potential_offsides

        if team:
            offsides = [o for o in offsides if o.attacking_team == team]

        if confirmed_only:
            offsides = [o for o in offsides if o.confirmed_offside]

        return offsides

    def get_confirmed_offsides(self) -> List[Dict[str, Any]]:
        """Get all confirmed offside events."""
        return self._confirmed_offsides.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get offside detection statistics."""
        total_potential = len(self._potential_offsides)
        # Count actual offside OFFENCES (discrete, de-duplicated events),
        # not per-frame offside positions.
        confirmed = len(self._confirmed_offsides)

        # By team
        team_a_potential = len([o for o in self._potential_offsides if o.attacking_team == 'A'])
        team_b_potential = len([o for o in self._potential_offsides if o.attacking_team == 'B'])

        return {
            'total_potential_offsides': total_potential,
            'confirmed_offsides': confirmed,
            'team_a_potential': team_a_potential,
            'team_b_potential': team_b_potential,
            'confirmation_rate': confirmed / total_potential if total_potential > 0 else 0.0
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self._current_offside_lines.clear()
        self._player_positions.clear()
        self._offside_line_history.clear()
        self._potential_offsides.clear()
        self._confirmed_offsides.clear()
        self._last_ball_position = None
        self._last_ball_velocity = 0.0
        self._ball_played_frame = None
        self._last_confirmed_offside_frame = -10000
