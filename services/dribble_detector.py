"""
Dribble Detection Module

Distinguishes between dribbling and passing, tracks ball control time,
calculates dribble success rate, and identifies take-ons.

DRIBBLE DETECTION LOGIC:

1. Dribble vs Pass:
   - Dribble: Player maintains close ball control while moving
   - Pass: Ball moves quickly away from player to teammate
   
2. Ball Control Time:
   - Track duration player keeps ball within control radius
   - Control radius: ~1 meter (varies by player speed)
   
3. Dribble Success:
   - Successful: Player maintains possession throughout dribble
   - Unsuccessful: Ball lost to opponent or out of bounds
   
4. Take-on Detection:
   - Player dribbles past an opponent
   - Opponent was within challenge distance
   - Player maintains possession after passing opponent

METRICS:
- Total dribbles
- Dribble success rate
- Average dribble distance
- Take-ons completed
- Dribbles in dangerous areas
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class DribbleEvent:
    """Records a detected dribble."""
    player_id: int
    team: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    distance: float
    duration: float
    max_speed: float
    opponents_beaten: List[int] = field(default_factory=list)
    successful: bool = True
    in_dangerous_area: bool = False


@dataclass
class TakeOnEvent:
    """Records a successful take-on (beating a defender)."""
    dribbler_id: int
    defender_id: int
    timestamp: float
    frame_idx: int
    dribbler_team: str
    location: Tuple[float, float]
    successful: bool = True


class DribbleDetector:
    """
    Detects and analyzes dribbling events.
    
    Features:
    - Distinguishes dribbles from passes
    - Tracks ball control time per player
    - Calculates dribble success rate
    - Identifies take-ons and beaten defenders
    """

    # Control radius in pixels (approximately 1-1.5 meters)
    CONTROL_RADIUS_PX = 50.0  # Increased from 40.0 for better control detection
    MAX_DRIBBLE_SPEED_MPS = 9.0  # Increased from 8.0 for sprinting with ball
    MIN_DRIBBLE_DURATION = 0.3  # Decreased from 0.5 to catch quick dribbles
    MAX_DRIBBLE_DURATION = 12.0  # Maximum seconds before considering it possession
    TAKE_ON_DISTANCE = 50.0  # Decreased from 60.0 for more accurate take-on detection
    TAKE_ON_ANGLE_THRESHOLD = 45.0  # Degrees - max angle change for take-on
    MIN_TAKE_ON_PROGRESS = 20.0  # Minimum progress past defender to count as take-on
    DRIBBLE_DIRECTION_CHANGE_THRESHOLD = 30.0  # Degrees for direction change detection
    BALL_CARRIER_PROXIMITY = 80.0  # Max distance for ball carrier identification
    SUCCESSFUL_DRIBBLE_MIN_DISTANCE = 3.0  # Minimum meters gained to count as successful
    OPPONENT_PRESSURE_DISTANCE = 100.0  # Distance to consider opponent pressure
    DRIBBLE_COOLDOWN_FRAMES = 10  # Frames to wait before detecting new dribble for same player

    def __init__(
        self,
        control_radius_px: float = 50.0,
        min_dribble_duration: float = 0.3,
        max_dribble_speed_mps: float = 9.0,
        take_on_distance_px: float = 50.0,
    ):
        """
        Initialize dribble detector.
        
        Args:
            control_radius_px: Distance for ball control (pixels)
            min_dribble_duration: Minimum duration to count as dribble (seconds)
            max_dribble_speed_mps: Maximum realistic dribble speed
            take_on_distance_px: Distance threshold for take-on detection
        """
        self.control_radius = control_radius_px
        self.min_duration = min_dribble_duration
        self.max_speed = max_dribble_speed_mps
        self.take_on_distance = take_on_distance_px
        
        # Additional parameters for improved detection
        self.take_on_angle_threshold = self.TAKE_ON_ANGLE_THRESHOLD
        self.min_take_on_progress = self.MIN_TAKE_ON_PROGRESS
        self.opponent_pressure_distance = self.OPPONENT_PRESSURE_DISTANCE
        self.dribble_cooldown_frames = self.DRIBBLE_COOLDOWN_FRAMES
        self.successful_dribble_min_distance = self.SUCCESSFUL_DRIBBLE_MIN_DISTANCE

        # Active dribbles: {player_id: dribble_data}
        self._active_dribbles: Dict[int, Dict] = {}
        
        # Dribble cooldown tracking
        self._last_dribble_end_frame: Dict[int, int] = {}

        # Completed dribbles
        self._dribble_events: List[DribbleEvent] = []
        self._take_on_events: List[TakeOnEvent] = []

        # Player statistics
        self._player_dribble_stats: Dict[int, Dict] = {}

        # Frame data
        self._frame_width: int = 1920
        self._meter_per_px: float = 105.0 / 1920.0  # Default scale

    def set_scale(self, meter_per_px: float) -> None:
        """Set pixel to meter conversion."""
        self._meter_per_px = meter_per_px

    def update(
        self,
        player_id: int,
        player_position: Tuple[float, float],
        player_team: str,
        ball_position: Tuple[float, float],
        ball_velocity: float,
        all_players: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float
    ) -> Optional[DribbleEvent]:
        """
        Update dribble detection for a player.
        
        Args:
            player_id: Player track ID
            player_position: (x, y) player position
            player_team: Team label
            ball_position: (x, y) ball position
            ball_velocity: Ball velocity (pixels/second)
            all_players: Dict of all player positions
            frame_idx: Current frame index
            timestamp: Current timestamp
            
        Returns:
            DribbleEvent if dribble completed, None otherwise
        """
        # Calculate distance to ball
        distance_to_ball = math.sqrt(
            (player_position[0] - ball_position[0]) ** 2 +
            (player_position[1] - ball_position[1]) ** 2
        )

        # Convert ball velocity to m/s
        ball_speed_mps = ball_velocity * self._meter_per_px

        # Check if player has ball control
        has_control = (
            distance_to_ball <= self.control_radius and
            ball_speed_mps <= self.max_speed
        )

        completed_dribble = None

        # Check cooldown period
        last_end_frame = self._last_dribble_end_frame.get(player_id, -100)
        if frame_idx - last_end_frame < self.dribble_cooldown_frames:
            return None
        
        if has_control:
            if player_id not in self._active_dribbles:
                # Start new dribble
                self._active_dribbles[player_id] = {
                    'start_frame': frame_idx,
                    'start_time': timestamp,
                    'start_position': player_position,
                    'last_position': player_position,
                    'previous_positions': [player_position],  # Track for direction analysis
                    'team': player_team,
                    'total_distance': 0.0,
                    'max_speed': 0.0,
                    'opponents_nearby': [],
                    'opponents_beaten': [],
                    'opponents_challenged': [],  # Track opponents who attempted challenge
                    'frames': 0,
                    'direction_changes': 0,
                    'pressure_situations': 0,
                    'in_opponent_half': self._is_in_opponent_half(player_position, player_team),
                }

                # Record initial opponents nearby
                self._active_dribbles[player_id]['opponents_nearby'] = \
                    self._get_nearby_opponents(player_id, player_team, player_position, all_players)
            else:
                # Continue dribble
                dribble = self._active_dribbles[player_id]

                # Update distance
                prev_pos = dribble['last_position']
                step_distance = math.sqrt(
                    (player_position[0] - prev_pos[0]) ** 2 +
                    (player_position[1] - prev_pos[1]) ** 2
                )
                dribble['total_distance'] += step_distance
                dribble['last_position'] = player_position
                dribble['frames'] += 1
                
                # Track position history for direction analysis
                dribble['previous_positions'].append(player_position)
                if len(dribble['previous_positions']) > 10:
                    dribble['previous_positions'].pop(0)

                # Update max speed
                dribble['max_speed'] = max(dribble['max_speed'], ball_speed_mps)
                
                # Track pressure situations
                nearby_opponents = self._get_nearby_opponents(
                    player_id, player_team, player_position, all_players
                )
                if len(nearby_opponents) >= 2:
                    dribble['pressure_situations'] += 1

                # Check for take-ons with improved logic
                self._check_take_ons_improved(player_id, player_position, dribble, all_players, frame_idx, timestamp)

        else:
            # Player lost ball control
            if player_id in self._active_dribbles:
                completed_dribble = self._finalize_dribble(
                    player_id, player_position, frame_idx, timestamp, successful=False
                )

        # Check for possession change (someone else has ball)
        current_possessor = self._find_ball_possessor(ball_position, all_players)
        if current_possessor is not None and current_possessor != player_id:
            if player_id in self._active_dribbles:
                # Dribble ended due to possession change
                successful = self._active_dribbles[player_id]['team'] == all_players.get(current_possessor, ('', '', player_team))[2]
                completed_dribble = self._finalize_dribble(
                    player_id, player_position, frame_idx, timestamp, successful=successful
                )

        return completed_dribble

    def _get_nearby_opponents(
        self,
        player_id: int,
        player_team: str,
        player_position: Tuple[float, float],
        all_players: Dict[int, Tuple[float, float, str]]
    ) -> List[Tuple[int, float]]:
        """
        Get opponents within challenge distance.
        
        Returns:
            List of (opponent_id, distance) tuples
        """
        nearby = []
        for other_id, (x, y, team) in all_players.items():
            if other_id == player_id or team == player_team:
                continue

            distance = math.sqrt(
                (player_position[0] - x) ** 2 +
                (player_position[1] - y) ** 2
            )

            if distance <= self.take_on_distance * 2:  # Detection range
                nearby.append((other_id, distance))

        return sorted(nearby, key=lambda x: x[1])

    def _is_in_opponent_half(self, position: Tuple[float, float], team: str) -> bool:
        """Check if position is in opponent's half."""
        x = position[0]
        if team == 'A':
            return x > self._frame_width * 0.5
        else:
            return x < self._frame_width * 0.5
    
    def _calculate_progress_past_defender(
        self,
        player_pos: Tuple[float, float],
        defender_pos: Tuple[float, float],
        team: str
    ) -> float:
        """Calculate how far player has progressed past defender."""
        if team == 'A':
            # Progress is positive when moving right (increasing x)
            return player_pos[0] - defender_pos[0]
        else:
            # Progress is positive when moving left (decreasing x)
            return defender_pos[0] - player_pos[0]
    
    def _check_take_ons(
        self,
        player_id: int,
        player_position: Tuple[float, float],
        dribble: Dict,
        all_players: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float
    ) -> None:
        """Check if player has beaten any defenders (legacy method)."""
        self._check_take_ons_improved(player_id, player_position, dribble, all_players, frame_idx, timestamp)
    
    def _check_take_ons_improved(
        self,
        player_id: int,
        player_position: Tuple[float, float],
        dribble: Dict,
        all_players: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float
    ) -> None:
        """
        Improved take-on detection with better accuracy.
        
        A take-on is counted when:
        1. Player was initially close to opponent (within take_on_distance)
        2. Player has moved significantly past opponent (min_take_on_progress)
        3. Player maintains possession
        4. Player is moving in attacking direction
        """
        for opponent_id, initial_distance in dribble['opponents_nearby']:
            if opponent_id in dribble['opponents_beaten']:
                continue

            # Get current opponent position
            if opponent_id not in all_players:
                continue

            opp_x, opp_y, _ = all_players[opponent_id]

            # Calculate current distance
            current_distance = math.sqrt(
                (player_position[0] - opp_x) ** 2 +
                (player_position[1] - opp_y) ** 2
            )
            
            # Calculate progress past defender
            progress = self._calculate_progress_past_defender(
                player_position, (opp_x, opp_y), dribble['team']
            )

            # Check if player has moved past opponent with sufficient progress
            if (current_distance > self.take_on_distance and 
                initial_distance <= self.take_on_distance * 1.5 and
                progress > self.min_take_on_progress):
                
                # Check if player is now ahead of opponent (in attacking direction)
                if self._is_ahead_of(player_position, (opp_x, opp_y), dribble['team']):
                    dribble['opponents_beaten'].append(opponent_id)
                    dribble['opponents_challenged'].append(opponent_id)

                    take_on = TakeOnEvent(
                        dribbler_id=player_id,
                        defender_id=opponent_id,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        dribbler_team=dribble['team'],
                        location=player_position,
                        successful=True
                    )
                    self._take_on_events.append(take_on)
                    
        # Also check for defenders who came to challenge but were evaded
        for player_oid, (ox, oy, oteam) in all_players.items():
            if oteam == dribble['team'] or player_oid == player_id:
                continue
            
            if player_oid in dribble['opponents_beaten'] or player_oid in dribble['opponents_challenged']:
                continue
            
            # Check if defender is close (challenging)
            distance = math.sqrt(
                (player_position[0] - ox) ** 2 +
                (player_position[1] - oy) ** 2
            )
            
            if distance < self.opponent_pressure_distance:
                dribble['opponents_challenged'].append(player_oid)
                
                # Check if player has beaten this challenger
                progress = self._calculate_progress_past_defender(
                    player_position, (ox, oy), dribble['team']
                )
                
                if progress > self.min_take_on_progress and self._is_ahead_of(player_position, (ox, oy), dribble['team']):
                    dribble['opponents_beaten'].append(player_oid)
                    
                    take_on = TakeOnEvent(
                        dribbler_id=player_id,
                        defender_id=player_oid,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        dribbler_team=dribble['team'],
                        location=player_position,
                        successful=True
                    )
                    self._take_on_events.append(take_on)

    def _is_ahead_of(
        self,
        player_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        team: str
    ) -> bool:
        """Check if player is ahead of opponent in attacking direction."""
        if team == 'A':
            # Team A attacks right (increasing x)
            return player_pos[0] > opponent_pos[0]
        else:
            # Team B attacks left (decreasing x)
            return player_pos[0] < opponent_pos[0]

    def _find_ball_possessor(
        self,
        ball_position: Tuple[float, float],
        all_players: Dict[int, Tuple[float, float, str]]
    ) -> Optional[int]:
        """Find which player currently has the ball."""
        min_distance = float('inf')
        possessor = None

        for player_id, (x, y, team) in all_players.items():
            distance = math.sqrt(
                (ball_position[0] - x) ** 2 +
                (ball_position[1] - y) ** 2
            )

            if distance < min_distance and distance <= self.control_radius:
                min_distance = distance
                possessor = player_id

        return possessor

    def _finalize_dribble(
        self,
        player_id: int,
        end_position: Tuple[float, float],
        end_frame: int,
        end_time: float,
        successful: bool
    ) -> Optional[DribbleEvent]:
        """
        Finalize an active dribble and create DribbleEvent.
        
        Returns:
            DribbleEvent if dribble meets criteria, None otherwise
        """
        dribble = self._active_dribbles.pop(player_id)
        
        # Track cooldown
        self._last_dribble_end_frame[player_id] = end_frame

        duration = end_time - dribble['start_time']

        # Only count if meets minimum duration
        if duration < self.min_duration:
            return None
        
        # Calculate actual distance gained in attacking direction
        distance_m = dribble['total_distance'] * self._meter_per_px
        
        # Determine success based on multiple factors
        # 1. Player maintained possession (not tackled)
        # 2. Made progress in attacking direction
        # 3. Distance gained is significant
        start_x = dribble['start_position'][0]
        end_x = end_position[0]
        team = dribble['team']
        
        # Calculate progress in attacking direction
        if team == 'A':
            progress_m = (end_x - start_x) * self._meter_per_px
        else:
            progress_m = (start_x - end_x) * self._meter_per_px
        
        # Consider successful if:
        # - Player maintained possession AND
        # - Made some progress OR beat an opponent
        is_successful = successful and (progress_m > 0 or len(dribble['opponents_beaten']) > 0)
        
        # Override success if significant distance was covered
        if distance_m >= self.successful_dribble_min_distance:
            is_successful = True

        # Check if in dangerous area (attacking third)
        in_dangerous = self._is_in_dangerous_area(end_position, dribble['team'])
        
        # Check if dribble was under pressure
        under_pressure = dribble.get('pressure_situations', 0) > 0
        
        # Check if dribble was in opponent's half
        in_opponent_half = dribble.get('in_opponent_half', False)

        event = DribbleEvent(
            player_id=player_id,
            team=dribble['team'],
            start_frame=dribble['start_frame'],
            end_frame=end_frame,
            start_time=dribble['start_time'],
            end_time=end_time,
            start_position=dribble['start_position'],
            end_position=end_position,
            distance=distance_m,
            duration=duration,
            max_speed=dribble['max_speed'],
            opponents_beaten=dribble['opponents_beaten'],
            successful=is_successful,
            in_dangerous_area=in_dangerous
        )
        
        # Add extra metadata for analysis
        event.progress_m = progress_m
        event.under_pressure = under_pressure
        event.in_opponent_half = in_opponent_half
        event.opponents_challenged = dribble.get('opponents_challenged', [])

        self._dribble_events.append(event)
        self._update_player_stats(event)

        return event

    def _is_in_dangerous_area(self, position: Tuple[float, float], team: str) -> bool:
        """Check if position is in dangerous attacking area."""
        x = position[0]
        attacking_third = self._frame_width * 0.66

        if team == 'A':
            return x > attacking_third
        else:
            return x < (self._frame_width - attacking_third)

    def _update_player_stats(self, event: DribbleEvent) -> None:
        """Update statistics for a player."""
        pid = event.player_id

        if pid not in self._player_dribble_stats:
            self._player_dribble_stats[pid] = {
                'total_dribbles': 0,
                'successful_dribbles': 0,
                'total_distance': 0.0,
                'total_duration': 0.0,
                'take_ons': 0,
                'dribbles_in_dangerous_area': 0
            }

        stats = self._player_dribble_stats[pid]
        stats['total_dribbles'] += 1
        if event.successful:
            stats['successful_dribbles'] += 1
        stats['total_distance'] += event.distance
        stats['total_duration'] += event.duration
        stats['take_ons'] += len(event.opponents_beaten)
        if event.in_dangerous_area:
            stats['dribbles_in_dangerous_area'] += 1

    def finalize_all(self, end_frame: int, end_time: float) -> None:
        """Finalize all active dribbles at end of processing."""
        for player_id in list(self._active_dribbles.keys()):
            dribble = self._active_dribbles[player_id]
            self._finalize_dribble(
                player_id, dribble['last_position'], end_frame, end_time, successful=True
            )

    def get_dribble_events(
        self,
        player_id: int = None,
        team: str = None,
        successful_only: bool = False
    ) -> List[DribbleEvent]:
        """
        Get dribble events with optional filtering.
        
        Args:
            player_id: Filter by player
            team: Filter by team
            successful_only: Only return successful dribbles
            
        Returns:
            List of DribbleEvent objects
        """
        events = self._dribble_events

        if player_id is not None:
            events = [e for e in events if e.player_id == player_id]

        if team:
            events = [e for e in events if e.team == team]

        if successful_only:
            events = [e for e in events if e.successful]

        return events

    def get_take_on_events(
        self,
        dribbler_id: int = None,
        team: str = None
    ) -> List[TakeOnEvent]:
        """
        Get take-on events with optional filtering.
        
        Args:
            dribbler_id: Filter by dribbler
            team: Filter by team
            
        Returns:
            List of TakeOnEvent objects
        """
        events = self._take_on_events

        if dribbler_id is not None:
            events = [e for e in events if e.dribbler_id == dribbler_id]

        if team:
            events = [e for e in events if e.dribbler_team == team]

        return events

    def get_player_stats(self, player_id: int = None) -> Dict[str, Any]:
        """
        Get dribble statistics.
        
        Args:
            player_id: Specific player ID or None for all players
            
        Returns:
            Dictionary of statistics
        """
        if player_id is not None:
            return self._player_dribble_stats.get(player_id, {
                'total_dribbles': 0,
                'successful_dribbles': 0,
                'total_distance': 0.0,
                'total_duration': 0.0,
                'take_ons': 0,
                'dribbles_in_dangerous_area': 0
            })

        # Aggregate all players
        total_stats = {
            'total_dribbles': 0,
            'successful_dribbles': 0,
            'total_distance': 0.0,
            'total_duration': 0.0,
            'take_ons': 0,
            'dribbles_in_dangerous_area': 0
        }

        for stats in self._player_dribble_stats.values():
            for key in total_stats:
                total_stats[key] += stats[key]

        # Calculate success rate
        if total_stats['total_dribbles'] > 0:
            total_stats['success_rate'] = (
                total_stats['successful_dribbles'] / total_stats['total_dribbles'] * 100
            )
        else:
            total_stats['success_rate'] = 0.0

        # Average distance per dribble
        if total_stats['total_dribbles'] > 0:
            total_stats['avg_distance'] = (
                total_stats['total_distance'] / total_stats['total_dribbles']
            )
        else:
            total_stats['avg_distance'] = 0.0

        return total_stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dribble statistics."""
        stats = self.get_player_stats()

        # Add team breakdown
        team_stats = {'A': {'dribbles': 0, 'take_ons': 0, 'successful_dribbles': 0, 'pressure_dribbles': 0}, 
                      'B': {'dribbles': 0, 'take_ons': 0, 'successful_dribbles': 0, 'pressure_dribbles': 0}}

        for event in self._dribble_events:
            team = event.team
            if team in team_stats:
                team_stats[team]['dribbles'] += 1
                team_stats[team]['take_ons'] += len(event.opponents_beaten)
                if event.successful:
                    team_stats[team]['successful_dribbles'] += 1
                # Check for pressure attribute (added in improved detection)
                if hasattr(event, 'under_pressure') and event.under_pressure:
                    team_stats[team]['pressure_dribbles'] += 1

        stats['by_team'] = team_stats
        stats['total_take_ons'] = len(self._take_on_events)
        
        # Calculate take-on success rate
        if self._take_on_events:
            successful_take_ons = sum(1 for t in self._take_on_events if t.successful)
            stats['take_on_success_rate'] = (successful_take_ons / len(self._take_on_events)) * 100
        else:
            stats['take_on_success_rate'] = 0.0
        
        # Add average dribble metrics
        if self._dribble_events:
            stats['avg_dribble_distance'] = sum(d.distance for d in self._dribble_events) / len(self._dribble_events)
            stats['avg_dribble_duration'] = sum(d.duration for d in self._dribble_events) / len(self._dribble_events)
            stats['avg_opponents_beaten_per_dribble'] = sum(len(d.opponents_beaten) for d in self._dribble_events) / len(self._dribble_events)
        else:
            stats['avg_dribble_distance'] = 0.0
            stats['avg_dribble_duration'] = 0.0
            stats['avg_opponents_beaten_per_dribble'] = 0.0

        return stats

    def reset(self) -> None:
        """Reset all tracking data."""
        self._active_dribbles.clear()
        self._dribble_events.clear()
        self._take_on_events.clear()
        self._player_dribble_stats.clear()
