"""
Possession Tracker Module - Enhanced with Tactical Context

This module implements possession detection - identifying which player has control of the ball.

ENHANCEMENTS:
1. Pressure metrics - Count opponents pressing the possessor
2. Tactical zones - Track possession in Defensive/Midfield/Attacking thirds
3. Duration analysis - Categorize possessions as Short/Medium/Long
4. Enhanced context - Provides rich tactical insights for coaches

WHY THESE ENHANCEMENTS MATTER:
- Pressure: Shows defensive intensity, helps identify when team struggles under press
- Zones: Reveals where team builds attacks, defensive vs attacking style
- Duration: Shows playing style (quick direct play vs patient build-up)

KEY INSIGHT:
Possession = player closest to ball (within threshold distance)
"""

from typing import List, Tuple, Optional, Dict, Any
import math
from collections import deque


class PossessionTracker:
    """
    Tracks ball possession by identifying the closest player to the ball.
    
    ENHANCED with tactical context: pressure, zones, duration analysis.
    """

    def __init__(self, distance_threshold: float = 50.0, max_history: int = 1000):
        """
        Initialize the possession tracker.

        Args:
            distance_threshold: Maximum distance in pixels for possession (default: 50px)
            max_history: Maximum number of possession events to store
        """
        self.distance_threshold = distance_threshold
        self.max_history = max_history

        # Current state
        self.current_possessor: Optional[int] = None
        self.current_team: Optional[str] = None
        self.possession_start_time: Optional[float] = None
        self.possession_start_frame: Optional[int] = None

        # Historical data
        self.possession_history: List[Dict[str, Any]] = []

        # Statistics
        self.team_possession_time: Dict[str, float] = {'A': 0.0, 'B': 0.0}
        self.player_possession_time: Dict[int, float] = {}
        self.player_touch_count: Dict[int, int] = {}

        # Last processed frame to avoid duplicates
        self.last_frame_idx: Optional[int] = None
        
        # NEW: Tactical context tracking
        self.zone_possession_time = {
            'A': {'Defensive': 0.0, 'Midfield': 0.0, 'Attacking': 0.0},
            'B': {'Defensive': 0.0, 'Midfield': 0.0, 'Attacking': 0.0}
        }
        self.pressure_events = []  # Track high-pressure possessions
        self.possession_durations = []  # Track all possession durations
        self.zone_changes = 0
        self.last_zone = None

    def detect_possession(
        self,
        ball_pos: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float
    ) -> Optional[Tuple[int, str, float]]:
        """
        Detect which player has possession of the ball.

        Args:
            ball_pos: (x, y) position of the ball in pixels
            player_positions: Dict mapping player_id to (x, y, team) tuples
            frame_idx: Current frame index in the video
            timestamp: Current timestamp in seconds

        Returns:
            Tuple of (player_id, team, distance) if possession detected, None otherwise
        """
        # Skip if we already processed this frame
        if frame_idx == self.last_frame_idx:
            if self.current_possessor is not None:
                return (self.current_possessor, self.current_team, 0.0)
            return None

        self.last_frame_idx = frame_idx

        if not player_positions or ball_pos is None:
            return None

        # Calculate distance from ball to each player
        min_distance = float('inf')
        closest_player = None
        closest_team = None

        ball_x, ball_y = ball_pos

        for player_id, (player_x, player_y, team) in player_positions.items():
            # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
            distance = math.sqrt(
                (player_x - ball_x) ** 2 +
                (player_y - ball_y) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_player = player_id
                closest_team = team

        # Check if closest player is within threshold
        if min_distance <= self.distance_threshold:
            # Possession detected
            new_possessor = closest_player
            new_team = closest_team

            # Check if possession changed
            if new_possessor != self.current_possessor:
                self._update_possession(
                    new_possessor,
                    new_team,
                    frame_idx,
                    timestamp
                )

            return (new_possessor, new_team, min_distance)
        else:
            # Ball is loose (no player close enough)
            if self.current_possessor is not None:
                # Possession ended
                self._update_possession(None, None, frame_idx, timestamp)

            return None

    def _update_possession(
        self,
        new_possessor: Optional[int],
        new_team: Optional[str],
        frame_idx: int,
        timestamp: float
    ) -> None:
        """
        Internal method to update possession state and log changes.
        """
        # Calculate duration of previous possession
        if self.current_possessor is not None and self.possession_start_time is not None:
            duration = timestamp - self.possession_start_time

            # Update team possession time
            if self.current_team in self.team_possession_time:
                self.team_possession_time[self.current_team] += duration

            # Update player possession time
            if self.current_possessor not in self.player_possession_time:
                self.player_possession_time[self.current_possessor] = 0.0
            self.player_possession_time[self.current_possessor] += duration
            
            # NEW: Track possession duration
            self.possession_durations.append(duration)
            
            # NEW: Track zone possession time
            if self.last_zone and self.current_team:
                if self.last_zone in self.zone_possession_time[self.current_team]:
                    self.zone_possession_time[self.current_team][self.last_zone] += duration

            # Log possession event to history
            event = {
                'player_id': self.current_possessor,
                'team': self.current_team,
                'start_frame': self.possession_start_frame,
                'end_frame': frame_idx,
                'start_time': self.possession_start_time,
                'end_time': timestamp,
                'duration': duration
            }
            self.possession_history.append(event)

            # Limit history size
            if len(self.possession_history) > self.max_history:
                self.possession_history.pop(0)

        # Update current state
        self.current_possessor = new_possessor
        self.current_team = new_team
        self.possession_start_time = timestamp if new_possessor is not None else None
        self.possession_start_frame = frame_idx if new_possessor is not None else None

        # Update touch count
        if new_possessor is not None:
            if new_possessor not in self.player_touch_count:
                self.player_touch_count[new_possessor] = 0
            self.player_touch_count[new_possessor] += 1

    def get_current_possessor(self) -> Optional[int]:
        """Get the ID of the player currently in possession."""
        return self.current_possessor

    def get_current_team(self) -> Optional[str]:
        """Get the team currently in possession."""
        return self.current_team

    def get_possession_percentage(self) -> Dict[str, float]:
        """Calculate possession percentage for each team."""
        total_time = sum(self.team_possession_time.values())

        if total_time == 0:
            return {'A': 0.0, 'B': 0.0}

        return {
            team: (time_val / total_time) * 100
            for team, time_val in self.team_possession_time.items()
        }

    def get_player_possession_stats(self) -> Dict[int, Dict[str, float]]:
        """Get detailed possession statistics for each player."""
        stats = {}

        for player_id in self.player_possession_time.keys():
            total_time = self.player_possession_time.get(player_id, 0.0)
            touch_count = self.player_touch_count.get(player_id, 0)

            stats[player_id] = {
                'total_time': total_time,
                'touch_count': touch_count,
                'avg_possession_duration': total_time / touch_count if touch_count > 0 else 0.0
            }

        return stats

    def get_possession_history(self) -> List[Dict[str, Any]]:
        """Get chronological history of possession events."""
        return self.possession_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive possession statistics."""
        return {
            'current_possessor': self.current_possessor,
            'current_team': self.current_team,
            'possession_percentage': self.get_possession_percentage(),
            'team_possession_time': self.team_possession_time.copy(),
            'player_stats': self.get_player_possession_stats(),
            'total_events': len(self.possession_history),
            # NEW: Enhanced metrics
            'zone_stats': self.get_zone_statistics(),
            'pressure_stats': self.get_pressure_statistics(),
            'duration_stats': self.get_duration_statistics(),
            'zone_changes': self.zone_changes
        }

    def reset(self) -> None:
        """Reset all tracking data to initial state."""
        self.current_possessor = None
        self.current_team = None
        self.possession_start_time = None
        self.possession_start_frame = None
        self.possession_history.clear()
        self.team_possession_time = {'A': 0.0, 'B': 0.0}
        self.player_possession_time.clear()
        self.player_touch_count.clear()
        self.last_frame_idx = None
        # NEW
        self.zone_possession_time = {
            'A': {'Defensive': 0.0, 'Midfield': 0.0, 'Attacking': 0.0},
            'B': {'Defensive': 0.0, 'Midfield': 0.0, 'Attacking': 0.0}
        }
        self.pressure_events.clear()
        self.possession_durations.clear()
        self.zone_changes = 0
        self.last_zone = None

    # ============================================================
    # NEW: TACTICAL CONTEXT METHODS
    # ============================================================

    def calculate_pressure(
        self,
        possessor_id: int,
        possessor_pos: Tuple[float, float],
        possessor_team: str,
        all_player_positions: Dict[int, Tuple[float, float, str]],
        pressure_threshold: float = 150.0
    ) -> int:
        """
        Calculate number of opponents pressuring the possessor.
        
        Pressure = opponents within threshold distance of ball carrier.
        Key tactical metric showing defensive intensity.
        
        WHY PRESSURE MATTERS:
        - High pressure = opponent playing aggressive defense
        - Low pressure = team has space to build attacks
        - Coaches use this to evaluate defensive effectiveness
        
        Args:
            possessor_id: ID of player with possession
            possessor_pos: Position of possessor (x, y)
            possessor_team: Team of possessor ('A' or 'B')
            all_player_positions: Dict of all player positions
            pressure_threshold: Distance threshold for pressure (default: 150px ≈ 15m)
            
        Returns:
            Number of opponents applying pressure (0-11)
        """
        pressure_count = 0
        
        for player_id, (px, py, team) in all_player_positions.items():
            # Skip possessor and teammates
            if player_id == possessor_id or team == possessor_team:
                continue
            
            # Calculate distance to possessor
            distance = math.sqrt(
                (px - possessor_pos[0]) ** 2 +
                (py - possessor_pos[1]) ** 2
            )
            
            if distance <= pressure_threshold:
                pressure_count += 1
        
        return pressure_count
    
    def get_possession_zone(
        self,
        ball_pos: Tuple[float, float],
        possessor_team: str,
        frame_width: int
    ) -> str:
        """
        Determine which tactical zone the possession is in.
        
        Zones (for Team A attacking right):
        - Team A: Defensive Third (left), Midfield (center), Attacking Third (right)
        - Team B: Defensive Third (right), Midfield (center), Attacking Third (left)
        
        WHY ZONES MATTER:
        - Shows where team builds attacks
        - "Possession in attacking third" = creating chances
        - "Possession in defensive third" = building from back
        - Coaches use this for tactical analysis
        
        Args:
            ball_pos: Ball position (x, y)
            possessor_team: Team in possession ('A' or 'B')
            frame_width: Field width in pixels
            
        Returns:
            Zone string: "Defensive", "Midfield", or "Attacking"
        """
        x = ball_pos[0]
        
        # Divide field into thirds
        third = frame_width / 3.0
        
        if possessor_team == 'A':
            # Team A attacks right
            if x < third:
                return "Defensive"
            elif x < 2 * third:
                return "Midfield"
            else:
                return "Attacking"
        else:
            # Team B attacks left
            if x < third:
                return "Attacking"
            elif x < 2 * third:
                return "Midfield"
            else:
                return "Defensive"
    
    def detect_possession_with_context(
        self,
        ball_pos: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float, str]],
        frame_idx: int,
        timestamp: float,
        frame_width: int,
        frame_height: int
    ) -> Optional[Dict]:
        """
        Enhanced possession detection with tactical context.
        
        Returns possession info PLUS:
        - Pressure count (how many opponents nearby)
        - Tactical zone (Defensive/Midfield/Attacking)
        - Duration of current possession
        
        This gives coaches rich tactical insights:
        - "Team A has 70% possession in attacking third under high pressure"
        - "Team B loses possession quickly when pressed"
        
        Args:
            ball_pos: Ball position (x, y)
            player_positions: Dict of player positions {id: (x, y, team)}
            frame_idx: Current frame
            timestamp: Current time
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Dict with possession info and tactical context, or None
        """
        # Standard possession detection
        result = self.detect_possession(ball_pos, player_positions, frame_idx, timestamp)
        
        if result is None:
            return None
        
        player_id, team, distance = result
        
        # Calculate pressure
        possessor_pos = player_positions[player_id][:2]
        pressure = self.calculate_pressure(
            player_id, possessor_pos, team, player_positions
        )
        
        # Get zone
        zone = self.get_possession_zone(ball_pos, team, frame_width)
        
        # Track zone possession time (if this is a new zone)
        if zone != self.last_zone and self.current_team is not None:
            if self.last_zone is not None:
                self.zone_changes += 1
        self.last_zone = zone
        
        # Calculate duration of current possession
        duration = 0.0
        if self.possession_start_time is not None:
            duration = timestamp - self.possession_start_time
        
        # Track high pressure events (3+ opponents pressing)
        if pressure >= 3:
            self.pressure_events.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'player_id': player_id,
                'team': team,
                'pressure_count': pressure,
                'zone': zone
            })
        
        return {
            'player_id': player_id,
            'team': team,
            'distance': distance,
            'pressure': pressure,
            'zone': zone,
            'duration': duration
        }
    
    def get_zone_statistics(self) -> Dict:
        """
        Get possession statistics by tactical zone.
        
        Returns:
            Dict with zone possession percentages for each team
        """
        stats = {}
        
        for team in ['A', 'B']:
            total_time = sum(self.zone_possession_time[team].values())
            
            if total_time > 0:
                stats[team] = {
                    zone: (time / total_time) * 100
                    for zone, time in self.zone_possession_time[team].items()
                }
            else:
                stats[team] = {
                    'Defensive': 0.0,
                    'Midfield': 0.0,
                    'Attacking': 0.0
                }
        
        return stats
    
    def get_pressure_statistics(self) -> Dict:
        """
        Get statistics about possession under pressure.
        
        Returns:
            Dict with pressure statistics
        """
        if not self.pressure_events:
            return {
                'total_high_pressure_events': 0,
                'avg_pressure_count': 0.0,
                'max_pressure_count': 0,
                'pressure_events': []
            }
        
        pressure_counts = [e['pressure_count'] for e in self.pressure_events]
        
        return {
            'total_high_pressure_events': len(self.pressure_events),
            'avg_pressure_count': sum(pressure_counts) / len(pressure_counts),
            'max_pressure_count': max(pressure_counts),
            'pressure_events': self.pressure_events[-20:]  # Last 20 events
        }
    
    def get_duration_statistics(self) -> Dict:
        """
        Get statistics about possession durations.
        
        Reveals playing style:
        - Short possessions (<2s) = direct, fast-paced play
        - Long possessions (>5s) = patient build-up play
        
        Returns:
            Dict with duration statistics
        """
        if not self.possession_durations:
            return {
                'avg_duration': 0.0,
                'max_duration': 0.0,
                'min_duration': 0.0,
                'total_possessions': 0,
                'short_possessions': 0,
                'medium_possessions': 0,
                'long_possessions': 0,
                'short_pct': 0.0,
                'medium_pct': 0.0,
                'long_pct': 0.0
            }
        
        durations = self.possession_durations
        
        # Categorize possessions
        short = [d for d in durations if d < 2.0]
        medium = [d for d in durations if 2.0 <= d < 5.0]
        long = [d for d in durations if d >= 5.0]
        
        return {
            'avg_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'total_possessions': len(durations),
            'short_possessions': len(short),  # <2s
            'medium_possessions': len(medium),  # 2-5s
            'long_possessions': len(long),  # >5s
            'short_pct': (len(short) / len(durations)) * 100,
            'medium_pct': (len(medium) / len(durations)) * 100,
            'long_pct': (len(long) / len(durations)) * 100
        }
