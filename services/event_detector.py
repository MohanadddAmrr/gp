"""
Event Detector Module - Pass and Shot Detection

This module implements:
1. Pass detection - identifying when ball moves from Player A to Player B
2. Shot detection - identifying when ball moves rapidly toward goal

PASS DETECTION LOGIC:
Pass detected when ALL conditions met:
1. Possession changed (A -> B)
2. Ball velocity > 2 m/s (not just rolling)
3. Distance > 5 meters (not a dribble or touch)
4. Distance < 45 meters (filter out tracking errors)
5. If same team -> successful pass
6. If different team -> interception

SHOT DETECTION LOGIC:
Shot detected when ALL conditions met:
1. Ball velocity > 10 m/s (fast enough to be intentional)
2. Ball velocity < 50 m/s (filter out tracking errors)
3. Ball direction points toward goal (±30 degrees)
4. Player had possession recently (within last 1 second)
5. Ball is in attacking third of the field
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np


class EventDetector:
    """Detects football events: passes and shots."""

    def __init__(
        self,
        min_velocity_mps: float = 2.0,
        min_distance_m: float = 5.0,
        max_distance_m: float = 45.0,
        shot_velocity_threshold_mps: float = 5.0,
        shot_max_velocity_mps: float = 200.0,
        shot_angle_threshold_deg: float = 30.0,
        possession_memory_sec: float = 1.0
    ):
        """
        Initialize the event detector.
        
        Args:
            min_velocity_mps: Min ball velocity for pass detection (m/s)
            min_distance_m: Min distance for pass detection (m)
            max_distance_m: Max distance for pass detection (m)
            shot_velocity_threshold_mps: Min ball velocity for shot detection (m/s)
            shot_max_velocity_mps: Max realistic ball velocity for shots (m/s) - filters noise
            shot_angle_threshold_deg: Max angle deviation from goal for shots (degrees)
            possession_memory_sec: Time window to remember last possessor for shots (seconds)
        """
        # Pass detection parameters
        self.min_velocity_mps = min_velocity_mps
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m

        # Shot detection parameters
        self.shot_velocity_threshold_mps = shot_velocity_threshold_mps
        self.shot_max_velocity_mps = shot_max_velocity_mps
        self.shot_angle_threshold_deg = shot_angle_threshold_deg
        self.possession_memory_sec = possession_memory_sec

        # Pass tracking state
        self.last_possessor: Optional[int] = None
        self.last_possessor_team: Optional[str] = None
        self.last_possessor_position: Optional[Tuple[float, float]] = None
        self.last_ball_velocity_mps: float = 0.0

        # Shot tracking state
        self.last_possession_time: float = 0.0
        self.last_possession_player: Optional[int] = None
        self.last_possession_team: Optional[str] = None
        self.last_shot_frame: int = -100  # Cooldown to prevent duplicate shots

        # Event history
        self.pass_events: List[Dict[str, Any]] = []
        self.shot_events: List[Dict[str, Any]] = []

        # Pass statistics
        self.total_passes = 0
        self.completed_passes = 0
        self.intercepted_passes = 0
        self.team_passes = {'A': {'attempted': 0, 'completed': 0},
                           'B': {'attempted': 0, 'completed': 0}}
        self.player_passes: Dict[int, Dict[str, int]] = {}

        # Pass direction tracking
        self.forward_passes = 0
        self.backward_passes = 0
        self.lateral_passes = 0

        # Pass distance tracking
        self.total_pass_distance = 0.0
        self.pass_distances: List[float] = []

        # Shot statistics
        self.total_shots = 0
        self.team_shots = {'A': 0, 'B': 0}
        self.player_shots: Dict[int, int] = {}
        self.shot_velocities: List[float] = []

    def detect_pass(
        self,
        current_possessor: Optional[int],
        current_team: Optional[str],
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_velocity_mps: float,
        meter_per_px: float,
        frame_idx: int,
        timestamp: float,
        frame_width: int = 1920
    ) -> Optional[Dict[str, Any]]:
        """
        Detect pass events when possession changes.
        
        Returns pass event dict if pass detected, None otherwise.
        """
        pass_event = None

        # Check if possession changed
        if (current_possessor is not None and
            self.last_possessor is not None and
            current_possessor != self.last_possessor):

            # Condition 1: Ball must be moving fast enough
            if self.last_ball_velocity_mps < self.min_velocity_mps:
                self._update_pass_state(current_possessor, current_team,
                                        player_positions, ball_velocity_mps, timestamp)
                return None

            # Condition 2: Distance between passer and receiver
            if (self.last_possessor_position is not None and
                current_possessor in player_positions):

                receiver_pos = player_positions[current_possessor][:2]

                distance_px = math.sqrt(
                    (receiver_pos[0] - self.last_possessor_position[0]) ** 2 +
                    (receiver_pos[1] - self.last_possessor_position[1]) ** 2
                )

                distance_m = distance_px * meter_per_px

                if distance_m < self.min_distance_m:
                    self._update_pass_state(current_possessor, current_team,
                                            player_positions, ball_velocity_mps, timestamp)
                    return None

                if distance_m > self.max_distance_m:
                    self._update_pass_state(current_possessor, current_team,
                                            player_positions, ball_velocity_mps, timestamp)
                    return None

                # Determine outcome
                outcome = 'complete' if self.last_possessor_team == current_team else 'intercepted'

                # Calculate direction
                direction = self._calculate_pass_direction(
                    self.last_possessor_position,
                    receiver_pos,
                    self.last_possessor_team,
                    frame_width
                )

                pass_event = {
                    'type': 'pass',
                    'passer_id': self.last_possessor,
                    'passer_team': self.last_possessor_team,
                    'receiver_id': current_possessor,
                    'receiver_team': current_team,
                    'distance_m': round(distance_m, 2),
                    'velocity_mps': round(self.last_ball_velocity_mps, 2),
                    'outcome': outcome,
                    'direction': direction,
                    'frame': frame_idx,
                    'timestamp': round(timestamp, 3)
                }

                self._record_pass(pass_event)

        self._update_pass_state(current_possessor, current_team,
                                player_positions, ball_velocity_mps, timestamp)

        return pass_event

    def detect_shot(
        self,
        ball_position: Optional[Tuple[float, float]],
        ball_direction: Tuple[float, float],
        ball_velocity_mps: float,
        frame_idx: int,
        timestamp: float,
        frame_width: int,
        frame_height: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect shot attempts toward goal.
        
        Shot detected when:
        1. Ball velocity > threshold (10 m/s default)
        2. Ball velocity < max threshold (50 m/s) - filters tracking noise
        3. Ball direction points toward goal (±30 degrees)
        4. Player had possession recently (within 1 second)
        5. Ball is in attacking third
        
        Args:
            ball_position: Current ball (x, y) position
            ball_direction: Normalized direction vector (dx, dy)
            ball_velocity_mps: Ball velocity in meters/second
            frame_idx: Current frame number
            timestamp: Current timestamp
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Shot event dict if shot detected, None otherwise
        """
        # Cooldown: prevent detecting same shot multiple times
        if frame_idx - self.last_shot_frame < 25:  # ~1 second at 25fps
            return None
        
        # Condition 1: Ball must be moving fast enough
        if ball_velocity_mps < self.shot_velocity_threshold_mps:
            return None
        
        # Condition 2: Filter out unrealistic velocities (tracking noise/errors)
        # Professional shots max out around 35-40 m/s (140 km/h)
        if ball_velocity_mps > self.shot_max_velocity_mps:
            return None
        
        # Condition 3: Must have had possession recently
        if self.last_possession_player is None:
            return None
        
        time_since_possession = timestamp - self.last_possession_time
        if time_since_possession > self.possession_memory_sec:
            return None
        
        if ball_position is None:
            return None
        
        ball_x, ball_y = ball_position
        
        # Determine which goal to check based on last possessor's team
        # Team A attacks right (goal at x=width), Team B attacks left (goal at x=0)
        if self.last_possession_team == 'A':
            goal_x = frame_width
            goal_y = frame_height / 2.0
            # Ball should be in attacking third (right side)
            attacking_third_start = frame_width * (2.0 / 3.0)
            if ball_x < attacking_third_start:
                return None  # Not in attacking third
        else:  # Team B
            goal_x = 0
            goal_y = frame_height / 2.0
            # Ball should be in attacking third (left side)
            attacking_third_end = frame_width * (1.0 / 3.0)
            if ball_x > attacking_third_end:
                return None  # Not in attacking third
        
        # Condition 4: Ball direction must point toward goal
        # Calculate vector from ball to goal
        to_goal_x = goal_x - ball_x
        to_goal_y = goal_y - ball_y
        
        # Normalize the to-goal vector
        to_goal_magnitude = math.sqrt(to_goal_x**2 + to_goal_y**2)
        if to_goal_magnitude < 0.001:
            return None
        
        to_goal_x /= to_goal_magnitude
        to_goal_y /= to_goal_magnitude
        
        # Get ball direction (already normalized)
        dir_x, dir_y = ball_direction
        
        # Check if direction vector is valid
        dir_magnitude = math.sqrt(dir_x**2 + dir_y**2)
        if dir_magnitude < 0.001:
            return None
        
        # Calculate angle between ball direction and goal direction
        # Using dot product: cos(angle) = a · b / (|a| |b|)
        dot_product = dir_x * to_goal_x + dir_y * to_goal_y
        
        # Clamp to avoid floating point errors with acos
        dot_product = max(-1.0, min(1.0, dot_product))
        
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Check if angle is within threshold
        if angle_deg > self.shot_angle_threshold_deg:
            return None
        
        # Calculate distance to goal
        distance_to_goal_px = math.sqrt(
            (goal_x - ball_x)**2 + (goal_y - ball_y)**2
        )
        
        # Shot detected!
        shot_event = {
            'type': 'shot',
            'shooter_id': self.last_possession_player,
            'shooter_team': self.last_possession_team,
            'velocity_mps': round(ball_velocity_mps, 2),
            'angle_to_goal_deg': round(angle_deg, 1),
            'distance_to_goal_px': round(distance_to_goal_px, 1),
            'ball_position': (round(ball_x, 1), round(ball_y, 1)),
            'frame': frame_idx,
            'timestamp': round(timestamp, 3)
        }
        
        self._record_shot(shot_event)
        self.last_shot_frame = frame_idx
        
        return shot_event

    def _update_pass_state(
        self,
        current_possessor: Optional[int],
        current_team: Optional[str],
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_velocity_mps: float,
        timestamp: float
    ) -> None:
        """Update internal state for pass detection."""
        self.last_possessor = current_possessor
        self.last_possessor_team = current_team
        self.last_ball_velocity_mps = ball_velocity_mps

        if current_possessor is not None and current_possessor in player_positions:
            self.last_possessor_position = player_positions[current_possessor][:2]
            # Also update shot tracking state
            self.last_possession_time = timestamp
            self.last_possession_player = current_possessor
            self.last_possession_team = current_team
        else:
            self.last_possessor_position = None

    def _calculate_pass_direction(
        self,
        passer_pos: Tuple[float, float],
        receiver_pos: Tuple[float, float],
        passer_team: str,
        frame_width: int
    ) -> str:
        """Calculate pass direction relative to attacking goal."""
        dx = receiver_pos[0] - passer_pos[0]
        dy = receiver_pos[1] - passer_pos[1]

        if passer_team == 'B':
            dx = -dx

        if abs(dx) < 0.001:
            return 'lateral'

        ratio = abs(dy) / abs(dx)

        if ratio > 2:
            return 'lateral'
        elif dx > 0:
            return 'forward'
        else:
            return 'backward'

    def _record_pass(self, pass_event: Dict[str, Any]) -> None:
        """Record pass event and update statistics."""
        self.pass_events.append(pass_event)

        passer_id = pass_event['passer_id']
        passer_team = pass_event['passer_team']
        outcome = pass_event['outcome']
        direction = pass_event['direction']
        distance = pass_event['distance_m']

        self.total_passes += 1
        if outcome == 'complete':
            self.completed_passes += 1
        else:
            self.intercepted_passes += 1

        if passer_team in self.team_passes:
            self.team_passes[passer_team]['attempted'] += 1
            if outcome == 'complete':
                self.team_passes[passer_team]['completed'] += 1

        if passer_id not in self.player_passes:
            self.player_passes[passer_id] = {'attempted': 0, 'completed': 0}
        self.player_passes[passer_id]['attempted'] += 1
        if outcome == 'complete':
            self.player_passes[passer_id]['completed'] += 1

        if direction == 'forward':
            self.forward_passes += 1
        elif direction == 'backward':
            self.backward_passes += 1
        else:
            self.lateral_passes += 1

        self.total_pass_distance += distance
        self.pass_distances.append(distance)

    def _record_shot(self, shot_event: Dict[str, Any]) -> None:
        """Record shot event and update statistics."""
        self.shot_events.append(shot_event)
        
        shooter_id = shot_event['shooter_id']
        shooter_team = shot_event['shooter_team']
        velocity = shot_event['velocity_mps']
        
        self.total_shots += 1
        
        if shooter_team in self.team_shots:
            self.team_shots[shooter_team] += 1
        
        if shooter_id not in self.player_shots:
            self.player_shots[shooter_id] = 0
        self.player_shots[shooter_id] += 1
        
        self.shot_velocities.append(velocity)

    def get_pass_events(self) -> List[Dict[str, Any]]:
        """Get all detected pass events."""
        return self.pass_events.copy()

    def get_shot_events(self) -> List[Dict[str, Any]]:
        """Get all detected shot events."""
        return self.shot_events.copy()

    def get_pass_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pass statistics."""
        accuracy = (self.completed_passes / self.total_passes * 100
                   if self.total_passes > 0 else 0.0)

        team_accuracy = {}
        for team, stats in self.team_passes.items():
            if stats['attempted'] > 0:
                team_accuracy[team] = round(stats['completed'] / stats['attempted'] * 100, 1)
            else:
                team_accuracy[team] = 0.0

        avg_distance = (sum(self.pass_distances) / len(self.pass_distances)
                       if self.pass_distances else 0.0)

        if self.total_passes > 0:
            forward_pct = round(self.forward_passes / self.total_passes * 100, 1)
            backward_pct = round(self.backward_passes / self.total_passes * 100, 1)
            lateral_pct = round(self.lateral_passes / self.total_passes * 100, 1)
        else:
            forward_pct = backward_pct = lateral_pct = 0.0

        return {
            'total_passes': self.total_passes,
            'completed_passes': self.completed_passes,
            'intercepted_passes': self.intercepted_passes,
            'pass_accuracy': round(accuracy, 1),
            'team_passes': self.team_passes,
            'team_accuracy': team_accuracy,
            'direction': {
                'forward': self.forward_passes,
                'backward': self.backward_passes,
                'lateral': self.lateral_passes,
                'forward_pct': forward_pct,
                'backward_pct': backward_pct,
                'lateral_pct': lateral_pct
            },
            'distance': {
                'avg_m': round(avg_distance, 1),
                'max_m': round(max(self.pass_distances), 1) if self.pass_distances else 0.0,
                'min_m': round(min(self.pass_distances), 1) if self.pass_distances else 0.0
            },
            'player_passes': {
                pid: {
                    'attempted': s['attempted'],
                    'completed': s['completed'],
                    'accuracy': round(s['completed'] / s['attempted'] * 100, 1) if s['attempted'] > 0 else 0.0
                }
                for pid, s in self.player_passes.items()
            }
        }

    def get_shot_statistics(self) -> Dict[str, Any]:
        """Get comprehensive shot statistics."""
        avg_velocity = (sum(self.shot_velocities) / len(self.shot_velocities)
                       if self.shot_velocities else 0.0)
        
        return {
            'total_shots': self.total_shots,
            'team_shots': self.team_shots.copy(),
            'player_shots': {
                pid: count for pid, count in self.player_shots.items()
            },
            'velocity': {
                'avg_mps': round(avg_velocity, 1),
                'max_mps': round(max(self.shot_velocities), 1) if self.shot_velocities else 0.0,
                'min_mps': round(min(self.shot_velocities), 1) if self.shot_velocities else 0.0
            }
        }

    def get_passing_network(self) -> Dict[str, Any]:
        """Build passing network from completed passes."""
        pass_counts: Dict[Tuple[int, int], int] = {}

        for event in self.pass_events:
            if event['outcome'] == 'complete':
                pair = (event['passer_id'], event['receiver_id'])
                pass_counts[pair] = pass_counts.get(pair, 0) + 1

        edges = [{'from': p[0], 'to': p[1], 'count': c} for p, c in pass_counts.items()]
        edges.sort(key=lambda x: x['count'], reverse=True)

        nodes = set()
        for pair in pass_counts.keys():
            nodes.add(pair[0])
            nodes.add(pair[1])

        return {
            'nodes': list(nodes),
            'edges': edges,
            'most_common_combinations': edges[:5] if edges else []
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        # Pass state
        self.last_possessor = None
        self.last_possessor_team = None
        self.last_possessor_position = None
        self.last_ball_velocity_mps = 0.0
        
        # Shot state
        self.last_possession_time = 0.0
        self.last_possession_player = None
        self.last_possession_team = None
        self.last_shot_frame = -100
        
        # Pass events and stats
        self.pass_events.clear()
        self.total_passes = 0
        self.completed_passes = 0
        self.intercepted_passes = 0
        self.team_passes = {'A': {'attempted': 0, 'completed': 0},
                           'B': {'attempted': 0, 'completed': 0}}
        self.player_passes.clear()
        self.forward_passes = 0
        self.backward_passes = 0
        self.lateral_passes = 0
        self.total_pass_distance = 0.0
        self.pass_distances.clear()
        
        # Shot events and stats
        self.shot_events.clear()
        self.total_shots = 0
        self.team_shots = {'A': 0, 'B': 0}
        self.player_shots.clear()
        self.shot_velocities.clear()
