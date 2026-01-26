"""
Event Detector Module - Pass Detection

This module implements pass detection - identifying when ball moves from Player A to Player B.

PASS DETECTION LOGIC:
Pass detected when ALL conditions met:
1. Possession changed (A -> B)
2. Ball velocity > 2 m/s (not just rolling)
3. Distance > 5 meters (not a dribble or touch)
4. Distance < 45 meters (filter out tracking errors)
5. If same team -> successful pass
6. If different team -> interception
"""

from typing import Dict, List, Optional, Tuple, Any
import math


class EventDetector:
    """Detects football events, primarily passes."""

    def __init__(
        self,
        min_velocity_mps: float = 2.0,
        min_distance_m: float = 5.0,
        max_distance_m: float = 45.0
    ):
        self.min_velocity_mps = min_velocity_mps
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m

        # Pass tracking state
        self.last_possessor: Optional[int] = None
        self.last_possessor_team: Optional[str] = None
        self.last_possessor_position: Optional[Tuple[float, float]] = None
        self.last_ball_velocity_mps: float = 0.0

        # Event history
        self.pass_events: List[Dict[str, Any]] = []

        # Statistics
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

        # Distance tracking
        self.total_pass_distance = 0.0
        self.pass_distances: List[float] = []

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
                self._update_state(current_possessor, current_team,
                                  player_positions, ball_velocity_mps)
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
                    # Too short - dribble or close control, not a pass
                    self._update_state(current_possessor, current_team,
                                      player_positions, ball_velocity_mps)
                    return None

                if distance_m > self.max_distance_m:
                    # Too far - likely a tracking error, not a real pass
                    self._update_state(current_possessor, current_team,
                                      player_positions, ball_velocity_mps)
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

        self._update_state(current_possessor, current_team,
                          player_positions, ball_velocity_mps)

        return pass_event

    def _update_state(
        self,
        current_possessor: Optional[int],
        current_team: Optional[str],
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_velocity_mps: float
    ) -> None:
        self.last_possessor = current_possessor
        self.last_possessor_team = current_team
        self.last_ball_velocity_mps = ball_velocity_mps

        if current_possessor is not None and current_possessor in player_positions:
            self.last_possessor_position = player_positions[current_possessor][:2]
        else:
            self.last_possessor_position = None

    def _calculate_pass_direction(
        self,
        passer_pos: Tuple[float, float],
        receiver_pos: Tuple[float, float],
        passer_team: str,
        frame_width: int
    ) -> str:
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

    def get_pass_events(self) -> List[Dict[str, Any]]:
        return self.pass_events.copy()

    def get_pass_statistics(self) -> Dict[str, Any]:
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

    def get_passing_network(self) -> Dict[str, Any]:
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
        self.last_possessor = None
        self.last_possessor_team = None
        self.last_possessor_position = None
        self.last_ball_velocity_mps = 0.0
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
