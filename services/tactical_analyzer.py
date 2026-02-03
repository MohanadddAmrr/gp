"""
Tactical Analysis Integration Module

Combines all tactical modules to provide comprehensive tactical analysis:
- Formation detection
- Offside detection
- Set piece recognition
- Dribble detection
- Pressing intensity
- Transition tracking

This module serves as the main interface for all tactical analysis,
aggregating data from individual tactical modules and generating
comprehensive tactical reports.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from services.formation_detector import FormationDetector, FormationSnapshot
from services.offside_detector import OffsideDetector, PotentialOffside
from services.set_piece_detector import SetPieceDetector, SetPieceEvent, SetPieceType
from services.dribble_detector import DribbleDetector, DribbleEvent, TakeOnEvent
from services.pitch_transform import PitchTransform, TacticalHeatmapGenerator


@dataclass
class TacticalReport:
    """Comprehensive tactical report for a match or period."""
    timestamp: float
    duration: float
    
    # Formation analysis
    formations: Dict[str, str] = field(default_factory=dict)
    formation_changes: List[Dict] = field(default_factory=list)
    
    # Offside analysis
    offsides: int = 0
    potential_offsides: int = 0
    
    # Set pieces
    set_pieces: Dict[str, int] = field(default_factory=dict)
    
    # Dribbling
    dribbles: int = 0
    take_ons: int = 0
    dribble_success_rate: float = 0.0
    
    # Pressing
    pressing_intensity: Dict[str, float] = field(default_factory=dict)
    
    # Transitions
    transitions: Dict[str, int] = field(default_factory=dict)
    
    # Team shape
    team_shape: Dict[str, Dict] = field(default_factory=dict)


class TacticalAnalyzer:
    """
    Main tactical analysis interface combining all tactical modules.
    
    Integrates:
    - Formation detection
    - Offside detection
    - Set piece recognition
    - Dribble detection
    - Pressing intensity analysis
    - Transition tracking
    """

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        enable_formation: bool = True,
        enable_offside: bool = True,
        enable_set_pieces: bool = True,
        enable_dribbles: bool = True,
        enable_pitch_transform: bool = True,
    ):
        """
        Initialize tactical analyzer.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            enable_formation: Enable formation detection
            enable_offside: Enable offside detection
            enable_set_pieces: Enable set piece detection
            enable_dribbles: Enable dribble detection
            enable_pitch_transform: Enable pitch transform
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Initialize tactical modules
        self.formation_detector = FormationDetector(pitch_length_px=frame_width) if enable_formation else None
        self.offside_detector = OffsideDetector() if enable_offside else None
        self.set_piece_detector = SetPieceDetector() if enable_set_pieces else None
        self.dribble_detector = DribbleDetector() if enable_dribbles else None
        self.pitch_transform = PitchTransform() if enable_pitch_transform else None
        self.heatmap_generator = None
        if enable_pitch_transform:
            self.pitch_transform.calibrate_from_points([
                (0, 0),
                (frame_width / 2, 0),
                (frame_width, 0),
                (0, frame_height / 2),
                (frame_width / 2, frame_height / 2),
                (frame_width, frame_height / 2),
                (0, frame_height),
                (frame_width / 2, frame_height),
                (frame_width, frame_height),
            ])
            self.heatmap_generator = TacticalHeatmapGenerator(self.pitch_transform)

        # Set piece detector frame dimensions
        if self.set_piece_detector:
            self.set_piece_detector.set_frame_dimensions(frame_width, frame_height)

        # Transition tracking
        self._last_possession_team: Optional[str] = None
        self._transitions: List[Dict[str, Any]] = []
        self._transition_counts: Dict[str, int] = {'defense_to_attack': 0, 'attack_to_defense': 0}

        # Pressing intensity tracking
        self._pressing_events: List[Dict] = []
        self._pressing_by_team: Dict[str, List[float]] = {'A': [], 'B': []}

        # Match data
        self._start_time: Optional[float] = None
        self._frame_count: int = 0
        
        
    def calculate_ppda(self) -> Dict[str, Any]:
        """
        Calculate PPDA (Passes Per Defensive Action) for both teams.
        
        PPDA = Opponent passes in their defensive 60% / Defensive actions
        Lower PPDA = More aggressive pressing
        
        Returns:
            Dict with PPDA stats for both teams
        """
        # Count passes and defensive actions by zone
        team_a_passes_allowed = 0  # Passes by B in B's defensive zone
        team_b_passes_allowed = 0  # Passes by A in A's defensive zone
        team_a_def_actions = max(1, len([e for e in self._pressing_events if e.get('team') == 'A']))
        team_b_def_actions = max(1, len([e for e in self._pressing_events if e.get('team') == 'B']))
        
        # Get pass counts from event history if available
        if hasattr(self, '_pass_events'):
            for pass_event in self._pass_events:
                passer_team = pass_event.get('passer_team')
                x_pos = pass_event.get('x', 0.5)
                
                # Team A's defensive zone is x < 0.4 (left side)
                # Team B's defensive zone is x > 0.6 (right side)
                if passer_team == 'A' and x_pos < 0.4:
                    team_b_passes_allowed += 1
                elif passer_team == 'B' and x_pos > 0.6:
                    team_a_passes_allowed += 1
        
        # Calculate PPDA
        ppda_team_a = team_b_passes_allowed / team_a_def_actions
        ppda_team_b = team_a_passes_allowed / team_b_def_actions
        
        # Interpret pressing intensity
        def interpret_ppda(ppda: float) -> str:
            if ppda < 6:
                return "Very High Press"
            elif ppda < 10:
                return "High Press"
            elif ppda < 15:
                return "Medium Press"
            elif ppda < 20:
                return "Low Press"
            else:
                return "Very Low Press"
        
        return {
            'team_a': {
                'ppda': round(ppda_team_a, 2),
                'interpretation': interpret_ppda(ppda_team_a),
                'defensive_actions': team_a_def_actions,
                'opponent_passes_allowed': team_b_passes_allowed
            },
            'team_b': {
                'ppda': round(ppda_team_b, 2),
                'interpretation': interpret_ppda(ppda_team_b),
                'defensive_actions': team_b_def_actions,
                'opponent_passes_allowed': team_a_passes_allowed
            }
        }
    
    def get_pressing_zones(self) -> Dict[str, Any]:
        """
        Analyze where each team applies pressing.
        
        Returns:
            Dict with pressing zone analysis
        """
        zones = {
            'A': {'high': 0, 'mid': 0, 'low': 0},
            'B': {'high': 0, 'mid': 0, 'low': 0}
        }
        
        for event in self._pressing_events:
            team = event.get('team')
            x = event.get('x', 0.5)
            
            if team not in zones:
                continue
            
            # Determine zone based on position
            if team == 'A':
                # Team A attacks right, so high press is x > 0.6
                if x > 0.6:
                    zones[team]['high'] += 1
                elif x > 0.4:
                    zones[team]['mid'] += 1
                else:
                    zones[team]['low'] += 1
            else:
                # Team B attacks left, so high press is x < 0.4
                if x < 0.4:
                    zones[team]['high'] += 1
                elif x < 0.6:
                    zones[team]['mid'] += 1
                else:
                    zones[team]['low'] += 1
        
        return zones


    def update(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        ball_position: Optional[Tuple[float, float]],
        ball_velocity: float,
        current_possessor: Optional[int],
        current_team: Optional[str],
        frame_idx: int,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Update all tactical analysis modules.
        
        Args:
            player_positions: {player_id: (x, y, team)}
            ball_position: (x, y) ball position or None
            ball_velocity: Ball velocity
            current_possessor: Current ball possessor ID
            current_team: Current team in possession
            frame_idx: Current frame index
            timestamp: Current timestamp
            
        Returns:
            Dictionary of detected events in this frame
        """
        events = {
            'formation': None,
            'offside': None,
            'set_piece': None,
            'dribble': None,
            'transition': None
        }

        if self._start_time is None:
            self._start_time = timestamp

        self._frame_count = frame_idx

        # Update formation detector
        if self.formation_detector:
            for player_id, (x, y, team) in player_positions.items():
                self.formation_detector.update_player_position(
                    player_id, (x, y), team, timestamp, frame_idx
                )

            # Detect formations periodically (every 30 frames ~ 1 second)
            if frame_idx % 30 == 0:
                for team in ['A', 'B']:
                    formation = self.formation_detector.detect_formation(
                        team, frame_idx, timestamp
                    )
                    if formation:
                        events['formation'] = formation

        # Update offside detector
        if self.offside_detector:
            self.offside_detector.update_player_positions(
                player_positions, frame_idx, timestamp
            )

            if ball_position:
                offside_event = self.offside_detector.update_ball_state(
                    ball_position, ball_velocity, frame_idx, timestamp
                )
                if offside_event:
                    events['offside'] = offside_event

        # Update set piece detector
        if self.set_piece_detector and ball_position:
            set_piece = self.set_piece_detector.update(
                ball_position, ball_velocity, player_positions,
                frame_idx, timestamp, current_possessor, current_team
            )
            if set_piece:
                events['set_piece'] = set_piece

        # Update dribble detector
        if self.dribble_detector and current_possessor and ball_position:
            # Get current possessor position
            if current_possessor in player_positions:
                px, py, team = player_positions[current_possessor]
                dribble = self.dribble_detector.update(
                    current_possessor, (px, py), team,
                    ball_position, ball_velocity, player_positions,
                    frame_idx, timestamp
                )
                if dribble:
                    events['dribble'] = dribble

        # Track transitions
        if current_team and current_team != self._last_possession_team:
            if self._last_possession_team is not None:
                transition = self._detect_transition(
                    self._last_possession_team, current_team, timestamp, frame_idx
                )
                if transition:
                    events['transition'] = transition
            self._last_possession_team = current_team

        # Track pressing intensity
        self._track_pressing(player_positions, current_possessor, current_team, timestamp)

        # Update heatmap generator
        if self.heatmap_generator:
            for player_id, (x, y, team) in player_positions.items():
                self.heatmap_generator.add_position(player_id, (x, y), team)

        return events

    def _detect_transition(
        self,
        from_team: str,
        to_team: str,
        timestamp: float,
        frame_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect possession transition (defense to attack or vice versa).
        
        Returns:
            Transition event dict or None
        """
        if from_team == to_team:
            return None

        # Determine transition type based on zones
        # Simplified: any possession change is a transition
        transition_type = 'possession_change'

        event = {
            'type': transition_type,
            'timestamp': timestamp,
            'frame_idx': frame_idx,
            'from_team': from_team,
            'to_team': to_team
        }

        self._transitions.append(event)

        # Categorize as defense->attack or attack->defense
        # This is simplified - would need zone info for accurate categorization
        if from_team == 'A':
            self._transition_counts['attack_to_defense'] += 1
        else:
            self._transition_counts['defense_to_attack'] += 1

        return event

    def _track_pressing(
        self,
        player_positions: Dict[int, Tuple[float, float, str]],
        possessor_id: Optional[int],
        possessor_team: Optional[str],
        timestamp: float
    ) -> None:
        """Track pressing intensity."""
        if possessor_id is None or possessor_team is None:
            return

        # Get possessor position
        if possessor_id not in player_positions:
            return

        px, py, _ = player_positions[possessor_id]

        # Count opponents within pressing distance
        pressing_distance = 150.0  # pixels
        pressing_count = 0

        for player_id, (x, y, team) in player_positions.items():
            if team == possessor_team or player_id == possessor_id:
                continue

            distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if distance <= pressing_distance:
                pressing_count += 1

        # Record pressing intensity for opposing team
        opposing_team = 'B' if possessor_team == 'A' else 'A'
        self._pressing_by_team[opposing_team].append(pressing_count)

        # Keep only recent history
        max_history = 300  # 10 seconds at 30fps
        if len(self._pressing_by_team[opposing_team]) > max_history:
            self._pressing_by_team[opposing_team] = self._pressing_by_team[opposing_team][-max_history:]

    def get_current_formations(self) -> Dict[str, Optional[str]]:
        """Get current formations for both teams."""
        if not self.formation_detector:
            return {'A': None, 'B': None}

        return {
            'A': self.formation_detector.get_current_formation('A'),
            'B': self.formation_detector.get_current_formation('B')
        }

    def get_formation_statistics(self) -> Dict[str, Any]:
        """Get formation statistics."""
        if not self.formation_detector:
            return {}

        return self.formation_detector.get_formation_statistics()

    def get_team_shape_metrics(self) -> Dict[str, Dict]:
        """Get team shape metrics for both teams."""
        if not self.formation_detector:
            return {}

        return {
            'A': self.formation_detector.get_team_shape_metrics('A'),
            'B': self.formation_detector.get_team_shape_metrics('B')
        }

    def get_offside_statistics(self) -> Dict[str, Any]:
        """Get offside detection statistics."""
        if not self.offside_detector:
            return {}

        return self.offside_detector.get_statistics()

    def get_set_piece_statistics(self) -> Dict[str, Any]:
        """Get set piece statistics."""
        if not self.set_piece_detector:
            return {}

        return self.set_piece_detector.get_statistics()

    def get_dribble_statistics(self) -> Dict[str, Any]:
        """Get dribble statistics."""
        if not self.dribble_detector:
            return {}

        return self.dribble_detector.get_statistics()

    def get_pressing_intensity(self) -> Dict[str, float]:
        """
        Get current pressing intensity for both teams.
        
        Returns:
            Dictionary with pressing intensity scores
        """
        intensity = {}

        for team in ['A', 'B']:
            pressures = self._pressing_by_team[team]
            if pressures:
                avg_pressure = sum(pressures) / len(pressures)
                # Normalize to 0-100 scale (assuming max 5 players pressing)
                intensity[team] = min(100, (avg_pressure / 5.0) * 100)
            else:
                intensity[team] = 0.0

        return intensity

    def get_transition_statistics(self) -> Dict[str, Any]:
        """Get transition statistics."""
        return {
            'total_transitions': len(self._transitions),
            'defense_to_attack': self._transition_counts['defense_to_attack'],
            'attack_to_defense': self._transition_counts['attack_to_defense'],
            'recent_transitions': self._transitions[-10:]  # Last 10
        }

    def get_tactical_heatmaps(self) -> Dict[str, np.ndarray]:
        """Get tactical heatmaps for both teams."""
        if not self.heatmap_generator:
            return {}

        return {
            'team_A': self.heatmap_generator.get_team_heatmap('A'),
            'team_B': self.heatmap_generator.get_team_heatmap('B')
        }

    def generate_report(self, end_time: float) -> TacticalReport:
        """
        Generate comprehensive tactical report.
        
        Args:
            end_time: End timestamp
            
        Returns:
            TacticalReport object
        """
        duration = end_time - (self._start_time or end_time)

        # Get current formations
        current_formations = self.get_current_formations()

        # Get formation changes
        formation_changes = []
        if self.formation_detector:
            formation_changes = self.formation_detector.get_formation_changes()

        # Get offside stats
        offside_stats = self.get_offside_statistics()

        # Get set piece stats
        set_piece_stats = self.get_set_piece_statistics()

        # Get dribble stats
        dribble_stats = self.get_dribble_statistics()

        # Get pressing intensity
        pressing = self.get_pressing_intensity()

        # Get transitions
        transitions = self.get_transition_statistics()

        # Get team shape
        shape = self.get_team_shape_metrics()

        report = TacticalReport(
            timestamp=end_time,
            duration=duration,
            formations=current_formations,
            formation_changes=formation_changes,
            offsides=offside_stats.get('confirmed_offsides', 0),
            potential_offsides=offside_stats.get('total_potential_offsides', 0),
            set_pieces=set_piece_stats.get('by_type', {}),
            dribbles=dribble_stats.get('total_dribbles', 0),
            take_ons=dribble_stats.get('total_take_ons', 0),
            dribble_success_rate=dribble_stats.get('success_rate', 0.0),
            pressing_intensity=pressing,
            transitions=transitions,
            team_shape=shape
        )

        return report

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get all tactical statistics combined."""
        return {
            'formations': self.get_formation_statistics(),
            'team_shape': self.get_team_shape_metrics(),
            'offsides': self.get_offside_statistics(),
            'set_pieces': self.get_set_piece_statistics(),
            'dribbles': self.get_dribble_statistics(),
            'pressing_intensity': self.get_pressing_intensity(),
            'transitions': self.get_transition_statistics()
        }

    def finalize(self, end_time: float) -> TacticalReport:
        """
        Finalize analysis and generate report.
        
        Args:
            end_time: End timestamp
            
        Returns:
            TacticalReport object
        """
        # Finalize dribble detector
        if self.dribble_detector:
            self.dribble_detector.finalize_all(self._frame_count, end_time)

        return self.generate_report(end_time)

    def reset(self) -> None:
        """Reset all tactical analysis."""
        if self.formation_detector:
            self.formation_detector.reset()
        if self.offside_detector:
            self.offside_detector.reset()
        if self.set_piece_detector:
            self.set_piece_detector.reset()
        if self.dribble_detector:
            self.dribble_detector.reset()
        if self.heatmap_generator:
            self.heatmap_generator.reset()

        self._last_possession_team = None
        self._transitions.clear()
        self._transition_counts = {'defense_to_attack': 0, 'attack_to_defense': 0}
        self._pressing_events.clear()
        self._pressing_by_team = {'A': [], 'B': []}
        self._start_time = None
        self._frame_count = 0
