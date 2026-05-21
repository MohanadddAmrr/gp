"""
Expected Goals (xG) Calculator Module

Calculates expected goals based on shot position, angle, body part, and context.
Uses statistical models to estimate the probability of a shot resulting in a goal.

Key Features:
- Shot quality calculation based on position and angle
- Body part tracking (foot, head, other)
- Shot type classification (open play, set piece, header)
- xG comparison with actual goals
- Shot quality trends over time
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from collections import defaultdict


class ShotType(Enum):
    """Types of shots."""
    OPEN_PLAY = "open_play"
    SET_PIECE = "set_piece"
    HEADER = "header"
    PENALTY = "penalty"
    FREE_KICK = "free_kick"
    CORNER = "corner"


class BodyPart(Enum):
    """Body parts used for shots."""
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    HEAD = "head"
    OTHER = "other"


class ShotOutcome(Enum):
    """Outcomes of shots."""
    GOAL = "goal"
    SAVED = "saved"
    BLOCKED = "blocked"
    OFF_TARGET = "off_target"
    POST = "post"
    UNKNOWN = "unknown"


@dataclass
class ShotEvent:
    """Represents a shot event with all relevant data."""
    timestamp: float
    frame: int
    shooter_id: int
    shooter_team: str
    
    # Position (normalized 0-1)
    x: float
    y: float
    
    # Shot characteristics
    shot_type: ShotType = ShotType.OPEN_PLAY
    body_part: BodyPart = BodyPart.RIGHT_FOOT
    outcome: ShotOutcome = ShotOutcome.UNKNOWN
    
    # Shot metrics
    distance_to_goal: float = 0.0  # meters
    angle_to_goal: float = 0.0  # degrees
    velocity_mps: float = 0.0
    
    # Context
    big_chance: bool = False
    assisted: bool = False
    assist_type: Optional[str] = None  # 'pass', 'cross', 'through_ball'
    pressure_level: int = 0  # 0-3 scale
    
    # xG
    xg_value: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class xGStats:
    """Statistics for xG analysis."""
    total_shots: int = 0
    total_xg: float = 0.0
    actual_goals: int = 0
    
    # By shot type
    xg_by_type: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    shots_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    goals_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # By body part
    xg_by_body_part: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    shots_by_body_part: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # By team
    xg_by_team: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    shots_by_team: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    goals_by_team: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Quality metrics
    big_chances: int = 0
    big_chances_converted: int = 0
    
    # Time series
    xg_timeline: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, cumulative_xg)


class xGCalculator:
    """
    Expected Goals Calculator.
    
    Calculates xG using a statistical model based on:
    - Distance to goal
    - Angle to goal
    - Shot type
    - Body part
    - Context (big chance, pressure, etc.)
    
    The model uses base probabilities adjusted by various factors.
    """
    
    # Pitch dimensions (meters)
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    GOAL_WIDTH = 7.32
    GOAL_DEPTH = 2.44
    
    # Goal positions (normalized coordinates)
    GOAL_LEFT = {'x': 0.0, 'y': 0.5}
    GOAL_RIGHT = {'x': 1.0, 'y': 0.5}
    
    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        goal_width_m: float = 7.32,
        distance_decay: float = 0.08,  # Changed from 0.12 - less aggressive decay
        angle_weight: float = 0.4,      # Changed from 0.3 - more weight on angle
        min_xg: float = 0.01,
        max_xg: float = 0.95
    ):
        self.PITCH_LENGTH = pitch_length_m
        self.PITCH_WIDTH = pitch_width_m
        self.GOAL_WIDTH = goal_width_m
        self.distance_decay = distance_decay
        self.angle_weight = angle_weight
        self.min_xg = min_xg
        self.max_xg = max_xg
        
        # Initialize statistics
        self.stats = xGStats()
        self.shots: List[ShotEvent] = []
        
    def calculate_distance_to_goal(self, x: float, y: float, attacking_right: bool = True) -> float:
        """
        Calculate distance from shot position to goal center.
        
        Args:
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            attacking_right: Whether attacking the right goal
            
        Returns:
            Distance in meters
        """
        goal = self.GOAL_RIGHT if attacking_right else self.GOAL_LEFT
        
        # Convert to meters
        x_m = x * self.PITCH_LENGTH
        y_m = y * self.PITCH_WIDTH
        goal_x_m = goal['x'] * self.PITCH_LENGTH
        goal_y_m = goal['y'] * self.PITCH_WIDTH
        
        distance = math.sqrt((x_m - goal_x_m) ** 2 + (y_m - goal_y_m) ** 2)
        return distance
    
    def calculate_angle_to_goal(self, x: float, y: float, attacking_right: bool = True) -> float:
        """
        Calculate angle to goal from shot position.
        
        Args:
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            attacking_right: Whether attacking the right goal
            
        Returns:
            Angle in degrees (0-180)
        """
        goal = self.GOAL_RIGHT if attacking_right else self.GOAL_LEFT
        
        # Convert to meters
        x_m = x * self.PITCH_LENGTH
        y_m = y * self.PITCH_WIDTH
        goal_x_m = goal['x'] * self.PITCH_LENGTH
        goal_y_m_left = (goal['y'] * self.PITCH_WIDTH) - (self.GOAL_WIDTH / 2)
        goal_y_m_right = (goal['y'] * self.PITCH_WIDTH) + (self.GOAL_WIDTH / 2)
        
        # Calculate angles to goal posts
        angle_left = math.atan2(goal_y_m_left - y_m, goal_x_m - x_m)
        angle_right = math.atan2(goal_y_m_right - y_m, goal_x_m - x_m)
        
        # Angle between the posts
        angle = abs(math.degrees(angle_right - angle_left))
        return min(angle, 180.0)
    
    
    def calculate_xg(
        self,
        x: float,
        y: float,
        shot_type: ShotType = ShotType.OPEN_PLAY,
        body_part: BodyPart = BodyPart.RIGHT_FOOT,
        big_chance: bool = False,
        pressure_level: int = 0,
        attacking_right: bool = True,
        num_defenders: int = 0,
        goalkeeper_position: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Calculate expected goals for a shot with improved model.
        
        Args:
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            shot_type: Type of shot
            body_part: Body part used
            big_chance: Whether it's a big chance
            pressure_level: Defensive pressure (0-3)
            attacking_right: Whether attacking right goal
            num_defenders: Number of defenders between shooter and goal
            goalkeeper_position: Optional GK position for advanced calculation
            
        Returns:
            xG value (0-1)
        """
        # Handle penalties separately
        if shot_type == ShotType.PENALTY:
            return 0.76
        
        # Calculate distance and angle
        distance = self.calculate_distance_to_goal(x, y, attacking_right)
        angle = self.calculate_angle_to_goal(x, y, attacking_right)
        
        # Improved base xG calculation using logistic regression-style formula
        # Based on historical xG models
        distance_factor = 1.0 / (1.0 + math.exp(0.1 * (distance - 12)))  # Sigmoid decay
        angle_factor = min(angle / 60.0, 1.0)  # Normalize angle contribution
        
        # Combine factors with weighted average
        base_xg = (0.6 * distance_factor + 0.4 * angle_factor)
        
        # Central position bonus (shots from central areas are more dangerous)
        central_bonus = 1.0 + 0.2 * (1.0 - abs(y - 0.5) * 2)  # Max bonus at y=0.5
        base_xg *= central_bonus
        
        # Apply shot type modifiers (recalibrated)
        type_modifiers = {
            ShotType.OPEN_PLAY: 1.0,
            ShotType.SET_PIECE: 0.80,
            ShotType.HEADER: 0.65,  # Headers are harder
            ShotType.FREE_KICK: 0.08,  # Direct free kicks are rare goals
            ShotType.CORNER: 0.03,  # Very rare from corner kick directly
            ShotType.PENALTY: 1.0
        }
        xg = base_xg * type_modifiers.get(shot_type, 1.0)
        
        # Apply body part modifiers
        body_modifiers = {
            BodyPart.LEFT_FOOT: 0.95,  # Slightly lower for weak foot
            BodyPart.RIGHT_FOOT: 1.0,
            BodyPart.HEAD: 0.75,
            BodyPart.OTHER: 0.60
        }
        xg *= body_modifiers.get(body_part, 1.0)
        
        # Big chance modifier
        if big_chance:
            xg *= 1.4
        
        # Pressure modifier (more granular)
        pressure_modifiers = {0: 1.0, 1: 0.90, 2: 0.75, 3: 0.55}
        xg *= pressure_modifiers.get(pressure_level, 0.75)
        
        # Defender blocking modifier
        if num_defenders > 0:
            defender_penalty = max(0.3, 1.0 - (num_defenders * 0.15))
            xg *= defender_penalty
        
        # Clamp to valid range
        xg = max(self.min_xg, min(self.max_xg, xg))
        
        return round(xg, 3)

    
    def add_shot(
        self,
        timestamp: float,
        frame: int,
        shooter_id: int,
        shooter_team: str,
        x: float,
        y: float,
        shot_type: ShotType = ShotType.OPEN_PLAY,
        body_part: BodyPart = BodyPart.RIGHT_FOOT,
        outcome: ShotOutcome = ShotOutcome.UNKNOWN,
        velocity_mps: float = 0.0,
        big_chance: bool = False,
        assisted: bool = False,
        pressure_level: int = 0,
        attacking_right: bool = True,
        metadata: Dict[str, Any] = None
    ) -> ShotEvent:
        """
        Add a shot and calculate its xG.
        
        Args:
            timestamp: Time in seconds
            frame: Frame number
            shooter_id: Player ID
            shooter_team: Team ('A' or 'B')
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            shot_type: Type of shot
            body_part: Body part used
            outcome: Shot outcome
            velocity_mps: Ball velocity
            big_chance: Whether it's a big chance
            assisted: Whether shot was assisted
            pressure_level: Defensive pressure (0-3)
            attacking_right: Whether attacking right goal
            metadata: Additional metadata
            
        Returns:
            ShotEvent object
        """
        # Calculate distance and angle
        distance = self.calculate_distance_to_goal(x, y, attacking_right)
        angle = self.calculate_angle_to_goal(x, y, attacking_right)
        
        # Calculate xG
        xg = self.calculate_xg(
            x, y, shot_type, body_part, big_chance, pressure_level, attacking_right
        )
        
        # Create shot event
        shot = ShotEvent(
            timestamp=timestamp,
            frame=frame,
            shooter_id=shooter_id,
            shooter_team=shooter_team,
            x=x,
            y=y,
            shot_type=shot_type,
            body_part=body_part,
            outcome=outcome,
            distance_to_goal=distance,
            angle_to_goal=angle,
            velocity_mps=velocity_mps,
            big_chance=big_chance,
            assisted=assisted,
            pressure_level=pressure_level,
            xg_value=xg,
            metadata=metadata or {}
        )
        
        self.shots.append(shot)
        self._update_stats(shot)
        
        return shot
    
    def _update_stats(self, shot: ShotEvent):
        """Update statistics with a new shot."""
        self.stats.total_shots += 1
        self.stats.total_xg += shot.xg_value
        
        if shot.outcome == ShotOutcome.GOAL:
            self.stats.actual_goals += 1
            self.stats.goals_by_team[shot.shooter_team] += 1
            self.stats.goals_by_type[shot.shot_type.value] += 1
            
            if shot.big_chance:
                self.stats.big_chances_converted += 1
        
        # By type
        self.stats.xg_by_type[shot.shot_type.value] += shot.xg_value
        self.stats.shots_by_type[shot.shot_type.value] += 1
        
        # By body part
        self.stats.xg_by_body_part[shot.body_part.value] += shot.xg_value
        self.stats.shots_by_body_part[shot.body_part.value] += 1
        
        # By team
        self.stats.xg_by_team[shot.shooter_team] += shot.xg_value
        self.stats.shots_by_team[shot.shooter_team] += 1
        
        # Big chances
        if shot.big_chance:
            self.stats.big_chances += 1
        
        # Timeline
        self.stats.xg_timeline.append((shot.timestamp, self.stats.total_xg))
    
    def update_shot_outcome(self, frame: int, outcome: ShotOutcome):
        """Update the outcome of a shot."""
        for shot in self.shots:
            if shot.frame == frame and shot.outcome == ShotOutcome.UNKNOWN:
                shot.outcome = outcome
                if outcome == ShotOutcome.GOAL:
                    self.stats.actual_goals += 1
                    self.stats.goals_by_team[shot.shooter_team] += 1
                    self.stats.goals_by_type[shot.shot_type.value] += 1
                    if shot.big_chance:
                        self.stats.big_chances_converted += 1
                break
    
    def get_shot_quality_rating(self, shot: ShotEvent) -> str:
        """
        Get a qualitative rating for a shot.
        
        Returns:
            Rating string: 'Excellent', 'Good', 'Average', 'Poor'
        """
        if shot.xg_value >= 0.3:
            return 'Excellent'
        elif shot.xg_value >= 0.15:
            return 'Good'
        elif shot.xg_value >= 0.05:
            return 'Average'
        else:
            return 'Poor'
    
    def get_team_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Get xG comparison between teams.
        
        Returns:
            Dictionary with team comparison stats
        """
        comparison = {}
        
        for team in ['A', 'B']:
            shots = self.stats.shots_by_team.get(team, 0)
            xg = self.stats.xg_by_team.get(team, 0.0)
            goals = self.stats.goals_by_team.get(team, 0)
            
            comparison[team] = {
                'shots': shots,
                'xg': round(xg, 2),
                'goals': goals,
                'xg_per_shot': round(xg / shots, 3) if shots > 0 else 0.0,
                'conversion_rate': round(goals / shots * 100, 1) if shots > 0 else 0.0,
                'xg_diff': round(goals - xg, 2)  # Positive = overperforming
            }
        
        return comparison
    
    def get_shot_map_data(self, team: str = None) -> List[Dict]:
        """
        Get shot data for visualization.
        
        Args:
            team: Filter by team (optional)
            
        Returns:
            List of shot data dictionaries
        """
        data = []
        
        for shot in self.shots:
            if team and shot.shooter_team != team:
                continue
                
            data.append({
                'x': shot.x,
                'y': shot.y,
                'xg': shot.xg_value,
                'outcome': shot.outcome.value,
                'shot_type': shot.shot_type.value,
                'distance': round(shot.distance_to_goal, 1),
                'angle': round(shot.angle_to_goal, 1),
                'big_chance': shot.big_chance,
                'rating': self.get_shot_quality_rating(shot)
            })
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive xG statistics."""
        return {
            'total_shots': self.stats.total_shots,
            'total_xg': round(self.stats.total_xg, 2),
            'actual_goals': self.stats.actual_goals,
            'xg_difference': round(self.stats.actual_goals - self.stats.total_xg, 2),
            'conversion_rate': round(self.stats.actual_goals / self.stats.total_shots * 100, 1) 
                              if self.stats.total_shots > 0 else 0.0,
            'big_chances': self.stats.big_chances,
            'big_chance_conversion': round(self.stats.big_chances_converted / self.stats.big_chances * 100, 1)
                                   if self.stats.big_chances > 0 else 0.0,
            'by_type': {
                shot_type: {
                    'shots': self.stats.shots_by_type.get(shot_type, 0),
                    'xg': round(self.stats.xg_by_type.get(shot_type, 0.0), 2),
                    'goals': self.stats.goals_by_type.get(shot_type, 0)
                }
                for shot_type in self.stats.shots_by_type.keys()
            },
            'by_body_part': {
                part: {
                    'shots': self.stats.shots_by_body_part.get(part, 0),
                    'xg': round(self.stats.xg_by_body_part.get(part, 0.0), 2)
                }
                for part in self.stats.shots_by_body_part.keys()
            },
            'team_comparison': self.get_team_comparison(),
            'xg_timeline': self.stats.xg_timeline
        }
    
    def get_shots_by_quality(self, min_xg: float = 0.0, max_xg: float = 1.0) -> List[ShotEvent]:
        """Get shots filtered by xG range."""
        return [s for s in self.shots if min_xg <= s.xg_value <= max_xg]
    
    def get_big_chances(self) -> List[ShotEvent]:
        """Get all big chances."""
        return [s for s in self.shots if s.big_chance]
    
    def reset(self):
        """Reset all data."""
        self.shots = []
        self.stats = xGStats()
