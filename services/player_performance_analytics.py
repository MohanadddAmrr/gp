"""
Player Performance Analytics Module

Provides comprehensive player performance analysis based on video tracking data.
This module tracks individual player metrics, generates performance reports,
and provides insights for player development and match preparation.

Key Features:
- Real-time player performance tracking
- Physical metrics (distance, speed, sprints)
- Technical metrics (passes, shots, touches)
- Tactical metrics (positioning, pressing, defensive actions)
- Performance ratings and form tracking
- Comparative analysis between players
- Historical performance trends
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    PHYSICAL = "physical"
    TECHNICAL = "technical"
    TACTICAL = "tactical"
    MENTAL = "mental"
    OVERALL = "overall"


class PositionCategory(Enum):
    """Player position categories."""
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"


@dataclass
class PhysicalMetrics:
    """Physical performance metrics for a player."""
    # Distance
    total_distance_m: float = 0.0
    sprint_distance_m: float = 0.0
    high_intensity_distance_m: float = 0.0
    walking_distance_m: float = 0.0
    
    # Speed
    avg_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    
    # Sprints
    sprints: int = 0
    sprint_attempts: List[Dict] = field(default_factory=list)
    
    # Acceleration
    accelerations: int = 0
    decelerations: int = 0
    
    # Workload
    workload_score: float = 0.0  # 0-100
    fatigue_index: float = 0.0   # 0-100
    
    # Time-based
    time_in_possession: float = 0.0
    time_defending: float = 0.0
    time_attacking: float = 0.0


@dataclass
class TechnicalMetrics:
    """Technical performance metrics for a player."""
    # Passing
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0.0
    forward_passes: int = 0
    backward_passes: int = 0
    long_passes: int = 0
    key_passes: int = 0
    
    # Shooting
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    xg: float = 0.0
    shot_conversion_rate: float = 0.0
    
    # Ball control
    touches: int = 0
    touches_in_box: int = 0
    dribbles_attempted: int = 0
    dribbles_successful: int = 0
    
    # Defensive
    tackles: int = 0
    tackles_successful: int = 0
    interceptions: int = 0
    clearances: int = 0
    blocks: int = 0
    
    # Aerial
    aerial_duels: int = 0
    aerial_duels_won: int = 0


@dataclass
class TacticalMetrics:
    """Tactical performance metrics for a player."""
    # Positioning
    avg_position_x: float = 0.5
    avg_position_y: float = 0.5
    position_variance: float = 0.0
    
    # Heatmap zones
    time_in_defensive_third: float = 0.0
    time_in_middle_third: float = 0.0
    time_in_attacking_third: float = 0.0
    
    # Pressing
    pressing_actions: int = 0
    pressing_success_rate: float = 0.0
    avg_pressing_distance: float = 0.0
    
    # Defensive positioning
    recoveries: int = 0
    recovery_zone: str = ""
    
    # Attacking positioning
    runs_in_behind: int = 0
    runs_into_box: int = 0
    offside_count: int = 0
    
    # Team play
    pass_network_connections: int = 0
    most_connected_teammate: Optional[int] = None


@dataclass
class PerformanceRating:
    """Overall performance rating breakdown."""
    overall: float = 5.0  # 0-10
    physical: float = 5.0
    technical: float = 5.0
    tactical: float = 5.0
    mental: float = 5.0
    
    # Component scores
    attacking_contribution: float = 5.0
    defensive_contribution: float = 5.0
    creativity: float = 5.0
    work_rate: float = 5.0
    decision_making: float = 5.0
    
    # Context
    minutes_played: float = 0.0
    position: str = ""
    match_importance: float = 1.0  # Multiplier


@dataclass
class PlayerMatchData:
    """Complete player data for a single match."""
    player_id: int
    team: str
    match_id: str
    timestamp: datetime
    
    # Identity
    name: str = "Unknown"
    jersey_number: int = 0
    position: str = "Unknown"
    
    # Metrics
    physical: PhysicalMetrics = field(default_factory=PhysicalMetrics)
    technical: TechnicalMetrics = field(default_factory=TechnicalMetrics)
    tactical: TacticalMetrics = field(default_factory=TacticalMetrics)
    rating: PerformanceRating = field(default_factory=PerformanceRating)
    
    # Timeline
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    speed_history: List[Tuple[float, float]] = field(default_factory=list)  # (speed, timestamp)
    event_history: List[Dict] = field(default_factory=list)


class PlayerPerformanceAnalytics:
    """
    Player Performance Analytics Engine.
    
    Tracks and analyzes individual player performance from video data,
    providing comprehensive metrics and insights.
    
    Features:
    - Real-time performance tracking
    - Multi-dimensional performance analysis
    - Form and consistency tracking
    - Comparative player analysis
    - Position-specific metrics
    """
    
    # Speed thresholds (m/s)
    WALKING_THRESHOLD = 2.0
    JOGGING_THRESHOLD = 4.0
    RUNNING_THRESHOLD = 5.5
    SPRINTING_THRESHOLD = 7.0
    
    # Distance thresholds (meters)
    SPRINT_MIN_DISTANCE = 10.0
    HIGH_INTENSITY_MIN_DISTANCE = 5.0
    
    def __init__(self):
        """Initialize the player performance analytics engine."""
        # Current match data
        self.current_match_data: Dict[int, PlayerMatchData] = {}
        
        # Historical data
        self.player_history: Dict[int, List[PlayerMatchData]] = defaultdict(list)
        
        # Tracking state
        self.player_positions: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        self.player_speeds: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        # Calculated metrics cache
        self.metrics_cache: Dict[int, Dict[str, Any]] = {}
        
        # Form tracking
        self.form_history: Dict[int, List[float]] = defaultdict(list)  # player_id -> [ratings]
        
    def register_player(
        self,
        player_id: int,
        team: str,
        name: str = "Unknown",
        jersey_number: int = 0,
        position: str = "Unknown",
        match_id: str = ""
    ):
        """
        Register a player for tracking.
        
        Args:
            player_id: Unique player identifier
            team: Team identifier ('A' or 'B')
            name: Player name
            jersey_number: Jersey number
            position: Playing position
            match_id: Match identifier
        """
        self.current_match_data[player_id] = PlayerMatchData(
            player_id=player_id,
            team=team,
            match_id=match_id,
            timestamp=datetime.now(),
            name=name,
            jersey_number=jersey_number,
            position=position
        )
    
    def update_position(
        self,
        player_id: int,
        x: float,
        y: float,
        timestamp: float,
        team: str = None
    ):
        """
        Update player position.
        
        Args:
            player_id: Player identifier
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            timestamp: Time in seconds
            team: Team identifier (optional)
        """
        if player_id not in self.current_match_data:
            self.register_player(player_id, team or "Unknown")
        
        self.player_positions[player_id].append((x, y, timestamp))
        
        # Update tactical metrics
        player_data = self.current_match_data[player_id]
        
        # Track zone time
        if x < 0.33:
            player_data.tactical.time_in_defensive_third += 0.1  # Assuming 10Hz
        elif x > 0.67:
            player_data.tactical.time_in_attacking_third += 0.1
        else:
            player_data.tactical.time_in_middle_third += 0.1
    
    def update_speed(
        self,
        player_id: int,
        speed_mps: float,
        timestamp: float
    ):
        """
        Update player speed.
        
        Args:
            player_id: Player identifier
            speed_mps: Speed in meters per second
            timestamp: Time in seconds
        """
        if player_id not in self.current_match_data:
            return
        
        self.player_speeds[player_id].append((speed_mps, timestamp))
        
        player_data = self.current_match_data[player_id]
        physical = player_data.physical
        
        # Update max speed
        if speed_mps > physical.max_speed_mps:
            physical.max_speed_mps = speed_mps
        
        # Categorize movement
        dt = 0.1  # Assuming 10Hz updates
        distance = speed_mps * dt
        physical.total_distance_m += distance
        
        if speed_mps >= self.SPRINTING_THRESHOLD:
            physical.sprint_distance_m += distance
            physical.high_intensity_distance_m += distance
        elif speed_mps >= self.RUNNING_THRESHOLD:
            physical.high_intensity_distance_m += distance
        elif speed_mps <= self.WALKING_THRESHOLD:
            physical.walking_distance_m += distance
    
    def record_event(
        self,
        player_id: int,
        event_type: str,
        success: bool = True,
        metadata: Dict = None
    ):
        """
        Record a player event.
        
        Args:
            player_id: Player identifier
            event_type: Type of event (pass, shot, tackle, etc.)
            success: Whether the action was successful
            metadata: Additional event data
        """
        if player_id not in self.current_match_data:
            return
        
        player_data = self.current_match_data[player_id]
        technical = player_data.technical
        
        event = {
            'type': event_type,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        player_data.event_history.append(event)
        
        # Update technical metrics based on event type
        if event_type == 'pass':
            technical.passes_attempted += 1
            if success:
                technical.passes_completed += 1
            
            # Update pass accuracy
            if technical.passes_attempted > 0:
                technical.pass_accuracy = (
                    technical.passes_completed / technical.passes_attempted * 100
                )
        
        elif event_type == 'shot':
            technical.shots += 1
            if success:
                technical.shots_on_target += 1
            
            # Update conversion rate
            if technical.shots > 0:
                technical.shot_conversion_rate = (
                    technical.goals / technical.shots * 100
                )
        
        elif event_type == 'goal':
            technical.goals += 1
            technical.shot_conversion_rate = (
                technical.goals / max(technical.shots, 1) * 100
            )
        
        elif event_type == 'tackle':
            technical.tackles += 1
            if success:
                technical.tackles_successful += 1
        
        elif event_type == 'interception':
            technical.interceptions += 1
        
        elif event_type == 'touch':
            technical.touches += 1
        
        elif event_type == 'dribble':
            technical.dribbles_attempted += 1
            if success:
                technical.dribbles_successful += 1
    
    def calculate_physical_metrics(self, player_id: int) -> PhysicalMetrics:
        """
        Calculate comprehensive physical metrics for a player.
        
        Args:
            player_id: Player identifier
            
        Returns:
            PhysicalMetrics object
        """
        if player_id not in self.current_match_data:
            return PhysicalMetrics()
        
        player_data = self.current_match_data[player_id]
        physical = player_data.physical
        
        # Calculate average speed
        speeds = [s for s, _ in self.player_speeds.get(player_id, [])]
        if speeds:
            physical.avg_speed_mps = np.mean(speeds)
        
        # Count sprints
        sprint_count = 0
        in_sprint = False
        for speed, _ in self.player_speeds.get(player_id, []):
            if speed >= self.SPRINTING_THRESHOLD and not in_sprint:
                sprint_count += 1
                in_sprint = True
            elif speed < self.RUNNING_THRESHOLD:
                in_sprint = False
        
        physical.sprints = sprint_count
        
        # Calculate workload score (0-100)
        # Based on distance covered, sprints, and high-intensity work
        expected_distance = 8000  # Expected distance in meters (90 min)
        distance_score = min(100, (physical.total_distance_m / expected_distance) * 100)
        sprint_score = min(100, (physical.sprints / 20) * 100)  # 20 sprints = max
        
        physical.workload_score = (distance_score * 0.6) + (sprint_score * 0.4)
        
        # Calculate fatigue index (simplified)
        # Higher workload in later stages = higher fatigue
        physical.fatigue_index = min(100, physical.workload_score * 0.8)
        
        return physical
    
    def calculate_tactical_metrics(self, player_id: int) -> TacticalMetrics:
        """
        Calculate tactical metrics for a player.
        
        Args:
            player_id: Player identifier
            
        Returns:
            TacticalMetrics object
        """
        if player_id not in self.current_match_data:
            return TacticalMetrics()
        
        player_data = self.current_match_data[player_id]
        tactical = player_data.tactical
        
        # Calculate average position
        positions = self.player_positions.get(player_id, [])
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            tactical.avg_position_x = np.mean(xs)
            tactical.avg_position_y = np.mean(ys)
            tactical.position_variance = np.var(xs) + np.var(ys)
        
        return tactical
    
    def calculate_performance_rating(self, player_id: int) -> PerformanceRating:
        """
        Calculate overall performance rating for a player.
        
        Args:
            player_id: Player identifier
            
        Returns:
            PerformanceRating object
        """
        if player_id not in self.current_match_data:
            return PerformanceRating()
        
        player_data = self.current_match_data[player_id]
        rating = PerformanceRating()
        
        # Calculate physical rating
        physical = self.calculate_physical_metrics(player_id)
        distance_score = min(10, (physical.total_distance_m / 10000) * 10)
        sprint_score = min(10, (physical.sprints / 15) * 10)
        rating.physical = (distance_score + sprint_score) / 2
        
        # Calculate technical rating
        technical = player_data.technical
        if technical.passes_attempted > 0:
            pass_score = (technical.pass_accuracy / 100) * 10
        else:
            pass_score = 5.0
        
        if technical.shots > 0:
            shot_score = (technical.shot_conversion_rate / 100) * 10
        else:
            shot_score = 5.0
        
        rating.technical = (pass_score + shot_score) / 2
        
        # Calculate tactical rating
        tactical = self.calculate_tactical_metrics(player_id)
        # Based on positioning and contributions
        rating.tactical = 5.0 + (tactical.pressing_actions * 0.1)
        rating.tactical = min(10, max(0, rating.tactical))
        
        # Calculate overall rating
        rating.overall = (
            rating.physical * 0.25 +
            rating.technical * 0.35 +
            rating.tactical * 0.30 +
            rating.mental * 0.10
        )
        
        # Component scores
        rating.attacking_contribution = self._calculate_attacking_score(player_id)
        rating.defensive_contribution = self._calculate_defensive_score(player_id)
        rating.work_rate = rating.physical
        
        player_data.rating = rating
        return rating
    
    def _calculate_attacking_score(self, player_id: int) -> float:
        """Calculate attacking contribution score."""
        if player_id not in self.current_match_data:
            return 5.0
        
        technical = self.current_match_data[player_id].technical
        
        # Goals and assists weighted heavily
        goal_score = min(10, technical.goals * 2)
        shot_score = min(10, (technical.shots_on_target / max(technical.shots, 1)) * 10)
        pass_score = min(10, (technical.key_passes / max(technical.passes_attempted, 1)) * 100)
        
        return (goal_score * 0.4 + shot_score * 0.3 + pass_score * 0.3)
    
    def _calculate_defensive_score(self, player_id: int) -> float:
        """Calculate defensive contribution score."""
        if player_id not in self.current_match_data:
            return 5.0
        
        technical = self.current_match_data[player_id].technical
        
        tackle_score = min(10, (technical.tackles_successful / max(technical.tackles, 1)) * 10)
        interception_score = min(10, technical.interceptions * 0.5)
        
        return (tackle_score + interception_score) / 2
    
    def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get comprehensive player performance summary.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Summary dictionary
        """
        if player_id not in self.current_match_data:
            return {}
        
        player_data = self.current_match_data[player_id]
        
        # Calculate all metrics
        physical = self.calculate_physical_metrics(player_id)
        tactical = self.calculate_tactical_metrics(player_id)
        rating = self.calculate_performance_rating(player_id)
        
        return {
            'player_info': {
                'id': player_id,
                'name': player_data.name,
                'team': player_data.team,
                'jersey_number': player_data.jersey_number,
                'position': player_data.position
            },
            'physical': {
                'total_distance_m': round(physical.total_distance_m, 1),
                'sprint_distance_m': round(physical.sprint_distance_m, 1),
                'max_speed_mps': round(physical.max_speed_mps, 2),
                'avg_speed_mps': round(physical.avg_speed_mps, 2),
                'sprints': physical.sprints,
                'workload_score': round(physical.workload_score, 1)
            },
            'technical': {
                'passes': f"{player_data.technical.passes_completed}/{player_data.technical.passes_attempted}",
                'pass_accuracy': round(player_data.technical.pass_accuracy, 1),
                'shots': player_data.technical.shots,
                'shots_on_target': player_data.technical.shots_on_target,
                'goals': player_data.technical.goals,
                'tackles': player_data.technical.tackles,
                'interceptions': player_data.technical.interceptions,
                'touches': player_data.technical.touches
            },
            'tactical': {
                'avg_position': (round(tactical.avg_position_x, 2), round(tactical.avg_position_y, 2)),
                'time_in_attacking_third': round(tactical.time_in_attacking_third, 1),
                'pressing_actions': tactical.pressing_actions
            },
            'rating': {
                'overall': round(rating.overall, 1),
                'physical': round(rating.physical, 1),
                'technical': round(rating.technical, 1),
                'tactical': round(rating.tactical, 1),
                'attacking': round(rating.attacking_contribution, 1),
                'defensive': round(rating.defensive_contribution, 1)
            }
        }
    
    def get_team_summary(self, team: str) -> Dict[str, Any]:
        """
        Get performance summary for entire team.
        
        Args:
            team: Team identifier ('A' or 'B')
            
        Returns:
            Team summary dictionary
        """
        team_players = [
            pid for pid, data in self.current_match_data.items()
            if data.team == team
        ]
        
        if not team_players:
            return {}
        
        # Aggregate metrics
        total_distance = sum(
            self.current_match_data[pid].physical.total_distance_m
            for pid in team_players
        )
        total_passes = sum(
            self.current_match_data[pid].technical.passes_attempted
            for pid in team_players
        )
        total_shots = sum(
            self.current_match_data[pid].technical.shots
            for pid in team_players
        )
        total_goals = sum(
            self.current_match_data[pid].technical.goals
            for pid in team_players
        )
        
        # Calculate average rating
        ratings = [
            self.calculate_performance_rating(pid).overall
            for pid in team_players
        ]
        avg_rating = np.mean(ratings) if ratings else 0
        
        # Top performers
        player_summaries = [
            (pid, self.calculate_performance_rating(pid).overall)
            for pid in team_players
        ]
        player_summaries.sort(key=lambda x: x[1], reverse=True)
        top_performers = [
            {
                'player_id': pid,
                'name': self.current_match_data[pid].name,
                'rating': round(rating, 1)
            }
            for pid, rating in player_summaries[:3]
        ]
        
        return {
            'team': team,
            'players_tracked': len(team_players),
            'aggregate_stats': {
                'total_distance_m': round(total_distance, 1),
                'total_passes': total_passes,
                'total_shots': total_shots,
                'total_goals': total_goals
            },
            'average_rating': round(avg_rating, 1),
            'top_performers': top_performers
        }
    
    def compare_players(
        self,
        player_ids: List[int],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple players on specified metrics.
        
        Args:
            player_ids: List of player IDs to compare
            metrics: List of metrics to compare (None = all)
            
        Returns:
            Comparison dictionary
        """
        if not player_ids:
            return {}
        
        metrics = metrics or ['overall_rating', 'distance', 'pass_accuracy', 'goals']
        
        comparison = {
            'players': [],
            'rankings': {}
        }
        
        for pid in player_ids:
            if pid not in self.current_match_data:
                continue
            
            summary = self.get_player_summary(pid)
            comparison['players'].append({
                'player_id': pid,
                'name': summary['player_info']['name'],
                'position': summary['player_info']['position'],
                'metrics': {
                    'overall_rating': summary['rating']['overall'],
                    'distance': summary['physical']['total_distance_m'],
                    'pass_accuracy': summary['technical']['pass_accuracy'],
                    'goals': summary['technical']['goals']
                }
            })
        
        # Create rankings
        for metric in metrics:
            sorted_players = sorted(
                comparison['players'],
                key=lambda p: p['metrics'].get(metric, 0),
                reverse=True
            )
            comparison['rankings'][metric] = [
                {'name': p['name'], 'value': p['metrics'].get(metric, 0)}
                for p in sorted_players
            ]
        
        return comparison
    
    def finalize_match(self, match_id: str = "") -> Dict[str, Any]:
        """
        Finalize match data and generate comprehensive report.
        
        Args:
            match_id: Match identifier
            
        Returns:
            Final match report
        """
        report = {
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
            'players': {},
            'team_summaries': {}
        }
        
        # Generate player reports
        for player_id in self.current_match_data:
            report['players'][player_id] = self.get_player_summary(player_id)
            
            # Store in history
            self.player_history[player_id].append(
                self.current_match_data[player_id]
            )
            
            # Update form history
            rating = self.current_match_data[player_id].rating.overall
            self.form_history[player_id].append(rating)
        
        # Generate team summaries
        report['team_summaries']['A'] = self.get_team_summary('A')
        report['team_summaries']['B'] = self.get_team_summary('B')
        
        return report
    
    def get_player_form_trend(
        self,
        player_id: int,
        num_matches: int = 5
    ) -> Dict[str, Any]:
        """
        Get player's form trend over recent matches.
        
        Args:
            player_id: Player identifier
            num_matches: Number of matches to analyze
            
        Returns:
            Form trend data
        """
        history = self.form_history.get(player_id, [])
        
        if not history:
            return {
                'player_id': player_id,
                'matches_analyzed': 0,
                'current_form': 0,
                'trend': 'stable',
                'consistency': 0
            }
        
        recent = history[-num_matches:]
        current_form = recent[-1] if recent else 0
        avg_form = np.mean(recent)
        
        # Determine trend
        if len(recent) >= 2:
            first_half = np.mean(recent[:len(recent)//2])
            second_half = np.mean(recent[len(recent)//2:])
            
            if second_half > first_half + 0.5:
                trend = 'improving'
            elif second_half < first_half - 0.5:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Calculate consistency (lower std = more consistent)
        consistency = 10 - min(10, np.std(recent)) if len(recent) > 1 else 10
        
        return {
            'player_id': player_id,
            'matches_analyzed': len(recent),
            'current_form': round(current_form, 1),
            'average_form': round(avg_form, 1),
            'trend': trend,
            'consistency': round(consistency, 1),
            'form_history': [round(r, 1) for r in recent]
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.current_match_data = {}
        self.player_positions = defaultdict(list)
        self.player_speeds = defaultdict(list)
        self.metrics_cache = {}
