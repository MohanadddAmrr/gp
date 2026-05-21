"""
AI Tactical Recommendations Module

Provides intelligent tactical recommendations based on real-time video analysis data.
This module uses machine learning and statistical analysis to generate actionable
insights for coaches and analysts.

Key Features:
- Real-time tactical recommendations based on match data
- Pattern recognition for opponent weaknesses
- Formation effectiveness analysis
- Player positioning optimization
- Set-piece strategy recommendations
- In-game adjustment suggestions
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime

from services.tactical_analyzer import TacticalAnalyzer, TacticalReport
from services.tactical_insights import TacticalInsightsEngine, TacticalAdjustment
from services.advanced_analytics import AdvancedAnalytics, Zone
from services.event_detector import EventDetector
from services.possession_tracker import PossessionTracker


class RecommendationPriority(Enum):
    """Priority levels for tactical recommendations."""
    CRITICAL = 5  # Immediate action required
    HIGH = 4      # Strong recommendation
    MEDIUM = 3    # Consider implementing
    LOW = 2       # Optional adjustment
    INFO = 1      # Informational only


class RecommendationCategory(Enum):
    """Categories of tactical recommendations."""
    FORMATION = "formation"
    PRESSING = "pressing"
    ATTACKING = "attacking"
    DEFENSIVE = "defensive"
    TRANSITIONS = "transitions"
    SET_PIECES = "set_pieces"
    PLAYER_POSITIONING = "player_positioning"
    SUBSTITUTION = "substitution"
    OPPONENT_EXPLOIT = "opponent_exploit"


@dataclass
class TacticalRecommendation:
    """A single tactical recommendation with full context."""
    id: str
    timestamp: float
    priority: RecommendationPriority
    category: RecommendationCategory
    
    # Content
    title: str
    description: str
    reasoning: str
    expected_outcome: str
    
    # Data backing
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0  # 0-1
    
    # Implementation
    actionable: bool = True
    implementation_steps: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    
    # Target
    target_team: str = ""  # 'A', 'B', or both
    target_players: List[int] = field(default_factory=list)
    applicable_timeframe: str = "immediate"  # immediate, next_half, future_matches


@dataclass
class MatchContext:
    """Current match context for recommendation generation."""
    timestamp: float
    score_a: int = 0
    score_b: int = 0
    possession_a: float = 50.0
    possession_b: float = 50.0
    
    # Time context
    match_minute: float = 0.0
    period: str = "first_half"  # first_half, second_half, extra_time
    
    # Game state
    leading_team: Optional[str] = None
    goal_difference: int = 0
    momentum: str = "neutral"  # team_a, team_b, neutral
    
    # Recent events
    recent_goals: List[Dict] = field(default_factory=list)
    recent_cards: List[Dict] = field(default_factory=list)
    substitutions_made: Dict[str, int] = field(default_factory=dict)


@dataclass
class PlayerPerformanceSnapshot:
    """Real-time player performance data."""
    player_id: int
    team: str
    
    # Physical
    distance_covered: float = 0.0
    current_speed: float = 0.0
    fatigue_score: float = 0.0  # 0-100, higher = more fatigued
    
    # Technical
    pass_accuracy: float = 0.0
    touches: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    
    # Tactical
    heatmap_zone: str = ""
    pressing_contributions: int = 0
    defensive_actions: int = 0
    
    # Form
    form_rating: float = 50.0  # 0-100


class AITacticalRecommendations:
    """
    AI-Powered Tactical Recommendations Engine.
    
    Analyzes real-time match data from video tracking to generate
    intelligent tactical recommendations for coaches.
    
    Features:
    - Real-time pattern analysis
    - Opponent weakness exploitation
    - Formation optimization
    - Player performance-based suggestions
    - Game state-aware recommendations
    """
    
    def __init__(
        self,
        tactical_analyzer: TacticalAnalyzer = None,
        insights_engine: TacticalInsightsEngine = None,
        advanced_analytics: AdvancedAnalytics = None
    ):
        """
        Initialize the AI tactical recommendations engine.
        
        Args:
            tactical_analyzer: Existing tactical analyzer instance
            insights_engine: Existing insights engine instance
            advanced_analytics: Existing advanced analytics instance
        """
        self.tactical_analyzer = tactical_analyzer or TacticalAnalyzer()
        self.insights_engine = insights_engine or TacticalInsightsEngine()
        self.advanced_analytics = advanced_analytics or AdvancedAnalytics()
        
        # Recommendation history
        self.recommendations: List[TacticalRecommendation] = []
        self.recommendation_counter = 0
        
        # Match context
        self.match_context = MatchContext(timestamp=0.0)
        
        # Player performance tracking
        self.player_snapshots: Dict[int, PlayerPerformanceSnapshot] = {}
        
        # Pattern tracking
        self.detected_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.opponent_weaknesses: Dict[str, List[str]] = defaultdict(list)
        
        # Effectiveness tracking
        self.implemented_recommendations: Dict[str, Dict] = {}
        self.success_metrics: Dict[str, float] = {}
        
    def _generate_id(self) -> str:
        """Generate unique recommendation ID."""
        self.recommendation_counter += 1
        timestamp = datetime.now().strftime("%H%M%S")
        return f"REC-{timestamp}-{self.recommendation_counter:04d}"
    
    def update_match_context(
        self,
        score_a: int,
        score_b: int,
        possession_a: float,
        possession_b: float,
        match_minute: float,
        period: str = "first_half"
    ):
        """
        Update current match context.
        
        Args:
            score_a: Team A score
            score_b: Team B score
            possession_a: Team A possession percentage
            possession_b: Team B possession percentage
            match_minute: Current match minute
            period: Match period
        """
        self.match_context.score_a = score_a
        self.match_context.score_b = score_b
        self.match_context.possession_a = possession_a
        self.match_context.possession_b = possession_b
        self.match_context.match_minute = match_minute
        self.match_context.period = period
        
        # Calculate goal difference and leading team
        self.match_context.goal_difference = score_a - score_b
        if score_a > score_b:
            self.match_context.leading_team = 'A'
        elif score_b > score_a:
            self.match_context.leading_team = 'B'
        else:
            self.match_context.leading_team = None
    
    def update_player_performance(
        self,
        player_id: int,
        team: str,
        distance_covered: float = None,
        current_speed: float = None,
        pass_accuracy: float = None,
        touches: int = None,
        fatigue_score: float = None,
        form_rating: float = None
    ):
        """
        Update player performance snapshot.
        
        Args:
            player_id: Player identifier
            team: Team ('A' or 'B')
            distance_covered: Total distance covered in meters
            current_speed: Current speed in m/s
            pass_accuracy: Pass accuracy percentage
            touches: Number of ball touches
            fatigue_score: Fatigue score 0-100
            form_rating: Current form rating 0-100
        """
        if player_id not in self.player_snapshots:
            self.player_snapshots[player_id] = PlayerPerformanceSnapshot(
                player_id=player_id,
                team=team
            )
        
        snapshot = self.player_snapshots[player_id]
        
        if distance_covered is not None:
            snapshot.distance_covered = distance_covered
        if current_speed is not None:
            snapshot.current_speed = current_speed
        if pass_accuracy is not None:
            snapshot.pass_accuracy = pass_accuracy
        if touches is not None:
            snapshot.touches = touches
        if fatigue_score is not None:
            snapshot.fatigue_score = fatigue_score
        if form_rating is not None:
            snapshot.form_rating = form_rating
    
    def analyze_opponent_weaknesses(
        self,
        team: str = 'B'
    ) -> Dict[str, Any]:
        """
        Analyze opponent weaknesses based on tracking data.
        
        Args:
            team: Opponent team to analyze ('A' or 'B')
            
        Returns:
            Dictionary of identified weaknesses
        """
        weaknesses = {
            'defensive_gaps': [],
            'pressing_triggers': [],
            'transition_vulnerabilities': [],
            'set_piece_weaknesses': [],
            'player_matchups': []
        }
        
        # Analyze zone control from advanced analytics
        zone_control = self.advanced_analytics.get_zone_control_summary()
        
        # Find zones where opponent is weak
        for zone_name, control in zone_control.get('by_zone', {}).items():
            team_control = control.get(f'team_{team.lower()}', 0)
            if team_control < 30:
                weaknesses['defensive_gaps'].append({
                    'zone': zone_name,
                    'control_percentage': team_control,
                    'exploitation_strategy': f"Attack through {zone_name} - opponent has low control"
                })
        
        # Analyze pressing effectiveness
        pressure_stats = self.advanced_analytics.get_pressure_statistics()
        opponent_pressures = pressure_stats.get(team, {})
        
        if opponent_pressures.get('avg_intensity', 0) < 40:
            weaknesses['pressing_triggers'].append({
                'type': 'low_pressing_intensity',
                'description': 'Opponent shows low pressing intensity - build up play from back',
                'avg_intensity': opponent_pressures.get('avg_intensity', 0)
            })
        
        # Analyze transition success
        progression_stats = self.advanced_analytics.get_progression_statistics()
        opponent_progressions = progression_stats.get(team, {})
        
        if opponent_progressions.get('success_rate', 100) < 50:
            weaknesses['transition_vulnerabilities'].append({
                'type': 'poor_transition_defense',
                'description': 'Opponent struggles with defensive transitions - exploit with quick counters',
                'success_rate': opponent_progressions.get('success_rate', 0)
            })
        
        return weaknesses
    
    def generate_formation_recommendations(
        self,
        current_formations: Dict[str, str]
    ) -> List[TacticalRecommendation]:
        """
        Generate formation-based recommendations.
        
        Args:
            current_formations: Current formations for both teams
            
        Returns:
            List of formation recommendations
        """
        recommendations = []
        
        # Get team A's current formation
        team_a_formation = current_formations.get('A', '4-3-3')
        team_b_formation = current_formations.get('B', '4-3-3')
        
        # Analyze formation matchup
        if team_b_formation in ['4-3-3', '4-2-3-1'] and team_a_formation == '4-3-3':
            # Recommend exploiting wide areas
            rec = TacticalRecommendation(
                id=self._generate_id(),
                timestamp=self.match_context.match_minute,
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.FORMATION,
                title="Exploit Wide Areas",
                description="Opponent playing narrow formation. Switch to wider formation or instruct full-backs to push higher.",
                reasoning=f"Opponent's {team_b_formation} leaves space on the flanks. Our {team_a_formation} can exploit this with overlapping runs.",
                expected_outcome="Create 2v1 situations on wings and deliver dangerous crosses",
                supporting_data={
                    'opponent_formation': team_b_formation,
                    'current_formation': team_a_formation,
                    'exploitable_zones': ['wide_left', 'wide_right']
                },
                confidence_score=0.75,
                implementation_steps=[
                    "Instruct full-backs to push higher up the pitch",
                    "Wingers should stay wide to stretch defense",
                    "Central midfielders to provide cover for full-backs"
                ],
                target_team='A'
            )
            recommendations.append(rec)
        
        # Check if losing and need more attacking shape
        if self.match_context.leading_team == 'B' and self.match_context.match_minute > 60:
            rec = TacticalRecommendation(
                id=self._generate_id(),
                timestamp=self.match_context.match_minute,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.FORMATION,
                title="Switch to More Attacking Formation",
                description="Consider switching to 3-4-3 or 4-2-4 to increase attacking threat",
                reasoning=f"Trailing by {abs(self.match_context.goal_difference)} goals with {90 - self.match_context.match_minute:.0f} minutes remaining. Need more attacking presence.",
                expected_outcome="Increased attacking options and more shots on goal",
                supporting_data={
                    'current_score': f"{self.match_context.score_a}-{self.match_context.score_b}",
                    'time_remaining': 90 - self.match_context.match_minute
                },
                confidence_score=0.8,
                risk_level="medium",
                target_team='A'
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_pressing_recommendations(
        self,
        pressing_stats: Dict[str, Any]
    ) -> List[TacticalRecommendation]:
        """
        Generate pressing-related recommendations.
        
        Args:
            pressing_stats: Pressing statistics from tactical analyzer
            
        Returns:
            List of pressing recommendations
        """
        recommendations = []
        
        team_a_ppda = pressing_stats.get('team_a', {}).get('ppda', 15)
        team_b_ppda = pressing_stats.get('team_b', {}).get('ppda', 15)
        
        # If opponent has high PPDA (low pressing), recommend high press
        if team_b_ppda > 12:
            rec = TacticalRecommendation(
                id=self._generate_id(),
                timestamp=self.match_context.match_minute,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.PRESSING,
                title="Implement High Press",
                description="Opponent vulnerable to high pressing. Apply immediate pressure after losing possession.",
                reasoning=f"Opponent PPDA of {team_b_ppda:.1f} indicates they struggle under pressure. High press can force turnovers in dangerous areas.",
                expected_outcome="Win possession in attacking third and create scoring opportunities",
                supporting_data={
                    'opponent_ppda': team_b_ppda,
                    'pressing_intensity': 'high'
                },
                confidence_score=0.8,
                implementation_steps=[
                    "First defender presses immediately after losing ball",
                    "Cut off passing lanes to full-backs",
                    "Forcing play towards the touchlines",
                    "Midfielders support press from behind"
                ],
                target_team='A'
            )
            recommendations.append(rec)
        
        # If our PPDA is too high (not pressing enough) when losing
        if team_a_ppda > 15 and self.match_context.leading_team == 'B':
            rec = TacticalRecommendation(
                id=self._generate_id(),
                timestamp=self.match_context.match_minute,
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.PRESSING,
                title="Increase Pressing Intensity",
                description="Current pressing intensity too low. Need to win ball back quicker.",
                reasoning=f"PPDA of {team_a_ppda:.1f} shows we're allowing opponent too much time on the ball while trailing.",
                expected_outcome="Regain possession faster and reduce opponent's attacking threat",
                supporting_data={
                    'current_ppda': team_a_ppda,
                    'target_ppda': 8
                },
                confidence_score=0.7,
                target_team='A'
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_substitution_recommendations(
        self,
        fatigue_threshold: float = 70.0
    ) -> List[TacticalRecommendation]:
        """
        Generate substitution recommendations based on player performance.
        
        Args:
            fatigue_threshold: Fatigue level threshold for substitution recommendation
            
        Returns:
            List of substitution recommendations
        """
        recommendations = []
        
        for player_id, snapshot in self.player_snapshots.items():
            # Check for fatigue
            if snapshot.fatigue_score > fatigue_threshold:
                rec = TacticalRecommendation(
                    id=self._generate_id(),
                    timestamp=self.match_context.match_minute,
                    priority=RecommendationPriority.MEDIUM,
                    category=RecommendationCategory.SUBSTITUTION,
                    title=f"Consider Substituting Player {player_id}",
                    description=f"Player showing signs of fatigue (fatigue score: {snapshot.fatigue_score:.1f})",
                    reasoning=f"High fatigue levels indicate declining performance. Fresh player could provide needed energy.",
                    expected_outcome="Maintain intensity and reduce injury risk",
                    supporting_data={
                        'player_id': player_id,
                        'fatigue_score': snapshot.fatigue_score,
                        'distance_covered': snapshot.distance_covered
                    },
                    confidence_score=min(snapshot.fatigue_score / 100, 0.9),
                    target_team=snapshot.team,
                    target_players=[player_id]
                )
                recommendations.append(rec)
            
            # Check for poor form
            if snapshot.form_rating < 40 and snapshot.touches > 10:
                rec = TacticalRecommendation(
                    id=self._generate_id(),
                    timestamp=self.match_context.match_minute,
                    priority=RecommendationPriority.LOW,
                    category=RecommendationCategory.SUBSTITUTION,
                    title=f"Player {player_id} Below Par",
                    description=f"Player form rating low ({snapshot.form_rating:.1f}). Consider substitution.",
                    reasoning="Multiple unsuccessful actions and low impact on the game.",
                    expected_outcome="Improve team performance with more effective player",
                    supporting_data={
                        'player_id': player_id,
                        'form_rating': snapshot.form_rating,
                        'pass_accuracy': snapshot.pass_accuracy
                    },
                    confidence_score=0.6,
                    target_team=snapshot.team,
                    target_players=[player_id]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def generate_opponent_exploit_recommendations(
        self,
        weaknesses: Dict[str, Any]
    ) -> List[TacticalRecommendation]:
        """
        Generate recommendations to exploit opponent weaknesses.
        
        Args:
            weaknesses: Dictionary of opponent weaknesses
            
        Returns:
            List of exploitation recommendations
        """
        recommendations = []
        
        # Exploit defensive gaps
        for gap in weaknesses.get('defensive_gaps', []):
            rec = TacticalRecommendation(
                id=self._generate_id(),
                timestamp=self.match_context.match_minute,
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.OPPONENT_EXPLOIT,
                title=f"Attack {gap['zone'].replace('_', ' ').title()}",
                description=gap['exploitation_strategy'],
                reasoning=f"Opponent only has {gap['control_percentage']:.1f}% control in this zone",
                expected_outcome="Create high-quality chances in under-defended areas",
                supporting_data={
                    'target_zone': gap['zone'],
                    'opponent_control': gap['control_percentage']
                },
                confidence_score=0.75,
                implementation_steps=[
                    f"Direct attacks through {gap['zone']}",
                    "Use quick combinations to exploit space",
                    "Look for through balls into the channel"
                ],
                target_team='A'
            )
            recommendations.append(rec)
        
        # Exploit low pressing
        for trigger in weaknesses.get('pressing_triggers', []):
            rec = TacticalRecommendation(
                id=self._generate_id(),
                timestamp=self.match_context.match_minute,
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.OPPONENT_EXPLOIT,
                title="Build from the Back",
                description=trigger['description'],
                reasoning=f"Opponent's low pressing intensity ({trigger['avg_intensity']:.1f}) allows time for controlled buildup",
                expected_outcome="Draw opponent out of shape and create space behind",
                supporting_data={
                    'opponent_pressing_intensity': trigger['avg_intensity']
                },
                confidence_score=0.7,
                target_team='A'
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_all_recommendations(
        self,
        tactical_report: TacticalReport = None,
        current_formations: Dict[str, str] = None,
        pressing_stats: Dict[str, Any] = None
    ) -> List[TacticalRecommendation]:
        """
        Generate comprehensive set of tactical recommendations.
        
        Args:
            tactical_report: Current tactical report
            current_formations: Current formations for both teams
            pressing_stats: Pressing statistics
            
        Returns:
            List of all generated recommendations
        """
        all_recommendations = []
        
        # Formation recommendations
        if current_formations:
            all_recommendations.extend(
                self.generate_formation_recommendations(current_formations)
            )
        
        # Pressing recommendations
        if pressing_stats:
            all_recommendations.extend(
                self.generate_pressing_recommendations(pressing_stats)
            )
        
        # Substitution recommendations
        all_recommendations.extend(
            self.generate_substitution_recommendations()
        )
        
        # Opponent weakness exploitation
        opponent_weaknesses = self.analyze_opponent_weaknesses('B')
        all_recommendations.extend(
            self.generate_opponent_exploit_recommendations(opponent_weaknesses)
        )
        
        # Sort by priority
        priority_order = {
            RecommendationPriority.CRITICAL: 5,
            RecommendationPriority.HIGH: 4,
            RecommendationPriority.MEDIUM: 3,
            RecommendationPriority.LOW: 2,
            RecommendationPriority.INFO: 1
        }
        all_recommendations.sort(
            key=lambda r: priority_order.get(r.priority, 0),
            reverse=True
        )
        
        # Store recommendations
        self.recommendations.extend(all_recommendations)
        
        return all_recommendations
    
    def get_top_recommendations(
        self,
        n: int = 5,
        min_confidence: float = 0.5
    ) -> List[TacticalRecommendation]:
        """
        Get top N recommendations filtered by confidence.
        
        Args:
            n: Number of recommendations to return
            min_confidence: Minimum confidence score (0-1)
            
        Returns:
            List of top recommendations
        """
        # Filter by confidence
        filtered = [
            r for r in self.recommendations
            if r.confidence_score >= min_confidence
        ]
        
        # Sort by priority (highest first), then by confidence (highest first)
        priority_order = {
            RecommendationPriority.CRITICAL: 5,
            RecommendationPriority.HIGH: 4,
            RecommendationPriority.MEDIUM: 3,
            RecommendationPriority.LOW: 2,
            RecommendationPriority.INFO: 1
        }
        
        filtered.sort(
            key=lambda r: (priority_order.get(r.priority, 0), r.confidence_score),
            reverse=True
        )
        
        return filtered[:n]
    
    def mark_recommendation_implemented(
        self,
        recommendation_id: str,
        outcome: str = "pending"
    ):
        """
        Mark a recommendation as implemented and track outcome.
        
        Args:
            recommendation_id: ID of the recommendation
            outcome: Outcome of implementation (success, failure, pending)
        """
        self.implemented_recommendations[recommendation_id] = {
            'implemented_at': self.match_context.match_minute,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_recommendation_report(
        self,
        team: str = 'A'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive recommendation report.
        
        Args:
            team: Team to generate report for
            
        Returns:
            Report dictionary
        """
        team_recommendations = [
            r for r in self.recommendations
            if r.target_team == team or r.target_team == ''
        ]
        
        # Group by category
        by_category = defaultdict(list)
        for rec in team_recommendations:
            by_category[rec.category.value].append({
                'id': rec.id,
                'title': rec.title,
                'priority': rec.priority.name,
                'confidence': rec.confidence_score,
                'description': rec.description
            })
        
        # Get top actionable recommendations
        actionable = [
            r for r in team_recommendations
            if r.actionable and r.confidence_score > 0.6
        ][:5]
        
        return {
            'team': team,
            'total_recommendations': len(team_recommendations),
            'by_category': dict(by_category),
            'top_actionable': [
                {
                    'id': r.id,
                    'title': r.title,
                    'category': r.category.value,
                    'priority': r.priority.name,
                    'expected_outcome': r.expected_outcome,
                    'implementation_steps': r.implementation_steps
                }
                for r in actionable
            ],
            'match_context': {
                'minute': self.match_context.match_minute,
                'score': f"{self.match_context.score_a}-{self.match_context.score_b}",
                'possession': f"{self.match_context.possession_a:.1f}%-{self.match_context.possession_b:.1f}%"
            }
        }
    
    def reset(self):
        """Reset all recommendation data."""
        self.recommendations = []
        self.recommendation_counter = 0
        self.player_snapshots = {}
        self.detected_patterns = defaultdict(list)
        self.opponent_weaknesses = defaultdict(list)
        self.implemented_recommendations = {}
        self.match_context = MatchContext(timestamp=0.0)
