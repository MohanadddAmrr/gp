"""
Tactical Insights Engine Module

Analyzes match data to provide tactical insights and recommendations.
Identifies successful/unsuccessful patterns, team strengths/weaknesses,
and suggests tactical adjustments.

Key Features:
- Pattern recognition for successful plays
- Team strength and weakness analysis
- Tactical adjustment recommendations
- Playing style comparison between teams
- Formation effectiveness analysis
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


class PatternType(Enum):
    """Types of tactical patterns."""
    PASSING_SEQUENCE = "passing_sequence"
    ATTACKING_MOVE = "attacking_move"
    DEFENSIVE_ACTION = "defensive_action"
    TRANSITION = "transition"
    SET_PIECE = "set_piece"
    PRESSING = "pressing"


class PatternOutcome(Enum):
    """Outcomes of tactical patterns."""
    SUCCESSFUL = "successful"
    UNSUCCESSFUL = "unsuccessful"
    NEUTRAL = "neutral"


class StrengthArea(Enum):
    """Areas where teams can be strong/weak."""
    ATTACKING_THIRD = "attacking_third"
    MIDFIELD = "midfield"
    DEFENSIVE_THIRD = "defensive_third"
    WIDE_AREAS = "wide_areas"
    CENTRAL_AREAS = "central_areas"
    TRANSITIONS = "transitions"
    SET_PIECES = "set_pieces"
    PRESSING = "pressing"
    COUNTER_ATTACK = "counter_attack"
    BALL_RETENTION = "ball_retention"


class TacticalStyle(Enum):
    """Tactical playing styles."""
    POSSESSION = "possession_based"
    DIRECT = "direct_play"
    COUNTER_ATTACK = "counter_attacking"
    HIGH_PRESS = "high_press"
    LOW_BLOCK = "low_block"
    WING_PLAY = "wing_play"
    TIKI_TAKA = "tiki_taka"
    LONG_BALL = "long_ball"


@dataclass
class TacticalPattern:
    """Represents a tactical pattern."""
    pattern_type: PatternType
    outcome: PatternOutcome
    team: str
    
    # Timing
    start_time: float
    end_time: float
    duration: float = 0.0
    
    # Participants
    players_involved: List[int] = field(default_factory=list)
    start_zone: str = ""
    end_zone: str = ""
    
    # Metrics
    pass_count: int = 0
    distance_covered: float = 0.0
    xg_generated: float = 0.0
    
    # Description
    description: str = ""
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class TeamStrengths:
    """Team strengths analysis."""
    team: str
    
    # Strength ratings (0-100)
    strengths: Dict[StrengthArea, float] = field(default_factory=dict)
    
    # Key metrics
    top_patterns: List[str] = field(default_factory=list)
    success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommended_focus: List[str] = field(default_factory=list)


@dataclass
class TacticalAdjustment:
    """Tactical adjustment recommendation."""
    priority: int  # 1-5, 5 being highest
    category: str
    current_issue: str
    recommendation: str
    expected_impact: str
    based_on_data: List[str] = field(default_factory=list)


@dataclass
class StyleComparison:
    """Comparison of playing styles between teams."""
    team_a_style: TacticalStyle
    team_b_style: TacticalStyle
    
    # Metrics comparison
    possession_diff: float = 0.0
    pass_length_diff: float = 0.0
    pressing_intensity_diff: float = 0.0
    tempo_diff: float = 0.0
    
    # Analysis
    key_differences: List[str] = field(default_factory=list)
    matchup_advantages: Dict[str, str] = field(default_factory=dict)


class TacticalInsightsEngine:
    """
    Tactical Insights Engine.
    
    Analyzes match patterns to provide tactical insights:
    - Identifies successful and unsuccessful patterns
    - Analyzes team strengths and weaknesses
    - Suggests tactical adjustments
    - Compares playing styles
    """
    
    def __init__(self):
        """Initialize the tactical insights engine."""
        self.patterns: List[TacticalPattern] = []
        self.team_strengths: Dict[str, TeamStrengths] = {}
        self.adjustments: List[TacticalAdjustment] = []
        
        # Match data storage
        self.pass_sequences: List[Dict] = []
        self.attacking_moves: List[Dict] = []
        self.defensive_actions: List[Dict] = []
        self.transitions: List[Dict] = []
        
        # Zone definitions (normalized 0-1)
        self.zones = {
            'defensive_left': (0.0, 0.0, 0.33, 0.5),
            'defensive_right': (0.0, 0.5, 0.33, 1.0),
            'defensive_center': (0.0, 0.33, 0.33, 0.67),
            'midfield_left': (0.33, 0.0, 0.67, 0.5),
            'midfield_right': (0.33, 0.5, 0.67, 1.0),
            'midfield_center': (0.33, 0.33, 0.67, 0.67),
            'attacking_left': (0.67, 0.0, 1.0, 0.5),
            'attacking_right': (0.67, 0.5, 1.0, 1.0),
            'attacking_center': (0.67, 0.33, 1.0, 0.67),
        }
    
    def add_pattern(
        self,
        pattern_type: PatternType,
        outcome: PatternOutcome,
        team: str,
        start_time: float,
        end_time: float,
        players_involved: List[int] = None,
        start_zone: str = "",
        end_zone: str = "",
        pass_count: int = 0,
        distance_covered: float = 0.0,
        xg_generated: float = 0.0,
        description: str = ""
    ) -> TacticalPattern:
        """Add a tactical pattern."""
        pattern = TacticalPattern(
            pattern_type=pattern_type,
            outcome=outcome,
            team=team,
            start_time=start_time,
            end_time=end_time,
            players_involved=players_involved or [],
            start_zone=start_zone,
            end_zone=end_zone,
            pass_count=pass_count,
            distance_covered=distance_covered,
            xg_generated=xg_generated,
            description=description
        )
        
        self.patterns.append(pattern)
        return pattern
    
    def analyze_pass_sequence(
        self,
        passes: List[Dict],
        team: str,
        outcome: PatternOutcome,
        xg_generated: float = 0.0
    ) -> TacticalPattern:
        """
        Analyze a passing sequence.
        
        Args:
            passes: List of pass events
            team: Team in possession
            outcome: Success/failure of the sequence
            xg_generated: xG generated by the sequence
            
        Returns:
            TacticalPattern object
        """
        if not passes:
            return None
        
        start_time = passes[0].get('timestamp', 0)
        end_time = passes[-1].get('timestamp', 0)
        
        players = list(set(p.get('passer_id') for p in passes))
        
        # Determine zones
        first_pass = passes[0]
        last_pass = passes[-1]
        
        start_zone = self._get_zone(
            first_pass.get('x', 0.5), 
            first_pass.get('y', 0.5)
        )
        end_zone = self._get_zone(
            last_pass.get('x', 0.5),
            last_pass.get('y', 0.5)
        )
        
        description = f"{len(passes)}-pass sequence from {start_zone} to {end_zone}"
        
        return self.add_pattern(
            pattern_type=PatternType.PASSING_SEQUENCE,
            outcome=outcome,
            team=team,
            start_time=start_time,
            end_time=end_time,
            players_involved=players,
            start_zone=start_zone,
            end_zone=end_zone,
            pass_count=len(passes),
            xg_generated=xg_generated,
            description=description
        )
    
    def _get_zone(self, x: float, y: float) -> str:
        """Determine which zone a position falls into."""
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return "unknown"
    
    def analyze_team_strengths(self, team: str) -> TeamStrengths:
        """
        Analyze strengths for a specific team.
        
        Args:
            team: Team identifier ('A' or 'B')
            
        Returns:
            TeamStrengths object
        """
        team_patterns = [p for p in self.patterns if p.team == team]
        
        if not team_patterns:
            return TeamStrengths(team=team)
        
        # Calculate success rates by pattern type
        success_rates = {}
        for pattern_type in PatternType:
            type_patterns = [p for p in team_patterns if p.pattern_type == pattern_type]
            if type_patterns:
                successful = len([p for p in type_patterns if p.outcome == PatternOutcome.SUCCESSFUL])
                success_rates[pattern_type.value] = successful / len(type_patterns) * 100
        
        # Calculate zone effectiveness
        zone_success = defaultdict(lambda: {'success': 0, 'total': 0})
        for pattern in team_patterns:
            if pattern.outcome == PatternOutcome.SUCCESSFUL:
                zone_success[pattern.start_zone]['success'] += 1
            zone_success[pattern.start_zone]['total'] += 1
        
        # Determine strengths by area
        strengths = {}
        
        # Attacking third strength
        attacking_patterns = [
            p for p in team_patterns 
            if 'attacking' in p.end_zone and p.outcome == PatternOutcome.SUCCESSFUL
        ]
        strengths[StrengthArea.ATTACKING_THIRD] = min(100, len(attacking_patterns) * 10)
        
        # Midfield control
        midfield_patterns = [
            p for p in team_patterns
            if 'midfield' in p.start_zone and p.outcome == PatternOutcome.SUCCESSFUL
        ]
        strengths[StrengthArea.MIDFIELD] = min(100, len(midfield_patterns) * 10)
        
        # Defensive solidity
        defensive_patterns = [
            p for p in team_patterns
            if p.pattern_type == PatternType.DEFENSIVE_ACTION and p.outcome == PatternOutcome.SUCCESSFUL
        ]
        strengths[StrengthArea.DEFENSIVE_THIRD] = min(100, len(defensive_patterns) * 15)
        
        # Transition speed
        transition_patterns = [
            p for p in team_patterns
            if p.pattern_type == PatternType.TRANSITION and p.outcome == PatternOutcome.SUCCESSFUL
        ]
        avg_transition_time = np.mean([p.duration for p in transition_patterns]) if transition_patterns else 5.0
        transition_score = max(0, 100 - (avg_transition_time * 10))
        strengths[StrengthArea.TRANSITIONS] = transition_score
        
        # Ball retention (passing success)
        passing_patterns = [p for p in team_patterns if p.pattern_type == PatternType.PASSING_SEQUENCE]
        if passing_patterns:
            successful_passes = len([p for p in passing_patterns if p.outcome == PatternOutcome.SUCCESSFUL])
            retention_score = (successful_passes / len(passing_patterns)) * 100
        else:
            retention_score = 50
        strengths[StrengthArea.BALL_RETENTION] = retention_score
        
        # Identify top patterns
        pattern_success = {}
        for pattern_type in PatternType:
            type_patterns = [p for p in team_patterns if p.pattern_type == pattern_type]
            if type_patterns:
                successful = len([p for p in type_patterns if p.outcome == PatternOutcome.SUCCESSFUL])
                pattern_success[pattern_type.value] = successful / len(type_patterns)
        
        top_patterns = sorted(
            pattern_success.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Generate recommendations
        recommendations = []
        
        # Find weak areas
        weak_areas = [
            area for area, score in strengths.items()
            if score < 40
        ]
        
        for area in weak_areas:
            if area == StrengthArea.ATTACKING_THIRD:
                recommendations.append("Focus on final third penetration and creativity")
            elif area == StrengthArea.MIDFIELD:
                recommendations.append("Improve midfield control and passing triangles")
            elif area == StrengthArea.DEFENSIVE_THIRD:
                recommendations.append("Strengthen defensive organization and positioning")
            elif area == StrengthArea.TRANSITIONS:
                recommendations.append("Work on quicker transitions when winning possession")
        
        # Find strong areas to exploit
        strong_areas = [
            area for area, score in strengths.items()
            if score > 70
        ]
        
        for area in strong_areas:
            recommendations.append(f"Exploit strength in {area.value.replace('_', ' ')}")
        
        strengths_obj = TeamStrengths(
            team=team,
            strengths=strengths,
            top_patterns=[p[0] for p in top_patterns],
            success_rates=success_rates,
            recommended_focus=recommendations
        )
        
        self.team_strengths[team] = strengths_obj
        return strengths_obj
    
    def generate_tactical_adjustments(self) -> List[TacticalAdjustment]:
        """
        Generate tactical adjustment recommendations.
        
        Returns:
            List of TacticalAdjustment objects
        """
        adjustments = []
        
        for team in ['A', 'B']:
            if team not in self.team_strengths:
                self.analyze_team_strengths(team)
            
            strengths = self.team_strengths[team]
            
            # Check for low ball retention
            retention_score = strengths.strengths.get(StrengthArea.BALL_RETENTION, 50)
            if retention_score < 50:
                adjustments.append(TacticalAdjustment(
                    priority=4,
                    category="Possession",
                    current_issue="Low pass completion rate",
                    recommendation="Simplify passing options and focus on shorter passes",
                    expected_impact="Improved ball retention and control",
                    based_on_data=[f"Ball retention score: {retention_score:.0f}/100"]
                ))
            
            # Check for weak attacking third
            attacking_score = strengths.strengths.get(StrengthArea.ATTACKING_THIRD, 50)
            if attacking_score < 40:
                adjustments.append(TacticalAdjustment(
                    priority=5,
                    category="Attack",
                    current_issue="Struggling to create chances in final third",
                    recommendation="Increase width and look for crossing opportunities",
                    expected_impact="More entries into dangerous areas",
                    based_on_data=[f"Attacking third effectiveness: {attacking_score:.0f}/100"]
                ))
            
            # Check for slow transitions
            transition_score = strengths.strengths.get(StrengthArea.TRANSITIONS, 50)
            if transition_score < 40:
                adjustments.append(TacticalAdjustment(
                    priority=3,
                    category="Transitions",
                    current_issue="Slow to transition from defense to attack",
                    recommendation="Position players higher up the pitch when in possession",
                    expected_impact="Quicker counter-attacking opportunities",
                    based_on_data=[f"Transition speed score: {transition_score:.0f}/100"]
                ))
            
            # Check pattern diversity
            team_patterns = [p for p in self.patterns if p.team == team]
            pattern_types = set(p.pattern_type for p in team_patterns)
            if len(pattern_types) < 3:
                adjustments.append(TacticalAdjustment(
                    priority=3,
                    category="Variety",
                    current_issue="Predictable play patterns",
                    recommendation="Introduce more varied attacking patterns",
                    expected_impact="Harder for opponents to defend against",
                    based_on_data=[f"Only {len(pattern_types)} pattern types used"]
                ))
        
        # Sort by priority
        adjustments.sort(key=lambda a: a.priority, reverse=True)
        self.adjustments = adjustments
        
        return adjustments
    
    def compare_playing_styles(
        self,
        team_a_stats: Dict[str, Any],
        team_b_stats: Dict[str, Any]
    ) -> StyleComparison:
        """
        Compare playing styles between two teams.
        
        Args:
            team_a_stats: Statistics for team A
            team_b_stats: Statistics for team B
            
        Returns:
            StyleComparison object
        """
        # Determine styles based on metrics
        team_a_style = self._determine_style(team_a_stats)
        team_b_style = self._determine_style(team_b_stats)
        
        # Calculate differences
        possession_diff = abs(
            team_a_stats.get('possession', 50) - 
            team_b_stats.get('possession', 50)
        )
        
        pass_length_diff = abs(
            team_a_stats.get('avg_pass_length', 15) - 
            team_b_stats.get('avg_pass_length', 15)
        )
        
        pressing_diff = abs(
            team_a_stats.get('pressing_intensity', 50) - 
            team_b_stats.get('pressing_intensity', 50)
        )
        
        tempo_diff = abs(
            team_a_stats.get('passes_per_minute', 10) - 
            team_b_stats.get('passes_per_minute', 10)
        )
        
        # Identify key differences
        differences = []
        
        if possession_diff > 15:
            dominant = "A" if team_a_stats.get('possession', 50) > team_b_stats.get('possession', 50) else "B"
            differences.append(f"Team {dominant} dominates possession ({possession_diff:.0f}% difference)")
        
        if pass_length_diff > 5:
            longer = "A" if team_a_stats.get('avg_pass_length', 15) > team_b_stats.get('avg_pass_length', 15) else "B"
            differences.append(f"Team {longer} plays more direct (longer average passes)")
        
        if pressing_diff > 20:
            higher = "A" if team_a_stats.get('pressing_intensity', 50) > team_b_stats.get('pressing_intensity', 50) else "B"
            differences.append(f"Team {higher} employs higher pressing intensity")
        
        # Determine matchup advantages
        advantages = {}
        
        if team_a_style == TacticalStyle.HIGH_PRESS and team_b_style == TacticalStyle.POSSESSION:
            advantages['A'] = "Pressing can disrupt possession-based buildup"
        elif team_b_style == TacticalStyle.HIGH_PRESS and team_a_style == TacticalStyle.POSSESSION:
            advantages['B'] = "Pressing can disrupt possession-based buildup"
        
        if team_a_style == TacticalStyle.COUNTER_ATTACK and team_b_style == TacticalStyle.HIGH_PRESS:
            advantages['A'] = "Counter-attacking exploits space behind high press"
        elif team_b_style == TacticalStyle.COUNTER_ATTACK and team_a_style == TacticalStyle.HIGH_PRESS:
            advantages['B'] = "Counter-attacking exploits space behind high press"
        
        comparison = StyleComparison(
            team_a_style=team_a_style,
            team_b_style=team_b_style,
            possession_diff=possession_diff,
            pass_length_diff=pass_length_diff,
            pressing_intensity_diff=pressing_diff,
            tempo_diff=tempo_diff,
            key_differences=differences,
            matchup_advantages=advantages
        )
        
        return comparison
    
    def _determine_style(self, stats: Dict[str, Any]) -> TacticalStyle:
        """Determine playing style from statistics."""
        possession = stats.get('possession', 50)
        pass_length = stats.get('avg_pass_length', 15)
        pressing = stats.get('pressing_intensity', 50)
        long_balls = stats.get('long_ball_pct', 10)
        
        if pressing > 70:
            return TacticalStyle.HIGH_PRESS
        elif possession > 60 and pass_length < 15:
            return TacticalStyle.TIKI_TAKA
        elif possession > 55:
            return TacticalStyle.POSSESSION
        elif long_balls > 25:
            return TacticalStyle.LONG_BALL
        elif pass_length > 20:
            return TacticalStyle.DIRECT
        else:
            return TacticalStyle.COUNTER_ATTACK
    
    def get_successful_patterns(self, team: str = None, min_success_rate: float = 0.5) -> List[Dict]:
        """
        Get patterns with high success rates.
        
        Args:
            team: Filter by team (optional)
            min_success_rate: Minimum success rate (0-1)
            
        Returns:
            List of successful pattern summaries
        """
        patterns_to_check = self.patterns
        if team:
            patterns_to_check = [p for p in patterns_to_check if p.team == team]
        
        # Group by type and calculate success rates
        pattern_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'examples': []})
        
        for pattern in patterns_to_check:
            key = (pattern.pattern_type.value, pattern.start_zone, pattern.end_zone)
            pattern_stats[key]['total'] += 1
            if pattern.outcome == PatternOutcome.SUCCESSFUL:
                pattern_stats[key]['success'] += 1
            pattern_stats[key]['examples'].append(pattern)
        
        # Filter by success rate
        successful_patterns = []
        for key, stats in pattern_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            if success_rate >= min_success_rate and stats['total'] >= 3:
                successful_patterns.append({
                    'pattern_type': key[0],
                    'start_zone': key[1],
                    'end_zone': key[2],
                    'success_rate': round(success_rate * 100, 1),
                    'occurrences': stats['total'],
                    'avg_duration': round(np.mean([p.duration for p in stats['examples']]), 2),
                    'avg_passes': round(np.mean([p.pass_count for p in stats['examples']]), 1)
                })
        
        # Sort by success rate
        successful_patterns.sort(key=lambda p: p['success_rate'], reverse=True)
        
        return successful_patterns
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate a comprehensive tactical insights report."""
        # Analyze both teams
        for team in ['A', 'B']:
            self.analyze_team_strengths(team)
        
        # Generate adjustments
        adjustments = self.generate_tactical_adjustments()
        
        # Get successful patterns
        successful_patterns = self.get_successful_patterns()
        
        return {
            'team_strengths': {
                team: {
                    'strengths': {k.value: v for k, v in ts.strengths.items()},
                    'top_patterns': ts.top_patterns,
                    'success_rates': ts.success_rates,
                    'recommendations': ts.recommended_focus
                }
                for team, ts in self.team_strengths.items()
            },
            'tactical_adjustments': [
                {
                    'priority': adj.priority,
                    'category': adj.category,
                    'issue': adj.current_issue,
                    'recommendation': adj.recommendation,
                    'expected_impact': adj.expected_impact,
                    'based_on': adj.based_on_data
                }
                for adj in adjustments
            ],
            'successful_patterns': successful_patterns,
            'total_patterns_analyzed': len(self.patterns),
            'pattern_breakdown': {
                pattern_type.value: len([p for p in self.patterns if p.pattern_type == pattern_type])
                for pattern_type in PatternType
            }
        }
    
    def reset(self):
        """Reset all data."""
        self.patterns = []
        self.team_strengths = {}
        self.adjustments = []
        self.pass_sequences = []
        self.attacking_moves = []
        self.defensive_actions = []
        self.transitions = []
