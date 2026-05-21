"""
Opponent Analysis Module

Analyzes historical match data to identify opponent patterns and tendencies.
Generates scout reports with formation predictions, favorite zones, and passing patterns.

Key Features:
- Pattern recognition from historical data
- Opponent tendency identification
- Formation prediction
- Scout report generation
- Player-specific analysis
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta


class Formation(Enum):
    """Common football formations."""
    F_4_4_2 = "4-4-2"
    F_4_3_3 = "4-3-3"
    F_4_2_3_1 = "4-2-3-1"
    F_3_5_2 = "3-5-2"
    F_5_3_2 = "5-3-2"
    F_4_5_1 = "4-5-1"
    F_3_4_3 = "3-4-3"
    F_4_1_4_1 = "4-1-4-1"
    UNKNOWN = "unknown"


class PlayStyle(Enum):
    """Playing styles."""
    POSSESSION = "possession_based"
    DIRECT = "direct"
    COUNTER_ATTACK = "counter_attacking"
    HIGH_PRESS = "high_press"
    LOW_BLOCK = "low_block"
    WING_PLAY = "wing_play"
    BALANCED = "balanced"


class ZonePreference(Enum):
    """Zone preferences."""
    LEFT_WING = "left_wing"
    RIGHT_WING = "right_wing"
    CENTER = "center"
    LEFT_HALF = "left_half_space"
    RIGHT_HALF = "right_half_space"


@dataclass
class OpponentProfile:
    """Profile of an opponent team."""
    team_name: str
    
    # Formation tendencies
    formations_used: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    formation_transitions: List[Dict] = field(default_factory=list)
    preferred_formation: str = ""
    
    # Playing style
    primary_style: PlayStyle = PlayStyle.BALANCED
    secondary_style: PlayStyle = PlayStyle.BALANCED
    style_consistency: float = 0.0  # 0-1
    
    # Zone preferences
    zone_preferences: Dict[str, float] = field(default_factory=dict)
    favorite_attacking_zone: str = ""
    favorite_buildup_zone: str = ""
    
    # Passing patterns
    avg_pass_length: float = 0.0
    pass_direction_bias: Dict[str, float] = field(default_factory=dict)
    preferred_passing_network: Dict[str, List[str]] = field(default_factory=dict)
    
    # Key players
    key_players: Dict[str, List[int]] = field(default_factory=dict)
    danger_players: List[int] = field(default_factory=list)
    
    # Defensive tendencies
    pressing_intensity: float = 0.0  # 0-100
    defensive_line_height: float = 0.5  # 0-1 (0=deep, 1=high)
    
    # Set pieces
    set_piece_threat: float = 0.0  # 0-100
    set_piece_strengths: List[str] = field(default_factory=list)
    
    # Match data
    matches_analyzed: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ScoutReport:
    """Scout report for an upcoming opponent."""
    opponent_name: str
    report_date: datetime = field(default_factory=datetime.now)
    
    # Predicted lineup/formation
    predicted_formation: str = ""
    formation_confidence: float = 0.0
    
    # Key threats
    main_threats: List[Dict] = field(default_factory=list)
    danger_zones: List[str] = field(default_factory=list)
    
    # Tactical analysis
    expected_style: PlayStyle = PlayStyle.BALANCED
    tactical_summary: str = ""
    
    # Vulnerabilities
    weaknesses: List[str] = field(default_factory=list)
    exploitable_patterns: List[str] = field(default_factory=list)
    
    # Recommendations
    defensive_recommendations: List[str] = field(default_factory=list)
    attacking_recommendations: List[str] = field(default_factory=list)
    
    # Historical context
    head_to_head: Dict[str, Any] = field(default_factory=dict)
    recent_form: List[str] = field(default_factory=list)


@dataclass
class PatternCluster:
    """Cluster of similar patterns."""
    pattern_type: str
    frequency: int
    success_rate: float
    avg_start_zone: str
    avg_end_zone: str
    typical_duration: float
    key_players: List[int] = field(default_factory=list)


class OpponentAnalyzer:
    """
    Opponent Analysis Engine.
    
    Analyzes historical match data to identify opponent patterns:
    - Formation tendencies and transitions
    - Favorite zones and passing patterns
    - Player roles and key threats
    - Tactical vulnerabilities
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize the opponent analyzer.
        
        Args:
            db_manager: Optional database manager for historical data
        """
        self.db_manager = db_manager
        self.opponent_profiles: Dict[str, OpponentProfile] = {}
        self.pattern_history: List[Dict] = []
        
        # Analysis thresholds
        self.min_matches_for_profile = 3
        self.pattern_similarity_threshold = 0.8
    
    def analyze_match_data(
        self,
        team_name: str,
        match_data: Dict[str, Any]
    ) -> OpponentProfile:
        """
        Analyze match data to build/update opponent profile.
        
        Args:
            team_name: Name of the team
            match_data: Match statistics and events
            
        Returns:
            Updated OpponentProfile
        """
        # Get or create profile
        if team_name not in self.opponent_profiles:
            self.opponent_profiles[team_name] = OpponentProfile(team_name=team_name)
        
        profile = self.opponent_profiles[team_name]
        
        # Update formation data
        formation = match_data.get('formation', 'unknown')
        if formation:
            profile.formations_used[formation] += 1
        
        # Track formation transitions
        formation_changes = match_data.get('formation_changes', [])
        for change in formation_changes:
            profile.formation_transitions.append(change)
        
        # Determine preferred formation
        if profile.formations_used:
            profile.preferred_formation = max(
                profile.formations_used.items(),
                key=lambda x: x[1]
            )[0]
        
        # Analyze playing style
        style = self._analyze_playing_style(match_data)
        profile.primary_style = style
        
        # Analyze zone preferences
        self._analyze_zone_preferences(profile, match_data)
        
        # Analyze passing patterns
        self._analyze_passing_patterns(profile, match_data)
        
        # Identify key players
        self._identify_key_players(profile, match_data)
        
        # Analyze defensive tendencies
        profile.pressing_intensity = match_data.get('pressing_intensity', 50)
        profile.defensive_line_height = match_data.get('defensive_line_height', 0.5)
        
        # Analyze set pieces
        self._analyze_set_pieces(profile, match_data)
        
        # Update metadata
        profile.matches_analyzed += 1
        profile.last_updated = datetime.now()
        
        return profile
    
    def _analyze_playing_style(self, match_data: Dict[str, Any]) -> PlayStyle:
        """Determine playing style from match data."""
        possession = match_data.get('possession_percentage', 50)
        pass_accuracy = match_data.get('pass_accuracy', 75)
        avg_pass_length = match_data.get('avg_pass_length', 15)
        pressing = match_data.get('pressing_intensity', 50)
        
        # Decision tree for style
        if pressing > 70:
            return PlayStyle.HIGH_PRESS
        elif possession > 60 and pass_accuracy > 85:
            return PlayStyle.POSSESSION
        elif avg_pass_length > 20:
            return PlayStyle.DIRECT
        elif possession < 45 and pressing > 60:
            return PlayStyle.COUNTER_ATTACK
        else:
            return PlayStyle.BALANCED
    
    def _analyze_zone_preferences(self, profile: OpponentProfile, match_data: Dict):
        """Analyze zone preferences from match data."""
        zone_data = match_data.get('zone_activity', {})
        
        # Calculate zone preferences as percentages
        total_activity = sum(zone_data.values()) if zone_data else 1
        
        profile.zone_preferences = {
            zone: (count / total_activity * 100) if total_activity > 0 else 0
            for zone, count in zone_data.items()
        }
        
        # Determine favorite zones
        if profile.zone_preferences:
            sorted_zones = sorted(
                profile.zone_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )
            profile.favorite_attacking_zone = sorted_zones[0][0] if sorted_zones else ""
            profile.favorite_buildup_zone = sorted_zones[1][0] if len(sorted_zones) > 1 else ""
    
    def _analyze_passing_patterns(self, profile: OpponentProfile, match_data: Dict):
        """Analyze passing patterns from match data."""
        pass_data = match_data.get('passing', {})
        
        profile.avg_pass_length = pass_data.get('avg_length', 15)
        
        # Pass direction bias
        directions = pass_data.get('directions', {})
        total = sum(directions.values()) if directions else 1
        
        profile.pass_direction_bias = {
            direction: (count / total * 100) if total > 0 else 0
            for direction, count in directions.items()
        }
        
        # Preferred passing combinations
        combinations = pass_data.get('combinations', [])
        profile.preferred_passing_network = self._build_passing_network(combinations)
    
    def _build_passing_network(
        self,
        combinations: List[Dict]
    ) -> Dict[str, List[str]]:
        """Build passing network from combinations."""
        network = defaultdict(list)
        
        for combo in combinations:
            from_player = combo.get('from', '')
            to_player = combo.get('to', '')
            if from_player and to_player:
                network[from_player].append(to_player)
        
        return dict(network)
    
    def _identify_key_players(self, profile: OpponentProfile, match_data: Dict):
        """Identify key players from match data."""
        player_stats = match_data.get('player_stats', [])
        
        # Categorize players by role
        profile.key_players = {
            'top_scorer': [],
            'playmaker': [],
            'defensive_anchor': [],
            'wide_threat': []
        }
        
        for player in player_stats:
            player_id = player.get('player_id')
            
            # Top scorer
            if player.get('goals', 0) > 0:
                profile.key_players['top_scorer'].append(player_id)
            
            # Playmaker (high assists/passes)
            if player.get('assists', 0) > 0 or player.get('passes_completed', 0) > 50:
                profile.key_players['playmaker'].append(player_id)
            
            # Defensive anchor
            if player.get('tackles', 0) > 5 or player.get('interceptions', 0) > 5:
                profile.key_players['defensive_anchor'].append(player_id)
            
            # Wide threat
            if player.get('crosses', 0) > 5:
                profile.key_players['wide_threat'].append(player_id)
        
        # Identify danger players (most involved in attacks)
        attack_involvement = {}
        for player in player_stats:
            involvement = (
                player.get('shots', 0) * 3 +
                player.get('key_passes', 0) * 2 +
                player.get('dribbles', 0)
            )
            if involvement > 5:
                attack_involvement[player['player_id']] = involvement
        
        profile.danger_players = sorted(
            attack_involvement.keys(),
            key=lambda x: attack_involvement[x],
            reverse=True
        )[:5]
    
    def _analyze_set_pieces(self, profile: OpponentProfile, match_data: Dict):
        """Analyze set piece effectiveness."""
        set_pieces = match_data.get('set_pieces', {})
        
        # Calculate set piece threat
        goals_from_sp = set_pieces.get('goals_scored', 0)
        total_sp = set_pieces.get('total', 1)
        
        profile.set_piece_threat = min(100, (goals_from_sp / max(total_sp, 1)) * 500)
        
        # Identify strengths
        strengths = []
        if set_pieces.get('corners_successful', 0) > 3:
            strengths.append('corners')
        if set_pieces.get('free_kicks_successful', 0) > 2:
            strengths.append('free_kicks')
        if set_pieces.get('penalties_successful', 0) > 0:
            strengths.append('penalties')
        
        profile.set_piece_strengths = strengths
    
    def predict_formation(self, team_name: str, context: Dict = None) -> Tuple[str, float]:
        """
        Predict formation for upcoming match.
        
        Args:
            team_name: Name of the opponent
            context: Optional context (venue, competition, etc.)
            
        Returns:
            Tuple of (predicted_formation, confidence)
        """
        if team_name not in self.opponent_profiles:
            return (Formation.UNKNOWN.value, 0.0)
        
        profile = self.opponent_profiles[team_name]
        
        if not profile.formations_used:
            return (Formation.UNKNOWN.value, 0.0)
        
        # Get most common formation
        total_matches = sum(profile.formations_used.values())
        most_common = max(profile.formations_used.items(), key=lambda x: x[1])
        
        formation = most_common[0]
        confidence = most_common[1] / total_matches if total_matches > 0 else 0
        
        # Adjust based on context
        if context:
            venue = context.get('venue', '')
            if venue == 'home' and '4-3-3' in profile.formations_used:
                # Teams often play more attacking at home
                formation = '4-3-3'
                confidence = min(0.9, confidence + 0.1)
        
        return (formation, round(confidence, 2))
    
    def identify_patterns(
        self,
        team_name: str,
        min_frequency: int = 3
    ) -> List[PatternCluster]:
        """
        Identify recurring patterns for an opponent.
        
        Args:
            team_name: Name of the opponent
            min_frequency: Minimum occurrences to be considered a pattern
            
        Returns:
            List of PatternCluster objects
        """
        # Get patterns from history
        team_patterns = [
            p for p in self.pattern_history
            if p.get('team') == team_name
        ]
        
        if not team_patterns:
            return []
        
        # Cluster similar patterns
        clusters = defaultdict(list)
        
        for pattern in team_patterns:
            key = (pattern.get('type'), pattern.get('start_zone'), pattern.get('end_zone'))
            clusters[key].append(pattern)
        
        # Create pattern clusters
        result = []
        for (p_type, start_zone, end_zone), patterns in clusters.items():
            if len(patterns) >= min_frequency:
                successful = len([p for p in patterns if p.get('success', False)])
                
                cluster = PatternCluster(
                    pattern_type=p_type,
                    frequency=len(patterns),
                    success_rate=successful / len(patterns),
                    avg_start_zone=start_zone,
                    avg_end_zone=end_zone,
                    typical_duration=np.mean([p.get('duration', 0) for p in patterns]),
                    key_players=list(set(
                        player for p in patterns
                        for player in p.get('players', [])
                    ))
                )
                result.append(cluster)
        
        # Sort by frequency
        result.sort(key=lambda x: x.frequency, reverse=True)
        
        return result
    
    def generate_scout_report(
        self,
        opponent_name: str,
        our_team_name: str = None,
        context: Dict = None
    ) -> ScoutReport:
        """
        Generate a scout report for an opponent.
        
        Args:
            opponent_name: Name of the opponent
            our_team_name: Optional name of our team for head-to-head
            context: Optional match context
            
        Returns:
            ScoutReport object
        """
        if opponent_name not in self.opponent_profiles:
            # Try to load from database
            if self.db_manager:
                self._load_profile_from_db(opponent_name)
        
        profile = self.opponent_profiles.get(opponent_name)
        
        if not profile or profile.matches_analyzed < self.min_matches_for_profile:
            return ScoutReport(
                opponent_name=opponent_name,
                tactical_summary="Insufficient data for analysis"
            )
        
        report = ScoutReport(opponent_name=opponent_name)
        
        # Predict formation
        predicted_form, confidence = self.predict_formation(opponent_name, context)
        report.predicted_formation = predicted_form
        report.formation_confidence = confidence
        
        # Identify main threats
        report.main_threats = self._identify_threats(profile)
        report.danger_zones = [
            profile.favorite_attacking_zone,
            profile.favorite_buildup_zone
        ]
        
        # Expected style
        report.expected_style = profile.primary_style
        report.tactical_summary = self._generate_tactical_summary(profile)
        
        # Identify weaknesses
        report.weaknesses = self._identify_weaknesses(profile)
        report.exploitable_patterns = self._find_exploitable_patterns(opponent_name)
        
        # Generate recommendations
        report.defensive_recommendations = self._generate_defensive_recommendations(profile)
        report.attacking_recommendations = self._generate_attacking_recommendations(profile)
        
        # Head to head
        if our_team_name:
            report.head_to_head = self._get_head_to_head(our_team_name, opponent_name)
        
        # Recent form (placeholder - would come from database)
        report.recent_form = ['W', 'D', 'W', 'L', 'W']  # Example
        
        return report
    
    def _identify_threats(self, profile: OpponentProfile) -> List[Dict]:
        """Identify main threats from opponent profile."""
        threats = []
        
        # Set piece threat
        if profile.set_piece_threat > 60:
            threats.append({
                'type': 'set_pieces',
                'description': f"Dangerous from set pieces ({profile.set_piece_threat:.0f}% threat)",
                'severity': 'high' if profile.set_piece_threat > 75 else 'medium'
            })
        
        # Key players
        for role, players in profile.key_players.items():
            if players:
                threats.append({
                    'type': role,
                    'description': f"Has dangerous {role.replace('_', ' ')}(s)",
                    'players': players,
                    'severity': 'high'
                })
        
        # Zone threat
        if profile.favorite_attacking_zone:
            threats.append({
                'type': 'zone',
                'description': f"Attacks predominantly through {profile.favorite_attacking_zone}",
                'severity': 'medium'
            })
        
        return threats
    
    def _generate_tactical_summary(self, profile: OpponentProfile) -> str:
        """Generate tactical summary text."""
        summary_parts = []
        
        # Formation
        if profile.preferred_formation:
            summary_parts.append(
                f"Typically sets up in a {profile.preferred_formation} formation."
            )
        
        # Style
        summary_parts.append(
            f"Plays a {profile.primary_style.value.replace('_', ' ')} style."
        )
        
        # Pressing
        if profile.pressing_intensity > 70:
            summary_parts.append("Employs aggressive high pressing.")
        elif profile.pressing_intensity < 40:
            summary_parts.append("Prefers to sit deep and counter-press.")
        
        # Buildup
        if profile.favorite_buildup_zone:
            summary_parts.append(
                f"Builds play primarily through the {profile.favorite_buildup_zone}."
            )
        
        return " ".join(summary_parts)
    
    def _identify_weaknesses(self, profile: OpponentProfile) -> List[str]:
        """Identify opponent weaknesses."""
        weaknesses = []
        
        # Low pressing
        if profile.pressing_intensity < 40:
            weaknesses.append("Vulnerable to quick transitions due to low pressing intensity")
        
        # High line
        if profile.defensive_line_height > 0.7:
            weaknesses.append("High defensive line can be exploited with runs in behind")
        
        # One-sided attack
        zone_prefs = profile.zone_preferences
        if zone_prefs:
            max_zone = max(zone_prefs.values())
            if max_zone > 50:
                weaknesses.append("Over-reliance on one side of attack - can be unbalanced")
        
        # Weak set piece defense (assumed if not strong)
        if profile.set_piece_threat < 30:
            weaknesses.append("May be vulnerable defending set pieces")
        
        return weaknesses
    
    def _find_exploitable_patterns(self, team_name: str) -> List[str]:
        """Find patterns that can be exploited."""
        patterns = self.identify_patterns(team_name)
        exploitable = []
        
        for pattern in patterns:
            if pattern.success_rate < 0.4:
                exploitable.append(
                    f"They struggle with {pattern.pattern_type} from {pattern.avg_start_zone} "
                    f"(only {pattern.success_rate*100:.0f}% success)"
                )
        
        return exploitable
    
    def _generate_defensive_recommendations(self, profile: OpponentProfile) -> List[str]:
        """Generate defensive recommendations."""
        recommendations = []
        
        # Formation-specific
        if profile.preferred_formation in ['4-3-3', '3-4-3']:
            recommendations.append("Watch for overlapping wing-backs")
        
        # Zone-specific
        if profile.favorite_attacking_zone:
            recommendations.append(
                f"Double up on {profile.favorite_attacking_zone} to limit their main threat"
            )
        
        # Pressing
        if profile.pressing_intensity > 70:
            recommendations.append("Play through their press with quick one-touch passing")
            recommendations.append("Look for long balls over the top to exploit space")
        
        # Key players
        if profile.key_players.get('playmaker'):
            recommendations.append("Mark their playmaker closely - don't let them dictate tempo")
        
        # Set pieces
        if 'corners' in profile.set_piece_strengths:
            recommendations.append("Be extra vigilant defending corners")
        
        return recommendations
    
    def _generate_attacking_recommendations(self, profile: OpponentProfile) -> List[str]:
        """Generate attacking recommendations."""
        recommendations = []
        
        # Defensive vulnerabilities
        if profile.pressing_intensity < 40:
            recommendations.append("Build from the back - they won't press high")
        
        if profile.defensive_line_height > 0.7:
            recommendations.append("Make runs in behind their high line")
        
        # Space in wide areas
        if profile.preferred_formation in ['4-4-2', '4-1-4-1']:
            recommendations.append("Exploit space in wide areas between defense and midfield")
        
        # Transition opportunities
        if profile.primary_style == PlayStyle.POSSESSION:
            recommendations.append("Hit them on the counter when they lose possession")
        
        return recommendations
    
    def _get_head_to_head(self, team_a: str, team_b: str) -> Dict:
        """Get head-to-head record."""
        # Would query database in real implementation
        return {
            'matches_played': 0,
            'team_a_wins': 0,
            'team_b_wins': 0,
            'draws': 0,
            'note': 'No recent head-to-head data available'
        }
    
    def _load_profile_from_db(self, team_name: str):
        """Load opponent profile from database."""
        # Would implement database query here
        pass
    
    def export_profile_json(self, team_name: str) -> Dict:
        """Export opponent profile as JSON."""
        if team_name not in self.opponent_profiles:
            return {}
        
        profile = self.opponent_profiles[team_name]
        
        return {
            'team_name': profile.team_name,
            'matches_analyzed': profile.matches_analyzed,
            'preferred_formation': profile.preferred_formation,
            'formations_used': dict(profile.formations_used),
            'primary_style': profile.primary_style.value,
            'zone_preferences': profile.zone_preferences,
            'favorite_attacking_zone': profile.favorite_attacking_zone,
            'avg_pass_length': profile.avg_pass_length,
            'pressing_intensity': profile.pressing_intensity,
            'defensive_line_height': profile.defensive_line_height,
            'set_piece_threat': profile.set_piece_threat,
            'key_players': profile.key_players,
            'last_updated': profile.last_updated.isoformat()
        }
    
    def reset(self):
        """Reset all data."""
        self.opponent_profiles = {}
        self.pattern_history = []
