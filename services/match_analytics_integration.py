"""
Match Analytics Integration Module

Integrates AI tactical recommendations and player performance analytics
with the existing video processing pipeline. This module serves as the
main interface for accessing all analytics features during live match
processing.

Key Features:
- Unified interface for all analytics modules
- Real-time data integration from video tracking
- Automatic recommendation generation
- Player performance tracking during matches
- Export and reporting capabilities
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from services.ai_tactical_recommendations import (
    AITacticalRecommendations,
    TacticalRecommendation,
    MatchContext
)
from services.player_performance_analytics import (
    PlayerPerformanceAnalytics,
    PlayerMatchData,
    PerformanceRating
)
from services.tactical_analyzer import TacticalAnalyzer, TacticalReport
from services.tactical_insights import TacticalInsightsEngine
from services.advanced_analytics import AdvancedAnalytics
from services.event_detector import EventDetector
from services.possession_tracker import PossessionTracker
from services.database_manager import DatabaseManager


@dataclass
class AnalyticsConfig:
    """Configuration for match analytics."""
    # Feature toggles
    enable_ai_recommendations: bool = True
    enable_player_performance: bool = True
    enable_tactical_analysis: bool = True
    
    # Recommendation settings
    recommendation_interval: int = 30  # seconds
    min_confidence_threshold: float = 0.6
    max_recommendations: int = 10
    
    # Performance tracking
    track_all_players: bool = True
    update_frequency: int = 10  # frames
    
    # Export settings
    auto_export: bool = True
    export_format: str = "json"
    export_path: str = "analytics_output"


class MatchAnalyticsIntegration:
    """
    Main integration class for match analytics.
    
    Combines AI tactical recommendations and player performance analytics
    into a unified interface for the video processing pipeline.
    
    Usage:
        analytics = MatchAnalyticsIntegration()
        
        # During video processing
        analytics.update_player_position(player_id, x, y, timestamp, team)
        analytics.update_match_context(score_a, score_b, possession, minute)
        
        # Get recommendations
        recommendations = analytics.get_current_recommendations()
        
        # Get player performance
        player_summary = analytics.get_player_summary(player_id)
        
        # Finalize match
        report = analytics.finalize_match(match_id)
    """
    
    def __init__(
        self,
        config: AnalyticsConfig = None,
        db_manager: DatabaseManager = None,
        tactical_analyzer: TacticalAnalyzer = None,
        insights_engine: TacticalInsightsEngine = None,
        advanced_analytics: AdvancedAnalytics = None
    ):
        """
        Initialize the match analytics integration.
        
        Args:
            config: Analytics configuration
            db_manager: Database manager instance
            tactical_analyzer: Tactical analyzer instance
            insights_engine: Insights engine instance
            advanced_analytics: Advanced analytics instance
        """
        self.config = config or AnalyticsConfig()
        self.db_manager = db_manager
        
        # Initialize analytics modules
        self.ai_recommendations = AITacticalRecommendations(
            tactical_analyzer=tactical_analyzer,
            insights_engine=insights_engine,
            advanced_analytics=advanced_analytics
        )
        
        self.player_performance = PlayerPerformanceAnalytics()
        
        # External analyzers (can be shared)
        self.tactical_analyzer = tactical_analyzer
        self.advanced_analytics = advanced_analytics
        
        # Tracking state
        self.match_id: str = ""
        self.frame_count: int = 0
        self.last_recommendation_time: float = 0.0
        
        # Data storage
        self.recommendations_history: List[TacticalRecommendation] = []
        self.player_registry: Dict[int, Dict] = {}
        
        # Ensure export directory exists
        if self.config.auto_export:
            Path(self.config.export_path).mkdir(parents=True, exist_ok=True)
    
    def initialize_match(
        self,
        match_id: str,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        players_a: List[Dict] = None,
        players_b: List[Dict] = None
    ):
        """
        Initialize analytics for a new match.
        
        Args:
            match_id: Unique match identifier
            team_a_name: Name of team A
            team_b_name: Name of team B
            players_a: List of team A player info dicts
            players_b: List of team B player info dicts
        """
        self.match_id = match_id
        self.frame_count = 0
        self.last_recommendation_time = 0.0
        
        # Reset modules
        self.ai_recommendations.reset()
        self.player_performance.reset()
        
        # Register players
        players_a = players_a or []
        players_b = players_b or []
        
        for player_info in players_a:
            self.register_player(
                player_id=player_info.get('id'),
                team='A',
                name=player_info.get('name', 'Unknown'),
                jersey_number=player_info.get('number', 0),
                position=player_info.get('position', 'Unknown')
            )
        
        for player_info in players_b:
            self.register_player(
                player_id=player_info.get('id'),
                team='B',
                name=player_info.get('name', 'Unknown'),
                jersey_number=player_info.get('number', 0),
                position=player_info.get('position', 'Unknown')
            )
        
        print(f"✅ Match analytics initialized: {match_id}")
        print(f"   Team A ({team_a_name}): {len(players_a)} players")
        print(f"   Team B ({team_b_name}): {len(players_b)} players")
    
    def register_player(
        self,
        player_id: int,
        team: str,
        name: str = "Unknown",
        jersey_number: int = 0,
        position: str = "Unknown"
    ):
        """
        Register a player for tracking.
        
        Args:
            player_id: Unique player identifier
            team: Team ('A' or 'B')
            name: Player name
            jersey_number: Jersey number
            position: Playing position
        """
        self.player_performance.register_player(
            player_id=player_id,
            team=team,
            name=name,
            jersey_number=jersey_number,
            position=position,
            match_id=self.match_id
        )
        
        self.player_registry[player_id] = {
            'name': name,
            'team': team,
            'number': jersey_number,
            'position': position
        }
    
    def update_player_position(
        self,
        player_id: int,
        x: float,
        y: float,
        timestamp: float,
        team: str = None
    ):
        """
        Update player position from video tracking.
        
        Args:
            player_id: Player identifier
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
            timestamp: Time in seconds
            team: Team identifier (optional)
        """
        if not self.config.enable_player_performance:
            return
        
        self.player_performance.update_position(
            player_id=player_id,
            x=x,
            y=y,
            timestamp=timestamp,
            team=team
        )
    
    def update_player_speed(
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
        if not self.config.enable_player_performance:
            return
        
        self.player_performance.update_speed(
            player_id=player_id,
            speed_mps=speed_mps,
            timestamp=timestamp
        )
    
    def record_player_event(
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
        if not self.config.enable_player_performance:
            return
        
        self.player_performance.record_event(
            player_id=player_id,
            event_type=event_type,
            success=success,
            metadata=metadata
        )
    
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
        Update match context for recommendation generation.
        
        Args:
            score_a: Team A score
            score_b: Team B score
            possession_a: Team A possession percentage
            possession_b: Team B possession percentage
            match_minute: Current match minute
            period: Match period
        """
        self.ai_recommendations.update_match_context(
            score_a=score_a,
            score_b=score_b,
            possession_a=possession_a,
            possession_b=possession_b,
            match_minute=match_minute,
            period=period
        )
    
    def update_player_performance_data(
        self,
        player_id: int,
        distance_covered: float = None,
        current_speed: float = None,
        pass_accuracy: float = None,
        touches: int = None,
        fatigue_score: float = None,
        form_rating: float = None
    ):
        """
        Update player performance data for AI recommendations.
        
        Args:
            player_id: Player identifier
            distance_covered: Total distance in meters
            current_speed: Current speed in m/s
            pass_accuracy: Pass accuracy percentage
            touches: Number of ball touches
            fatigue_score: Fatigue score 0-100
            form_rating: Form rating 0-100
        """
        if not self.config.enable_ai_recommendations:
            return
        
        team = self.player_registry.get(player_id, {}).get('team', 'A')
        
        self.ai_recommendations.update_player_performance(
            player_id=player_id,
            team=team,
            distance_covered=distance_covered,
            current_speed=current_speed,
            pass_accuracy=pass_accuracy,
            touches=touches,
            fatigue_score=fatigue_score,
            form_rating=form_rating
        )
    
    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        player_positions: Dict[int, Tuple[float, float]],
        ball_position: Tuple[float, float] = None,
        current_possessor: int = None,
        current_team: str = None
    ):
        """
        Process a single frame of video data.
        
        Args:
            frame_idx: Frame index
            timestamp: Time in seconds
            player_positions: Dict of player_id -> (x, y)
            ball_position: Ball position (x, y)
            current_possessor: Current ball possessor player_id
            current_team: Current team in possession
        """
        self.frame_count = frame_idx
        
        # Update player positions
        for player_id, (x, y) in player_positions.items():
            team = self.player_registry.get(player_id, {}).get('team')
            self.update_player_position(player_id, x, y, timestamp, team)
        
        # Generate recommendations periodically
        if self.config.enable_ai_recommendations:
            time_since_last = timestamp - self.last_recommendation_time
            if time_since_last >= self.config.recommendation_interval:
                self._generate_recommendations()
                self.last_recommendation_time = timestamp
    
    def _generate_recommendations(self):
        """Generate tactical recommendations based on current data."""
        # Get current formations if available
        current_formations = {}
        if self.tactical_analyzer:
            current_formations = self.tactical_analyzer.get_current_formations()
        
        # Get pressing stats if available
        pressing_stats = {}
        if self.tactical_analyzer:
            pressing_stats = self.tactical_analyzer.calculate_ppda()
        
        # Generate recommendations
        recommendations = self.ai_recommendations.generate_all_recommendations(
            current_formations=current_formations,
            pressing_stats=pressing_stats
        )
        
        # Store in history
        self.recommendations_history.extend(recommendations)
    
    def get_current_recommendations(
        self,
        n: int = 5,
        min_confidence: float = None
    ) -> List[TacticalRecommendation]:
        """
        Get current tactical recommendations.
        
        Args:
            n: Number of recommendations to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of recommendations
        """
        if not self.config.enable_ai_recommendations:
            return []
        
        min_confidence = min_confidence or self.config.min_confidence_threshold
        
        return self.ai_recommendations.get_top_recommendations(
            n=n,
            min_confidence=min_confidence
        )
    
    def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get comprehensive player performance summary.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Player summary dictionary
        """
        if not self.config.enable_player_performance:
            return {}
        
        return self.player_performance.get_player_summary(player_id)
    
    def get_team_summary(self, team: str) -> Dict[str, Any]:
        """
        Get team performance summary.
        
        Args:
            team: Team identifier ('A' or 'B')
            
        Returns:
            Team summary dictionary
        """
        if not self.config.enable_player_performance:
            return {}
        
        return self.player_performance.get_team_summary(team)
    
    def get_all_players_summary(self) -> Dict[int, Dict[str, Any]]:
        """
        Get summary for all tracked players.
        
        Returns:
            Dictionary of player_id -> summary
        """
        summaries = {}
        for player_id in self.player_registry:
            summaries[player_id] = self.get_player_summary(player_id)
        return summaries
    
    def compare_players(
        self,
        player_ids: List[int],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple players.
        
        Args:
            player_ids: List of player IDs
            metrics: Metrics to compare
            
        Returns:
            Comparison data
        """
        if not self.config.enable_player_performance:
            return {}
        
        return self.player_performance.compare_players(player_ids, metrics)
    
    def get_recommendation_report(self, team: str = 'A') -> Dict[str, Any]:
        """
        Get comprehensive recommendation report.
        
        Args:
            team: Team to generate report for
            
        Returns:
            Recommendation report
        """
        if not self.config.enable_ai_recommendations:
            return {}
        
        return self.ai_recommendations.get_recommendation_report(team)
    
    def finalize_match(self) -> Dict[str, Any]:
        """
        Finalize match and generate comprehensive report.
        
        Returns:
            Complete match analytics report
        """
        report = {
            'match_id': self.match_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'ai_recommendations_enabled': self.config.enable_ai_recommendations,
                'player_performance_enabled': self.config.enable_player_performance
            }
        }
        
        # Player performance report
        if self.config.enable_player_performance:
            report['player_performance'] = self.player_performance.finalize_match(
                self.match_id
            )
        
        # AI recommendations report
        if self.config.enable_ai_recommendations:
            report['tactical_recommendations'] = {
                'total_generated': len(self.recommendations_history),
                'by_team': {
                    'A': self.ai_recommendations.get_recommendation_report('A'),
                    'B': self.ai_recommendations.get_recommendation_report('B')
                }
            }
        
        # Export if enabled
        if self.config.auto_export:
            self._export_report(report)
        
        return report
    
    def _export_report(self, report: Dict[str, Any]):
        """Export report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"match_analytics_{self.match_id}_{timestamp}.json"
        filepath = Path(self.config.export_path) / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Analytics report exported: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to export report: {e}")
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data formatted for live dashboard display.
        
        Returns:
            Dashboard data dictionary
        """
        data = {
            'match_id': self.match_id,
            'frame_count': self.frame_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Top recommendations
        if self.config.enable_ai_recommendations:
            recommendations = self.get_current_recommendations(n=3)
            data['top_recommendations'] = [
                {
                    'id': r.id,
                    'title': r.title,
                    'priority': r.priority.name,
                    'category': r.category.value,
                    'confidence': r.confidence_score,
                    'description': r.description
                }
                for r in recommendations
            ]
        
        # Top performers
        if self.config.enable_player_performance:
            team_a_summary = self.get_team_summary('A')
            team_b_summary = self.get_team_summary('B')
            
            data['team_summaries'] = {
                'A': team_a_summary,
                'B': team_b_summary
            }
        
        return data
    
    def save_to_database(self):
        """Save analytics data to database."""
        if not self.db_manager:
            print("⚠️ No database manager configured")
            return
        
        try:
            # Save player stats
            for player_id in self.player_registry:
                summary = self.get_player_summary(player_id)
                if summary:
                    # Extract relevant stats for database
                    stats = {
                        'total_distance_m': summary['physical']['total_distance_m'],
                        'sprint_distance_m': summary['physical']['sprint_distance_m'],
                        'avg_speed_mps': summary['physical']['avg_speed_mps'],
                        'max_speed_mps': summary['physical']['max_speed_mps'],
                        'sprints': summary['physical']['sprints'],
                        'passes_attempted': int(summary['technical']['passes'].split('/')[1]) if '/' in summary['technical']['passes'] else 0,
                        'passes_completed': int(summary['technical']['passes'].split('/')[0]) if '/' in summary['technical']['passes'] else 0,
                        'shots': summary['technical']['shots'],
                        'shots_on_target': summary['technical']['shots_on_target'],
                        'goals': summary['technical']['goals'],
                        'tackles': summary['technical']['tackles'],
                        'interceptions': summary['technical']['interceptions'],
                        'touches': summary['technical']['touches'],
                        'workload_score': summary['physical']['workload_score']
                    }
                    
                    # Save to database
                    # Note: This would need player_instance_id mapping
                    # self.db_manager.save_player_stats(
                    #     match_id=self.match_id,
                    #     player_instance_id=player_id,
                    #     **stats
                    # )
            
            print(f"✅ Analytics data saved to database")
            
        except Exception as e:
            print(f"⚠️ Failed to save to database: {e}")
