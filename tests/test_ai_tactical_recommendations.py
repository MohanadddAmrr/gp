"""
Tests for AI Tactical Recommendations Module

This module tests the AI-powered tactical recommendation system.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.ai_tactical_recommendations import (
    AITacticalRecommendations,
    TacticalRecommendation,
    RecommendationPriority,
    RecommendationCategory,
    MatchContext,
    PlayerPerformanceSnapshot
)


class TestAITacticalRecommendations(unittest.TestCase):
    """Test cases for AI Tactical Recommendations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ai_rec = AITacticalRecommendations()
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(len(self.ai_rec.recommendations), 0)
        self.assertEqual(self.ai_rec.recommendation_counter, 0)
        self.assertIsInstance(self.ai_rec.match_context, MatchContext)
    
    def test_update_match_context(self):
        """Test match context updates."""
        self.ai_rec.update_match_context(
            score_a=2,
            score_b=1,
            possession_a=55.0,
            possession_b=45.0,
            match_minute=35.5,
            period="first_half"
        )
        
        self.assertEqual(self.ai_rec.match_context.score_a, 2)
        self.assertEqual(self.ai_rec.match_context.score_b, 1)
        self.assertEqual(self.ai_rec.match_context.leading_team, 'A')
        self.assertEqual(self.ai_rec.match_context.goal_difference, 1)
    
    def test_update_player_performance(self):
        """Test player performance updates."""
        self.ai_rec.update_player_performance(
            player_id=1,
            team='A',
            distance_covered=5000.0,
            current_speed=5.5,
            pass_accuracy=85.0,
            touches=25,
            fatigue_score=45.0,
            form_rating=75.0
        )
        
        self.assertIn(1, self.ai_rec.player_snapshots)
        snapshot = self.ai_rec.player_snapshots[1]
        self.assertEqual(snapshot.distance_covered, 5000.0)
        self.assertEqual(snapshot.pass_accuracy, 85.0)
    
    def test_generate_formation_recommendations(self):
        """Test formation recommendation generation."""
        # Set up context - losing team should get attacking formation recommendation
        self.ai_rec.update_match_context(
            score_a=0,
            score_b=2,
            possession_a=45.0,
            possession_b=55.0,
            match_minute=65.0
        )
        
        current_formations = {'A': '4-3-3', 'B': '4-2-3-1'}
        recommendations = self.ai_rec.generate_formation_recommendations(current_formations)
        
        # Should generate at least one recommendation
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check recommendation structure
        rec = recommendations[0]
        self.assertIsInstance(rec, TacticalRecommendation)
        self.assertIsNotNone(rec.id)
        self.assertIsNotNone(rec.title)
        self.assertIsNotNone(rec.description)
    
    def test_generate_pressing_recommendations(self):
        """Test pressing recommendation generation."""
        pressing_stats = {
            'team_a': {'ppda': 15.5, 'interpretation': 'Low Press'},
            'team_b': {'ppda': 8.2, 'interpretation': 'High Press'}
        }
        
        recommendations = self.ai_rec.generate_pressing_recommendations(pressing_stats)
        
        # Should generate recommendations
        self.assertIsInstance(recommendations, list)
    
    def test_generate_substitution_recommendations(self):
        """Test substitution recommendation generation."""
        # Add fatigued player
        self.ai_rec.update_player_performance(
            player_id=1,
            team='A',
            fatigue_score=85.0  # High fatigue
        )
        
        # Add underperforming player
        self.ai_rec.update_player_performance(
            player_id=2,
            team='A',
            form_rating=30.0,  # Low form
            touches=15
        )
        
        recommendations = self.ai_rec.generate_substitution_recommendations(
            fatigue_threshold=70.0
        )
        
        # Should recommend substitution for fatigued player
        fatigued_recs = [r for r in recommendations if 'fatigue' in r.description.lower()]
        self.assertGreaterEqual(len(fatigued_recs), 1)
    
    def test_opponent_weakness_analysis(self):
        """Test opponent weakness analysis."""
        # This would need advanced analytics mock
        weaknesses = self.ai_rec.analyze_opponent_weaknesses('B')
        
        self.assertIn('defensive_gaps', weaknesses)
        self.assertIn('pressing_triggers', weaknesses)
        self.assertIn('transition_vulnerabilities', weaknesses)
    
    def test_recommendation_priority_ordering(self):
        """Test that recommendations are properly ordered by priority."""
        # Create recommendations with different priorities
        from dataclasses import replace
        
        rec_high = TacticalRecommendation(
            id="rec1",
            timestamp=0.0,
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.ATTACKING,
            title="High Priority",
            description="Test",
            reasoning="Test",
            expected_outcome="Test",
            confidence_score=0.8
        )
        
        rec_low = TacticalRecommendation(
            id="rec2",
            timestamp=0.0,
            priority=RecommendationPriority.LOW,
            category=RecommendationCategory.DEFENSIVE,
            title="Low Priority",
            description="Test",
            reasoning="Test",
            expected_outcome="Test",
            confidence_score=0.8
        )
        
        self.ai_rec.recommendations = [rec_low, rec_high]
        
        top_recs = self.ai_rec.get_top_recommendations(n=2)
        
        # High priority should come first
        self.assertEqual(top_recs[0].priority, RecommendationPriority.HIGH)
        self.assertEqual(top_recs[1].priority, RecommendationPriority.LOW)
    
    def test_confidence_filtering(self):
        """Test recommendation filtering by confidence."""
        rec_high_conf = TacticalRecommendation(
            id="rec1",
            timestamp=0.0,
            priority=RecommendationPriority.MEDIUM,
            category=RecommendationCategory.ATTACKING,
            title="High Confidence",
            description="Test",
            reasoning="Test",
            expected_outcome="Test",
            confidence_score=0.9
        )
        
        rec_low_conf = TacticalRecommendation(
            id="rec2",
            timestamp=0.0,
            priority=RecommendationPriority.MEDIUM,
            category=RecommendationCategory.DEFENSIVE,
            title="Low Confidence",
            description="Test",
            reasoning="Test",
            expected_outcome="Test",
            confidence_score=0.3
        )
        
        self.ai_rec.recommendations = [rec_high_conf, rec_low_conf]
        
        filtered = self.ai_rec.get_top_recommendations(
            n=10,
            min_confidence=0.5
        )
        
        # Only high confidence should remain
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].id, "rec1")
    
    def test_recommendation_report(self):
        """Test recommendation report generation."""
        # Add some recommendations
        self.ai_rec.update_match_context(
            score_a=1,
            score_b=0,
            possession_a=50.0,
            possession_b=50.0,
            match_minute=30.0
        )
        
        current_formations = {'A': '4-3-3', 'B': '4-3-3'}
        self.ai_rec.generate_formation_recommendations(current_formations)
        
        report = self.ai_rec.get_recommendation_report('A')
        
        self.assertIn('team', report)
        self.assertIn('total_recommendations', report)
        self.assertIn('by_category', report)
        self.assertIn('top_actionable', report)
        self.assertIn('match_context', report)
    
    def test_reset(self):
        """Test reset functionality."""
        # Add some data
        self.ai_rec.update_match_context(1, 0, 50.0, 50.0, 30.0)
        self.ai_rec.recommendations.append(
            TacticalRecommendation(
                id="test",
                timestamp=0.0,
                priority=RecommendationPriority.LOW,
                category=RecommendationCategory.ATTACKING,
                title="Test",
                description="Test",
                reasoning="Test",
                expected_outcome="Test",
                confidence_score=0.5
            )
        )
        
        # Reset
        self.ai_rec.reset()
        
        self.assertEqual(len(self.ai_rec.recommendations), 0)
        self.assertEqual(self.ai_rec.recommendation_counter, 0)


class TestMatchContext(unittest.TestCase):
    """Test cases for MatchContext dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        context = MatchContext(timestamp=0.0)
        
        self.assertEqual(context.score_a, 0)
        self.assertEqual(context.score_b, 0)
        self.assertEqual(context.possession_a, 50.0)
        self.assertEqual(context.possession_b, 50.0)
        self.assertEqual(context.match_minute, 0.0)


class TestPlayerPerformanceSnapshot(unittest.TestCase):
    """Test cases for PlayerPerformanceSnapshot dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        snapshot = PlayerPerformanceSnapshot(player_id=1, team='A')
        
        self.assertEqual(snapshot.distance_covered, 0.0)
        self.assertEqual(snapshot.pass_accuracy, 0.0)
        self.assertEqual(snapshot.form_rating, 50.0)


if __name__ == '__main__':
    unittest.main()
