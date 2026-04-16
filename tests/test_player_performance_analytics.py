"""
Tests for Player Performance Analytics Module

This module tests the player performance tracking and analysis system.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.player_performance_analytics import (
    PlayerPerformanceAnalytics,
    PhysicalMetrics,
    TechnicalMetrics,
    TacticalMetrics,
    PerformanceRating,
    PlayerMatchData,
    PerformanceMetric,
    PositionCategory
)


class TestPlayerPerformanceAnalytics(unittest.TestCase):
    """Test cases for Player Performance Analytics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analytics = PlayerPerformanceAnalytics()
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(len(self.analytics.current_match_data), 0)
        self.assertEqual(len(self.analytics.player_history), 0)
    
    def test_register_player(self):
        """Test player registration."""
        self.analytics.register_player(
            player_id=1,
            team='A',
            name="Test Player",
            jersey_number=10,
            position="MID",
            match_id="test_match"
        )
        
        self.assertIn(1, self.analytics.current_match_data)
        player_data = self.analytics.current_match_data[1]
        self.assertEqual(player_data.name, "Test Player")
        self.assertEqual(player_data.team, 'A')
        self.assertEqual(player_data.jersey_number, 10)
        self.assertEqual(player_data.position, "MID")
    
    def test_update_position(self):
        """Test position updates."""
        self.analytics.register_player(1, 'A')
        
        # Update position in attacking third
        self.analytics.update_position(1, 0.8, 0.5, 10.0, 'A')
        
        player_data = self.analytics.current_match_data[1]
        self.assertGreater(player_data.tactical.time_in_attacking_third, 0)
        
        # Update position in defensive third
        self.analytics.update_position(1, 0.2, 0.5, 20.0, 'A')
        self.assertGreater(player_data.tactical.time_in_defensive_third, 0)
    
    def test_update_speed(self):
        """Test speed updates."""
        self.analytics.register_player(1, 'A')
        
        # Update with sprint speed
        self.analytics.update_speed(1, 8.0, 10.0)
        
        physical = self.analytics.current_match_data[1].physical
        self.assertEqual(physical.max_speed_mps, 8.0)
        self.assertGreater(physical.total_distance_m, 0)
        self.assertGreater(physical.sprint_distance_m, 0)
    
    def test_record_event(self):
        """Test event recording."""
        self.analytics.register_player(1, 'A')
        
        # Record successful pass
        self.analytics.record_event(1, 'pass', success=True)
        technical = self.analytics.current_match_data[1].technical
        self.assertEqual(technical.passes_attempted, 1)
        self.assertEqual(technical.passes_completed, 1)
        self.assertEqual(technical.pass_accuracy, 100.0)
        
        # Record failed pass
        self.analytics.record_event(1, 'pass', success=False)
        self.assertEqual(technical.passes_attempted, 2)
        self.assertEqual(technical.passes_completed, 1)
        self.assertEqual(technical.pass_accuracy, 50.0)
    
    def test_shot_events(self):
        """Test shot event recording."""
        self.analytics.register_player(1, 'A')
        
        # Record shot on target
        self.analytics.record_event(1, 'shot', success=True)
        technical = self.analytics.current_match_data[1].technical
        self.assertEqual(technical.shots, 1)
        self.assertEqual(technical.shots_on_target, 1)
        
        # Record goal
        self.analytics.record_event(1, 'goal', success=True)
        self.assertEqual(technical.goals, 1)
    
    def test_calculate_physical_metrics(self):
        """Test physical metrics calculation."""
        self.analytics.register_player(1, 'A')
        
        # Add some speed data
        for i in range(10):
            speed = 5.0 + i * 0.5  # Increasing speed
            self.analytics.update_speed(1, speed, float(i))
        
        physical = self.analytics.calculate_physical_metrics(1)
        
        self.assertGreater(physical.total_distance_m, 0)
        self.assertGreater(physical.avg_speed_mps, 0)
        self.assertGreater(physical.workload_score, 0)
    
    def test_calculate_tactical_metrics(self):
        """Test tactical metrics calculation."""
        self.analytics.register_player(1, 'A')
        
        # Add position data
        self.analytics.update_position(1, 0.5, 0.5, 10.0, 'A')
        self.analytics.update_position(1, 0.6, 0.4, 20.0, 'A')
        self.analytics.update_position(1, 0.7, 0.6, 30.0, 'A')
        
        tactical = self.analytics.calculate_tactical_metrics(1)
        
        self.assertGreater(tactical.avg_position_x, 0)
        self.assertGreater(tactical.avg_position_y, 0)
    
    def test_calculate_performance_rating(self):
        """Test performance rating calculation."""
        self.analytics.register_player(1, 'A', position="MID")
        
        # Add some data
        self.analytics.update_speed(1, 6.0, 10.0)
        self.analytics.record_event(1, 'pass', success=True)
        self.analytics.record_event(1, 'pass', success=True)
        self.analytics.record_event(1, 'pass', success=False)
        
        rating = self.analytics.calculate_performance_rating(1)
        
        self.assertGreater(rating.overall, 0)
        self.assertLessEqual(rating.overall, 10)
        self.assertGreaterEqual(rating.physical, 0)
        self.assertGreaterEqual(rating.technical, 0)
    
    def test_get_player_summary(self):
        """Test player summary generation."""
        self.analytics.register_player(
            player_id=1,
            team='A',
            name="Test Player",
            jersey_number=10,
            position="MID"
        )
        
        # Add some data
        self.analytics.update_speed(1, 6.0, 10.0)
        self.analytics.record_event(1, 'pass', success=True)
        
        summary = self.analytics.get_player_summary(1)
        
        self.assertIn('player_info', summary)
        self.assertIn('physical', summary)
        self.assertIn('technical', summary)
        self.assertIn('tactical', summary)
        self.assertIn('rating', summary)
        
        self.assertEqual(summary['player_info']['name'], "Test Player")
    
    def test_get_team_summary(self):
        """Test team summary generation."""
        # Register multiple players for team A
        for i in range(1, 4):
            self.analytics.register_player(i, 'A', name=f"Player {i}")
            self.analytics.update_speed(i, 6.0 + i, 10.0)
            self.analytics.record_event(i, 'pass', success=True)
        
        summary = self.analytics.get_team_summary('A')
        
        self.assertEqual(summary['team'], 'A')
        self.assertEqual(summary['players_tracked'], 3)
        self.assertIn('aggregate_stats', summary)
        self.assertIn('average_rating', summary)
        self.assertIn('top_performers', summary)
    
    def test_compare_players(self):
        """Test player comparison."""
        # Register and add data for players
        for i in range(1, 4):
            self.analytics.register_player(i, 'A', name=f"Player {i}")
            self.analytics.update_speed(i, 5.0 + i, 10.0)
            self.analytics.record_event(i, 'pass', success=True)
            self.analytics.record_event(i, 'shot', success=True)
        
        comparison = self.analytics.compare_players([1, 2, 3])
        
        self.assertEqual(len(comparison['players']), 3)
        self.assertIn('rankings', comparison)
    
    def test_finalize_match(self):
        """Test match finalization."""
        self.analytics.register_player(1, 'A', name="Player 1")
        self.analytics.update_speed(1, 6.0, 10.0)
        
        report = self.analytics.finalize_match("test_match")
        
        self.assertEqual(report['match_id'], "test_match")
        self.assertIn('players', report)
        self.assertIn('team_summaries', report)
        
        # Check that data was stored in history
        self.assertIn(1, self.analytics.player_history)
    
    def test_get_player_form_trend(self):
        """Test form trend calculation."""
        # Simulate multiple matches
        for match_rating in [6.5, 7.0, 6.8, 7.2, 7.5]:
            self.analytics.form_history[1].append(match_rating)
        
        trend = self.analytics.get_player_form_trend(1, num_matches=5)
        
        self.assertEqual(trend['matches_analyzed'], 5)
        self.assertEqual(trend['current_form'], 7.5)
        self.assertIn('trend', trend)
        self.assertIn('consistency', trend)
    
    def test_reset(self):
        """Test reset functionality."""
        self.analytics.register_player(1, 'A')
        self.analytics.update_speed(1, 6.0, 10.0)
        
        self.analytics.reset()
        
        self.assertEqual(len(self.analytics.current_match_data), 0)
        self.assertEqual(len(self.analytics.player_positions), 0)
    
    def test_invalid_player_operations(self):
        """Test operations on non-existent players."""
        # Should not raise errors
        self.analytics.update_position(999, 0.5, 0.5, 10.0, 'A')
        self.analytics.update_speed(999, 6.0, 10.0)
        self.analytics.record_event(999, 'pass', success=True)
        
        # Player 999 should now be auto-registered and have data
        summary = self.analytics.get_player_summary(999)
        self.assertIn('player_info', summary)
        self.assertEqual(summary['player_info']['id'], 999)


class TestPhysicalMetrics(unittest.TestCase):
    """Test cases for PhysicalMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        metrics = PhysicalMetrics()
        
        self.assertEqual(metrics.total_distance_m, 0.0)
        self.assertEqual(metrics.max_speed_mps, 0.0)
        self.assertEqual(metrics.sprints, 0)
        self.assertEqual(metrics.workload_score, 0.0)


class TestTechnicalMetrics(unittest.TestCase):
    """Test cases for TechnicalMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        metrics = TechnicalMetrics()
        
        self.assertEqual(metrics.passes_attempted, 0)
        self.assertEqual(metrics.pass_accuracy, 0.0)
        self.assertEqual(metrics.shots, 0)
        self.assertEqual(metrics.goals, 0)


class TestTacticalMetrics(unittest.TestCase):
    """Test cases for TacticalMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        metrics = TacticalMetrics()
        
        self.assertEqual(metrics.avg_position_x, 0.5)
        self.assertEqual(metrics.avg_position_y, 0.5)
        self.assertEqual(metrics.pressing_actions, 0)


class TestPerformanceRating(unittest.TestCase):
    """Test cases for PerformanceRating dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        rating = PerformanceRating()
        
        self.assertEqual(rating.overall, 5.0)
        self.assertEqual(rating.physical, 5.0)
        self.assertEqual(rating.technical, 5.0)
        self.assertEqual(rating.tactical, 5.0)


class TestPlayerMatchData(unittest.TestCase):
    """Test cases for PlayerMatchData dataclass."""
    
    def test_initialization(self):
        """Test initialization."""
        from datetime import datetime
        
        data = PlayerMatchData(
            player_id=1,
            team='A',
            match_id="test",
            timestamp=datetime.now()
        )
        
        self.assertEqual(data.player_id, 1)
        self.assertEqual(data.team, 'A')
        self.assertIsInstance(data.physical, PhysicalMetrics)
        self.assertIsInstance(data.technical, TechnicalMetrics)
        self.assertIsInstance(data.tactical, TacticalMetrics)
        self.assertIsInstance(data.rating, PerformanceRating)


if __name__ == '__main__':
    unittest.main()
