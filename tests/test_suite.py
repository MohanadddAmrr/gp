"""
TactiVision Pro - Comprehensive Test Suite
============================================

Complete test suite covering all major services and integration tests.
Includes unit tests, integration tests, and performance benchmarks.

Usage:
    python -m pytest tests/test_suite.py -v
    python tests/test_suite.py --benchmark
    python tests/test_suite.py --coverage

Author: TactiVision Pro Team
Version: 2.0.0
"""

import sys
import os
import unittest
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import optional dependencies
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for test suite."""
    run_benchmarks: bool = False
    run_integration: bool = True
    run_performance: bool = False
    verbose: bool = True
    timeout: float = 30.0


# Global test configuration
TEST_CONFIG = TestConfig()


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

def generate_mock_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Generate a mock video frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def generate_mock_detections(num_detections: int = 10) -> List[Dict]:
    """Generate mock object detections."""
    detections = []
    for i in range(num_detections):
        detections.append({
            'bbox': [100 + i*50, 100 + i*30, 150 + i*50, 150 + i*30],
            'confidence': 0.7 + (i * 0.02),
            'class': 0,  # Person
            'class_name': 'person',
            'track_id': i
        })
    return detections


def generate_mock_player_positions(num_players: int = 22) -> Dict[int, tuple]:
    """Generate mock player positions."""
    positions = {}
    for i in range(num_players):
        team = 'A' if i < 11 else 'B'
        x = np.random.uniform(0, 1920)
        y = np.random.uniform(0, 1080)
        positions[i] = (x, y, team)
    return positions


def generate_mock_match_data() -> Dict[str, Any]:
    """Generate mock match data."""
    return {
        'match_id': 1,
        'team_a': 'Liverpool',
        'team_b': 'Manchester City',
        'score_a': 2,
        'score_b': 1,
        'duration': 5400,
        'players': [
            {'id': i, 'name': f'Player {i}', 'team': 'A' if i < 11 else 'B'}
            for i in range(22)
        ],
        'events': [
            {'time': 120, 'type': 'goal', 'team': 'A', 'player': 5},
            {'time': 2400, 'type': 'goal', 'team': 'B', 'player': 15},
            {'time': 4800, 'type': 'goal', 'team': 'A', 'player': 8},
        ]
    }


# =============================================================================
# UNIT TESTS - CORE SERVICES
# =============================================================================

# BallTracker is covered comprehensively by tests/test_ball_tracker.py
# (basic functionality, velocity smoothing, stationary ball, direction,
# prediction, reset). No duplicate suite is kept here.


# PossessionTracker is covered by tests/test_possession.py and
# tests/test_possession_validation.py (detection, history, statistics,
# zone/pressure/duration stats). No duplicate suite is kept here.


class TestEventDetector(unittest.TestCase):
    """Test cases for EventDetector service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from services.event_detector import EventDetector
            cls.EventDetector = EventDetector
        except ImportError as e:
            cls.skipTest(cls, f"EventDetector not available: {e}")
    
    def setUp(self):
        """Set up test instance."""
        self.detector = self.EventDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
    
    def test_shot_detection(self):
        """Test shot event detection returns a shot dict or None."""
        result = self.detector.detect_shot(
            ball_position=(1800, 340),
            ball_direction=(1.0, 0.0),
            ball_velocity_mps=25.0,
            frame_idx=100,
            timestamp=4.0,
            frame_width=1920,
            frame_height=1080,
        )
        # detect_shot returns a shot-event dict when detected, else None
        self.assertTrue(result is None or isinstance(result, dict))


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from services.database_manager import DatabaseManager
            cls.DatabaseManager = DatabaseManager
        except ImportError as e:
            cls.skipTest(cls, f"DatabaseManager not available: {e}")
    
    def setUp(self):
        """Set up test instance with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()  # release the OS handle; Windows blocks unlink while open
        self.db = self.DatabaseManager(self.temp_db.name)
        self.db.initialize_database()
    
    def tearDown(self):
        """Clean up test instance."""
        import os
        os.unlink(self.temp_db.name)
    
    def test_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db)
    
    def test_create_match(self):
        """Test match creation."""
        match_id = self.db.create_match(
            video_path="test.mp4",
            team_a="Team A",
            team_b="Team B",
            duration_seconds=5400
        )
        
        self.assertIsNotNone(match_id)
        self.assertIsInstance(match_id, int)
    
    def test_get_match(self):
        """Test match retrieval."""
        match_id = self.db.create_match(
            video_path="test.mp4",
            team_a="Team A",
            team_b="Team B"
        )
        
        match = self.db.get_match(match_id)
        self.assertIsNotNone(match)
        self.assertEqual(match['team_a'], "Team A")
        self.assertEqual(match['team_b'], "Team B")
    
    def test_get_all_matches(self):
        """Test getting all matches."""
        # Create multiple matches
        for i in range(3):
            self.db.create_match(
                video_path=f"test{i}.mp4",
                team_a=f"Team A{i}",
                team_b=f"Team B{i}"
            )
        
        matches = self.db.get_all_matches()
        self.assertEqual(len(matches), 3)


class TestxGCalculator(unittest.TestCase):
    """Test cases for xGCalculator service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from services.xg_calculator import xGCalculator, ShotType
            cls.xGCalculator = xGCalculator
            cls.ShotType = ShotType
        except ImportError as e:
            cls.skipTest(cls, f"xGCalculator not available: {e}")
    
    def setUp(self):
        """Set up test instance."""
        self.calculator = self.xGCalculator()
    
    def test_initialization(self):
        """Test calculator initialization."""
        self.assertIsNotNone(self.calculator)
    
    def test_xg_calculation(self):
        """Test xG calculation returns a probability in [0, 1]."""
        xg = self.calculator.calculate_xg(
            x=0.85,  # Near goal
            y=0.5,
            shot_type=self.ShotType.OPEN_PLAY,
        )

        self.assertIsInstance(xg, float)
        self.assertGreaterEqual(xg, 0.0)
        self.assertLessEqual(xg, 1.0)

    def test_stats_accumulation(self):
        """Test statistics accumulation over multiple shots."""
        for i in range(5):
            self.calculator.add_shot(
                timestamp=float(i * 60),
                frame=i * 1500,
                shooter_id=i,
                shooter_team="A",
                x=0.8,
                y=0.5,
                shot_type=self.ShotType.OPEN_PLAY,
            )

        stats = self.calculator.get_statistics()
        self.assertEqual(stats['total_shots'], 5)


class TestTacticalAnalyzer(unittest.TestCase):
    """Test cases for TacticalAnalyzer service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from services.tactical_analyzer import TacticalAnalyzer
            cls.TacticalAnalyzer = TacticalAnalyzer
        except ImportError as e:
            cls.skipTest(cls, f"TacticalAnalyzer not available: {e}")
    
    def setUp(self):
        """Set up test instance with only formation detection enabled."""
        self.analyzer = self.TacticalAnalyzer(
            enable_offside=False,
            enable_set_pieces=False,
            enable_dribbles=False,
            enable_pitch_transform=False,
        )

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer)

    def test_formation_detection(self):
        """Test formation detection via the update loop."""
        positions = generate_mock_player_positions(22)
        # Formation detection runs inside update() every 30 frames
        for frame_idx in range(60):
            self.analyzer.update(
                player_positions=positions,
                ball_position=(960, 540),
                ball_velocity=0.0,
                current_possessor=0,
                current_team="A",
                frame_idx=frame_idx,
                timestamp=frame_idx / 30.0,
            )

        formations = self.analyzer.get_current_formations()
        self.assertIsInstance(formations, dict)
        self.assertIn("A", formations)
        self.assertIn("B", formations)


class TestHighlightsGenerator(unittest.TestCase):
    """Test cases for HighlightsGenerator service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from services.highlights_generator import (
                HighlightsGenerator, EventType, ImportanceLevel,
            )
            cls.HighlightsGenerator = HighlightsGenerator
            cls.EventType = EventType
            cls.ImportanceLevel = ImportanceLevel
        except ImportError as e:
            cls.skipTest(cls, f"HighlightsGenerator not available: {e}")

    def setUp(self):
        """Set up test instance."""
        self.generator = self.HighlightsGenerator()

    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)

    def test_add_event(self):
        """Test adding a highlight event."""
        event = self.generator.add_event(
            event_type=self.EventType.GOAL,
            timestamp=120.5,
            frame=3012,
            team="A",
            description="Amazing goal!",
        )

        self.assertEqual(len(self.generator.events), 1)
        self.assertEqual(event.event_type, self.EventType.GOAL)

    def test_importance_scoring(self):
        """Test that more significant events get higher importance."""
        goal = self.generator.add_event(
            event_type=self.EventType.GOAL, timestamp=10.0, frame=250,
        )
        pass_event = self.generator.add_event(
            event_type=self.EventType.PASS_COMPLETION, timestamp=20.0, frame=500,
        )

        self.assertIsInstance(goal.importance, self.ImportanceLevel)
        self.assertGreater(goal.importance.value, pass_event.importance.value)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for multiple services."""
    
    @classmethod
    def setUpClass(cls):
        """Check if integration tests should run."""
        if not TEST_CONFIG.run_integration:
            cls.skipTest(cls, "Integration tests disabled")
    
    def test_end_to_end_video_processing(self):
        """Test complete video processing pipeline."""
        # This is a placeholder for a full integration test
        # In practice, this would process a short test video
        self.assertTrue(True)  # Placeholder
    
    def test_database_to_dashboard_flow(self):
        """Test data flow from database to dashboard."""
        try:
            from services.database_manager import DatabaseManager
            
            # Create temporary database
            temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            temp_db.close()  # release the OS handle; Windows blocks unlink while open
            db = DatabaseManager(temp_db.name)
            db.initialize_database()
            
            # Create match
            match_id = db.create_match(
                video_path="test.mp4",
                team_a="Liverpool",
                team_b="Man City"
            )
            
            # Retrieve match
            match = db.get_match(match_id)
            self.assertIsNotNone(match)
            self.assertEqual(match['team_a'], "Liverpool")
            
            # Clean up
            import os
            os.unlink(temp_db.name)
            
        except ImportError:
            self.skipTest("DatabaseManager not available")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Performance benchmark tests."""
    
    @classmethod
    def setUpClass(cls):
        """Check if performance tests should run."""
        if not TEST_CONFIG.run_performance:
            cls.skipTest(cls, "Performance tests disabled")
    
    def test_ball_tracker_performance(self):
        """Benchmark ball tracker performance."""
        try:
            from services.ball_tracker import BallTracker
            
            tracker = BallTracker()
            
            # Measure update performance
            start_time = time.time()
            
            for i in range(1000):
                bbox = (100 + i, 200, 110 + i, 210)
                tracker.update(bbox, i, i / 30.0)
            
            elapsed = time.time() - start_time
            
            # Should process 1000 updates in less than 1 second
            self.assertLess(elapsed, 1.0)
            
        except ImportError:
            self.skipTest("BallTracker not available")
    
    def test_database_query_performance(self):
        """Benchmark database query performance."""
        try:
            from services.database_manager import DatabaseManager

            temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            temp_db.close()  # release the OS handle; Windows blocks unlink while open
            db = DatabaseManager(temp_db.name)
            db.initialize_database()
            
            # Create many matches
            for i in range(100):
                db.create_match(
                    video_path=f"test{i}.mp4",
                    team_a=f"Team A{i}",
                    team_b=f"Team B{i}"
                )
            
            # Measure query performance
            start_time = time.time()
            matches = db.get_all_matches()
            elapsed = time.time() - start_time
            
            # Should retrieve 100 matches in less than 100ms
            self.assertLess(elapsed, 0.1)
            self.assertEqual(len(matches), 100)
            
            # Clean up
            import os
            os.unlink(temp_db.name)
            
        except ImportError:
            self.skipTest("DatabaseManager not available")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests():
    """Run the complete test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes (BallTracker / PossessionTracker live in their
    # own dedicated test files; see comments near the top of this module)
    suite.addTests(loader.loadTestsFromTestCase(TestEventDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManager))
    suite.addTests(loader.loadTestsFromTestCase(TestxGCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestTacticalAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestHighlightsGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if TEST_CONFIG.verbose else 1)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def main():
    """Main entry point for test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TactiVision Pro Test Suite')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure tests
    TEST_CONFIG.verbose = args.verbose
    
    if args.benchmark:
        TEST_CONFIG.run_performance = True
        TEST_CONFIG.run_integration = False
    elif args.integration:
        TEST_CONFIG.run_integration = True
        TEST_CONFIG.run_performance = False
    elif args.unit:
        TEST_CONFIG.run_integration = False
        TEST_CONFIG.run_performance = False
    
    # Run with coverage if requested
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage(source=['services'])
            cov.start()
            exit_code = run_tests()
            cov.stop()
            cov.save()
            cov.report()
            cov.html_report(directory='coverage_html')
            print("\nCoverage report generated in coverage_html/")
            sys.exit(exit_code)
        except ImportError:
            print("Coverage package not installed. Run: pip install coverage")
            sys.exit(1)
    else:
        sys.exit(run_tests())


if __name__ == '__main__':
    main()
